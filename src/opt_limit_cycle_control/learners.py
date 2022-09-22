from src.control.utils import dummy_trainloader, weighted_log_likelihood_loss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint


class OptEigManifoldLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, non_integral_task_loss_func: nn.Module, l_period=1.0, alpha_p=1.0, alpha_s=0.0,
                 alpha_mv=0.0, l_task_loss=1.0, l_task_loss_2=1.0, lr=0.001, sensitivity='autograd', opt_strategy=1.0,
                 spatial_dim=1, min_period=0, max_period=None, times=None, u0_init=None, u0_requires_grad=True):
        super().__init__()
        self.model = model
        self.spatial_dim = spatial_dim
        if u0_init is None:
            u0 = 0.0
            if spatial_dim == 2:
                u0 = [0.0, 0.0]
            self.u0 = torch.tensor(u0).view(1, self.spatial_dim).cuda()
        else:
            if not len(u0_init) == self.spatial_dim:
                raise ValueError("u0_init should be a list containing spatial_dim values indicating the initial value")
            else:
                self.u0 = torch.tensor(u0_init).view(1, self.spatial_dim).cuda()

        #self.u0 = torch.randn(1, self.spatial_dim).cuda()
        self.u0.requires_grad = u0_requires_grad
        self.non_integral_task_loss = non_integral_task_loss_func
        self.times = times
        self.num_times = None if self.times is None else self.times.size(0)
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint
        self.l_period = l_period
        self.alpha_p = alpha_p
        self.alpha_s = alpha_s
        self.alpha_mv = alpha_mv
        self.l_task_loss = l_task_loss
        self.l_task_loss_2 = l_task_loss_2
        self.lr = lr
        self.optimizer_strategy = opt_strategy
        self.half_period_index = None # will be defined later on in the create_t_list function
        self.minT = min_period
        self.maxT = max_period
        self.epoch = 0
        self.count = 0
        assert min_period >= 0, "the minimum period should always be larger or equal to 0"
        assert max_period is None or max_period >= 0, "the maximum period should always be larger or equal to 0"
        if self.times is not None:
            assert min_period >= max(times), "the minimum period should be larger or equal to the largest value in " \
                                             "self.times"
            assert max_period is None or max_period >= max(times), "the maximum period should be larger or equal to the " \
                                                                   "largest value in self.times"

    def create_t_list(self, num_points, T):
        if self.times is not None:
            output, inverse_indices = torch.unique(torch.cat([torch.linspace(0, 1, num_points).cuda(), self.times/T[0]]),
                                                   sorted=True, return_inverse=True)
            return output, inverse_indices[-self.num_times:]
        else:
            # output = torch.linspace(0, 1, num_points).cuda()
            output, half_period_index = torch.unique(torch.cat([torch.linspace(0, 1, num_points).cuda(), torch.tensor(0.5).unsqueeze(0).cuda()]),
                                                   sorted=True, return_inverse=True)
            self.half_period_index = half_period_index[-1].item()
            if hasattr(self.non_integral_task_loss, 'half_period_index'):
                self.non_integral_task_loss.half_period_index = self.half_period_index
            return output, [0]


    def forward(self, x):

        with torch.no_grad():
            output, indices = self.create_t_list(100, self.model.f.T)

            if hasattr(self.non_integral_task_loss, 'indices'):
                self.non_integral_task_loss.indices = indices

        return self.odeint(self.model, x, output.cuda(), method='midpoint').squeeze(1)

    def eig_mode_loss(self, init_cond, xT, indices):
        if self.alpha_s + self.alpha_p + self.alpha_mv == 0:
            return 0.0
        elif self.half_period_index is None:
            periodicity_loss = self.periodicity_loss(init_cond, xT)
            avg_variance = self.avg_vec_field_variance(xT)
            return (1.0 + 1.0 / avg_variance) * periodicity_loss
        else:
            periodicity_loss = self.periodicity_loss(init_cond, xT)
            sym_trajectory_loss = self.sym_trajectory_loss(xT, indices)
            middle_vel_loss = self.middle_vel_loss(xT)
            #avg_variance = self.avg_vec_field_variance(xT)
            alpha_sum = self.alpha_p + self.alpha_s + self.alpha_mv
            eig_loss = (self.alpha_p * periodicity_loss + self.alpha_s * sym_trajectory_loss + self.alpha_mv
                        * middle_vel_loss) / alpha_sum
            #return (1.0 + 1.0/avg_variance) * eig_loss
            return eig_loss

    def eigenmode_loss(self, xT):
        return torch.sum(torch.square(xT[self.half_period_index, 2:4]))

    def avg_vec_field_variance(self, xT):
        vector_field = self.model.f(torch.linspace(0, 1, 100).cuda(), xT)[:, 0:self.spatial_dim].view(-1,
                                                                                                      self.spatial_dim)
        return torch.mean(torch.var(vector_field, dim=0))

    def periodicity_loss(self, init_cond, xT):
        return torch.sum(torch.square(init_cond.squeeze(0)[0:self.spatial_dim] - xT[-1, :self.spatial_dim].cuda())) + \
               torch.sum(torch.square(init_cond.squeeze(0)[self.spatial_dim:2*self.spatial_dim] -
                                      xT[-1, self.spatial_dim:2*self.spatial_dim].cuda())) / 100

    def sym_trajectory_loss(self, xT, indices):
        sym_indices = (xT.shape[0] - 1) - indices
        return (torch.sum(torch.square(xT[indices, 0:self.spatial_dim] - xT[sym_indices, 0:self.spatial_dim])) +
                torch.sum(torch.square(xT[indices, self.spatial_dim:] + xT[sym_indices, self.spatial_dim:])) / 100) / indices.numel()

    def middle_vel_loss(self, xT):
        return torch.sum(torch.square(xT[self.half_period_index, self.spatial_dim:]))

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        period = self.model.f.T.data
        period = torch.abs(period)
        period = period.clamp(self.minT, self.maxT)
        self.model.f.T.data = period

    def beta_scheduler(self, epoch, max_epoch, R=0.5, M=4):
        tau = ((epoch) % (max_epoch / M)) / (max_epoch / M)
        if tau <= R:
            beta = 2 * (epoch % (max_epoch / M)) / (max_epoch / M)
        else:
            beta = 1
        return beta

    def _training_step1(self, batch, batch_idx):
        # Solve the ODE forward in time for T seconds
        if self.count % 7 == 0:
            self.epoch += 1
        self.count += 1

        init_cond = torch.cat([self.u0, torch.zeros(1, self.spatial_dim + 1).cuda()], dim=1)
        xTl = self.forward(init_cond)
        xT, l = xTl[:, :-1], xTl[:, -1:]

        # Compute loss
        num_sym_check_instances = 25
        if self.half_period_index is not None:
            indices = torch.randperm(self.half_period_index-1)[:num_sym_check_instances]
        else:
            indices = None
        periodicity_loss = self.l_period * self.eig_mode_loss(init_cond, xT, indices)# self.eigenmode_loss(xT) #self.l_period * self.eig_mode_loss(init_cond, xT, indices)
        integral_task_loss = torch.abs(self.model.f.T[0]) * self.l_task_loss * l[-1]#torch.mean(l)
        non_integral_task_loss = self.l_task_loss_2 * self.non_integral_task_loss(xT)
        beta = 1#self.beta_scheduler(self.epoch, 200)
        print('beta: ', beta)
        print('epoch: ', self.epoch)
        print('periodicity loss', periodicity_loss)
        loss = beta * periodicity_loss + integral_task_loss + non_integral_task_loss
        print('                      ')
        print('                      ')
        print('periodicity loss multiplied by beta', periodicity_loss)
        print('task loss', non_integral_task_loss)
        print('integral loss', integral_task_loss)
        print('                      ')
        # log training data
        self.logger.experiment.log(
            {
                'periodicity loss': periodicity_loss,
                'integral task loss': integral_task_loss,
                'non-integral task loss': non_integral_task_loss,
                'train loss': loss,
                'nfe': self.model.nfe,
                'q_max': xT[:, 0].max(),
                'p_max': xT[:, 1].max(),
                'q_min': xT[:, 0].min(),
                'p_min': xT[:, 1].min(),
                'xT_mean': xT.mean(),
                'xT_std': xT.std()
            }
        )

        self.model.nfe = 0
        return {'loss': loss}

    def training_step(self, batch, batch_idx, optimizer_idx=1):
        if self.optimizer_strategy == 1.0:
            return self._training_step1(batch, batch_idx)
        else:
            print("Not Implemented")
            pass

    def configure_optimizers(self):
        if self.optimizer_strategy == 1.0:
            if self.u0.requires_grad:
                params = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.model.f.T, 'lr': self.lr},
                          {'params': self.u0, 'lr': self.lr}]
            else:
                params = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.model.f.T, 'lr': self.lr}]
            optimizer = torch.optim.Adam(params)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
            return ({"optimizer": optimizer, "lr_scheduler": scheduler, "frequency": 1})
        else:
            print("Not Implemented")
            pass

    def train_dataloader(self):
        return dummy_trainloader()


# Define the Integral Cost Function
class ControlEffort(nn.Module):
    # control effort integral cost
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            if x.shape[1] == 2:
                q = x[:,:1].requires_grad_(True)
            else:
                q = x[:, :2].requires_grad_(True)
            u = self.f._energy_shaping(q)

        #return torch.sum(torch.abs(u), dim=1, keepdim=False) #L1-norm
        return torch.sum(torch.square(u), dim=1, keepdim=False)  # L2-norm


class CloseToActualPositionAtHalfPeriod(nn.Module):
    # Given a time series as input, this cost function measures how close we are to certain destinations at specific
    # pre-specified times
    def __init__(self, dest_angle, l1, l2):
        super().__init__()
        self.dest_angle = dest_angle.reshape(-1, 1).cuda()
        self.l1 = l1
        self.l2 = l2
        self.half_period_index = None

        # Calculating self.dest
        # compute the desired double pendulum position
        x_pos_first_joint = self.l1 * torch.sin(self.dest_angle[0])
        y_pos_first_joint = -self.l1 * torch.cos(self.dest_angle[0])
        x2 = x_pos_first_joint + self.l2 * torch.sin(self.dest_angle[1] + self.dest_angle[0])
        y2 = y_pos_first_joint - self.l2 * torch.cos(self.dest_angle[1] + self.dest_angle[0])

        self.dest = torch.tensor([x2, y2]).reshape(-1, 1).cuda()

    def forward(self, xt):
        # xt[:, 0] for pendulum
        if xt.shape[1] == 2:
            raise NotImplementedError
        else:

            # compute double pendulum position
            x_pos_first_joint = self.l1 * torch.sin(xt[self.half_period_index, 0])
            y_pos_first_joint = -self.l1 * torch.cos(xt[self.half_period_index, 0])
            x2 = x_pos_first_joint + self.l2 * torch.sin(xt[self.half_period_index, 1] + xt[self.half_period_index, 0])
            y2 = y_pos_first_joint - self.l2 * torch.cos(xt[self.half_period_index, 1] + xt[self.half_period_index, 0])

            return (x2 - self.dest[0]) ** 2 + (y2 - self.dest[1]) ** 2

# Define the non-integral cost function
class CloseToPositions(nn.Module):
    # Given a time series as input, this cost function measures how close the average distance is to a set of points
    def __init__(self, dest):
        super().__init__()
        self.dest = dest.cuda()

    def forward(self, xt):
        # xt[:, 0] for pendulum
        if xt.shape[1] == 2:
            return torch.max(torch.min(torch.square(xt[:, 0] - self.dest), dim=1)[0])
        else:
            # return torch.max(torch.min(torch.square(xt[:, 0] - self.dest[:, 0].unsqueeze(1)), dim=1)[0]) + \
            #        torch.max(torch.min(torch.square(xt[:, 1] - self.dest[:, 1].unsqueeze(1)), dim=1)[0])\
            # return torch.max(torch.min(torch.sum(torch.square(xt[:, 0:2].unsqueeze(1) - self.dest), dim=2), dim=0)[0])
            return torch.sum(torch.min(torch.sum(torch.square(xt[:, 0:2].unsqueeze(1) - self.dest), dim=2), dim=0)[0])


class CloseToPositionsAtTime(nn.Module):
    # Given a time series as input, this cost function measures how close we are to certain destinations at specific
    # pre-specified times
    def __init__(self, dest):
        super().__init__()
        self.dest = dest.cuda()
        self.indices = torch.empty(1)

    def forward(self, xt):
        # xt[:, 0] for pendulum
        if xt.shape[1] == 2:
            return torch.sum(torch.square(xt[self.indices, 0].unsqueeze(1) - self.dest))
        else:
            return torch.sum(torch.sum(torch.square(xt[self.indices, 0:2] - self.dest), dim=1))

class CloseToPositionAtHalfPeriod(nn.Module):
    # Given a time series as input, this cost function measures how close we are to certain destinations at specific
    # pre-specified times
    def __init__(self, dest):
        super().__init__()
        self.dest = dest.reshape(-1, 1).cuda()
        self.half_period_index = None

    def forward(self, xt):
        # xt[:, 0] for pendulum
        if xt.shape[1] == 2:
            return torch.sum(torch.square(xt[self.half_period_index, 0].unsqueeze(1) - self.dest)) + \
                   torch.sum(torch.square(xt[self.half_period_index, 1].unsqueeze(1)))
        else:
            return torch.sum(torch.square(xt[self.half_period_index, 0:2].unsqueeze(1) - self.dest)) + \
                   torch.sum(torch.square(xt[self.half_period_index, 2:4]))
            # return torch.exp(torch.sum(torch.square(xt[self.half_period_index, 0:2].unsqueeze(1) - self.dest))) * \
            #        (1 + torch.sum(torch.square(xt[self.half_period_index, 2:4])))

            # pos_dist = torch.sum(torch.square(xt[self.half_period_index, 0:2].unsqueeze(1) - self.dest))
            # vel_zero_dist = torch.sum(torch.square(xt[self.half_period_index, 2:4]))
            # epsilon = 0.025
            # scaling = 10 ** (-4)
            #
            # return pos_dist + torch.sigmoid(-(torch.square(pos_dist) - epsilon) / scaling) * vel_zero_dist

            # return torch.exp(torch.sum(torch.square(xt[self.half_period_index, 0:2].unsqueeze(1) - self.dest))) * \
            #        (1 + torch.sum(torch.square(xt[self.half_period_index, 2:4])))


# def lagrange_multiplier_changer(lam, mult_const, weight_grad_norm, weight_grad_bound, max_val):
#     if weight_grad_norm <= weight_grad_bound:
#         return min(lam*mult_const, max_val)
#     else:
#         return lam


