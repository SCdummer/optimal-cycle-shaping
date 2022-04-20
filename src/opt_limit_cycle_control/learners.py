from src.control.utils import dummy_trainloader, weighted_log_likelihood_loss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint


class OptEigManifoldLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, non_integral_task_loss_func: nn.Module, l_period=1.0, l_task_loss=1.0,
                 l_task_loss_2=1.0, lr=0.001, sensitivity='autograd', opt_strategy=1.0, spatial_dim=1, min_period=0,
                 max_period=None, times=None):
        super().__init__()
        self.model = model
        self.spatial_dim = spatial_dim
        u0 = 0.0
        if spatial_dim == 2:
            u0 = [0.0, 0.0]
        self.u0 = torch.tensor(u0).view(1, self.spatial_dim).cuda()
        #self.u0 = torch.randn(1, self.spatial_dim).cuda()
        self.u0.requires_grad = True
        self.non_integral_task_loss = non_integral_task_loss_func
        self.times = times
        self.num_times = None if self.times is None else self.times.size(0)
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint
        self.l_period = l_period
        self.l_task_loss = l_task_loss
        self.l_task_loss_2 = l_task_loss_2
        self.lr = lr
        self.optimizer_strategy = opt_strategy
        self.minT = min_period
        self.maxT = max_period

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
            output = torch.linspace(0, 1, num_points).cuda()
            return output, [0]


    def forward(self, x):

        with torch.no_grad():
            output, indices = self.create_t_list(100, self.model.f.T)

            if hasattr(self.non_integral_task_loss, 'indices'):
                self.non_integral_task_loss.indices = indices

        return self.odeint(self.model, x, output.cuda(), method='midpoint').squeeze(1)

    def periodicity_loss(self, init_cond, xT):
        vector_field = self.model.f(torch.linspace(0, 1, 100).cuda(), xT)[:, 0:self.spatial_dim].view(-1, self.spatial_dim)
        avg_variance = torch.mean(torch.var(vector_field, dim=0))
        #max_diff = torch.max(torch.norm(vector_field, dim=1) ** 2)
        periodicity_loss = torch.norm(init_cond.squeeze(0)[0:2*self.spatial_dim] - xT[-1, :].cuda()) ** 2
        #return periodicity_loss * (1 + 1/max_diff)
        return periodicity_loss * (1 + 1/avg_variance)

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        period = self.model.f.T.data
        period = torch.abs(period)
        period = period.clamp(self.minT, self.maxT)
        self.model.f.T.data = period

    def _training_step1(self, batch, batch_idx):
        # Solve the ODE forward in time for T seconds
        init_cond = torch.cat([self.u0, torch.zeros(1, self.spatial_dim + 1).cuda()], dim=1)
        xTl = self.forward(init_cond)
        xT, l = xTl[:, :-1], xTl[:, -1:]

        # Compute loss
        periodicity_loss = self.l_period * self.periodicity_loss(init_cond, xT)
        integral_task_loss = self.l_task_loss * torch.abs(self.model.f.T[0]) * torch.mean(l)
        non_integral_task_loss = self.l_task_loss_2 * self.non_integral_task_loss(xT)
        loss = periodicity_loss + integral_task_loss + non_integral_task_loss
        print('                      ')
        print('                      ')
        print('periodicity loss', periodicity_loss)
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

    def _training_step2(self, batch, batch_idx, optimizer_idx):
        # Solve the ODE forward in time for T seconds
        init_cond = torch.cat([self.u0, torch.zeros(1, self.spatial_dim + 1).cuda()], dim=1)
        xTl = self.forward(init_cond)
        xT, l = xTl[:, :-1], xTl[:, -1:]

        # Compute loss
        if optimizer_idx == 0:
            loss_type = "task loss"
            integral_task_loss = self.l_task_loss * torch.abs(self.model.f.T[0]) * torch.mean(l)
            non_integral_task_loss = self.l_task_loss_2 * self.non_integral_task_loss(xT)
            loss = integral_task_loss + non_integral_task_loss
            print('                      ')
            print('                      ')
            print('task loss', non_integral_task_loss)
            print('integral loss', integral_task_loss)
            print('                      ')

            # log training data
            self.logger.experiment.log(
                {
                    'periodicity loss': None,
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

        else:
            loss_type = "periodicity loss"
            periodicity_loss = self.l_period * self.periodicity_loss(init_cond, xT)
            loss = periodicity_loss
            print('                      ')
            print('                      ')
            print('periodicity loss', periodicity_loss)
            print('                      ')
            # log training data
            self.logger.experiment.log(
                {
                    'periodicity loss': periodicity_loss,
                    'integral task loss': None,
                    'non-integral task loss': None,
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
            return self._training_step2(batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        if self.optimizer_strategy == 1.0:
            params = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.model.f.T, 'lr': self.lr},
                      {'params': self.u0, 'lr': self.lr}]
            optimizer = torch.optim.Adam(params)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
            return ({"optimizer": optimizer, "lr_scheduler": scheduler, "frequency": 1})
        else:
            params1 = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.u0, 'lr': self.lr}]
            params2 = [{'params': self.model.f.V.parameters(), 'lr': self.lr}]#, {'params': self.u0, 'lr': self.lr}]
            optimizer1 = torch.optim.Adam(params1)
            optimizer2 = torch.optim.Adam(params2)
            scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=.999)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=.999)
            return ({"optimizer": optimizer1, "lr_scheduler": scheduler1, "frequency": 2},
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2, "frequency": 10})

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
        return torch.abs(torch.sum(u, dim=1, keepdim=False))


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


