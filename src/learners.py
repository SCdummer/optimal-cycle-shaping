from src.utils import dummy_trainloader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint
from src.losses import CloseToPositionAtHalfPeriod, CloseToActualPositionAtHalfPeriod, EigenmodeLoss

class OptEigenManifoldLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, use_target_angles=True, target=None, l1=1.0, l2=1.0, T=1.0, alpha_1=1.0,
                 lambda_1=1.0, lambda_2=1.0, alpha_eff=1.0, alpha_task=1.0, lr=0.001, sensitivity='autograd',
                 spatial_dim=1, min_period=0, max_period=None, times=None, u0_init=None, u0_requires_grad=True,
                 training_epochs=1, use_betascheduler=False):
        super().__init__()

        self.model = model
        self.spatial_dim = spatial_dim

        if u0_init is None:
            u0_init = torch.tensor([0.0, 0.0])
            self.u0 = u0_init.view(1, self.spatial_dim).cuda()
        else:
            if not len(u0_init) == self.spatial_dim:
                raise ValueError("u0_init should be a list containing spatial_dim values indicating the initial value")
            else:
                self.u0 = torch.tensor(u0_init).view(1, self.spatial_dim).cuda()
        self.u0.requires_grad = u0_requires_grad

        if use_target_angles:
            self.task_loss = CloseToPositionAtHalfPeriod(target)
        else:
            self.task_loss = CloseToActualPositionAtHalfPeriod(target, l1, l2)

        self.eigenmode_loss = EigenmodeLoss(spatial_dim=spatial_dim, lambda_1=lambda_1, lambda_2=lambda_2,
                                            alpha_1=alpha_1, half_period_index=None)
        self.times = times
        self.num_times = None if self.times is None else self.times.size(0)
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint
        self.alpha_1 = alpha_1
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_eff = alpha_eff
        self.alpha_task = alpha_task
        self.lr = lr
        self.half_period_index = None
        self.minT = min_period
        self.maxT = max_period
        self.epoch = 0
        self.count = 0
        self.discretization_steps = 100
        self.training_epochs = training_epochs
        self.use_betascheduler = use_betascheduler

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
            output, half_period_index = torch.unique(torch.cat([torch.linspace(0, 1, num_points).cuda(), torch.tensor(0.5).unsqueeze(0).cuda()]),
                                                   sorted=True, return_inverse=True)
            self.half_period_index = half_period_index[-1].item()
            if hasattr(self.task_loss, 'half_period_index'):
                self.task_loss.half_period_index = self.half_period_index
                self.eigenmode_loss.half_period_index = self.half_period_index
            return output, [0]

    def forward(self, x):
        with torch.no_grad():
            output, indices = self.create_t_list(self.discretization_steps, self.model.f.T)
            if hasattr(self.task_loss, 'indices'):
                self.task_loss.indices = indices
        return self.odeint(self.model, x, output.cuda(), method='midpoint').squeeze(1)

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

    def training_step(self, batch, batch_idx):
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
        eigenmode_loss = self.eigenmode_loss.compute_loss(xT, indices)
        control_effort_loss = self.alpha_eff * torch.abs(self.model.f.T[0]) * l[-1]
        task_loss = self.alpha_task * self.task_loss(xT)

        if self.use_betascheduler:
            beta = self.beta_scheduler(self.epoch, self.training_epochs)
            print('beta: ', beta)
        else:
            beta = 1.0
        loss = task_loss + beta * control_effort_loss + eigenmode_loss
        print('                  ')
        print('==================')
        print('Epoch: ', self.epoch)
        print('Total loss', loss)
        print('Task loss', task_loss)
        print('Control-effort loss', control_effort_loss)
        print('Eigenmode loss', eigenmode_loss)
        self.model.nfe = 0
        return {'loss': loss}

    def configure_optimizers(self):
        if self.u0.requires_grad:
            params = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.model.f.T, 'lr': self.lr},
                      {'params': self.u0, 'lr': self.lr}]
        else:
            params = [{'params': self.model.f.V.parameters(), 'lr': self.lr}, {'params': self.model.f.T, 'lr': self.lr}]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return ({"optimizer": optimizer, "lr_scheduler": scheduler, "frequency": 1})

    def train_dataloader(self):
        return dummy_trainloader()



