from src.control.utils import dummy_trainloader, weighted_log_likelihood_loss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint


class OptEigManifoldLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, non_integral_task_loss_func: nn.Module, l_period, l_task_loss,
                 lr=0.001, sensitivity='autograd'):
        super().__init__()
        self.model = model
        self.spatial_dim = 1
        self.u0 = torch.randn(1, self.spatial_dim).cuda()
        self.u0.requires_grad = True
        self.non_integral_task_loss = non_integral_task_loss_func
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint
        self.l_period = l_period
        self.l_task_loss = l_task_loss
        self.lr = lr
        self.optimizer_strategy = 2.0

    def forward(self, x):
        return self.odeint(self.model, x, torch.linspace(0, 1, 100).cuda(), method='midpoint').squeeze(1)

    def _training_step1(self, batch, batch_idx):
        # Solve the ODE forward in time for T seconds
        init_cond = torch.cat([self.u0, torch.zeros(1, self.spatial_dim + 1).cuda()], dim=1)
        xTl = self.forward(init_cond)
        xT, l = xTl[:, :2], xTl[:, -1:]

        # Compute loss
        periodicity_loss = torch.norm(init_cond.squeeze(0)[0:2*self.spatial_dim] - xT[-1, :].cuda()) ** 2
        integral_task_loss = self.model.f.T[0] * torch.mean(l)
        non_integral_task_loss = self.non_integral_task_loss(xT)
        loss = self.l_period * periodicity_loss + self.l_task_loss * integral_task_loss + non_integral_task_loss

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
        xT, l = xTl[:, :2], xTl[:, -1:]

        # Compute loss
        if optimizer_idx == 0:
            loss_type = "task loss"
            integral_task_loss = self.model.f.T[0] * torch.mean(l)
            non_integral_task_loss = self.non_integral_task_loss(xT)
            loss = self.l_task_loss * integral_task_loss + non_integral_task_loss

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
            periodicity_loss = torch.norm(init_cond.squeeze(0)[0:2 * self.spatial_dim] - xT[-1, :].cuda()) ** 2
            loss = periodicity_loss

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.optimizer_strategy == 1:
            return self._training_step1(batch, batch_idx)
        else:
            return self._training_step2(batch, batch_idx, optimizer_idx)

    def configure_optimizers(self):
        if self.optimizer_strategy == 1:
            params = [{'params': self.model.parameters(), 'lr': self.lr}, {'params': self.u0, 'lr': self.lr}]
            optimizer = torch.optim.Adam(params)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
            return [optimizer], [scheduler]
        else:
            params1 = [{'params': self.model.parameters(), 'lr': self.lr}, {'params': self.u0, 'lr': self.lr}]
            params2 = [{'params': self.model.f.T, 'lr': self.lr}, {'params': self.u0, 'lr': self.lr}]
            optimizer1 = torch.optim.Adam(params1)
            optimizer2 = torch.optim.Adam(params2)
            scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=.999)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, gamma=.999)
            return ({"optimizer": optimizer1, "lr_scheduler": scheduler1, "frequency": 2},
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2, "frequency": 10})

    def train_dataloader(self):
        return dummy_trainloader()
