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

    def forward(self, x):
        return self.odeint(self.model, x, torch.linspace(0, 1, 100).cuda(), method='midpoint').squeeze(1)

    def training_step(self, batch, batch_idx):
        # Solve the ODE forward in time for T seconds
        xTl = self.forward(torch.cat([self.u0, torch.zeros(1, self.spatial_dim + 1).cuda()], dim=1))
        xT, l = xTl[:, :2], xTl[:, -1:]

        # Compute loss
        periodicity_loss = torch.norm(self.u0.squeeze(0) - torch.tensor(xT[-1, 0]).cuda()) ** 2
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

    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr': self.lr}, {'params': self.u0, 'lr': self.lr}]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return dummy_trainloader()
