from src.control.utils import dummy_trainloader, weighted_log_likelihood_loss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint

class EnergyShapingLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, prior_dist, target_dist, t_span, sensitivity='autograd'):
        super().__init__()
        self.model = model
        self.prior, self.target = prior_dist, target_dist
        self.t_span = t_span
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint

        self.batch_size = 2048
        self.lr = 1e-3
        self.weight = torch.Tensor([1., 1.]).reshape(1, 2)

    def forward(self, x):
        return self.odeint(self.model, x, self.t_span, method='midpoint')[-1]

    def training_step(self, batch, batch_idx):
        # sample a batch of initial conditions
        x0 = self.prior.sample((self.batch_size,))

        # Integrate the model
        x0 = torch.cat([x0, torch.zeros(self.batch_size, 1).to(x0)], 1)
        xTl = self(x0)
        xT, l = xTl[:, :2], xTl[:, -1:]

        # Compute loss
        terminal_loss = weighted_log_likelihood_loss(xT, self.target, self.weight.to(xT))
        integral_loss = torch.mean(l)
        loss = terminal_loss + 0.01 * integral_loss

        # log training data
        self.logger.experiment.log(
            {
                'terminal loss': terminal_loss,
                'integral_loss': integral_loss,
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return dummy_trainloader()