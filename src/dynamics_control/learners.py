from dynamics_control.utils import dummy_trainloader, weighted_log_likelihood_loss
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdiffeq import odeint, odeint_adjoint
from argparse import Namespace, ArgumentParser


class DynamicsControlLearner(pl.LightningModule):
    def __init__(self, hparams, model: nn.Module, prior_dist, target_dist, t_span, sensitivity='autograd'):
        super().__init__()
        hparams = Namespace(**hparams) if type(hparams) is dict else hparams
        self.hparams = hparams
        self.model = model
        self.prior, self.target = prior_dist, target_dist
        self.t_span = t_span
        self.odeint = odeint if sensitivity == 'autograd' else odeint_adjoint

        self.batch_size = 2048
        # self.lr = 1e-3
        self.weight = torch.Tensor([1., 1.]).reshape(1, 2)

    def forward(self, x):
        return self.odeint(self.model, x, self.t_span, method='midpoint')

    def training_step(self, batch, batch_idx):
        # sample a batch of initial conditions
        x0 = self.prior.sample((self.batch_size,))

        # Integrate the model
        x0 = torch.cat([x0, torch.zeros(self.batch_size, 1).to(x0)], 1)
        traj = self(x0)
        xTl = traj[-1]
        xT, l = xTl[:, :2], xTl[:, -1:]

        # use the above trajectories as training data to train system properties
        pred_loss = self.get_pred_loss(traj[..., :-1])

        # Compute loss
        terminal_loss = weighted_log_likelihood_loss(xT, self.target, self.weight.to(xT))
        integral_loss = torch.mean(l)
        loss = terminal_loss + 0.01 * integral_loss + pred_loss

        # log training data
        self.logger.experiment.log(
            {
                'terminal loss': terminal_loss,
                'integral_loss': integral_loss,
                'pred_loss': pred_loss,
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

    def get_pred_loss(self, xt):
        # xt: T, bs, n
        T, bs, n = xt.shape
        # xt_ = xt.reshape(T*bs, n)
        # ut_ = self.model._energy_shaping(xt_[:, 0:n//2]) + self.model._damping_injection(xt_)
        # ut = ut_.reshape(T, bs, -1)

        # chunk trajectories into predefined length, 5 for now.
        t_len = 5
        assert T % t_len == 0
        xt_chunks = torch.stack(torch.split(xt, t_len, dim=0), dim=1).reshape(t_len, -1, n)
        # ut_chunks = torch.stack(torch.split(xt, t_len, dim=0), dim=1).reshape(t_len, -1, 1)
        
        xt_chunks_pred = odeint(self.model.f.parametrized_forward, xt_chunks[0], self.t_span[0:t_len], method="midpoint")
        
        # compute absolute mean error
        return (xt_chunks - xt_chunks_pred).reshape(T, bs, n).abs().sum([0, 2]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return dummy_trainloader()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

        return parser