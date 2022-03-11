from argparse import ArgumentParser, Namespace
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from numpy import pi as pi

from dynamics_control.utils import prior_dist, target_dist, dummy_trainloader, weighted_log_likelihood_loss, PSD
from dynamics_control.models import ElasticPendulum, AugmentedDynamics
from dynamics_control.learners import DynamicsControlLearner
seed_everything(0)

# integral cost function
class ControlEffort(nn.Module):
    # control effort integral cost
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            q = x[:,:1].requires_grad_(True)
            u = self.f._energy_shaping(q) + self.f._damping_injection(x)
        return torch.abs(u)

# UNIFORM "prior" distribution of initial conditions x(0) 
prior = prior_dist(-2*pi, 2*pi, -2*pi, 2*pi)

# NORMAL target distribution for x(T)
target = target_dist([0, 0], [.001, .001])

# define NN and model
# vector field parametrized by a NN
hdim = 64
V = nn.Sequential(
        nn.Linear(1, hdim),
        nn.Softplus(), 
        nn.Linear(hdim, hdim),
        nn.Tanh(), 
        nn.Linear(hdim, 1))
K = nn.Sequential(
        nn.Linear(2, hdim),
        nn.Softplus(),
        nn.Linear(hdim, 1),
        nn.Softplus())

H = nn.Sequential(
    nn.Linear(2, hdim),
    nn.Softplus(),
    nn.Linear(hdim, hdim),
    nn.Softplus(),
    nn.Linear(hdim, 1)
)
g_net = nn.Sequential(
    nn.Linear(1, hdim),
    nn.Softplus(),
    nn.Linear(hdim, hdim),
    nn.Softplus(),
    nn.Linear(hdim, 2)
)

D = PSD(1, 64, 2)

for p in V[-1].parameters(): torch.nn.init.zeros_(p)
for p in K[-2].parameters(): torch.nn.init.zeros_(p)


f = ElasticPendulum(V, K, H, D, g_net)
aug_f = AugmentedDynamics(f, ControlEffort(f))

t_span = torch.linspace(0, 3, 30)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = DynamicsControlLearner.add_model_specific_args(parser)
    hparams = parser.parse_args()


 


    logger = WandbLogger(project="dynamics-control-learner", name='pend-adjoint')
    learner = DynamicsControlLearner(hparams, aug_f, prior, target, t_span)

    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         logger=logger)

    trainer.fit(learner)