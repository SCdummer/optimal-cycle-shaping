import sys ; sys.path.append('')
import matplotlib.pyplot as plt
import copy

from src.control.utils import prior_dist, target_dist, dummy_trainloader, weighted_log_likelihood_loss
from src.opt_limit_cycle_control.models import ControlledSystemNoDamping, AugmentedDynamics
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner
from src.opt_limit_cycle_control.utils import DummyDataModule

import torch
import torch.nn as nn
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Define the Integral Cost Function
class ControlEffort(nn.Module):
    # control effort integral cost
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            q = x[:,:1].requires_grad_(True)
            u = self.f._energy_shaping(q)
        return torch.abs(u)


# Define the non-integral cost function
class CloseToPositions(nn.Module):
    # Given a time series as input, this cost function measures how close the average distance is to a set of points
    def __init__(self, dest: torch.Tensor):
        super().__init__()
        self.dest = dest.reshape(-1, 1).cuda()

    def forward(self, xt):
        return torch.max(torch.abs(torch.min(xt[:, 0] - self.dest, dim=1)[0]))


# Define NNs and model
# vector field parametrized by a NN
hdim = 64
V = nn.Sequential(
          nn.Linear(1, hdim),
          nn.Softplus(),
          nn.Linear(hdim, hdim),
          nn.Tanh(),
          nn.Linear(hdim, 1))

# for p in V[-1].parameters(): torch.nn.init.zeros_(p)

f = ControlledSystemNoDamping(V).to(device)
aug_f = AugmentedDynamics(f, ControlEffort(f))

# Train the Energy shaping controller
learn = OptEigManifoldLearner(aug_f, CloseToPositions(torch.tensor([[0.1], [0.2]])), 0.02, 0.0000001).cuda()
print("\n")
print("Starting u0: ", learn.u0)
print("Starting T: ", learn.model.f.T)
print(learn.model.f.parameters())
initial_weights = copy.deepcopy([params for params in learn.model.f.parameters()])
print("\n")
learn.lr = 5e-3
logger = WandbLogger(project='optimal-cycle-shaping', name='pend_adjoint')

trainer = pl.Trainer(max_epochs=500, logger=logger, gpus=[0])
datloader = DummyDataModule(7)

trainer.fit(learn, datloader)
print("\n")
print("Final u0: ", learn.u0)
print("Final T: ", learn.model.f.T)
final_weights = [params for params in learn.model.f.parameters()]
difference = [final_weights[i].cuda() - initial_weights[i].cuda() for i in range(len(initial_weights))]
print("Final weights - initial weights:", sum([torch.norm(diff) for diff in difference]))

# Plotting the final results.
xT = odeint(learn.model.f.cuda(), torch.cat([learn.u0, torch.zeros(learn.u0.size()).cuda()], dim=1).cuda(),
            torch.linspace(0, 1, 1000).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()

fig, ax = plt.subplots()

ax.plot(xT[:, 0], xT[:, 1])
ax.set_title('Plot in phase space of the found solution')
ax.set_xlabel('position (q)')
ax.set_ylabel('momentum (p)')

plt.show()
