import sys
sys.path.append('')

from src.opt_limit_cycle_control.models import ControlledSystemNoDamping, AugmentedDynamics, ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions
from src.opt_limit_cycle_control.utils import DummyDataModule
from src.opt_limit_cycle_control.plotter import plot_trajectories

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

pendulum = False

if pendulum:
    v_in = 1
    v_out = 1
    hdim = 256
    rep = 1
    training_epochs = 400
    lr = 1e-3
    spatial_dim = 1
    opt_strategy = 1
    l_period_k = 1.0
    l_task_k = 0.0
    l_task_2_k = 1.0

    # angular targets for q1
    target = [0.2, 2.2]

    # vector field parametrized by a NN
    V = nn.Sequential(
        nn.Linear(v_in, hdim),
        nn.Softplus(),
        nn.Linear(hdim, hdim),
        nn.Tanh(),
        nn.Linear(hdim, v_out))

    f = ControlledSystemNoDamping(V).to(device)
    aug_f = AugmentedDynamics(f, ControlEffort(f))
else:
    v_in = 2
    v_out = 2
    hdim = 256
    training_epochs = 600
    lr = 1e-3
    spatial_dim = 2
    opt_strategy = 1
    l_period_k = 1.0
    l_task_k = 0.0
    l_task_2_k = 1.0

    # angular targets for q1 and q2
    target = [-0.78, 0.78, 0, 0]

    # vector field parametrized by a NN
    V = nn.Sequential(
        nn.Linear(v_in, hdim),
        nn.Softplus(),
        nn.Linear(hdim, hdim),
        nn.Tanh(),
        nn.Linear(hdim, v_out))
        #,
        #nn.Tanh())

    f = ControlledSystemDoublePendulum(V).to(device)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))

# Train the Energy shaping controller
learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositions(target), l_period=l_period_k,
                              l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy, spatial_dim=spatial_dim, lr=lr).cuda()

logger = WandbLogger(project='optimal-cycle-shaping', name='pend_adjoint')

trainer = pl.Trainer(max_epochs=training_epochs, logger=logger, gpus=[0])
datloader = DummyDataModule(7)

print('Initial period:', learn.model.f.T[0])
print('Initial Initial position:', learn.u0[0])

trainer.fit(learn, train_dataloader=datloader)

print('Final period:', learn.model.f.T)
print('Final Initial position:', learn.u0[0])

# Plotting the final results.
num_points = 1000
if pendulum:
    angles = torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)
else:
    angles = torch.cat([torch.linspace(-np.pi, np.pi, num_points).view(-1, 1), torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)], dim=1)
with torch.no_grad():
    xT = odeint(learn.model.f.cuda(), torch.cat([learn.u0, torch.zeros(learn.u0.size()).cuda()], dim=1).cuda(),
                torch.linspace(0, 1, num_points).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()
vu = learn.model.f(torch.linspace(0, 1, num_points).cuda(), angles.detach().cuda(), V_only=True).cuda().detach().cpu().numpy()

plot_trajectories(xT=xT, target=target, V=vu[:, 0:v_in].reshape(num_points, v_in), angles=angles.numpy().reshape(num_points, v_in),
                  u=vu[:, :v_in].reshape(num_points, v_in), l1=1, l2=1, pendulum=pendulum, plot3D=False)
print('Job is done!')

