import sys
sys.path.append('')

from src.opt_limit_cycle_control.models import ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions, \
    CloseToPositionsAtTime, CloseToPositionAtHalfPeriod, CloseToActualPositionAtHalfPeriod
from src.opt_limit_cycle_control.utils import DummyDataModule
from src.opt_limit_cycle_control.plotter import plot_trajectories, animate_single_dp_trajectory
from src.opt_limit_cycle_control.layers import KernelRegression, KernelFunc, ReluKernel

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

import numpy as np

import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the parameters needed for the problem
v_in = 2
v_out = 2
hdim = 100
training_epochs = 100
lr = 1e-3
spatial_dim = 2
opt_strategy = 1
l_period_k = 0.0
alpha_p = 0.95
alpha_s = 0.01
alpha_mv = 0.01
l_task_k = 0.015 * 0.0
l_task_2_k = 10

# lengths of the pendulum bars
l1, l2 = 1.0, 1.0

# angular targets for q1 and q2
target = torch.tensor([0.5, 0.5]).reshape(2, 1).cuda()
use_target_angles = False

# vector field parametrized by a NN
V = nn.Sequential(
    nn.Linear(v_in, hdim),
    nn.Softplus(),
    nn.Linear(hdim, hdim),
    nn.Tanh(),
    nn.Linear(hdim, v_out))
    #,
    #nn.Tanh())
# discretization_thetas = torch.linspace(-math.pi, math.pi, 50)[0:-1]
# kernel_locations = torch.cartesian_prod(discretization_thetas, discretization_thetas)
# scaling = discretization_thetas[1]-discretization_thetas[0]
# kernel = KernelFunc(ReluKernel(scaling), kernel_locations, is_torus=True)
# V = KernelRegression(kernel)

f = ControlledSystemDoublePendulum(V).to(device)
aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))

# Train the Energy shaping controller.
if use_target_angles:
    learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositionAtHalfPeriod(target),
                                  l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                  l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                  spatial_dim=spatial_dim, lr=lr, u0_init=[-0.5, -0.5], u0_requires_grad=False)
else:
    learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToActualPositionAtHalfPeriod(target, l1, l2),
                                  l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                  l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                  spatial_dim=spatial_dim, lr=lr, u0_init=[-0.5, -0.5], u0_requires_grad=False)

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
angles = torch.cat([torch.linspace(-np.pi, np.pi, num_points).view(-1, 1), torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)], dim=1)
with torch.no_grad():
    xT = odeint(learn.model.f.cuda(), torch.cat([learn.u0, torch.zeros(learn.u0.size()).cuda()], dim=1).cuda(),
                torch.linspace(0, 1, num_points).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()

vu = learn.model.f(torch.linspace(0, 1, num_points).cuda(), angles.detach().cuda(), V_only=True).cuda().detach().cpu().numpy()

plot_trajectories(xT=xT, target=target.reshape(1, -1).cpu(), V=vu[:, 0:v_in].reshape(num_points, v_in), angles=angles.numpy().reshape(num_points, v_in),
                  u=vu[:, :v_in].reshape(num_points, v_in), l1=1, l2=1, pendulum=False, plot3D=False)
print('Job is done!')

