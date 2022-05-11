import sys
sys.path.append('')

from src.opt_limit_cycle_control.models import ControlledSystemNoDamping, AugmentedDynamics, ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions, CloseToPositionsAtTime
from src.opt_limit_cycle_control.utils import DummyDataModule, load_eig_mode_double_pendulum
from src.opt_limit_cycle_control.plotter import plot_trajectories, animate_single_dp_trajectory

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
    training_epochs = 100
    lr = 1e-3
    spatial_dim = 1
    opt_strategy = 1
    l_period_k = 1.0
    l_task_k = 0.0
    l_task_2_k = 1.0

    # angular targets for q1
    target = torch.tensor([-0.5, 0.5]).reshape(2, 1).cuda()

    # If we use the CloseToPosition loss, use times = None. If we use the CloseToPosition at times loss, specify the
    # times at which we want to pass through the positions.

    #times = None
    times = torch.tensor([0.25, 0.75]).reshape(2).cuda()

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
    training_epochs = 800
    lr = 1e-3
    spatial_dim = 2
    opt_strategy = 1
    l_period_k = 1.0
    l_task_k = 0.0
    l_task_2_k = 1.0
    to_animate_eig_mode = False

    # angular targets for q1 and q2

    # Getting the data from some eigenmode
    n, k, i = 1, 0, 40
    time, traj = load_eig_mode_double_pendulum(n, k, i)

    # Getting some specific points from the trajectory
    num_targets = 10
    num_times = traj.shape[0]
    step_size = int(np.floor(num_times/(num_targets + 1)))
    indices = range(step_size, num_times-step_size, step_size)

    target = torch.Tensor(traj[indices, 0:2]).cuda()

    # If we use the CloseToPosition loss, use times = None.= None. If we use the CloseToPosition at times loss, specify the
    # times at which we want to pass through the positions.

    times = torch.Tensor(time[indices, 0]).cuda()
    #times = None

    # Create an animation of the chosen eigenmode
    if to_animate_eig_mode:
        animate_single_dp_trajectory(traj)

        input("You have time to take a look at the created gif. Press Enter to continue...")

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

# Train the Energy shaping controller. If you specify times, use CloseToPositionsAtTime and else use CloseToPositions
# as task loss
if times is None:
    learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositions(target),
                                  l_period=l_period_k,
                                  l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                  spatial_dim=spatial_dim, lr=lr)
else:
    learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositionsAtTime(target), l_period=l_period_k,
                              l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy, spatial_dim=spatial_dim, lr=lr,
                              times=times, min_period=max(times))

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

plot_trajectories(xT=xT, target=target.cpu(), V=vu[:, 0:v_in].reshape(num_points, v_in), angles=angles.numpy().reshape(num_points, v_in),
                  u=vu[:, :v_in].reshape(num_points, v_in), l1=1, l2=1, pendulum=pendulum, plot3D=False)
print('Job is done!')

