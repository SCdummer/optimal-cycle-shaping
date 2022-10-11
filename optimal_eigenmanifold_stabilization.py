import sys
sys.path.append('')

#from scipy.interpolate import interp1d


from src.opt_limit_cycle_control.models import ControlledSystemNoDamping, AugmentedDynamics, ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum, StabilizedSystemDoublePendulum
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions, \
    CloseToPositionsAtTime, CloseToPositionAtHalfPeriod, CloseToActualPositionAtHalfPeriod

from src.opt_limit_cycle_control.utils import DummyDataModule, load_eig_mode_double_pendulum, interp_torch, traj_to_qp
from src.opt_limit_cycle_control.plotter import plot_trajectories, animate_single_dp_trajectory
from src.opt_limit_cycle_control.layers import KernelRegression, KernelFunc, ReluKernel, FourierEncoding


import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

import numpy as np

import matplotlib.pyplot as plt

import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

### Load trained model

config = {"v_in": 2,
          "v_out": 2,
          "hdim": 256,
          "lr": 0.001,
          "spatial_dim": 2,
          "opt_strategy": 1,
          "l_period_k": 1.0,
          "alpha_p": 0.0,
          "alpha_s": 0.05,
          "alpha_mv": 0.95,
          "l_task_k": 0.0,
          "l_task_2_k": 10,
          "l1": 1.0,
          "l2": 1.0,
          "use_target_angles": False,
          "u0_init": [-0.5, -0.5],
          "u0_requires_grad": False,
          "target": [0.75, 0.75],
          "training_epochs": 600}

v_in = config['v_in']
v_out = config['v_out']
hdim = config['hdim']
lr = config['lr']
spatial_dim = config['spatial_dim']
opt_strategy = config['opt_strategy']
l_period_k = config['l_period_k']
alpha_p = config['alpha_p'] 
alpha_s = config['alpha_s']
alpha_mv = config['alpha_mv']
l_task_k = config['l_task_k']
l_task_2_k = config['l_task_2_k']
l1 = config['l1']
l2 = config['l2']
use_target_angles = config['use_target_angles']
u0_init = config['u0_init']
u0_requires_grad = config['u0_requires_grad']
target = config['target']
training_epochs = config['training_epochs']

V = nn.Sequential(
    FourierEncoding(v_in),
    nn.Linear(2 * v_in, hdim),
    nn.Tanh(),
    nn.Linear(hdim, hdim),
    nn.Tanh(),
    nn.Linear(hdim, 1))


### Set-up double pendulum for control
pendulum = False
if not pendulum:
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

    target = torch.Tensor(traj[indices, 0:2]).to(device)

    # If we use the CloseToPosition loss, use times = None.= None. If we use the CloseToPosition at times loss, specify the
    # times at which we want to pass through the positions.

    times = torch.Tensor(time[indices, 0]).to(device)
    #times = None

    # Create an animation of the chosen eigenmode
    if to_animate_eig_mode:
        animate_single_dp_trajectory(traj)

        input("You have time to take a look at the created gif. Press Enter to continue...")

    # vector field parametrized by a NN
#    (V,u0,T) =torch.load('V_u_T_DoublePendulum_02_08_22.pt')
    
    V = V.to(device) 

    f = ControlledSystemDoublePendulum(V,T_initial=1.5,T_requires_grad = False)
    #f = StabilizedSystemDoublePendulum(V,a_M,a_E,x_t).to(device) # TODO: load from json
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))
    
    if use_target_angles:
        learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositionAtHalfPeriod(target),
                                      l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                      l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                       spatial_dim=spatial_dim, lr=lr, u0_init=u0_init, u0_requires_grad=False,
                                      training_epochs=training_epochs) #u0_init = [-0.5, -0.5]
    else:
        learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToActualPositionAtHalfPeriod(target, l1, l2),
                                      l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                      l_task_loss=l_task_k, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                      spatial_dim=spatial_dim, lr=lr, u0_init=u0_init, u0_requires_grad=False,
                                      training_epochs=training_epochs)
    learn.model.load_state_dict(torch.load('model_state_dict.pt'))
    
    u0 = torch.tensor(u0_init).to(device).unsqueeze(0) 
    T = torch.tensor(1.5).to(device).unsqueeze(0) 
    
    a_M = torch.tensor(0)
    a_E = torch.tensor(0)
    x_t = lambda t: torch.tensor((0,0,0,0)).float().to(device)
    
    V = learn.model.f.V
    f = StabilizedSystemDoublePendulum(V,a_M,a_E,x_t).to(device)
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
learn.u0 = u0
learn.model.f.T = torch.nn.Parameter(T)


# Plotting the final results.
num_points = 1000
if pendulum:
    angles = torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)
else:
    angles = torch.cat([torch.linspace(-np.pi, np.pi, num_points).view(-1, 1), torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)], dim=1)

with torch.no_grad():
    xT = odeint(learn.model.f.to(device), torch.cat([learn.u0, torch.zeros(learn.u0.size()).to(device)], dim=1).to(device),
                torch.linspace(0, 1, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()


a_M = torch.tensor(0)
a_E = torch.tensor(0)
AllTime = torch.linspace(0*T.detach().cpu().numpy()[0],T.detach().cpu().numpy()[0],num_points).to(device)
x_fun = interp_torch(AllTime,torch.tensor(xT).to(device)) 

learn.model.f._set_control_parameters(a_M.to(device),a_E.to(device),x_fun,xT)

with torch.no_grad():
    xT_ctrl = odeint(learn.model.f.to(device), torch.cat([learn.u0, torch.zeros(learn.u0.size()).to(device)], dim=1).to(device),
            torch.linspace(0, 1, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()




# Check energy:
Q,P = traj_to_qp(xT_ctrl)
n = len(Q)
E,AutE,Pot,Kin = torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n)

for i in range(n):
    E[i] = learn.model.f._energy(Q[i].to(device),P[i].to(device))
    AutE[i] = learn.model.f._autonomous_energy(Q[i].to(device),P[i].to(device))
    Pot[i] = learn.model.f._autonomous_potential(Q[i].to(device))
    Kin[i] = learn.model.f._autonomous_kinetic(Q[i].to(device),P[i].to(device))
AllTime = AllTime.detach().cpu().numpy();
handles = plt.plot(AllTime,E.detach().cpu().numpy(),AllTime,AutE.detach().cpu().numpy(),AllTime,Pot.detach().cpu().numpy(),AllTime,Kin.detach().cpu().numpy())
plt.legend(handles,('Total Energy','Autonomous Energy', 'Autonomous Potential','Kinetic Energy'))
plt.ylabel('Total Energy')
plt.xlabel('Time')
