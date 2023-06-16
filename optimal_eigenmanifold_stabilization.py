import sys
sys.path.append('')

#from scipy.interpolate import interp1d


from src.opt_limit_cycle_control.models import ControlledSystemNoDamping, AugmentedDynamics, ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum, StabilizedSystemDoublePendulum, StabilizedSystemDoublePendulumCosimo
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions, \
    CloseToPositionsAtTime, CloseToPositionAtHalfPeriod, CloseToActualPositionAtHalfPeriod

from src.opt_limit_cycle_control.utils import DummyDataModule, load_eig_mode_double_pendulum, interp_torch, traj_to_qp, find_orbit, numJ,numJ2
from src.opt_limit_cycle_control.plotter import plot_trajectories, animate_single_dp_trajectory
from src.opt_limit_cycle_control.layers import FourierEncoding


import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

import numpy as np

import matplotlib.pyplot as plt

import json

from matplotlib.ticker import FormatStrFormatter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

### Load trained model


with open('config.json') as f:
    config = json.load(f)

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
    COSIMO = True
    if COSIMO: 
        f = StabilizedSystemDoublePendulumCosimo(V,a_M,a_E,x_t).to(device) # Stabilizing controller may inject energy
    else: 
        f = StabilizedSystemDoublePendulum(V,a_M,a_E,x_t).to(device)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))


# Train the Energy shaping controller. If you specify times, use CloseToPositionsAtTime and else use CloseToPositions
# as task loss

#TODO: load state_dict

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

# =============================================================================
# # refine orbit:
# def psi(u0T):
#     xT =  odeint(learn.model.f.to(device), torch.cat([u0[:-1], torch.zeros(learn.u0.size()-1).to(device)], dim=1).to(device),
#                 torch.linspace(0, u0T[-1], num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()
#     return torch.norm(xT[num_points/2])
# =============================================================================



a_M = torch.tensor(10)
a_E = torch.tensor(10)
AllTime = torch.linspace(0*T.detach().cpu().numpy()[0],T.detach().cpu().numpy()[0],num_points).to(device)
x_fun = interp_torch(AllTime,torch.tensor(xT).to(device)) 

learn.model.f._set_control_parameters(a_E.to(device),a_M.to(device),x_fun,xT)


n_Periods = 3

learn.model.f._set_damping(0)

x0 = torch.cat([learn.u0+0.7, 5+torch.zeros(learn.u0.size()).to(device)], dim=1).to(device)

with torch.no_grad():
    xT_ctrl = odeint(learn.model.f.to(device), x0,
            torch.linspace(0, n_Periods, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()




# Check energy:
Q,P = traj_to_qp(xT_ctrl)
n = len(Q)
E,AutE,Pot,Kin = torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n)
E_des = learn.model.f.E_des 
F= torch.zeros((n,2))
E_In = torch.zeros(n)
TrajDist = torch.zeros(n)
MomentumDiff = torch.zeros(n)
DesiredPosition = torch.zeros((n,2))
DesiredMomentum = torch.zeros((n,2))
for i in range(n):
    E[i] = learn.model.f._energy(Q[i].to(device),P[i].to(device))
    AutE[i] = learn.model.f._autonomous_energy(Q[i].to(device),P[i].to(device))
    Pot[i] = learn.model.f._autonomous_potential(Q[i].to(device))
    Kin[i] = learn.model.f._autonomous_kinetic(Q[i].to(device),P[i].to(device))
    F[i] = learn.model.f._stabilizing_control(Q[i].to(device),P[i].to(device))
    E_In[i] = torch.inner(F[i].to(device),P[i].to(device)@learn.model.f._inv_mass_tensor(Q[i].to(device))) # E_In.sum()*T/n is the energy injected by the controller
    TrajDist[i] = learn.model.f._min_d(Q[i].to(device))
    DesiredPosition[i] = learn.model.f._q_des(Q[i].to(device))
    DesiredMomentum[i] = learn.model.f._p_des_raw(Q[i].to(device),P[i].to(device))
    MomentumDiff[i] = torch.norm(DesiredMomentum[i]-P[i])
    
    
AllTime_np = torch.linspace(0*T.detach().cpu().numpy()[0],n_Periods*T.detach().cpu().numpy()[0],num_points).detach().numpy();
handles = plt.plot(AllTime_np,E.detach().cpu().numpy(),AllTime_np,AutE.detach().cpu().numpy(),AllTime_np,Pot.detach().cpu().numpy(),AllTime_np,Kin.detach().cpu().numpy())
plt.legend(handles,('Total Energy','Autonomous Energy', 'Autonomous Potential','Kinetic Energy'))
plt.ylabel('Energy in J')
plt.xlabel('Time in s')

plt.figure()
plt.plot(AllTime_np,(E-E_des[0].cpu()).detach().cpu().numpy())
plt.ylabel('$\Delta E$ in J')
plt.xlabel('Time in s')

plt.figure()
plt.plot(AllTime_np,TrajDist.detach().cpu().numpy())
plt.xlabel('Time in s')
plt.ylabel('$dist(q,q_{des})$')

plt.figure()
plt.plot(AllTime_np,MomentumDiff.detach().cpu().numpy())
plt.xlabel('Time in s')
plt.ylabel('$ \|p - p_{des} \|_2$')

DesiredPosition = DesiredPosition.detach().cpu().numpy()
DesiredMomentum = DesiredMomentum.detach().cpu().numpy()
q0_des = DesiredPosition[:,0]
q1_des = DesiredPosition[:,1]
p0_des = DesiredMomentum[:,0]
p1_des = DesiredMomentum[:,1]


## Position and momentum in one large figure:
plt.figure()
fig, axs = plt.subplots(4, sharex=True)
fig.set_figheight(10)
h0 = axs[0].plot(AllTime_np,q0_des,AllTime_np,xT_ctrl[:,0])
h1 = axs[1].plot(AllTime_np,q1_des,AllTime_np,xT_ctrl[:,1])
h2 = axs[2].plot(AllTime_np,p0_des,AllTime_np,xT_ctrl[:,2])
h3 = axs[3].plot(AllTime_np,p1_des,AllTime_np,xT_ctrl[:,3])

axs[0].set(ylabel = 'rad')
axs[0].legend(h0,('Desired value','Actual value'),loc='upper right')
axs[0].set_title('$q_1(t)$',loc='left' )
axs[0].set_ylim(-1.5,1)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%i'))
axs[0].yaxis.set_ticks(np.arange(-1, 2, 1))

axs[1].set( ylabel = 'rad')
axs[1].set_title('$q_2(t)$',loc='left')

axs[2].set( ylabel = '$kg m^2 rad/s$')
axs[2].set_title('$p_1(t)$',loc='left')

axs[3].set( xlabel = 'Time in s', ylabel = '$kg m^2 rad/s$')
axs[3].set_title('$p_2(t)$',loc='left')
fig.align_labels()

#### Position:
plt.figure()
fig, axs = plt.subplots(2, sharex=True)
fig.set_figheight(5)

h0 = axs[0].plot(AllTime_np,q0_des,AllTime_np,xT_ctrl[:,0])
h1 = axs[1].plot(AllTime_np,q1_des,AllTime_np,xT_ctrl[:,1])

axs[0].set(ylabel = 'rad')
axs[0].legend(h0,('Desired value','Actual value'),loc='upper right')
axs[0].set_title('$q_1(t)$',loc='left' )
axs[0].set_ylim(-1.5,1)
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%i'))
axs[0].yaxis.set_ticks(np.arange(-1, 2, 1))

axs[1].set( ylabel = 'rad')
axs[1].set_title('$q_2(t)$',loc='left')
axs[1].set( xlabel = 'Time in s', ylabel = '$kg m^2 rad/s$')

fig.align_labels()

#### Momentum:
plt.figure()
fig, axs = plt.subplots(2, sharex=True)
fig.set_figheight(5)

h2 = axs[0].plot(AllTime_np,p0_des,AllTime_np,xT_ctrl[:,2])
h3 = axs[1].plot(AllTime_np,p1_des,AllTime_np,xT_ctrl[:,3])

axs[0].set( ylabel = '$kg m^2 rad/s$')
axs[0].set_title('$p_1(t)$',loc='left')
axs[0].legend(h2,('Desired value','Actual value'),loc='upper right')

axs[1].set( xlabel = 'Time in s', ylabel = '$kg m^2 rad/s$')
axs[1].set_title('$p_2(t)$',loc='left')

fig.align_labels()




# Check Distance plot:
k = 100 # spatial resolution CTRL-4 to comment, CTRL-5 to undo
d = (torch.arange(k+1).to(device)-k/2)/k*4
d_x = d #+ u0[0,0]
d_y = d +1#+ u0[0,1]
DistFun = np.zeros((k+1,k+1))
for i in range(k):
    for j in range(k):
        QC = torch.tensor((d_x[i],d_y[j]),device=device).unsqueeze(0)
        DistFun[j,i] = learn.model.f._min_d(QC) # This has to be in order (j,i), contourf takes first slot for y data, second slot for x data

X,Y = np.meshgrid(d_x.cpu().numpy(),d_y.cpu().numpy())

plt.figure()
plt.contourf(X,Y,DistFun,20)
q0 = xT_ctrl[:,0]; q1 = xT_ctrl[:,1]
plt.plot(q0,q1,'r--')

fig = plt.figure()
ax = plt.axes()
ax.contourf(X,Y,DistFun,20)
p1 = ax.scatter(q0, q1, c=np.linspace(0, T.cpu()*n_Periods, xT.shape[0], endpoint=True), cmap='twilight', s=10)
pt0 = ax.scatter(q0[0], q1[0], s=30, color="none", edgecolor="blue")
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
plt.xlabel('$q_1$')
plt.ylabel('$q_2$')
cbar = fig.colorbar(p1)
cbar.set_label('time', rotation=90)


def psi(x0):
    xT_ctrl = odeint(learn.model.f.to(device), x0.to(device),
                     torch.linspace(0, 2, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()
    Q,P = traj_to_qp(xT_ctrl)
    return torch.norm(torch.tensor(learn.model.f._phase_dist(Q[-1].to(device),P[-1].to(device))).to(device)).unsqueeze(0).unsqueeze(0) #- learn.model.f._phase_dist(Q[0].to(device),P[0].to(device))

x0 = torch.cat([learn.u0, torch.zeros(learn.u0.size()).to(device)], dim=1).to(device)
dpsidx = numJ2(psi,x0,torch.tensor(1e-1).to(device))


#handles = plt.plot(AllTime,xT_ctrl[:,0],AllTime,xT_ctrl[:,1],AllTime,xT_ctrl[:,2],AllTime,xT_ctrl[:,3])
#plt.legend(handles, ('q1','q2', 'p1', 'p2'))
#plt.xlabel('Time')

# =============================================================================
# 
# vu = learn.model.f(torch.linspace(0, n_Periods, num_points).to(device), angles.detach().to(device), V_only=True).to(device).detach().cpu().numpy()
# 
# plot_trajectories(xT=xT_ctrl, target=target.cpu(), V=vu[:, 0:v_in].reshape(num_points, v_in), angles=angles.numpy().reshape(num_points, v_in),
#                   u=vu[:, :v_in].reshape(num_points, v_in), l1=1, l2=1, pendulum=pendulum)
# 
# =============================================================================