import sys
sys.path.append('')


from src.opt_limit_cycle_control.models import ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum, \
    StabilizedSystemDoublePendulum, StabilizedSystemDoublePendulumCosimo
from src.opt_limit_cycle_control.learners import OptEigenManifoldLearner

from src.opt_limit_cycle_control.losses import ControlEffort

from src.opt_limit_cycle_control.utils import interp_torch, traj_to_qp
from src.opt_limit_cycle_control.layers import FourierEncoding


import torch
import torch.nn as nn

from torchdiffeq import odeint

import numpy as np

import matplotlib.pyplot as plt

import os
import json

from matplotlib.ticker import FormatStrFormatter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the parameters needed for the problem
v_in = 2
v_out = 2
spatial_dim = 2


def from_specs_to_variables(specs):
    """"
    This function takes the loaded specs.json file as input and assigns all the values to the relevant variables!
    """

    hdim = specs["hdim"]
    training_epochs = specs["epochs"]
    lr = specs["lr"]
    l_period_k = specs["l_period_k"]
    alpha_1 = specs["alpha_1"]
    lambda_1 = specs["lambda_1"]
    lambda_2 = specs["lambda_2"]
    alpha_eff = specs["alpha_eff"]
    alpha_task = specs["alpha_task"]
    T_initial = specs["T_initial"]
    T_requires_grad = specs["T_requires_grad"]
    target = torch.Tensor(specs["target"]).reshape(v_in, 1).to(device)
    u0_init = specs["u0_init"]
    use_target_angles = specs["use_target_angles"]
    use_betascheduler = specs["use_betascheduler"]
    l1 = specs["l1"]
    l2 = specs["l2"]

    return hdim, training_epochs, lr, l_period_k, alpha_1, lambda_1, lambda_2, alpha_eff, alpha_task, T_initial, \
           T_requires_grad, target, u0_init, use_target_angles, use_betascheduler, l1, l2


if __name__ == "__main__":

    # Define an argument parser
    import argparse

    arg_parser = argparse.ArgumentParser(description="Find the eigenmode solving a pick-and-place task.")

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json'")

    arg_parser.add_argument(
        "--alpha_E",
        "-a_E",
        dest="a_E",
        default=10,
        type=int,
        help="The alpha_E stabilizing control parameter")

    arg_parser.add_argument(
        "--alpha_M",
        "-a_M",
        dest="a_M",
        default=10,
        type=int,
        help="The alpha_M stabilizing control parameter")

    arg_parser.add_argument(
        "--damping",
        "-b",
        dest="b",
        default=0,
        type=float,
        help="The amount of damping present in the system")

    arg_parser.add_argument(
        "--init_angle_1",
        "-q0_1",
        dest="q0_1",
        type=float,
        help="The initial value of q1")

    arg_parser.add_argument(
        "--init_angle_2",
        "-q0_2",
        dest="q0_2",
        type=float,
        help="The initial value of q2")

    arg_parser.add_argument(
        "--init_momentum_1",
        "-p0_1",
        dest="p0_1",
        type=float,
        help="The initial value of p1")

    arg_parser.add_argument(
        "--init_momentum_2",
        "-p0_2",
        dest="p0_2",
        type=float,
        help="The initial value of p2")

    arg_parser.add_argument(
        '--injection_energy_eig_mode_stabillizing_controller',
        dest='eig_mode_stabillizing_controller_injects_energy',
        action='store_true',
        help="If specified, we use the controller where the eigenmode stabillizing part injects energy"
    )
    arg_parser.set_defaults(eig_mode_stabillizing_controller_injects_energy=False)

    # Parse the arguments
    args = arg_parser.parse_args()

    # Define the folder where we will save the generated figures
    save_dir = os.path.join(args.experiment_directory, "Figures", "Figures_stabilizing_controller")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load the specs.json file
    specs = json.load(open(os.path.join(args.experiment_directory, "specs.json")))

    # Get the variable names from the specs.json file
    hdim, training_epochs, lr, l_period_k, alpha_1, lambda_1, lambda_2, alpha_eff, alpha_task, T_initial, \
    T_requires_grad, target, u0_init, use_target_angles, use_betascheduler, l1, l2 = from_specs_to_variables(specs)

    # Set u0_requires_grad to false as it is set to false in learn_opt_eigenmode.py
    u0_requires_grad = False

    # Initialize the potential neural network
    V = nn.Sequential(
        FourierEncoding(v_in),
        nn.Linear(2 * v_in, hdim),
        nn.Tanh(),
        nn.Linear(hdim, hdim),
        nn.Tanh(),
        nn.Linear(hdim, 1)).to(device)

    # Load the double pendulum model
    f = ControlledSystemDoublePendulum(V, T_initial=1.0, T_requires_grad=False)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))
    learn = OptEigenManifoldLearner(model=aug_f, use_target_angles=use_target_angles, target=target, l1=l1, l2=l2,
                                    T=T_initial, alpha_1=alpha_1, lambda_1=lambda_1, lambda_2=lambda_2,
                                    alpha_eff=alpha_eff, alpha_task=alpha_task, spatial_dim=spatial_dim,
                                    lr=lr, u0_init=u0_init, u0_requires_grad=False, training_epochs=training_epochs,
                                    use_betascheduler=use_betascheduler)
    learn.model.load_state_dict(torch.load(os.path.join(args.experiment_directory, 'model_state_dict.pt')))

    # Calculate the eigenmode trajectory for 1000 time points between 0 and 1.
    num_points = 1000
    with torch.no_grad():
        xT = odeint(learn.model.f.to(device), torch.cat([learn.u0, torch.zeros(learn.u0.size()).to(device)], dim=1).to(device),
                    torch.linspace(0, 1, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()

    # Get the initial location of the learned eigenmode and get the learned period of the eigenmode
    u0 = torch.tensor(u0_init).to(device).unsqueeze(0)
    T = learn.model.f.T.to(device)

    # Load the potential V
    V = learn.model.f.V

    # Define the control parameters alpha_M and alpha_E
    a_M = torch.tensor(args.a_M)
    a_E = torch.tensor(args.a_E)

    # Define the function that interpolates the values on the eigenmode
    AllTime = torch.linspace(0 * T.detach().cpu().numpy()[0], T.detach().cpu().numpy()[0], num_points).to(device)
    x_fun = interp_torch(AllTime, torch.tensor(xT).to(device))

    # Previously, we defined f, aug_f, and learn without stabilizing controller. Now we want to define the same model
    # WITH a stabilizing controller

    # Define a function that can be used to (easily) initialize the controlled model
    x_fun_init = lambda t: torch.tensor((0,0,0,0)).float().to(device)

    # If we want to use the controller where the eigenmode stabilizing part CAN inject energy, define ...
    if args.eig_mode_stabillizing_controller_injects_energy:
        f = StabilizedSystemDoublePendulumCosimo(V, a_M.to(device), a_E.to(device), x_fun_init).to(device) # Stabilizing controller may inject energy
    else:
        f = StabilizedSystemDoublePendulum(V, a_M.to(device), a_E.to(device), x_fun_init).to(device)

    # Set the correct period
    f.T = torch.nn.Parameter(T)

    # Properly set the parameters of the model
    f._set_control_parameters(a_E.to(device), a_M.to(device), x_fun, xT)

    # Set the damping to a parameter
    f._set_damping(args.b)

    # Define the number of periods of the eigenmode that we want to use to control the trajectory towards the eigenmode
    n_Periods = 3

    # If both the initial angles are given, do ...
    if args.q0_1 is not None and args.q0_2 is not None:
        q0 = torch.cat([args.q0_1, args.q0_2])
    else:
        print("NOTE: either q0_1 or q0_2 is not supplied to the argument parser! Using the Default option for q0 "
              "instead...")
        q0 = u0 + 0.7

    # If both the initial momenta are given, do ...
    if args.p0_1 is not None and args.p0_2 is not None:
        p0 = torch.cat([args.p0_1, args.p0_2])
    else:
        print("NOTE: either p0_1 or p0_2 is not supplied to the argument parser! Using the Default option for p0 "
              "instead...")
        p0 = 5+torch.zeros(learn.u0.size()).to(device)

    # Define the initial condition of the ODE
    x0 = torch.cat([q0, p0], dim=1).to(device)

    # Find the trajectory of the controlled system
    with torch.no_grad():
        xT_ctrl = odeint(f.to(device), x0,
                torch.linspace(0, n_Periods, num_points).to(device), method='midpoint').squeeze(1).cpu().detach().numpy()

    # Get the coordinates q and the momenta p of the controlled trajectory
    Q,P = traj_to_qp(xT_ctrl)

    # Initialize some quantities
    n = len(Q)
    E,AutE,Pot,Kin = torch.zeros(n),torch.zeros(n),torch.zeros(n),torch.zeros(n)
    E_des = f.E_des
    F= torch.zeros((n,2))
    TrajDist = torch.zeros(n)
    MomentumDiff = torch.zeros(n)
    DesiredPosition = torch.zeros((n,2))
    DesiredMomentum = torch.zeros((n,2))

    # At every point in time of the controlled trajectory, calculate the full energy, the autonomous energy, the
    # autonomous potential, the autonomous kinetic energy, the stabilizing control input, the distance to the desired
    # eigenmode trajectory, the desired position, the desired momentum, and the difference between the actual momentum
    # and the desired momentum
    for i in range(n):
        E[i] = f._energy(Q[i].to(device),P[i].to(device))
        AutE[i] = f._autonomous_energy(Q[i].to(device),P[i].to(device))
        Pot[i] = f._autonomous_potential(Q[i].to(device))
        Kin[i] = f._autonomous_kinetic(Q[i].to(device),P[i].to(device))
        F[i] = f._stabilizing_control(Q[i].to(device),P[i].to(device))
        TrajDist[i] = f._min_d(Q[i].to(device))
        DesiredPosition[i] = f._q_des(Q[i].to(device))
        DesiredMomentum[i] = f._p_des_raw(Q[i].to(device),P[i].to(device))
        MomentumDiff[i] = torch.norm(DesiredMomentum[i]-P[i])

    # Start generating and saving the plots
    AllTime_np = torch.linspace(0*T.detach().cpu().numpy()[0],n_Periods*T.detach().cpu().numpy()[0],num_points).detach().numpy();
    handles = plt.plot(AllTime_np,E.detach().cpu().numpy(),AllTime_np,AutE.detach().cpu().numpy(),AllTime_np,Pot.detach().cpu().numpy(),AllTime_np,Kin.detach().cpu().numpy())
    plt.legend(handles,('Total Energy','Autonomous Energy', 'Autonomous Potential', 'Kinetic Energy'))
    plt.ylabel('Energy in J')
    plt.xlabel('Time in s')
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_energy_vs_time.png"))

    plt.figure()
    plt.plot(AllTime_np,(E-E_des[0].cpu()).detach().cpu().numpy())
    plt.ylabel('$\Delta E$ in J')
    plt.xlabel('Time in s')
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_difference_energy_desired_energy_vs_time.png"))

    plt.figure()
    plt.plot(AllTime_np,TrajDist.detach().cpu().numpy())
    plt.xlabel('Time in s')
    plt.ylabel('$dist(q,q_{des})$')
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_dist_to_learned_eigenmode_q_vs_time.png"))

    plt.figure()
    plt.plot(AllTime_np,MomentumDiff.detach().cpu().numpy())
    plt.xlabel('Time in s')
    plt.ylabel('$ \|p - p_{des} \|_2$')
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_dist_to_learned_eigenmode_p_vs_time.png"))

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
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_states_"
                                       "AND_desired_trajectory_states_vs_time.png"))

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
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_coordinates_"
                                       "AND_desired_trajectory_coordinates_vs_time.png"))

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
    plt.savefig(os.path.join(save_dir, "controlled_trajectory_momenta_"
                                       "AND_desired_trajectory_momenta_vs_time.png"))

    # Check Distance plot:
    k = 100 # spatial resolution
    d = (torch.arange(k+1).to(device)-k/2)/k*4
    d_x = d #+ u0[0,0]
    d_y = d +1#+ u0[0,1]
    DistFun = np.zeros((k+1,k+1))
    for i in range(k):
        for j in range(k):
            QC = torch.tensor((d_x[i],d_y[j]),device=device).unsqueeze(0)
            DistFun[j,i] = f._min_d(QC) # This has to be in order (j,i), contourf takes first slot for y data, second slot for x data

    X,Y = np.meshgrid(d_x.cpu().numpy(),d_y.cpu().numpy())

    plt.figure()
    plt.contourf(X,Y,DistFun,20)
    q0 = xT_ctrl[:,0]; q1 = xT_ctrl[:,1]
    plt.plot(q0,q1,'r--')
    plt.savefig(os.path.join(save_dir, "distance_to_eigenmode_and_coordinates_controlled_trajectory.png"))

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
    plt.savefig(os.path.join(save_dir, "distance_to_eigenmode_and_"
                                       "coordinates_controlled_trajectory_colored_by_time.png"))
