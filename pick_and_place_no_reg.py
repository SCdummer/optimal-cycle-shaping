import sys
sys.path.append('')

from src.opt_limit_cycle_control.models import ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.opt_limit_cycle_control.learners import OptEigManifoldLearner, ControlEffort, CloseToPositions, \
    CloseToPositionsAtTime, CloseToPositionAtHalfPeriod, CloseToActualPositionAtHalfPeriod
from src.opt_limit_cycle_control.utils import DummyDataModule
from src.opt_limit_cycle_control.plotter import plot_trajectories, animate_single_dp_trajectory
from src.opt_limit_cycle_control.layers import KernelRegression, KernelFunc, ReluKernel, FourierEncoding

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchdiffeq import odeint

import numpy as np

import os
import json

import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the parameters needed for the problem
v_in = 2
v_out = 2
hdim = 64
training_epochs = 500
lr = 1e-3
spatial_dim = 2
opt_strategy = 1
l_period_k = 1.0
alpha_p = 0.0
alpha_s = 0.05
alpha_mv = 0.95
l_task_k = 0.0
l_task_2_k = 10

# lengths of the pendulum bars
l1, l2 = 1.0, 1.0

# angular targets for q1 and q2
use_target_angles = False

config = {
    "v_in": v_in,
    "v_out": v_out,
    "hdim": hdim,
    "lr": lr,
    "spatial_dim": spatial_dim,
    "opt_strategy": opt_strategy,
    "l_period_k": l_period_k,
    "alpha_p": alpha_p,
    "alpha_s": alpha_s,
    "alpha_mv": alpha_mv,
    "l_task_k": l_task_k,
    "l_task_2_k": l_task_2_k,
    "l1": l1,
    "l2": l2,
    "use_target_angles": use_target_angles
}

# vector field parametrized by a NN
V = nn.Sequential(
    FourierEncoding(v_in),
    nn.Linear(2 * v_in, hdim),
    nn.Tanh(),
    nn.Linear(hdim, hdim),
    nn.Tanh(),
    nn.Linear(hdim, 1))


def compute_opt_eigenmode(target, u0_init, training_epochs, saving_dir):

    # Create the model
    f = ControlledSystemDoublePendulum(V).to(device)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))

    # Create the config file
    config = {
        "v_in": v_in,
        "v_out": v_out,
        "hdim": hdim,
        "lr": lr,
        "spatial_dim": spatial_dim,
        "opt_strategy": opt_strategy,
        "l_period_k": l_period_k,
        "alpha_p": alpha_p,
        "alpha_s": alpha_s,
        "alpha_mv": alpha_mv,
        "l_task_k": 0.0,
        "l_task_2_k": l_task_2_k,
        "l1": l1,
        "l2": l2,
        "use_target_angles": use_target_angles,
        "u0_init": tuple(u0_init),
        "u0_requires_grad": False,
        "target": (target[0].item(), target[1].item()),
        "training_epochs": training_epochs
    }

    # Train the Energy shaping controller.
    if use_target_angles:
        learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToPositionAtHalfPeriod(target),
                                    l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                    l_task_loss=0.0, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                     spatial_dim=spatial_dim, lr=lr, u0_init=u0_init, u0_requires_grad=False) #u0_init = [-0.5, -0.5]
    else:
        learn = OptEigManifoldLearner(model=aug_f, non_integral_task_loss_func=CloseToActualPositionAtHalfPeriod(target, l1, l2),
                                    l_period=l_period_k, alpha_p=alpha_p, alpha_s=alpha_s, alpha_mv=alpha_mv,
                                    l_task_loss=0.0, l_task_loss_2=l_task_2_k, opt_strategy=opt_strategy,
                                    spatial_dim=spatial_dim, lr=lr, u0_init=u0_init, u0_requires_grad=False)

    logger = WandbLogger(project='optimal-cycle-shaping', name='pend_adjoint')

    trainer = pl.Trainer(max_epochs=training_epochs, logger=logger, gpus=[0])
    datloader = DummyDataModule(7)

    print('Initial period:', learn.model.f.T[0])
    print('Initial Initial position:', learn.u0[0])

    trainer.fit(learn, train_dataloader=datloader)

    print('Final period:', learn.model.f.T)
    print('Final Initial position:', learn.u0[0])

    print("\n Now creating the plots!")

    # Plotting the final results.
    num_points = 1000
    num_data = 200
    angles = torch.cat([torch.linspace(-np.pi, np.pi, num_points).view(-1, 1), torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)], dim=1)

    q1, q2 = torch.tensor(np.meshgrid(np.linspace(-np.pi, np.pi, num_data).astype('float32'), np.linspace(-np.pi, np.pi, num_data).astype('float32')))

    q1 = q1.reshape((num_data*num_data, 1))
    q2 = q2.reshape((num_data*num_data, 1))
    q = torch.cat([q1, q2], dim=1)

    with torch.no_grad():
        xT = odeint(learn.model.f.cuda(), torch.cat([learn.u0, torch.zeros(learn.u0.size()).cuda()], dim=1).cuda(),
                    torch.linspace(0, 1, num_points).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()

    vu = learn.model.f(torch.linspace(0, 1, num_points).cuda(), q.detach().cuda(), V_only=True).cuda().detach().cpu().numpy()

    vu2 = learn.model.f(torch.linspace(0, 1, num_points).cuda(), angles.detach().cuda(),
                       V_only=True).cuda().detach().cpu().numpy()
    T = learn.model.f.T[0].item()
    plotting_dir = os.path.join(saving_dir, "Figures")
    plot_trajectories(xT=xT, target=target.reshape(1, -1).cpu(), V=vu[:, 0].reshape(num_data*num_data, 1),
                      angles=angles.numpy().reshape(num_points, v_in), u=vu[:, -v_in:].reshape(num_data*num_data, v_in),
                      l1=1, l2=1, pendulum=False, plot3D=False, c_eff_penalty=l_task_k, T=T, q1=q1.cpu().numpy(), q2=q2.cpu().numpy(), u2=vu2[:, -v_in:].reshape(num_points, v_in), plotting_dir=plotting_dir)
    print("Created and saved the plots")

    print("Saving the model")
    torch.save(learn.model.state_dict(), os.path.join(saving_dir, "model_state_dict.pt"))

    print("Saving the hyperparameters and, if applicable, the learned initial condition in JSON format")
    with open(os.path.join(saving_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    print('Job is done!')

if __name__ == "__main__":

    if not os.path.isdir("Experiments"):
        os.mkdir("Experiments")

    training_epochs = [600, 600, 600]
    u0_inits = [[0.0, 0.0], [-0.5, -0.5], [math.pi - 0.5, 0.0]]
    targets = [torch.tensor([1.5, 1.5]), torch.tensor([0.75, 0.75]), torch.tensor([math.pi + 0.5, 0.0])]

    for i in range(len(targets)):
        target = targets[i].reshape(2, 1).to(device)
        u0_init = u0_inits[i]
        saving_dir = os.path.join("Experiments", "DoublePendulum_{}_{}_to_{}_{}_no_reg".format(u0_init[0], u0_init[1],
                                                                                               target[0].item(),
                                                                                               target[1].item()))
        if not os.path.isdir(saving_dir):
            os.mkdir(saving_dir)
            os.mkdir(os.path.join(saving_dir, "Figures"))

        compute_opt_eigenmode(target, u0_init, training_epochs[i], saving_dir)
