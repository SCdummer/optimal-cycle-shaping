import sys
sys.path.append('')

from src.models import ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.learners import OptEigenManifoldLearner
from src.losses import ControlEffort
from src.utils import DummyDataModule
from src.plotter import plot_trajectories
from src.layers import FourierEncoding

import torch
import torch.nn as nn

import pytorch_lightning as pl
from torchdiffeq import odeint

import numpy as np

import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the parameters needed for the problem
v_in = 2
v_out = 2
spatial_dim = 2

# lengths of the pendulum bars
l1, l2 = 1.0, 1.0


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


def compute_opt_eigenmode(specs, saving_dir):

    # Get the variable names from the specs.json file
    hdim, training_epochs, lr, l_period_k, alpha_1, lambda_1, lambda_2, alpha_eff, alpha_task, T_initial, \
    T_requires_grad, target, u0_init, use_target_angles, use_betascheduler, l1, l2 = from_specs_to_variables(specs)

    # Initialize the potential neural network
    V = nn.Sequential(
        FourierEncoding(v_in),
        nn.Linear(2 * v_in, hdim),
        nn.Tanh(),
        nn.Linear(hdim, hdim),
        nn.Tanh(),
        nn.Linear(hdim, 1))

    # Create the model
    f = ControlledSystemDoublePendulum(V, T_initial=T_initial, T_requires_grad=T_requires_grad).to(device)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))

    # Train the Energy shaping controller.
    learn = OptEigenManifoldLearner(model=aug_f, use_target_angles=use_target_angles, target=target, l1=l1, l2=l2,
                                    T=T_initial, alpha_1=alpha_1, lambda_1=lambda_1, lambda_2=lambda_2,
                                    alpha_eff=alpha_eff, alpha_task=alpha_task, spatial_dim=spatial_dim,
                                    lr=lr, u0_init=u0_init, u0_requires_grad=False, training_epochs=training_epochs,
                                    use_betascheduler=use_betascheduler)

    trainer = pl.Trainer(max_epochs=training_epochs, gpus=[0])
    datloader = DummyDataModule(7)

    print('Initial period:', learn.model.f.T[0])
    print('Initial Initial position:', learn.u0[0])

    trainer.fit(learn, train_dataloader=datloader)

    print('Final period:', learn.model.f.T)
    print('Final Initial position:', learn.u0[0])

    print("\n Now creating the plots!")

    # Plotting the final results.

    # Getting a grid of angles with 200 points along each grid direction. q1 contains the first coordinate of each
    # point in the grid and q2 contains the second coordinate of each point in the grid.
    num_data = 200
    q1, q2 = torch.tensor(np.meshgrid(np.linspace(-np.pi, np.pi, num_data).astype('float32'), np.linspace(-np.pi, np.pi, num_data).astype('float32')))

    # Reshape q1 and q2 into vectors and concatenate them into one vector
    q1 = q1.reshape((num_data*num_data, 1))
    q2 = q2.reshape((num_data*num_data, 1))
    q = torch.cat([q1, q2], dim=1)

    # Calculate the eigenmode trajectory for 1000 time points between 0 and 1.
    num_points = 1000
    with torch.no_grad():
        xT = odeint(learn.model.f.cuda(), torch.cat([learn.u0, torch.zeros(learn.u0.size()).cuda()], dim=1).cuda(),
                    torch.linspace(0, 1, num_points).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()

    # Calculate the potential on the grid (q1, q2) (whose vectorized form is q!)
    vu = learn.model.f(torch.linspace(0, 1, num_points).cuda(), q.detach().cuda(), V_only=True).cuda().detach().cpu().numpy()

    # Calculate the potential values along the eigenmode trajectory
    vu2 = learn.model.f(torch.linspace(0, 1, num_points).cuda(), torch.tensor(xT[..., 0:2]).detach().cuda(),
                       V_only=True).cuda().detach().cpu().numpy()

    # Get the learned period of the eigenmode.
    T = learn.model.f.T[0].item()

    # Create the desired plots
    plotting_dir = os.path.join(saving_dir, "Figures", "Figures_eigenmode")
    plot_trajectories(xT=xT, target=target.reshape(1, -1).cpu(), V=vu[:, 0].reshape(num_data*num_data, 1),
                      angles=None, u=None, l1=l1, l2=l2, alpha_eff=alpha_eff, T=T, q1=q1.cpu().numpy(),
                      q2=q2.cpu().numpy(), u2=vu2[:, -v_in:].reshape(num_points, v_in), plotting_dir=plotting_dir)
    print("Created and saved the plots")

    print("Saving the model")
    torch.save(learn.model.state_dict(), os.path.join(saving_dir, "model_state_dict.pt"))

    print('Job is done!')


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

    # Parse the arguments
    args = arg_parser.parse_args()

    # Load the specs.json file
    specs = json.load(open(os.path.join(args.experiment_directory, "specs.json")))

    # Create a subdirectory in the experiment directory called 'Figures'. Also make a subdirectory in the latter folder
    # called 'Figures_eigenmode'. After finishing training, some figures will be saved in the subfolder
    # 'Figures_eigenmode'.
    if not os.path.isdir(os.path.join(args.experiment_directory, "Figures", "Figures_eigenmode")):
        os.makedirs(os.path.join(args.experiment_directory, "Figures", "Figures_eigenmode"))

    # Find the eigenmode solving a pick-and-place task
    compute_opt_eigenmode(specs, args.experiment_directory)
