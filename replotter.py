import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

from src.opt_limit_cycle_control.plotter import plot_trajectories
from src.opt_limit_cycle_control.models import ControlledSystemDoublePendulum, AugmentedDynamicsDoublePendulum
from src.opt_limit_cycle_control.learners import ControlEffort
from src.opt_limit_cycle_control.layers import FourierEncoding

import os
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def creating_plots(model, u0, saving_dir, target, v_in, l_task_k):
    print("\n Now creating the plots!")

    # Plotting the final results.
    num_points = 1000
    num_data = 200
    angles = torch.cat(
        [torch.linspace(-np.pi, np.pi, num_points).view(-1, 1), torch.linspace(-np.pi, np.pi, num_points).view(-1, 1)],
        dim=1)

    q1, q2 = torch.tensor(np.meshgrid(np.linspace(-np.pi, np.pi, num_data).astype('float32'),
                                      np.linspace(-np.pi, np.pi, num_data).astype('float32')))

    q1 = q1.reshape((num_data * num_data, 1))
    q2 = q2.reshape((num_data * num_data, 1))
    q = torch.cat([q1, q2], dim=1)

    with torch.no_grad():
        xT = odeint(model.f.cuda(), torch.cat([u0, torch.zeros(u0.size()).cuda()], dim=1).cuda(),
                    torch.linspace(0, 1, num_points).cuda(), method='midpoint').squeeze(1).cpu().detach().numpy()

    vu = model.f(torch.linspace(0, 1, num_points).cuda(), q.detach().cuda(),
                       V_only=True).cuda().detach().cpu().numpy()

    vu2 = model.f(torch.linspace(0, 1, num_points).cuda(), torch.tensor(xT[..., 0:2]).detach().cuda(),
                        V_only=True).cuda().detach().cpu().numpy()
    T = model.f.T[0].item()
    plotting_dir = os.path.join(saving_dir, "Figures")
    plot_trajectories(xT=xT, target=target.reshape(1, -1).cpu(), V=vu[:, 0].reshape(num_data * num_data, 1),
                      angles=angles.numpy().reshape(num_points, v_in),
                      u=vu[:, -v_in:].reshape(num_data * num_data, v_in),
                      l1=1, l2=1, pendulum=False, plot3D=False, c_eff_penalty=l_task_k, T=T, q1=q1.cpu().numpy(),
                      q2=q2.cpu().numpy(), u2=vu2[:, -v_in:].reshape(num_points, v_in), plotting_dir=plotting_dir)
    print("Created and saved the plots")


if __name__ == "__main__":

    ### Things to specify yourself ###

    # Specify the initial point and the target
    u0_init = [0.0, 0.0]
    target = torch.tensor([1.5, 1.5])

    T_requires_grad = False
    T_initial = 3.0

    # Give the main directory name in which you save the different kind of experiments
    main_dir = "Experiments to use in paper"

    # Give the folder name
    #folder_name = os.path.join("Experiments to use in paper", "Two decent experiments", "30-09-2022_14h-36m-49s")
    folder_name = os.path.join("30-09-2022_17h-13m-25s")
    #folder_name = "03-10-2022_20h-01m-43s"

    # Indicate if you use the 'no_reg' file or not
    use_no_reg_file = True

    ### Finished specifying things yourself ###

    if use_no_reg_file:
        string_to_add = '_no_reg'
    else:
        string_to_add = ''

    # Get the full directory
    directory = os.path.join(main_dir, "DoublePendulum_{}_{}_to_{}_{}".format(u0_init[0], u0_init[1],
                                                                                               target[0].item(),
                                                                                               target[1].item()) +
                             string_to_add, folder_name)



    # Load the config file
    with open(os.path.join(directory, 'config.json')) as f:
        config = json.load(f)

    # Loading all the variables into memory
    for key, val in config.items():
        exec(key + '=val')

    # Initializing the model
    V = nn.Sequential(
        FourierEncoding(v_in),
        nn.Linear(2 * v_in, hdim),
        nn.Tanh(),
        nn.Linear(hdim, hdim),
        nn.Tanh(),
        nn.Linear(hdim, 1))

    # Creating the model
    f = ControlledSystemDoublePendulum(V, T_initial=T_initial, T_requires_grad=T_requires_grad).to(device)
    aug_f = AugmentedDynamicsDoublePendulum(f, ControlEffort(f))

    # Creating the final model
    aug_f.load_state_dict(torch.load(os.path.join(directory, 'model_state_dict.pt')))
    aug_f.eval()

    # We forgot to add the correct l_task_k coefficient to the config file. We retrieve it from the figures that we
    # already made.
    filename = [filename for filename in os.listdir(os.path.join(directory, "Figures")) if filename.startswith("DoublePendulumTrajectory_")][0]
    l_task_k_string = filename.split("DoublePendulumTrajectory_", 1)[1][:-5]
    l_task_k = float(l_task_k_string)

    # Actually recreating the plots
    creating_plots(aug_f, torch.tensor(u0_init).reshape(1, 2).to(device), directory, torch.tensor(target).reshape(2, 1).to(device), v_in, l_task_k)



