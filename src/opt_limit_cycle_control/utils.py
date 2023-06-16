import torch
from torch.utils import data as data
from pytorch_lightning import LightningDataModule
import os.path
import scipy.io
from src.opt_limit_cycle_control.torch_cubic_spline_interp import interp_func


class DummyDataModule(LightningDataModule):
    def __init__(
        self,
        num_samples: int = 1
    ):
        super().__init__()
        self.num_samples = num_samples

    def train_dataloader(self):
        # dummy trainloader for Lightning learner
        dummy = data.DataLoader(
            data.TensorDataset(
                torch.arange(self.num_samples, dtype=torch.float32).view(self.num_samples, 1)
            ),
            batch_size=1,
            shuffle=False
        )

        return dummy


def dummy_trainloader():
    # dummy trainloader for Lightning learner
    dummy = data.DataLoader(
        data.TensorDataset(
            torch.Tensor(1, 1),
            torch.Tensor(1, 1)
        ),
        batch_size=1,
        shuffle=False
    )
    return dummy


def load_eig_mode_double_pendulum(n, k, i):
    """"
    This function loads a precalculated eigenmode of the double pendulum.
    Inputs:
    - n:        This determines which of the two eigenmanifolds we pick. The value can be 1 or 2.
    - k:        This determines the value of k we use. What this value is depends on the files present in
                DP_Eigenmodes/DP_Eigenmodes. If Trajectories_DP_k_k.mat is present, we can use the value of k. E.g. if
                Trajectories_DP_k_0.mat is present, we can use k = 0.
    - i:        This takes the ith trajectory in the list of trajectories on the eigenmanifold.

    Output:
    - time:     A (t_points, 1) numpy array containing the times at which we sample the trajectory. Here we have
                that t_points is the number of time points at which we sample.
    - traj:     A (t_points, 4) numpy array with:
                - column 1: q1
                - column 2: q2
                - column 3: dq1/dt
                - column 4: dq2/dt
                Here t_points again corresponds to the number of time points at which we sample these values.
af
    NOTE: traj[i,:] is sampled at time time[i,:]

    """

    # Checking whether the value of n is correct
    if not (n == 1 or n == 2):
        raise ValueError("The value of n should be 1 or 2")

    # Getting the name of the file we want load
    filename = "Trajectories_DP_k_"+str(k)+".mat"

    # Get exact location of the file
    matfile_dir = os.path.join("DP_Eigenmodes/DP_Eigenmodes/", filename)

    # Check whether any files exist at all in the directory. If not, give an error.
    if len(os.listdir("DP_Eigenmodes/DP_Eigenmodes")) == 0:
        raise FileNotFoundError("No files in the DP_Eigenmodes/DP_Eigenmodes directory")
    else:
        # Check wheter Trajectories_DP_k_k.mat files exist in the trajectory. If not, give an error.
        found_file = False
        for file in os.listdir("DP_Eigenmodes/DP_Eigenmodes"):
            if file.startswith("Trajectories_DP_k"):
                backup_file = file
                found_file = True
                break
        if not found_file:
            raise FileNotFoundError("No Trajectories_DP_k_k.mat file present in the directory")

    # Check if the file for our specific value of k exists in the trajectory
    if os.path.exists(matfile_dir):
        matfile = scipy.io.loadmat(matfile_dir)
    else:
        # If not, use the backup file in the directory
        matfile_dir = os.path.join("DP_Eigenmodes/DP_Eigenmodes/", backup_file)
        filename = backup_file
        Warning("The k you have chosen does not correspond to a Trajectories_DP_k_k.mat file in the " +
                "DP_Eigenmodes/DP_Eigenmodes directory. We continue by using " + backup_file + " instead")
        matfile = scipy.io.loadmat(os.path.join(matfile_dir))

    # Check if the value of i is correct
    if not (1 <= i <= matfile["Y"][n - 1, 0].shape[0]):
        error_message = "The value of i should be between 1 and " + str(matfile["Y"][n - 1, 0].shape[0]) + ". Here " + \
                        str(matfile["Y"][n - 1, 0].shape[0]) + " is the number of trajectories in " + filename + \
                        " corresponding to eigenmanifold n."
        raise ValueError(error_message)

    # Get the trajectory data and the time data
    traj = matfile["Y"][n - 1, 0][i - 1][0]
    time = matfile["Y"][n - 1, 2][i - 1][0]

    return time, traj


def interp_torch(x,y):
    return interp_func(x,y)


def numJ(f,x,dx):
    "Numerical Jacobian Matrix using center difference"
    nx, nf = x.size()[0], f(x).size()[0]
    I, dfdx = torch.eye(nx).to(x), torch.zeros((nf,nx)).to(x)
    for i in range(nx):
        dfdx[:,i] = (f(x+dx*I[:,i]) - f(x-dx*I[:,i]))/(2*dx)
    return dfdx


def numJ2(f,x,dx):
    "Numerical Jacobian Matrix using center difference"
    nx, nf = x.squeeze().size(0), f(x).size(-1)#f(x).squeeze().size(-1)
    I, dfdx = torch.eye(nx).to(x), torch.zeros((nf,nx)).to(x)
    for i in range(nx):
        dfdx[:,i] = (f(x+dx*I[:,i]) - f(x-dx*I[:,i]))/(2*dx)
    return dfdx


def cuberoot(coeff): 
    # Only returns one of the real solutions
    # if a =/= 0, p and q are real and 4p^3+27q^2 > 0, then there is a real root 
    a,b,c,d = coeff[0],coeff[1],coeff[2],coeff[3]
    if (a > 0):
        p = (3*a*c-b**2)/(3*a**2)
        q = (2*b**3-9*a*b*c + 27*a**2*d)/(27*a**3)
        m = torch.sqrt((q**2)/4+(p**3)/27)
        r = torch.pow(-q/2+m,1/3)+torch.pow(-q/2-m,1/3)
    elif(b>0):
        p = c/b
        q = d/b
        r = -p/2 + torch.sqrt(p**2/4-q)
    else:
        r = -d/c;
    return r


def traj_to_qp(xT):
    xT = torch.tensor(xT)
    n = xT.size(0)
    Q,P = [],[]
    for i in range(n):
        q = xT[i,:2].unsqueeze(0)
        p = xT[i,2:].unsqueeze(0)
        Q += [q]
        P += [p]
    return Q,P
