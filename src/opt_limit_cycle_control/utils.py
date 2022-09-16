import torch
from torch.utils import data as data
from torch.distributions import MultivariateNormal, Uniform, Normal
from pytorch_lightning import LightningDataModule
import numpy as np
import os.path
import scipy.io
from scipy.interpolate import interp1d
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


def log_likelihood_loss(x, target):
    # negative log likelihood loss
    return -torch.mean(target.log_prob(x))


def weighted_log_likelihood_loss(x, target, weight):
    # weighted negative log likelihood loss
    log_prob = target.log_prob(x)
    weighted_log_p = weight * log_prob
    return -torch.mean(weighted_log_p.sum(1))


def weighted_L2_loss(xh, x, weight):
    # weighted squared error loss
    e = xh - x
    return torch.einsum('ij, jk, ik -> i', e, weight, e).mean()


def prior_dist(q_min, q_max, p_min, p_max, device='cpu'):
    # uniform "prior" distribution of initial conditions x(0)=[q(0),p(0)]
    lb = torch.Tensor([q_min, p_min]).to(device)
    ub = torch.Tensor([q_max, p_max]).to(device)
    return Uniform(lb, ub)


def target_dist(mu, sigma, device='cpu'):
    # normal target distribution of terminal states x(T)
    mu, sigma = torch.Tensor(mu).reshape(1, 2).to(device), torch.Tensor(sigma).reshape(1, 2).to(device)
    return Normal(mu, torch.sqrt(sigma))


def multinormal_target_dist(mu, sigma, device='cpu'):
    # normal target distribution of terminal states x(T)
    mu, sigma = torch.Tensor(mu).to(device), torch.Tensor(sigma).to(device)
    return MultivariateNormal(mu, sigma*torch.eye(mu.shape[0]).to(device))


def PoU_S1(q):
    # Partition of unity for one chart of circle (other is 1 minus this)
    return 1/2*(1+np.cos(q))


def PoU_Sn(q_Sn):
    # Extension of PoU_S1 to the 2**n chart-regions of the n-Torus
    PoU_Raw_1 = PoU_S1(q_Sn);                                                   # PoUs for chart 1
    PoU_Raw_2 = 1 - PoU_Raw_1;                                                  # PoUs for chart 2

    n = len(q_Sn);            
    PoU_Fin = np.ones((2**n));           
    for i in range(n):
        ind1 = ind(i,2**n);                                                     # components to be multiplied by PoU of chart 1
        ind2 = ind1 + 2**(n-i-1);                                               # components to be multiplied by PoU of chart 2
        for k in range(2**(n-1)):
            PoU_Fin[int(ind1[k])] = PoU_Fin[int(ind1[k])]*PoU_Raw_1[i];
            PoU_Fin[int(ind2[k])] = PoU_Fin[int(ind2[k])]*PoU_Raw_2[i];
    
    PoU_Fin = PoU_Fin/np.sum(PoU_Fin);                                          # normalization
    
    return PoU_Fin


def ind(i,N):
    if i == 0:
        ind1 = np.array([i for i in range(int(N/2))])                           # first half
    elif i > 0:
        ind1 = ind(i-1,N/2);                                                    # find first half of first half
        ind1 = np.concatenate((ind1,ind1+N/2))
    return ind1


def chart_Sn(q_Sn):
    # local diffeomorphism: globally surjective, but not injective
    q_1 = np.mod(q_Sn+np.pi, 2*np.pi)-np.pi;
    q_2 = np.mod(q_Sn, 2*np.pi)-np.pi;
    
    n = len(q_Sn);            
    q_all = np.ones((2**n,n));           
    for i in range(n):
        ind1 = ind(i,2**n);                                                     # components in chart 1
        ind2 = ind1 + 2**(n-i-1);                                               # components in chart 2
        for k in range(2**(n-1)):
            q_all[int(ind1[k]),i] = q_1[i];
            q_all[int(ind2[k]),i] = q_2[i];

    return q_all


def chart_inv_Sn(q_all):
    # inverse unique if for all i: |q_Sn[i]| < pi  
    return q_all[0,:]


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

    NOTE: traj[i,:] is sampled at time time[i,:]

    """

    # Checking whether the value of n is correct
    if not (n == 1 or n == 2):
        raise ValueError("The value of n should be 1 or 2")

    # Getting the name of the file we want load
    filename = "Trajectories_DP_k_".join(str(k)).join(".mat")

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

def interp_via_scipy(x,y, device, kind='linear',axis=-1,copy=True,bounds_error=None,fill_value=np.nan,assume_sorted=False):
    X = x.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    fnp = interp1d(X,np.transpose(Y),kind=kind,axis=axis,copy=copy,bounds_error=bounds_error,fill_value=fill_value,assume_sorted=assume_sorted)
    f = lambda x: torch.tensor(fnp(x.detach().cpu().numpy())).to(device).float().squeeze()
    return f

def interp_torch(x,y):
    return interp_func(x,y)

def numJ(f,x,dx):
    "Numerical Jacobian Matrix using center difference"
    nx, nf = x.size()[0], f(x).size()[0]
    I, dfdx = torch.eye(nx).to(x), torch.zeros((nf,nx)).to(x)
    for i in range(nx):
        dfdx[:,i] = (f(x+dx*I[:,i]) - f(x-dx*I[:,i]))/(2*dx)
    return dfdx

def find_orbit(f,x,dx,n=5):
    # try n newton iterations to find fixed point x_ = f(x_)
    x_ = x
    for i in range(n):
        df_ = f(x_)-x_
        dfdx = numJ(f,x,dx)
        x_ = x - torch.inv(dfdx)@df_
    df_, df = f(x_)-x_, f(x)-x
    choice = torch.where(torch.norm(df_)-torch.norm(df)<0,1,0)
    return choice*x_ + torch.abs((choice-1))*x

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