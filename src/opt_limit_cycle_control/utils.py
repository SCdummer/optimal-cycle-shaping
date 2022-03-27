import torch
from torch.utils import data as data
from torch.distributions import MultivariateNormal, Uniform, Normal
from pytorch_lightning import LightningDataModule
import numpy as np


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
    

