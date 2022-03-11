import torch
import torch.nn as nn
from torch.utils import data as data
from torch.distributions import MultivariateNormal, Uniform, Normal
import numpy as np

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



def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM

class PSD(torch.nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        if diag_dim == 1:
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, diag_dim)

            for l in [self.linear1, self.linear2, self.linear3]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization
            
            self.nonlinearity = nn.Softplus()
        else:
            assert diag_dim > 1
            self.diag_dim = diag_dim
            self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.linear4 = torch.nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

            for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
                torch.nn.init.orthogonal_(l.weight) # use a principled initialization
            
            self.nonlinearity = nn.Softplus()


    def forward(self, q):
        if self.diag_dim == 1:
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            return h*h + 0.1
        else:
            bs = q.shape[0]
            h = self.nonlinearity( self.linear1(q) )
            h = self.nonlinearity( self.linear2(h) )
            h = self.nonlinearity( self.linear3(h) )
            diag, off_diag = torch.split(self.linear4(h), [self.diag_dim, self.off_diag_dim], dim=1)
            # diag = torch.nn.functional.relu( self.linear4(h) )

            L = torch.diag_embed(diag)

            ind = np.tril_indices(self.diag_dim, k=-1)
            flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            L = torch.flatten(L, start_dim=1)
            L[:, flat_ind] = off_diag
            L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

            D = torch.bmm(L, L.permute(0, 2, 1))
            D[:, 0, 0] = D[:, 0, 0] + 0.01
            D[:, 1, 1] = D[:, 1, 1] + 0.01
            return D