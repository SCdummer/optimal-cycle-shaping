import torch
from torch.utils import data as data
from torch.distributions import MultivariateNormal, Uniform, Normal
from pytorch_lightning import LightningDataModule


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

