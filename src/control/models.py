import torch; import torch.nn as nn
from torch.autograd import grad as grad

# parameters
m, k, l, qr, b, g = 1., 0.5, 1, 0, 0.01, 9.81

class ControlledSystem(nn.Module):
    # Elastic Pendulum Model
    def __init__(self, V, K):
        super().__init__()
        self.V, self.K, self.n = V, K, 1

    def forward(self, t, x):
        # Evaluates the closed-loop vector field
        with torch.set_grad_enabled(True):
            q, p = x[..., :self.n], x[..., self.n:]
            q = q.requires_grad_(True)
            # compute control action
            u = self._energy_shaping(q) + self._damping_injection(x)
            # compute dynamics
            dxdt = self._dynamics(q, p, u)
        return dxdt

    def _dynamics(self, q, p, u):
        # controlled elastic pendulum dynamics
        dqdt = p / m
        dpdt = -k * (q - qr) - m * g * l * torch.sin(q) - b * p / m + u
        return torch.cat([dqdt, dpdt], 1)

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

    def _damping_injection(self, x):
        # damping injection control action
        return -self.K(x) * x[:, self.n:] / m

    def _autonomous_energy(self, x):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        return (m * x[:, 1:] ** 2) / 2. + (k * (x[:, :1] - qr) ** 2) / 2 \
               + m * g * l * (1 - torch.cos(x[:, :1]))

    def _energy(self, x):
        # Hamiltonian (total energy) of the CONTROLLED system
        return (m * x[:, 1:] ** 2) / 2. + (k * (x[:, :1] - qr) ** 2) / 2 \
               + m * g * l * (1 - torch.cos(x[:, :1])) + self.V(x[:, :1])


class AugmentedDynamics(nn.Module):
    # "augmented" vector field to take into account integral loss functions
    def __init__(self, f, int_loss):
        super().__init__()
        self.f = f
        self.int_loss = int_loss
        self.nfe = 0.

    def forward(self, t, x):
        self.nfe += 1
        x = x[:,:2]
        dxdt = self.f(t, x)
        dldt = self.int_loss(t, x)
        return torch.cat([dxdt, dldt], 1)