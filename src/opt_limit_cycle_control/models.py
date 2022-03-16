import torch; import torch.nn as nn
from torch.autograd import grad as grad

# parameters pendulum
m, k, l, qr, b, g = 1., 0.5, 1, 0, 0.01, 9.81

# parameters double pendulum
m1, m2, k1, k2, l1, l2, qr1, qr2, b1, b2 = 1., 1., 0.5, 0.5, 1, 1, 0, 0, 0.01, 0.01

class ControlledSystemNoDamping(nn.Module):
    # Elastic Pendulum Model
    def __init__(self, V):
        super().__init__()
        self.V, self.T, self.n = V, torch.nn.Parameter(torch.tensor([1.0])), 1

    def forward(self, t, x):
        # Evaluates the closed-loop vector field
        with torch.set_grad_enabled(True):
            q, p = x[..., 0], x[..., 1]
            q = q.requires_grad_(True)
            # compute control action
            u = self._energy_shaping(q)
            # compute dynamics
            dxdt = torch.abs(self.T[0]) * self._dynamics(q, p, u)

        return dxdt

    def _dynamics(self, q, p, u):
        # controlled elastic pendulum dynamics
        dqdt = p / m
        dpdt = -k * (q - qr) - m * g * l * torch.sin(q) - b * p / m + u
        return torch.cat([dqdt, dpdt])

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

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
        return torch.cat([dxdt, dldt])



class ControlledSystemDoublePendulum(nn.Module):
    # Elastic Double Pendulum Model
    def __init__(self, V):
        super().__init__()
        self.V, self.T, self.n = V, torch.nn.Parameter(torch.tensor([1.0])), 1

    def forward(self, t, x):
        # Evaluates the closed-loop vector field
        with torch.set_grad_enabled(True):
            q1, q2, p1, p2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            q1 = q1.requires_grad_(True)
            q2 = q2.requires_grad_(True)
            # compute control action
            q = torch.cat([q1, q2])
            p = torch.cat([p1, p2])
            u = self._energy_shaping(q)
            # compute dynamics
            dxdt = torch.abs(self.T[0]) * self._dynamics(q, p, u)

        return dxdt

    def _dynamics(self, q, p, u):
        u1 = u[0].unsqueeze(-1)
        u2 = u[1].unsqueeze(-1)
        q1 = q[0].unsqueeze(-1)
        q2 = q[1].unsqueeze(-1)
        p1 = p[0].unsqueeze(-1)
        p2 = p[0].unsqueeze(-1)
        # controlled elastic double pendulum dynamics
        dq1dt = (l2 * p1 - l1 * p2 * torch.cos(q1-q2)) / (l1**2 * l2 * (m1 + m2 * torch.sin(q1-q2)**2))
        dq2dt = (-m2 * l2 * p1 * torch.cos(q1-q2) + (m1 + m2) * l1 * p2) / (m2 * l1 * l2**2 * (m1 + m2 * torch.sin(q1-q2)**2))

        h1 = (p1 * p2 * torch.sin(q1-q2)) / (l1 * l2 * ((m1 + m2 * torch.sin(q1-q2)**2)))
        h2 = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * p1 * p2 * torch.cos(q1-q2)) / (2 * l1**2 * l2**2 * (m1 + m2 * torch.sin(q1-q2)**2))


        dp1dt = -(m1 + m2) * g * l1 * torch.sin(q1) - h1 + h2 * torch.sin(2 * (q1-q2)) + u1
        dp2dt = -m2 * g * l2 * torch.sin(q2) + h1 - h2 * torch.sin(2 * (q1-q2)) + u2

        # TODO: add spring and damping to these equations

        return torch.cat([dq1dt, dq2dt, dp1dt, dp2dt])

    def _energy_shaping(self, q):
        # energy shaping control action
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        return -dVdx

    def _autonomous_energy(self, x):
        # Hamiltonian (total energy) of the UNCONTROLLED system
        pass

    def _energy(self, x):
        # Hamiltonian (total energy) of the CONTROLLED system
        pass


class AugmentedDynamicsDoublePendulum(nn.Module):
    # "augmented" vector field to take into account integral loss functions
    def __init__(self, f, int_loss):
        super().__init__()
        self.f = f
        self.int_loss = int_loss
        self.nfe = 0.

    def forward(self, t, x):
        self.nfe += 1
        x = x[:,:4]
        dxdt = self.f(t, x)
        dldt = self.int_loss(t, x)
        return torch.cat([dxdt, dldt])