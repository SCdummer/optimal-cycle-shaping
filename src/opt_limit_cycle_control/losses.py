import torch
import torch.nn as nn

# Define the Integral Cost Function
class ControlEffort(nn.Module):
    # control effort integral cost
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            if x.shape[1] == 2:
                q = x[:, :1].requires_grad_(True)
            else:
                q = x[:, :2].requires_grad_(True)
            u = self.f._energy_shaping(q)
        return torch.sum(torch.square(u), dim=1, keepdim=False)  # L2-norm

class CloseToActualPositionAtHalfPeriod(nn.Module):
    # Given a time series as input, this cost function measures how close we are to certain destinations at specific
    # pre-specified times
    def __init__(self, dest_angle, l1, l2):
        super().__init__()
        self.dest_angle = dest_angle.reshape(-1, 1).cuda()
        self.l1 = l1
        self.l2 = l2
        self.half_period_index = None

        # Calculating self.dest
        # compute the desired double pendulum position
        x_pos_first_joint = self.l1 * torch.sin(self.dest_angle[0])
        y_pos_first_joint = -self.l1 * torch.cos(self.dest_angle[0])
        x2 = x_pos_first_joint + self.l2 * torch.sin(self.dest_angle[1] + self.dest_angle[0])
        y2 = y_pos_first_joint - self.l2 * torch.cos(self.dest_angle[1] + self.dest_angle[0])

        self.dest = torch.tensor([x2, y2]).reshape(-1, 1).cuda()

    def forward(self, xt):
        # compute double pendulum position
        x_pos_first_joint = self.l1 * torch.sin(xt[self.half_period_index, 0])
        y_pos_first_joint = -self.l1 * torch.cos(xt[self.half_period_index, 0])
        x2 = x_pos_first_joint + self.l2 * torch.sin(xt[self.half_period_index, 1] + xt[self.half_period_index, 0])
        y2 = y_pos_first_joint - self.l2 * torch.cos(xt[self.half_period_index, 1] + xt[self.half_period_index, 0])

        return (x2 - self.dest[0]) ** 2 + (y2 - self.dest[1]) ** 2

class CloseToPositionAtHalfPeriod(nn.Module):
    # Given a time series as input, this cost function measures how close we are to certain destinations at specific
    # pre-specified times
    def __init__(self, dest):
        super().__init__()
        self.dest = dest.reshape(-1, 1).cuda()
        self.half_period_index = None

    def forward(self, xt, spatial_dim=2):
        return torch.sum(torch.square(xt[self.half_period_index, 0:spatial_dim].unsqueeze(1) - self.dest)) + \
                torch.sum(torch.square(xt[self.half_period_index, spatial_dim:]))

class EigenmodeLoss():
    def __init__(self, spatial_dim=1, lambda_1=1.0, lambda_2=1.0, alpha_1=1.0, half_period_index=1):

        self.spatial_dim = spatial_dim
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_1 = alpha_1
        self.half_period_index = half_period_index

    def compute_loss(self, xT, indices):
        sym_trajectory_loss = self.sym_trajectory_loss(xT, indices)
        middle_vel_loss = self.middle_vel_loss(xT)
        alpha_sum = self.lambda_1 + self.lambda_2
        eig_loss = (self.lambda_1 * sym_trajectory_loss + self.lambda_2 * middle_vel_loss) / alpha_sum
        return eig_loss

    def smooth_max(self, input):
        abs_val = torch.sum(torch.abs(input), dim=-1)
        return torch.max(abs_val)

    def sym_trajectory_loss(self, xT, indices):
        sym_indices = (xT.shape[0] - 1) - indices
        return self.smooth_max(xT[indices, 0:self.spatial_dim] - xT[sym_indices, 0:self.spatial_dim]) + \
               self.alpha_1 * self.smooth_max(xT[indices, self.spatial_dim:] + xT[sym_indices, self.spatial_dim:])

    def middle_vel_loss(self, xT):
        return torch.sum(torch.square(xT[self.half_period_index, self.spatial_dim:]))


