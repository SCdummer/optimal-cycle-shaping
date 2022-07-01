import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    def __init__(self, spatial_dim: int):
        super().__init__()
        self.spatial_dim = spatial_dim

    def forward(self, x):
        """"
        x:  batch_size x spatial_dim
        """
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class KernelRegression(nn.Module):
    def __init__(self, kernel):
        super(KernelRegression, self).__init__()
        self.kernel = kernel
        self.lin = nn.Linear(self.kernel.out_dim, 1)

        # Initializing the weights and biases to zero
        for param in self.lin.parameters():
            param.data = nn.parameter.Parameter(torch.zeros_like(param))

    def forward(self, x):
        return self.lin(self.kernel(x))


class KernelFunc(nn.Module):
    def __init__(self, kernel, kernel_locations, is_torus=True):
        super(KernelFunc, self).__init__()
        self.kernel_locations = kernel_locations.cuda()
        self.kernel = kernel
        self.is_torus = is_torus
        self.out_dim = self.kernel_locations.shape[0]

    def pretransform(self, x):
        xsin = torch.sin(x)
        xcos = torch.cos(x)
        return torch.cat([xsin, xcos], dim=1)

    def transform(self, x):
        if self.is_torus:
            x_tilde = (self.pretransform(x).unsqueeze(1) - self.pretransform(self.kernel_locations))
        else:
            x_tilde = x.unsqueeze(1) - self.kernel_locations

        return x_tilde

    def forward(self, x):
        return self.kernel(self.transform(x))


class ReluKernel(nn.Module):
    def __init__(self, scaling):
        super(ReluKernel, self).__init__()
        self.scaling = scaling

    def forward(self, x):
        return torch.relu(1 - torch.norm(x, dim=2)/self.scaling)



