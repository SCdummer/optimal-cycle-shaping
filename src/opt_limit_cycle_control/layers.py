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
