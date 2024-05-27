import numpy as np
import torch
import torch.nn as nn

class LinearParallel(nn.Module):
    """ Repurposed from the SDCD repository under an MIT liscence """
    def __init__(self, in_dim, out_dim, parallel_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parallel_dim = parallel_dim

        self.weight = nn.Parameter(torch.zeros(parallel_dim, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(parallel_dim, out_dim))
        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, parallel_dim, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, parallel_dim, out_dim)
        """
        xo = torch.einsum("npi, pio -> npo", x, self.weight) + self.bias
        return xo

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, parallel_dim={self.parallel_dim})"


class ParallelNet(nn.Module):
    """ n_parallel neural networks that take n_in variables to n_out outputs.
    
    Parameters:
    n_parallel: int
    n_in: int
    n_out: int
    hidden_width: int
    n_layers: int

    Methods:
    forward(x)
        Applies neural network to a tensor x that is [batch size, n_parallel, n_nodes].
    l1_mat(p)
        Return summed weights of first layer:
        \sum_i |W_{i, m}|^p where $m$ is the input dimension.
    """
    def __init__(self, n_parallel, n_in, n_out,
                 hidden_width, n_layers):
        super().__init__()

        first_layer = [LinearParallel(n_in, hidden_width, n_parallel),
                       nn.Dropout(0.1), nn.ReLU(), nn.BatchNorm1d(n_parallel)]
        final_layer = [LinearParallel(hidden_width, n_out, n_parallel)]
        
        layers = first_layer
        for n in range(n_layers-2):
            layers = layers + [LinearParallel(hidden_width, hidden_width, n_parallel),
                               nn.Dropout(0.5), nn.ReLU(), nn.BatchNorm1d(n_parallel)]
        layers = layers + final_layer
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)

    def l1_mat(self, p=1):
        return torch.abs(next(self.regressor.parameters())**p).sum(axis=-1)
