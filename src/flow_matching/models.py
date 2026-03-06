"""Implements torch models for flow and score matching."""

import torch
from torch import Tensor, nn


def make_mlp(dims: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers = []
    final_idx = len(dims) - 2
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < final_idx:
            layers.append(activation())
    return nn.Sequential(*layers)


class MLPVectorField(nn.Module):
    r"""A learnable MLP vector field of some corruption process $u_t^{ref}$.

    Represented by $u_t^{\theta}(x).

    Takes in (x, t) and returns the estimated marginal vector field at that point.

    Uses a MLP architecture with data dim `dim` and hidden dims `hidden_dims`.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: list[int],
        activation: type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.dim = dim
        input_dim = dim + 1  # x dim + t.
        output_dim = dim
        all_dims = [input_dim, *hidden_dims, output_dim]
        self.net = make_mlp(all_dims, activation)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes u^theta_t(x).

        Args:
            - x: state, shape (bs, dims)
            - t: time, shape (bs, 1)

        Returns:
            - u_t^{theta}: vector field, shape (bs, dim)
        """
        return self.net(torch.cat([x, t], dim=-1))


class MLPScore(nn.Module):
    """MLP-parameterization of learned score vector field."""

    def __init__(
        self, dim: int, hidden_dims: list[int], activation: type[nn.Module] = torch.nn.SiLU
    ):
        super().__init__()
        input_dim = dim + 1  # x dim + t
        output_dim = dim
        all_dims = [input_dim, *hidden_dims, output_dim]
        self.net = make_mlp(all_dims, activation)

    def forward(self, x: Tensor, t: Tensor):
        """Computes score at a given (x,t) coordinate.

        Args:
            - x: shape (bs, dim)
            - t: shape (bs, 1)

        Returns:
            - s_t^{theta}(x)
        """
        return self.net(torch.cat([x, t], dim=-1))
