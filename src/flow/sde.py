"""
Implementation of specific SDEs.
"""

import torch
from torch import Tensor

from flow.types.ode import SDE


class BrownianMotion(SDE):
    """A Brownian motion process. No drift and constant variance iid gaussian noise."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(xt)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(xt)


class OrnsteinUhlenbeckProcess(SDE):
    def __init__(self, theta: Tensor, sigma: Tensor) -> Tensor:
        self.theta = theta
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return -self.theta * torch.ones_like(xt)  # shape: (bs, dim)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(xt)  # shape: (bs, dim)
