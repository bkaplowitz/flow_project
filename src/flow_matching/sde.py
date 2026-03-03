"""Implementation of specific SDEs."""

import torch
from torch import Tensor

from flow_matching.base.dynamics import SDE
from flow_matching.base.probability import Density


class BrownianMotion(SDE):
    r"""A Brownian motion process, defined by $dX_t = \sigma dW_t$.

    Drift: $u_t(X_t)=0$.
    Diffusion: $\sigma_t(X_t) = \sigma$.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(xt)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(xt)


class OrnsteinUhlenbeckProcess(SDE):
    r"""An Ornstein Uhlenbeck process defined by $dX_t = -\theta X_t + \sigma dW_t$.

    Drift: $u_t(X_t) = -\theta X_t$.
    Diffusion: $\sigma_t(X_t) = \sigma$.
    """

    def __init__(self, theta: float, sigma: float) -> Tensor:
        self.theta = theta
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return -self.theta * xt  # shape: (bs, dim)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(xt)  # shape: (bs, dim)


class LangevinSDE(SDE):
    r"""Overdamped Langevin SDE.

    Defined by $dX_t = \frac{1}{2} \sigma^2 \nabla \log p(X_t)dt + \sigma dW_t$.

    Drift $u_t(X_t) = 1/2 \sigma^2 \grad \log p(X_t)$ for some target $p$ density.
    Diffusion $\sigma_t(X_t) = \sigma$.
    """

    def __init__(self, sigma: float, density: Density) -> Tensor:
        self.sigma = sigma
        self.density = density

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return 0.5 * self.sigma**2 * self.density.score(xt)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        return self.sigma * torch.ones_like(xt)
