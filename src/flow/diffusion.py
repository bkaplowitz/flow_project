"""
Implement common forms of probability paths
"""

import torch
from torch import Tensor

from flow.distributions import Gaussian
from flow.types.diffusion import Alpha, Beta, ConditionalProbabilityPath
from flow.types.probability import Sampleable


class LinearAlpha(Alpha):
    """
    Alpha_t =t
    """

    def __call__(self, t: Tensor) -> Tensor:
        return t

    def dt(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    """
    Beta_t = \\sqrt{1-t}
    """

    def __call__(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - t)

    def dt(self, t: Tensor) -> Tensor:
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)


# Diffusions
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    A gaussian conditional probability path, starting from initial gaussian distribution.

    """

    def __init__(self, p1: Sampleable, alpha: Alpha, beta: Beta):
        self.dim = p1.dim
        p0 = Gaussian.isotropic(p1.dim, 1.0)
        super().__init__(p0, p1)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int):
        return self.p1.sample(num_samples)  # x1 (num_samples, dim)

    def sample_conditional_path(self, x1: Tensor, t: Tensor) -> Tensor:
        """
        Sample xt ~ p_t(x|x1) = N(x; alpha_t * x1, beta_t**2 * I_d)

        Args:
            - x1: conditioning variable / data sample (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - xt: samples from p_t(x|x1) (num_samples, dims)
        """
        # alpha_t * x1 + beta_t * epsilon, where epsilon ~ N(0,I) = p0
        return self.alpha(t) * x1 + self.beta(t) * torch.randn_like(x1)

    def conditional_vector_field(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate the conditional vector field u_t(x|x1). Given by:
        u_t(x|x1) = (a'_t - (b'_t / b_t) * a_t) * x1 + (b'_t / b_t) * xt
        Note: Only defined for t in [0,1)

        Args:
            - xt: position variable (num_samples, dims)
            - x1: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dims)
        """
        dlogbt = self.beta.dt(t) / self.beta(t)
        return (self.alpha.dt(t) - dlogbt * self.alpha(t)) * x1 + dlogbt * xt

    def conditional_score(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Evaluates the conditional score of p_t(x|x1) = N(alpha_t * x1, beta_t**2 * I_d).
        Note: only defined on t in [0,1).

        Args:
            - xt: position variable (num_samples, dims)
            - x1: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_score: conditional score (num_samples, dims)
        """
        return (self.alpha(t) * x1 - xt) / (self.beta(t) ** 2)
