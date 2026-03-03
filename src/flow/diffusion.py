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

    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta):
        self.dim = p_data.dim
        p_simple = Gaussian.isotropic(p_data.dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int):
        return self.p_data.sample(num_samples)  # z (num_samples, dim)

    def sample_conditional_path(self, z: Tensor, t: Tensor) -> Tensor:
        """
        sample x ~ p(x|z) = N(x; alpha z, beta**2 I_d)

        Args:
            - z: conditioning variable/data (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - x: samples from p_t(x|z) (num_samples, dims)
        """

        # a_t z_{data} + beta_t * epsilon_t, where here epsilon_t ~ N(0,I) = p_simple.
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluate the conditional vector field u\\_t(x|z). We know this is given by:
        $u\\_t(x|z)=(\\dot{a}\\_t-\\frac{\\dot{b}\\_t}{b\\_t} a\\_t)z + \\frac{\\dot{b\\_t}{b\\_t}x$
        Note: Only defined for t in [0,1)

        Args:
            - x: position variable (num_samples, dims)
            - z: ccnditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dims)
        """
        # Samll exception for out of bounds error.
        # self.oob_check(t)
        # Main implementation
        dlogbt = self.beta.dt(t) / self.beta(t)
        return (self.alpha.dt(t) - dlogbt * self.alpha(t)) * z + dlogbt * x

    def conditional_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t ** 2 * I_d).
        Note: only defined on t in [0,1).

        Args:
            - x: position variable (num_samples, dims)
            - z: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_score: conditional score (num_samples, dims)
        """
        # self.oob_check(t)

        return (self.alpha(t) * z - x) / (self.beta(t) ** 2)
