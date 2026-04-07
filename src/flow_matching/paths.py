"""Implement common forms of probability paths."""

import torch
from einops.layers.torch import Rearrange
from torch import Tensor

from flow_matching.base.paths import (
    Alpha,
    Beta,
    ConditionalLabeledProbabilityPath,
    ConditionalProbabilityPath,
)
from flow_matching.base.probability import LabeledSampleable, Sampleable, SampleableDensity
from flow_matching.distributions import Gaussian, IsotropicGaussian


class LinearAlpha(Alpha):
    """Alpha_t = t."""

    def __call__(self, t: Tensor) -> Tensor:
        return t

    def dt(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


class SquareRootBeta(Beta):
    r"""Beta_t = \sqrt{1-t}."""

    def __call__(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - t)

    def dt(self, t: Tensor) -> Tensor:
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)


class LinearBeta(Beta):
    """Beta_t = 1-t."""

    def __call__(self, t: Tensor) -> Tensor:
        return 1 - t

    def dt(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


# Diffusions
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """A gaussian conditional probability path, starting from initial gaussian distribution."""

    def __init__(self, p1: SampleableDensity, alpha: Alpha, beta: Beta):
        self.dim = p1.dim
        p0 = Gaussian.isotropic(p1.dim, 1.0)
        super().__init__(p0, p1)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> Tensor:
        return self.p1.sample(num_samples)  # x1 (num_samples, dim)

    def sample_conditional_path(self, x1: Tensor, t: Tensor) -> Tensor:
        """Sample xt ~ p_t(x|x1) = N(x; alpha_t * x1, beta_t**2 * I_d).

        Args:
            - x1: conditioning variable / data sample (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - xt: samples from p_t(x|x1) (num_samples, dims)
        """
        # alpha_t * x1 + beta_t * epsilon, where epsilon ~ N(0,I) = p0
        return self.alpha(t) * x1 + self.beta(t) * torch.randn_like(x1, device=x1.device)

    def conditional_vector_field(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Evaluate the conditional vector field u_t(x|x1).

        Given by: u_t(x|x1) = (a'_t - (b'_t / b_t) * a_t) * x1 + (b'_t / b_t) * xt.
        Note: Only defined for t in [0,1).

        Args:
            - xt: position variable (num_samples, dims)
            - x1: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            conditional_vector_field: conditional vector field (num_samples, dims)
        """
        dlogbt = self.beta.dt(t) / self.beta(t)
        return (self.alpha.dt(t) - dlogbt * self.alpha(t)) * x1 + dlogbt * xt

    def conditional_score(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Evaluates the conditional score of p_t(x|x1) = N(alpha_t * x1, beta_t**2 * I_d).

        Note: only defined on t in [0,1).

        Args:
            - xt: position variable (num_samples, dims)
            - x1: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_score: conditional score (num_samples, dims)
        """
        return (self.alpha(t) * x1 - xt) / (self.beta(t) ** 2)


class GaussianConditionalLabeledProbabilityPath[T1: LabeledSampleable](
    ConditionalLabeledProbabilityPath
):
    """A gaussian conditional probability path, starting from initial gaussian distribution."""

    def __init__(self, p1: T1, alpha: Alpha, beta: Beta, p0_shape: list[int]):
        p0 = IsotropicGaussian(shape=p0_shape, std=1.0)
        super().__init__(p0, p1)
        self.alpha = alpha
        self.beta = beta
        self.rearrange_scalar = Rearrange(f"b-> b{' 1' * len(p0_shape)}")  # b-> b 1 1 1

    def sample_conditioning_variable(self, num_samples: int) -> Tensor:
        return self.p1.sample(num_samples)  # x1 (num_samples, dim)

    def sample_conditional_path(self, x1: Tensor, t: Tensor) -> Tensor:
        """Sample xt ~ p_t(x|x1) = N(x; alpha_t * x1, beta_t**2 * I_d).

        Args:
            - x1: conditioning variable / data sample (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - xt: samples from p_t(x|x1) (num_samples, dims)
        """
        # alpha_t * x1 + beta_t * epsilon, where epsilon ~ N(0,I) = p0
        alpha_t = self.rearrange_scalar(self.alpha(t))  # (b 1 1 1)
        beta_t = self.rearrange_scalar(self.beta(t))
        return alpha_t * x1 + beta_t * torch.randn_like(x1, device=x1.device)

    def conditional_vector_field(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Evaluate the conditional vector field u_t(x|x1).

        Given by: u_t(x|x1) = (a'_t - (b'_t / b_t) * a_t) * x1 + (b'_t / b_t) * xt.
        Note: Only defined for t in [0,1).

        Args:
            - xt: position variable (num_samples, ...)
            - x1: conditioning variable (num_samples, ...)
            - t: time (num_samples, 1)

        Returns:
            conditional_vector_field: conditional vector field (num_samples, ...)
        """
        alpha_t: Tensor = self.rearrange_scalar(self.alpha(t))
        beta_t: Tensor = self.rearrange_scalar(self.beta(t))
        dt_alpha_t: Tensor = self.rearrange_scalar(self.alpha.dt(t))
        dt_beta_t: Tensor = self.rearrange_scalar(self.beta.dt(t))

        dlogbt = dt_beta_t / beta_t
        return (dt_alpha_t - dlogbt * alpha_t) * x1 + dlogbt * xt

    def conditional_score(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Evaluates the conditional score of p_t(x|x1) = N(alpha_t * x1, beta_t**2 * I_d).

        Note: only defined on t in [0,1).

        Args:
            - xt: position variable (num_samples, ...)
            - x1: conditioning variable (num_samples, ...)
            - t: time (num_samples, 1)

        Returns:
            - conditional_score: conditional score (num_samples, ...)
        """
        alpha_t = self.rearrange_scalar(self.alpha(t))
        beta_t = self.rearrange_scalar(self.beta(t))

        return (alpha_t * x1 - xt) / (beta_t**2)


class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p0: Sampleable, p1: Sampleable):
        super().__init__(p0, p1)

    def sample_conditioning_variable(self, num_samples: int) -> Tensor:
        """Samples the conditioning variable from p1~p_data(x) (dataset).

        Args:
            - num_samples: the number of samples
        Returns:
            - x1: samples from p_data(x) (n_samples, dim).
        """
        return self.p1.sample(num_samples)

    def sample_conditional_path(self, x1: Tensor, t: Tensor) -> Tensor:
        """Samples the random variable X_t = (1-t)*X_0 + t X_1.

        Args:
            - x1: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)

        Returns:
            - xt: the samples from p_t(x|x1), (num_Samples, dim)
        """
        num_samples = x1.shape[0]
        x0 = self.p0.sample(num_samples).to(x1.device)
        return (1 - t) * x0 + t * x1

    def conditional_vector_field(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Samples the conditional random vector field u_t=(x1-xt)/(1-t).

        Args:
            - xt: sample where we evaluate conditional_vector_field, (num_samples, dim)
            - x1: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)

        Returns:
            - conditional_vector_field: u_t(x|x1), (num_samples, dim)
        """
        # Derivation: using u_t(x|z)
        # = (\dot{\alpha}_t - \dot{\beta_t}/\beta_t \alpha_t) z + \dot{\beta_t} / \beta_t x.
        # We have alpha_t = t, beta_t = (1-t).
        # \dot{alpha_t} = 1, \dot{beta_t} = -1.
        # So, (1+t/(1-t))*x1 -1/(1-t)* xt
        # Or: u_t = (x1-xt)/(1-t)
        return (x1 - xt) / (1 - t)

    def conditional_score(
        self, xt: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Not known for Linear Conditional Probability Paths."""
        raise AttributeError("You should not be calling this function!")
