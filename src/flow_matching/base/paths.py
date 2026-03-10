"""Conditional probability path abstract object.

Used for recovering a conditional vector field
and other primitives for diffusion models.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .probability import SampleableDensity


class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths."""

    def __init__(self, p0: SampleableDensity, p1: SampleableDensity) -> None:
        super().__init__()
        self.p0 = p0
        self.p1 = p1

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tensor:
        """Samples the conditioning variable x1 ~ p1(x).

        Args:
            num_samples: number of samples

        Returns:
            x1: samples from p1(x), (num_samples, dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, x1: Tensor, t: Tensor) -> Tensor:
        """Samples from the conditional distribution p_t(x|x1).

        Args:
            x1: conditioning/target variable, data (num_samples, dim)
            t: time (num_samples, 1)

        Returns:
            xt: samples from p_t(x|x1), (num_samples, dims)
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Evaluates the conditional vector field u_t(x|x1).

        Args:
            xt: position variable (num_samples, dims)
            x1: conditioning variable (num_samples, dim)
            t: time (num_samples, 1)

        Returns:
            conditional_vector_field: conditional vector field (num_samples, dims)
        """

    @abstractmethod
    def conditional_score(self, xt: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        r"""Evaluates the conditional score of p_t(x|x1).

        Args:
            xt: position variable (num_samples, dims)
            x1: conditioning variable (num_samples, dims)
            t: time (num_samples, 1)

        Returns:
            conditional_score: conditional score $\\grad_{x} \\log(p_t(x|x1))$. (num_samples, dim)
        """
        pass

    def sample_marginal_path(self, t: Tensor) -> Tensor:
        """Samples from the marginal distribution p_t(x) = ∫ p_t(x|x1) p1(x1) dx1.

        Args:
            t: time (num_samples, 1)

        Returns:
            xt: samples from p_t(x) (num_samples, dim)
        """
        num_samples = t.shape[0]
        x1 = self.sample_conditioning_variable(num_samples)
        xt = self.sample_conditional_path(x1, t)
        return xt

    @staticmethod
    def oob_check(t: Tensor) -> None:
        """Helper function to check t is in [0,1).

        Args:
            t: Tensor to check if strictly in [0,1)
        """
        t_below_zero = (t < torch.zeros_like(t)).any()
        t_above_one = (t >= torch.ones_like(t)).any()
        if t_below_zero or t_above_one:
            t_oob = "only defined for t in [0,1)."
            if t_above_one:
                t_oob += f"Found t max: {t.max()}>=1."
            if t_below_zero:
                t_oob += f"Found t min: {t.min()}< 0."
            raise ValueError(t_oob)


# a_t and b_t satisfy a_1=b_0=1 and a_0=b_1=0 and have time derivatives
class Alpha(ABC):
    def __init__(self):
        # Verify a_0=0
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1)), (
            "alpha at time 0 is not 0 as required."
        )
        # Verify a_1=1
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1)), (
            "Alpha at time 1 is not 1 as required."
        )

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        """Evaluates alpha_t. Should satisfy self(0.0)=0.0, self(1.0) = 1.0.

        Args:
            t: time (num_samples, 1).

        Returns:
            alpha_t (num_samples, 1)
        """
        pass

    def dt(self, t: Tensor) -> Tensor:
        """Evaluates d/dt alpha_t.

        Args:
            t: time (num_samples, 1).


        Returns:
            dadt: derivative of alpha w.r.t. t (num_samples, 1)
        """
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            a_t = self(t)
            da_dt = torch.autograd.grad(a_t.sum(), t)[0]
            return da_dt.detach()


# a_t and b_t satisfy a_1=b_0=1 and a_0=b_1=0 and have time derivatives
class Beta(ABC):
    def __init__(self):
        # Verify b_0=1
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1)), (
            "Beta at time 0 is not 1 as required."
        )
        # Verify b_1=0
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1)), (
            "Beta at time 1 is not 0 as required."
        )

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        """Evaluates beta_t. Should satisfy self(0.0)=1.0, self(1.0) = 0.0.

        Args:
            t: time (num_samples, 1)

        Returns:
            beta_t (num_samples, 1)
        """
        pass

    def dt(self, t: Tensor) -> Tensor:
        """Evaluates d/dt beta_t.

        Args:
            t: time (num_samples, 1)

        Returns:
            dbdt: derivative of beta_t w.r.t. t (num_samples, 1)
        """
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            b_t = self(t)
            db_dt = torch.autograd.grad(b_t.sum(), t)[0]
            return db_dt.detach()
