"""
Conditional probability path abstract object
used for recovering a conditional vector field
and other primitives for diffusion models.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from .probability import Sampleable


class ConditionalProbabilityPath(nn.Module, ABC):
    """Abstract base class for conditional probability paths"""

    def __init__(self, p_simple: Sampleable, p_data: Sampleable) -> None:
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tensor:
        """
        Samples the conditioning variable z

        Args:
            - num_samples: the number of samples
        Returns:
            -  z: samples from p(z), (num_samples, dim)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Samples from the conditional distribution p_t(x|z)

        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)

        Returns:
            - x: samples from p_t(x|z), (num_samples, dims)
        """

    @abstractmethod
    def conditional_vector_field(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)

        Args:
            - x: position variable (num_samples, dims)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)

        Returns:
            - conditional_vector_fieldl: conditional vector field (num_samples, dimm)
        """

    @abstractmethod
    def conditional_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Evaluates the conditional score of p_t(x|z)

        Args:
            - x: position variable (num_samples, dims)
            - z: conditioning variable (num_samples, dims)
            - t: time (num_samples, 1)

        Returns:
            - conditional_score: conditional score $\\grad_{x} \\log(p_t(x|z))$. (num_samples, dim)
        """
        pass

    def sample_marginal_path(self, t: Tensor) -> Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1)
        Returns:
            - x samples from p_t(x), (num_samples, dim)
        """
        num_samples = t.shape[0]
        # z ~ p(z)
        z = self.sample_conditioning_variable(num_samples)
        # x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t)
        return x


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
        """
        Evaluates alpha_t. Should satisfy self(0.0)=0.0, self(1.0) = 1.0
        Args:
            - t: time (num_samples, 1)

        Returns:
            - alpha_t (num_samples, 1)
        """
        pass

    def dt(self, t: Tensor) -> Tensor:
        """
        Evaluates d/dt alpha_t
        Args:
            - t: time (num_samples ,1 )

        Returns:
            - d / dt alpha_t (num_samples, 1)
        """
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            a_t = self(t)
            da_dt = torch.autograd.grad(a_t.sum(), t)[0]
            return da_dt.detach()


# a_t and b_t satisfy a_1=b_0=1 and a_0=b_1=0 and have time derivatives
class Beta(ABC):
    def __init__(self):
        # Verify a_0=0
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1)), (
            "Beta at time 0 is not 1 as required."
        )
        # Verify a_1=1
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1)), (
            "Beta at time 1 is not 0 as required."
        )

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        """
        Evaluates beta_t. Should satisfy self(0.0)=1.0, self(1.0) = 0.0
        Args:
            - t: time (num_samples, 1)

        Returns:
            - beta_t (num_samples, 1)
        """
        pass

    def dt(self, t: Tensor) -> Tensor:
        """
        Evaluates d/dt beta_t
        Args:
            - t: time (num_samples ,1 )

        Returns:
            - d / dt beta_t (num_samples, 1)
        """
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)
            b_t = self(t)
            db_dt = torch.autograd.grad(b_t.sum(), t)[0]
            return db_dt.detach()
