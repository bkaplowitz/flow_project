"""
Basic Densities and Probabilities
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Density(ABC):
    """
    Distribution with tractable density.
    """

    @abstractmethod
    def log_density(self, x: Tensor) -> Tensor:
        """
        Returns the log density at x.

        Args:
            - x: shape (bs, dim)
        Returns:
            - log_density: shape (bs, 1)
        """
        pass

    def score(self, x: Tensor) -> Tensor:
        """
        Returns the score at x: \\grad log_density(x) or dx log_density(x)

        Args:
            - x: (bs, dim)
        Returns:
            - score: (bs, dim)
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            log_p = self.log_density(x)  # (bs, 1)
            grad = torch.autograd.grad(log_p.sum(), x)[0]
        return grad.detach()


class Sampleable(ABC):
    """
    Distribution that can be sampled from.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> Tensor:
        """
        Returns samples from the distribution.

        Args:
            - num_samples: number of samples

        Returns:
            - samples: shape (num_samples, dim)
        """
        pass
