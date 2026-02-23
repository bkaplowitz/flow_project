"""
Basic Densities and Probabilities
"""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.func import jacrev, vmap


class Density(ABC):
    """
    Distribution with tractable densitty.
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
        Returns the score at x: dx log_density(x)

        Args:
            - x: (bs, dim)
        Returns:
            - score: (bs, dim)
        """
        x = x.unsqueeze(1)  # (bs, 1, dim)
        score = vmap(jacrev(self.log_density))(x)  # bs, 1, 1, 1, dim
        return score.squeeze((1, 2, 3))  # (bs, dim)


class Sampleable(ABC):
    """
    Distribution that can be sampled from.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tensor:
        """
        Returns a sample of z.

        Args:
            - num_samples: number of samples

        Returns:
            - samples: shape (num_samples, dim)
        """
