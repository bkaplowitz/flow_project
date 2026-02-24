"""
Basic ODE abstract base-classes.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class ODE(ABC):
    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """drift coefficient of associated ODE

        Args:
            - xt, state at time t shape (bs, dim)
            - t, time, shape ()
        Returns:
            - drift coefficient shape (bs, dim)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """drift coefficient of associated SDE.

        Args:
            - xt, state at time t shape (bs, dim)
            - t, time, shape ()
        Returns:
            - drift coefficient shape (bs, dim)
        """
        pass

    @abstractmethod
    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Returns the diffusion coefficient of the SDE.

        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape ()
        Returns:
            - diffusion coefficient: shape (batch_size, dim)
        """
        pass
