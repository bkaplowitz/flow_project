"""Basic ODE abstract base-classes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class ODE(ABC):
    """Represents an ODE with associated `drift_coef` method."""

    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Drift coefficient of associated ODE.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape ()

        Returns:
            drift coefficient shape (bs, ...)
        """
        pass


class SDE(ABC):
    """Represents a SDE with associated `drift_coef` and `diffusion_coef` methods."""

    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Drift coefficient of associated SDE.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape ()

        Returns:
            drift coefficient shape (bs, ...)
        """
        pass

    @abstractmethod
    def diffusion_coef(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Returns the diffusion coefficient of the SDE.

        Args:
            xt: state at time t, shape (batch_size, ...)
            t: time, shape ()

        Returns:
            diffusion coefficient: shape (batch_size, ...)
        """
        pass
