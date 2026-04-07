"""Basic ODE abstract base-classes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class ODE(ABC):
    """Represents an ODE with associated `drift_coef` method."""

    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient of associated ODE.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape (bs)

        Returns:
            drift coefficient shape (bs, ...)
        """
        pass


class ConditionedODE(ABC):
    """Represents an ODE with associated `drift_coef` method with  conditioning."""

    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Drift coefficient of associated ODE.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape (bs,)
            y: conditioning label, shape ()

        Returns:
            drift coefficient shape (bs, ...)
        """
        pass


class SDE(ABC):
    """Represents a SDE with associated `drift_coef` and `diffusion_coef` methods."""

    @abstractmethod
    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient of associated SDE.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape (bs,)

        Returns:
            drift coefficient shape (bs, ...)
        """
        pass

    @abstractmethod
    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns the diffusion coefficient of the SDE.

        Args:
            xt: state at time t, shape (batch_size, ...)
            t: time, shape (batch_size)

        Returns:
            diffusion coefficient: shape (batch_size, ...)
        """
        pass
