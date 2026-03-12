import torch
from torch import Tensor

from flow_matching.base.dynamics import ODE, SDE
from flow_matching.base.paths import ConditionalProbabilityPath
from flow_matching.models import MLPScore, MLPVectorField


class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, x1: Tensor):
        """Construct a conditional vector field for a given probability path.

        Args:
            path: The conditional probability path object that this is the vector field of.
            x1: the conditioning variable / data sample (1, dim)
        """
        super().__init__()
        self.path = path
        self.x1 = x1

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns the conditional vector field u_t(x|x1).

        Args:
            xt: state at time t, shape (bs, dim)
            t: time, shape (bs, 1)

        Returns:
            u_t(x|x1): shape (bs, dim)
        """
        bs = xt.shape[0]
        dims = self.x1.shape[1:]
        x1_expanded = self.x1.expand((bs, *dims))
        return self.path.conditional_vector_field(xt, x1_expanded, t)


class ConditionalVectorFieldSDE(SDE):
    r"""Construct a Langevin of associated conditional probability path.

    Langevin is relaed to score $score(X_t) = \nabla_x log p_t(X_t|x1)$ and given by:

    $dX_t = [u_t(X_t|x1) +1/2 sigma^2_t \nabla_x log p_t(X_t|x1)]dt + \sigma dW_t$.

    Args:
        path: The conditional probability path object that this is the vector field of.
        x1: the conditioning variable / data sample (1, dim)
    """

    def __init__(self, path: ConditionalProbabilityPath, x1: Tensor, sigma: float):

        self.path = path
        self.x1 = x1
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns the conditional vector field u_t(x|x1).

        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - u_t(x|x1): shape (bs, 1), drift coefficient of Langevin dynamics of original ODE.
        """
        bs = xt.shape[0]
        dims = self.x1.shape[1:]
        x1_expanded = self.x1.expand((bs, *dims))
        return self.path.conditional_vector_field(xt, x1_expanded, t) + (
            0.5 * self.sigma**2 * self.path.conditional_score(xt, x1_expanded, t)
        )

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns the diffusion coefficient of conditional vector field Langevin SDE.

        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - sigma_t, the diffusion coefficient. (bs, dim)
        """
        return self.sigma * torch.ones_like(xt)


class LearnedVectorFieldODE(ODE):
    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coef(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient for LearnedVectorODE.

        Args:
            - x: (bs, dim)
            - t: (bs, dim)

        Returns:
            - u_t: (bs, dim)
        """
        return self.net(xt, t)


class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        """A learned flow representing a langevin model.

        Has associated flow `flow_model` and score `score_model`.

        Args:
            - flow_model: the flow model object to which this vector field corresponds.
            - score_model: the score model associated with the vector field.
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns drift coefficient at xt, t.

        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - u_t(x|z) drift coefficient shape (bs, dim)
        """
        return self.flow_model(xt, t) + 0.5 * self.sigma**2 * self.score_model(xt, t)

    def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """Returns diffusion coefficient at xt, t.

        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - sigma_t, diffusion coefficient shape (bs, dim)
        """
        return self.sigma * torch.ones_like(xt)
