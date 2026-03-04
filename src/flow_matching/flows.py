from torch import Tensor

from flow_matching.base.dynamics import ODE
from flow_matching.base.paths import ConditionalProbabilityPath


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
