from torch import Tensor

from flow.types.diffusion import ConditionalProbabilityPath
from flow.types.ode import ODE


class ConditionalVectorField(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: Tensor):
        """
        Constructs a conditional vector field / flow that corresponds to a given probability path.

        Args:
            - path: The conditional probability path object that this is the vector field of.
            - z: the conditioning variable (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Returns the conditional vector field u_t(x|z)

        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - u_t(x|z): shape (bs, dim)
        """
        bs = xt.shape[0]
        dims = self.z.shape[1:]
        z_expanded = self.z.expand((bs, *dims))
        return self.path.conditional_vector_field(xt, z_expanded, t)
