"""Specific distributions"""

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor, nn

from flow.types.probability import Density, Sampleable


class Gaussian(nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian distribution.

    Wraps torch.distributions.MultivariateNormal.
    """

    def __init__(self, mean, cov):
        """
        mean: shape(2,)
        cov: shape (2,2)
        """
        super().__init__()
        # static
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self) -> D.MultivariateNormal:
        return D.MultivariateNormal(loc=self.mean, covariance_matrix=self.cov, validate_args=False)

    def sample(self, num_samples: int) -> Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        """
        Constructs an isotropic gaussian of dim `dim` and std `std`.
        """
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)


class GaussianMixture(nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian mixture model.

    Wraps torch.distributions.MixtureSameFamily.
    """

    def __init__(
        self,
        means: Tensor,  # n_modes x data_dim
        covs: Tensor,  # n_modes x data_dim x data_dim
        weights: Tensor,  # n_modes
    ):
        """
        Args:
            - means: means of distribution, shape (n_modes, dim (2))
            - covs : variances of distributions, shape (n_modes, dim (2), dim (2))
            - weights: weighting between distributions (n_modes,1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False
            ),
            validate_args=False,
        )

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(cls, nmodes: int, std: float, scale: float = 10.0, seed=0.0) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        # Diagonal cov matrix
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes) / nmodes  # uniform
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0) -> "GaussianMixture":
        # only select nmodes, exclude 2pi position duplicate
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        # embed means via polar coords
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
