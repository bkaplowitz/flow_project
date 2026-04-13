"""Specific distributions."""

from collections.abc import Callable
from functools import partial

import numpy as np
import torch
import torch.distributions as D
from sklearn.datasets import make_circles, make_moons
from torch import Tensor, nn

from flow_matching.base.probability import LabeledSampleable, Sampleable, SampleableDensity


class Gaussian(nn.Module, SampleableDensity):
    """Two-dimensional Gaussian distribution.

    Wraps torch.distributions.MultivariateNormal.
    Practically we would use torch.randn.
    """

    def __init__(self, mean: Tensor, cov: Tensor):
        """Initialize Gaussian.

        Args:
            mean: shape (2,)
            cov: shape (2, 2)
        """
        super().__init__()
        # static
        self.mean = nn.Buffer(mean)
        self.cov = nn.Buffer(cov)
        self._cached_dist = None

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self) -> D.MultivariateNormal:
        if self._cached_dist is None:
            self._cached_dist = D.MultivariateNormal(
                loc=self.mean, covariance_matrix=self.cov, validate_args=False
            )
        return self._cached_dist

    def _apply(self, fn, recurse=True):
        self._cached_dist = None
        return super()._apply(fn, recurse)

    def sample(self, num_samples: int) -> Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float = 1.0) -> "Gaussian":
        """Constructs an isotropic gaussian of dim `dim` and std `std`."""
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)


class IsotropicGaussian(nn.Module, Sampleable):
    """Sampleable wrapper around torch.randn."""

    def __init__(self, shape: list[int], std: float = 1.0):
        """shape: shape of sampled data."""
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(
            torch.zeros(1)
        )  # will automatically be moved when self.to is called.
        # Used for ensuring sampling on correct device.

    @property
    def dim(self) -> list[int]:
        return self.shape

    def sample(self, num_samples) -> Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)


class GaussianMixture(nn.Module, SampleableDensity):
    """Two-dimensional unlabelled Gaussian mixture model.

    Wraps torch.distributions.MixtureSameFamily.
    """

    def __init__(
        self,
        means: Tensor,  # n_modes x data_dim
        covs: Tensor,  # n_modes x data_dim x data_dim
        weights: Tensor,  # n_modes
    ):
        """Initialize GaussianMixture.

        Args:
            means: means of distribution, shape (n_modes, dim (2))
            covs: variances of distributions, shape (n_modes, dim (2), dim (2))
            weights: weighting between distributions (n_modes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.means = nn.Buffer(means)
        self.covs = nn.Buffer(covs)
        self.weights = nn.Buffer(weights)
        self._cached_dist = None

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        if self._cached_dist is None:
            self._cached_dist = D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )
        return self._cached_dist

    def _apply(self, fn, recurse=True):
        self._cached_dist = None
        return super()._apply(fn, recurse)

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


class LabeledGaussianMixture(nn.Module, LabeledSampleable):
    """A labelled version of GaussianMixture."""

    def __init__(self, means: Tensor, covs: Tensor, weights: Tensor):
        super().__init__()
        self.means = nn.Buffer(means)
        self.covs = nn.Buffer(covs)
        self.weights = nn.Buffer(weights)
        self._cached_dist = None

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        if self._cached_dist is None:
            self._cached_dist = D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )
        return self._cached_dist

    def _apply(self, fn, recurse=True):
        self._cached_dist = None
        return super()._apply(fn, recurse)

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> tuple[Tensor, Tensor]:
        """Samples num_samples from distribution.

        Args:
            - num_samples: the desired number of samples

        Returns:
            - samples: shape (num_samples, dims)
            - labels (num_samples)
        """
        # Choose amount of each mode
        # Perfor multinomial sampling on CPU to avoid device-side assertion errors
        # Sample labels of modes
        labels = torch.multinomial(
            self.weights.cpu(), num_samples=num_samples, replacement=True
        ).to(self.means.device)

        # Sample from each mode
        # samples = torch.zeros(num_samples, self.dim).to(self.means.device)
        selected_means = self.means[labels]
        selected_covs = self.covs[labels]
        # Broadcasts batched means, covs correctly in vectorized form so each drawing from own mode.
        samples = D.MultivariateNormal(loc=selected_means, covariance_matrix=selected_covs).sample()
        return samples, labels

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, seed=0.0
    ) -> "LabeledGaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale
        # Diagonal cov matrix
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes) / nmodes  # uniform
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0) -> "LabeledGaussianMixture":
        # only select nmodes, exclude 2pi position duplicate
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        # embed means via polar coords
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


class SampleableDataset(Sampleable):
    """Implements sklearn make_moon, make_circles, and a checkerboard distribution."""

    def __init__(
        self,
        device: torch.device,
        make_dist: Callable,
        noise: float = 0.05,
        scale: float = 5.0,
        offset: Tensor | None = None,
    ) -> None:
        """Makes a concrete type of sampleable of either circle, moon type.

        Can also be based on the custom function make_dist.

        Args:
        - noise: Stdev of noise added to data
        - scale: how much to scale data
        - offset: how much to shift (2,)
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros((2,))
        self.offset = offset.to(device)
        self.make_dist = make_dist

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> Tensor:
        """Samples from distribution.

        Args:
            - num_samples: number of samples to generate

        Returns:
            - torch.Tensor (num_samples, 3)

        """
        samples, _ = self.make_dist(n_samples=num_samples, noise=self.noise, random_state=None)
        return (
            self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device) + self.offset
        )

    @classmethod
    def Moon(
        cls,
        device: torch.device,
        noise: float = 0.05,
        scale: float = 5.0,
        offset: Tensor | None = None,
    ) -> "SampleableDataset":
        """Generates a sampleable instance with moons."""
        return cls(device, make_moons, noise, scale, offset)

    @classmethod
    def Circle(
        cls,
        device: torch.device,
        noise: float = 0.05,
        scale: float = 5.0,
        offset: Tensor | None = None,
        factor: float = 0.5,
    ) -> "SampleableDataset":
        """Generates a sampleable instance with circles."""
        _make_circles = partial(make_circles, factor=factor)
        return cls(device, _make_circles, noise, scale, offset)


class CheckerboardSampleable(Sampleable):
    def __init__(self, device: torch.device, grid_size: int = 3, scale: float = 5.0):
        """Initializes a sampleable checkerboard dataset.

        Args:
        - grid_size: number of gridpoitns for checkerboard
        - scale: how much to scale data.
        """
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> Tensor:
        """Samples from the distribution.

        Args:
            - num_samples: number of samples to generate.

        Returns:
            - torch.Tensor: shape (num_samples, 3)
        """
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)
        while samples.shape[0] < num_samples:
            # sample num_samples:
            # Samples centered around [-1, 1]
            new_samples = (torch.rand(num_samples, 2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0  # (bs,)
            y_mask = torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0  # (bs, )
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]
