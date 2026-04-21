"""Implements torch models for flow and score matching."""

import torch
from torch import Tensor, nn
from torch.nn import Sequential

from flow_matching.base.paths import Alpha, Beta


def make_mlp(
    dims: list[int], activation: type[nn.Module] = nn.SiLU, final_init: bool = False
) -> nn.Sequential:
    layers = []
    final_idx = len(dims) - 2
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < final_idx:
            layers.append(activation())

    net = nn.Sequential(*layers)
    if final_init:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class MLPVectorField(nn.Module):
    r"""A learnable MLP vector field of some corruption process $u_t^{ref}$.

    Represented by $u_t^{\theta}(x)$.

    Takes in $(x, t)$ and returns the estimated marginal vector field at that point.

    Uses a MLP architecture with data dim `dim` and hidden dims `hidden_dims`.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: list[int],
        activation: type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()
        self.dim = dim
        input_dim = dim + 1  # x dim + t.
        output_dim = dim
        all_dims = [input_dim, *hidden_dims, output_dim]
        self.net = make_mlp(all_dims, activation)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Computes u^theta_t(x).

        Args:
            - x: state, shape (bs, dims)
            - t: time, shape (bs, 1)

        Returns:
            - u_t^{theta}: vector field, shape (bs, dim)
        """
        return self.net(torch.cat([x, t], dim=-1))


class MLPScore(nn.Module):
    """MLP-parameterization of learned score vector field."""

    def __init__(
        self, dim: int, hidden_dims: list[int], activation: type[nn.Module] = torch.nn.SiLU
    ):
        super().__init__()
        input_dim = dim + 1  # x dim + t
        output_dim = dim
        all_dims = [input_dim, *hidden_dims, output_dim]
        self.net = make_mlp(all_dims, activation)

    def forward(self, x: Tensor, t: Tensor):
        """Computes score at a given (x,t) coordinate.

        Args:
            - x: shape (bs, dim)
            - t: shape (bs, 1)

        Returns:
            - s_t^{theta}(x)
        """
        return self.net(torch.cat([x, t], dim=-1))


class ScoreFromVectorField(nn.Module):
    """Parameterization of score via learned vector field (for Gaussian probability paths)."""

    def __init__(self, flow_model: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.flow_model = flow_model
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""Uses fact for Gaussian models $\nabla\log p^{ref}_t(x)=\frac{u_t^{ref}(x-a_t x)}{b_t}$.

        For Gaussian models:

        $a_t := \frac{\dot{a}_t}{a_t}$.

        $b_t := \beta^2_t (\frac{\dot{a}_t}{a_t} - \dot{\beta}_t \beta_t)$.

        Args:
            - x, state at time t, shape (bs, dim)
            - t: time, shape (bs, 1)

        Returns:
            - s_t^{theta} score estimated at time t, state x_t
        """
        a_t = self.alpha.dt(t) / self.alpha(t)
        b_t = self.beta(t) ** 2 * self.alpha.dt(t) / self.alpha(t) - self.beta.dt(t) * self.beta(t)
        return (self.flow_model(x, t) - a_t * x) / b_t


class MLPConditionalVectorField(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, class_dim: int, num_classes: int):
        super().__init__()
        self.mlp: Sequential = make_mlp(
            [dim + class_dim + 1, hidden_dim, hidden_dim, dim]
        )  # [x,embed(y), t]
        self.class_embedding = nn.Embedding(num_classes + 1, class_dim)  # num_classes + null

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Compute conditional vector field.

        Args:
            - x: shape (bs, dims)
            - t: shape (bs,)
            - y: shape (bs,)

        Returns:
            - u_t^{theta}(x|y): (b,c,h,w)
        """
        embed_y: Tensor = self.class_embedding(y)
        return self.mlp(torch.cat([x, embed_y, t.unsqueeze(-1)], dim=-1))


# Patch-based Diffusion Transformer (DiT) Related functions


class FourierEncoder(nn.Module):
    """Embeds a scalar 't' into a fourier space with learnable weights.

    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, embedding_dim: int):
        """Takes an embedding_dim of value 2d. Must be positive, even.

        Args:
            - embedding_dim: Number of cos + sin embeddings total to take scalar t into.
            One for each batch.
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "must be even."
        self.half_dim = embedding_dim // 2  # "d"
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: Tensor) -> Tensor:
        """Takes a tensor of time of size (bs,) and returns an embedding of size (bs, dim).

        Args:
            - t: shape (bs,)

        Returns:
            - embeddings: shape (bs, 2d)

        """
        freqs = (2 * torch.pi * self.weights * t).expand(-1, 1)
        embds = [torch.cos(freqs), torch.sin(freqs)]
        return torch.cat(embds, dim=1)  # bs, embedding_dim / bs, 2d


class Patchifier:
    pass
