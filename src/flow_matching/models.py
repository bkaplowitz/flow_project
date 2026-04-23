"""Implements torch models for flow and score matching."""

import torch
from einops.layers.torch import Rearrange
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
            - embeddings: shape (bs, 2d), 2d is embedding dim

        """
        freqs = (2 * torch.pi * self.weights * t).expand(-1, 1)
        scale = torch.sqrt(2 / self.half_dim)  # Ensures embedding sums to 1
        embds = [torch.cos(freqs), torch.sin(freqs)]
        return scale * torch.cat(embds, dim=1)  # bs, embedding_dim / bs, 2d


class Patchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
        """Takes an image valued tensor of shape b, c, 32, 32.

        It patchifies it to shape b (h/p * w/p) d
        where d is diffusion hidden transformer dim.
        It first applies a convolutional layer mapping the input of shape (b c 32 32) to
        b d h/p w/p
        and then rearranges from b d h/p w/p to
        b (h/p w/p) d = b n d, n tokens of dim d.
        """
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        assert patch_size >= 1, "Patch size must be equal to or larger than 1"
        #  H_out = floor((H_in + 2 padding - dilation (kernel_size - 1) -1 )+stride)/ stride
        # So we want stride = patch_size to tile the space.
        # We don't need any padding as the space is already divisible by 0.  padding=0
        # and this yields floor((H_in -(kernel_size-1) - 1 + stride)/stride) for dilation 1.
        # This simplifies to floor(H_in / patch_size) =  H_in / patch_size

        self.conv = torch.nn.Conv2d(
            in_channels=c_in, out_channels=dim, kernel_size=patch_size, stride=patch_size, padding=0
        )
        self.rearrange = Rearrange("b d h_out w_out-> b (h_out w_out) d")

    def forward(self, x: Tensor) -> Tensor:
        """Computes patchified version.

        Args:
        - x: (bs, c_in, img_size, img_size)

        Returns:
        - x: (bs, (img_width / patch_size * img_height/patch_size), d)
        """
        return self.rearrange(self.conv(x))


class MHA(nn.Module):
    """A nn module for multi-headed self-attention."""

    def __init__(self, dim: int, heads: int):
        """Initializes a multi-headed self-attention architecture block.

        Args:
            - dims: dimension of hidden layers
            - heads: number of heads
        """
        super().__init__()
        assert dim % heads == 0
        raise NotImplementedError("Fill me in.")


class DiffusionTransformerLayer(nn.Module):
    """A NN module implementing a DiT block."""

    def __init__(self, dim: int, heads: int):
        """Initializes a DiT Layer.

        Args:
            - dim: dimension of hidden layers
            - heads: number of attention heads
        """
        super().__init__()
        self.ffn = MLPVectorField(dim, [dim, 4 * dim, dim])
        self.mlp_conditioning = make_mlp(
            [dim, 4 * dim, 6 * dim], final_init=True
        )  # initialize to 0s in final layer.
        self.mha = MHA(dim, heads)

        # Initialize to 0 last layer done in self.mlp

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Computes the output of a diffusion transformer layer.

        Args:
            - x: b n d
            - c: b d
        Returns:
            - x: b n d
        """
        # Conditioning gating, scaling and bias
        tokens_normed = torch.nn.functional.layer_norm(x)
        shift_scale_bias = self.mlp_conditioning(c)
        # Get coefficients
        # each shape (b,d ) from output (b,6d)
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = torch.split(shift_scale_bias, 6, dim=1)
        # Shift and scale tokens normed
        # Attention + residual
        scaled_cond_latent = tokens_normed * (1 + gamma_1) + beta_1
        scaled_mha = self.mha(scaled_cond_latent) * alpha_1
        x = x + scaled_mha
        # feedforward + residual
        normed_x = torch.nn.functional.layer_norm(x)
        shift_scaled_ff_input = normed_x * (1 + gamma_2) + beta_2
        ff_out = alpha_2 * self.ffn(shift_scaled_ff_input)
        return x + ff_out


class DiffusionTransformer(nn.Module):
    """A NN Module implementing a latent diffusion transformer."""

    def __init__(self, depth: int, n_tokens: int, dim: int, **layer_kwargs):
        """Constructs a diffusion transformer DiT.

        Args:
            - n_tokens: sequence length for positional embeddings
            - dim: dimension of hidden layers
            - heads: number of attention heads
            - depth: number of hidden layers

        After patchifying our data is in shape (b,n,d), where:
        - b is batch size
        - n is # of tokens per image
        - d is dim of tokens.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.depth = depth
        self.dim = dim
        heads = layer_kwargs["heads"]
        self.dit_layers = nn.Sequential([DiffusionTransformerLayer(dim, heads) for _ in depth])

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Takes in patchified latent vars and embedded t,y.

        (either direct if classes, else prelearned and frozen clip for text)
        and returns u of shape C x H x W for predicted flow.

        Args:
            - x: patchified latent var, shape b, n, dim
            - c: conditioning embeddings (time + y_labels): shape b, dim
        Returns:
            - u: patchified predicted flow, shape b n d
        """
        return self.dit_layers(x, c)
