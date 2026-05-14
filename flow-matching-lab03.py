import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import torch as th
    from torch import Tensor, nn
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid

    from flow_matching.base.probability import LabeledSampleable

    class MNISTSampler(nn.Module, LabeledSampleable):
        """Sampleable wrapper for MNIST dataset."""

        def __init__(self):
            super().__init__()
            self.dataset = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1305,), (0.2891,)),
                    ]
                ),
            )

            self.dummy = nn.Buffer(th.zeros(1))  # handles automove

        def sample(self, num_samples: int) -> tuple[Tensor, Tensor]:
            """Samples from MNIST dataset.

            Args:
                - num_samples: the desired number of num_samples
            Returns:
                - samples: shape (bs, c, h, w)
                - labels: shape (bs, label_dim)
            """
            if num_samples > len(self.dataset):
                raise ValueError(f"num_samples exxeeds dataset size: {len(self.dataset)}")
            indices = th.randperm(len(self.dataset))[:num_samples]
            samples, labels = zip(*[self.dataset[i] for i in indices], strict=True)
            samples = th.stack(samples).to(self.dummy.device)  # typo
            labels = th.tensor(labels, dtype=th.int64).to(self.dummy.device)
            return samples, labels

    return MNISTSampler, Tensor, make_grid, th


@app.cell
def _(MNISTSampler, make_grid, th):
    import matplotlib.pyplot as plt

    from flow_matching.paths import (
        GaussianConditionalLabeledProbabilityPath,
        LinearAlpha,
        LinearBeta,
    )

    def sample_mnist_paths(num_rows: int = 3, num_cols: int = 3, num_timesteps: int = 5) -> None:
        device = th.device(
            "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
        )
        # Initialize sampler
        sampler = MNISTSampler().to(device)

        # Initialize probability path
        path = GaussianConditionalLabeledProbabilityPath(
            p1=sampler, alpha=LinearAlpha(), beta=LinearBeta(), p0_shape=[1, 32, 32]
        ).to(device)

        # sample
        num_samples = num_rows * num_cols
        x1, _ = path.p1.sample(num_samples)
        x1 = x1.view(-1, 1, 32, 32)
        # Setup plot
        fig, axs = plt.subplots(
            1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows)
        )
        t = th.linspace(0, 1, num_timesteps).to(device)
        for ti, tval in enumerate(t):
            t_expanded = tval.expand(num_samples)
            xt = path.sample_conditional_path(x1, t_expanded)
            grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1, 1))
            axs[ti].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
            axs[ti].axis("off")
            axs[ti].set_title(f"t={tval}", fontsize=64)
        plt.show()

    sample_mnist_paths()

    return (
        GaussianConditionalLabeledProbabilityPath,
        LinearAlpha,
        LinearBeta,
        plt,
    )


@app.cell
def _(
    GaussianConditionalLabeledProbabilityPath,
    LinearAlpha,
    LinearBeta,
    Tensor,
    th,
):
    # classifier free guidance
    import math

    from flow_matching.distributions import LabeledGaussianMixture
    from flow_matching.models import MLPConditionalVectorField
    from flow_matching.trainer import CFGTrainer
    from flow_matching.utils import get_device
    # Lbgm: labeled gaussian mixture

    def train_gmm() -> tuple[
        list[int],
        list[Tensor],
        LabeledGaussianMixture,
        MLPConditionalVectorField,
        GaussianConditionalLabeledProbabilityPath,
    ]:
        device = get_device()

        # Initialize GMM
        angles: list[float] = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
        means = 2 * th.tensor([[math.cos(a), math.sin(a)] for a in angles])
        covs = th.diag(th.tensor([0.2, 0.2])).expand(3, -1, -1)
        weights = th.tensor([1 / 3, 1 / 3, 1 / 3])
        gmm = LabeledGaussianMixture(means, covs, weights).to(device)

        # Initialize path
        path = GaussianConditionalLabeledProbabilityPath(
            gmm, alpha=LinearAlpha(), beta=LinearBeta(), p0_shape=[2]
        ).to(device)
        vector_field = MLPConditionalVectorField(
            dim=2, hidden_dim=2, class_dim=2, num_classes=3
        ).to(device)
        # Train vector field
        trainer = CFGTrainer(model=vector_field, path=path, eta=0.25, null_label=3)
        steps, losses = trainer.train(
            model=vector_field, num_epochs=3_000, lr=1e-3, batch_size=250, device=device
        )
        return steps, losses, gmm, vector_field, path

    steps_lbgm, losses_lbgm, gmm_lbgm, vf_lbgm, path_lbgm = train_gmm()

    return (
        LabeledGaussianMixture,
        MLPConditionalVectorField,
        get_device,
        losses_lbgm,
        steps_lbgm,
    )


@app.cell
def _(losses_lbgm, plt, steps_lbgm, th):
    plt.figure()
    # Stack as list of dim 0 tensors and dim 0 cannot be concatenated.
    losses_vec_lbgm = th.stack(losses_lbgm, dim=0).detach().cpu().numpy()
    plt.plot(steps_lbgm, losses_vec_lbgm)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Losses, GMM")
    plt.show()

    return


@app.cell
def _(
    LabeledGaussianMixture,
    MLPConditionalVectorField,
    get_device,
    plt,
    th,
    x_init,
):
    # Visualize results
    from flow_matching import EulerSimulator
    from flow_matching.base.paths import ConditionalLabeledProbabilityPath
    from flow_matching.flows import CFGVectorFieldODE

    def visualize_gmm_results(
        gmm: LabeledGaussianMixture,
        trained_model: MLPConditionalVectorField,
        path: ConditionalLabeledProbabilityPath,
        guidance_strength: float = 1.0,
        null_label: int = 3,
        batch_size: int = 250,
    ):
        device = get_device()
        fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 6))
        x1, _ = gmm.sample(batch_size)
        x1 = x1.detach().cpu().numpy()
        # Target
        t_ax = axs[0]
        t_ax.scatter(x1[:, 0], x1[:, 1], s=5, marker="*")
        t_ax.set_title("Target")
        # Panel 2: Condition CFG on each mode of LabeledGaussianMixture
        cond_ax = axs[1]
        vector_field = CFGVectorFieldODE(
            trained_model, guidance_scale=guidance_strength, null_label=null_label
        )
        simulator = EulerSimulator(vector_field)
        # Duplicates each entry n times after.
        # Block structure of [v_class_1,v_class_2, v_class_3]
        labels = th.arange(3).repeat_interleave(batch_size).to(device)
        x0 = path.p0.sample(3 * batch_size)  # (b 2) [bs , dims]
        ts = th.linspace(0, 1, 100).expand(3 * batch_size, -1).to(device)  # (bs, n_classes, t)
        xs = simulator.simulate(x0, ts, y=labels).detach().cpu().numpy()
        for idx in range(3):
            xs_idx = xs[idx * batch_size : (idx + 1) * batch_size].detach().cpu().numpy()
            cond_ax.scatter(xs_idx[:, 0], xs_idx[:, 1], s=5, label=f"Mode {idx}", marker="*")
        cond_ax.legend()
        cond_ax.set_title(f"CFG w/ Guidance Strength {guidance_strength:.2f}")

        # Panel 3 unconditioned
        uncond_ax = axs[2]
        batch_size_uncond = 3 * batch_size
        labels = th.ones(batch_size_uncond).long().to(device) * 3
        x0 = path.p0.sample(batch_size_uncond)
        ts = th.linspace(0, 1, 100).expand(batch_size_uncond, -1).to(device)  # (bs, n_classes, t)
        xs = simulator.simulate(x_init, ts, y=labels).detach().cpu().numpy()  # (bs 2)
        uncond_ax.scatter(xs[:, 0], xs[:, 1], s=5, label=f"Mode {null_label}", marker="*")
        uncond_ax.set_title("Unguided Samples")
        fig.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Making a Diffusion Transformer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start with a fourier encoder for time which maps a scalar time $t \in [0,1]$
     to an embedding of size 2d:
    $$
    t^{emb} = [\cos(2\pi w_1 t),\dots,\cos(2\pi w_d t), \sin(2\pi w_1 t),\dots,\sin(2\pi w_d t)]^T
    $$
    where $w_i \overset{iid}{\sim} \mathcal{N}(0,1)$
    for d weights.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    # Training utils for DiT
    from torchvision.utils import make_grid

    return (make_grid,)


if __name__ == "__main__":
    app.run()
