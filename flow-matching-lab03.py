import marimo

__generated_with = "0.22.4"
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

    return MNISTSampler, make_grid, th


@app.cell
def _(MNISTSampler, make_grid, th):
    import matplotlib.pyplot as plt

    from flow_matching.paths import (
        GaussianConditionalLabeledProbabilityPath,
        LinearAlpha,
        LinearBeta,
    )

    def sample_mnist_paths(num_rows=3, num_cols=3, num_timesteps=5):
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

    return


@app.cell
def _():
    # classifier free guidance

    return


if __name__ == "__main__":
    app.run()
