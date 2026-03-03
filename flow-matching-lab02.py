import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch

    from flow.diffusion import GaussianConditionalProbabilityPath, LinearAlpha, SquareRootBeta
    from flow.distributions import Gaussian, GaussianMixture
    from flow.plot import imshow_density

    return (
        Gaussian,
        GaussianConditionalProbabilityPath,
        GaussianMixture,
        LinearAlpha,
        SquareRootBeta,
        imshow_density,
        plt,
        torch,
    )


@app.cell
def _(torch):
    device = torch.device(  # noqa: F841
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return (device,)


@app.cell
def _(Gaussian, GaussianMixture, device, imshow_density, plt):
    PARAMS = {
        "scale": 15.0,
        "target_scale": 10.0,
        "target_std": 1.0,
    }
    p0 = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p1 = GaussianMixture.symmetric_2D(
        nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
    ).to(device)

    fig_gs, axes_gs = plt.subplots(1, 3, figsize=(24, 8))
    bins = 200

    scale = PARAMS["scale"]
    x_bounds_gs = [-scale, scale]
    y_bounds_gs = [-scale, scale]

    axes_gs[0].set_title("Heatmap of p0")
    axes_gs[0].set_xticks([])
    axes_gs[0].set_yticks([])
    imshow_density(
        density=p0,
        x_bounds=x_bounds_gs,
        y_bounds=y_bounds_gs,
        bins=bins,
        ax=axes_gs[0],
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )

    axes_gs[1].set_title("Heatmap of p1")
    axes_gs[1].set_xticks([])
    axes_gs[1].set_yticks([])
    imshow_density(
        density=p1,
        x_bounds=x_bounds_gs,
        y_bounds=y_bounds_gs,
        bins=bins,
        ax=axes_gs[1],
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )

    axes_gs[2].set_title("Heatmap of p0 and p1")
    axes_gs[2].set_xticks([])
    axes_gs[2].set_yticks([])
    imshow_density(
        density=p0,
        x_bounds=x_bounds_gs,
        y_bounds=y_bounds_gs,
        bins=bins,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p1,
        x_bounds=x_bounds_gs,
        y_bounds=y_bounds_gs,
        bins=bins,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )
    plt.show()

    return p0, p1


@app.cell
def _(
    GaussianConditionalProbabilityPath,
    GaussianMixture,
    LinearAlpha,
    SquareRootBeta,
    device,
    imshow_density,
    p0,
    p1,
    plt,
    torch,
):
    def plot_conditional_probability_path() -> None:
        # Construct conditional probability path
        PARAMS = {"scale": 15.0, "target_scale": 10.0, "target_std": 1.0}
        bins = 200
        # Construct conditional probability path
        path = GaussianConditionalProbabilityPath(
            p1=GaussianMixture.symmetric_2D(  # target distribution
                nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
            ).to(device),
            alpha=LinearAlpha(),
            beta=SquareRootBeta(),
        ).to(device)

        scale = PARAMS["scale"]
        x_bounds = [-scale, scale]
        y_bounds = [-scale, scale]

        plt.figure(figsize=(10, 10))
        plt.xlim(*x_bounds)
        plt.ylim(*y_bounds)
        plt.title("Gaussian Conditional Probability Path")

        # Plot source and target
        imshow_density(
            density=p0,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            bins=bins,
            vmin=-10,
            alpha=0.25,
            cmap=plt.get_cmap("Reds"),
        )
        imshow_density(
            density=p1,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            bins=bins,
            vmin=-10,
            alpha=0.25,
            cmap=plt.get_cmap("Blues"),
        )

        # Sample conditioning variable x1 ~ p1 (a single data point)
        x1 = path.sample_conditioning_variable(1)  # (1,2)
        ts = torch.linspace(0.0, 1.0, 7).to(device)

        # Plot x1
        plt.scatter(x1[:, 0].cpu(), x1[:, 1].cpu(), marker="*", color="red", s=75, label="x1")
        plt.xticks([])
        plt.yticks([])

        # Plot conditional probability path at each intermediate t
        num_samples = 1000
        for t in ts:
            x1_expanded = x1.expand(num_samples, 2)
            tt = t.unsqueeze(0).expand(num_samples, 1)  # (samples, 1)
            samples = path.sample_conditional_path(x1_expanded, tt)  # (samples, 2)
            plt.scatter(
                samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.25, s=8, label=f"t={t.item():.1f}"
            )

        plt.legend(prop={"size": 18}, markerscale=3)
        plt.show()

    plot_conditional_probability_path()

    return


if __name__ == "__main__":
    app.run()
