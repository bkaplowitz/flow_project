import marimo

__generated_with = "0.20.4"
app = marimo.App(auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from flow_matching import (
        ConditionalVectorFieldODE,
        ConditionalVectorFieldSDE,
        GaussianConditionalProbabilityPath,
        LinearAlpha,
        SquareRootBeta,
    )
    from flow_matching.distributions import Gaussian, GaussianMixture
    from flow_matching.plot import imshow_density, plot_conditional_probability_path, plot_flow_path

    return (
        ConditionalVectorFieldODE,
        ConditionalVectorFieldSDE,
        Gaussian,
        GaussianConditionalProbabilityPath,
        GaussianMixture,
        LinearAlpha,
        SquareRootBeta,
        imshow_density,
        mo,
        plot_conditional_probability_path,
        plot_flow_path,
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
    # initial distribution
    p0 = Gaussian.isotropic(dim=2, std=1.0).to(device)
    # target
    p1 = GaussianMixture.symmetric_2D(
        nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
    ).to(device)

    fig_gs, axes_gs = plt.subplots(1, 3, figsize=(24, 8))
    bins = 200

    scale = PARAMS["scale"]
    x_bounds_gs = (-scale, scale)
    y_bounds_gs = (-scale, scale)

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

    return


@app.cell
def _(plot_conditional_probability_path):

    plot_conditional_probability_path()

    return


@app.cell
def _(
    ConditionalVectorFieldODE,
    Gaussian,
    GaussianConditionalProbabilityPath,
    GaussianMixture,
    LinearAlpha,
    SquareRootBeta,
    device,
    plot_flow_path,
    torch,
):
    # Plot ODE flow
    params: dict[str, float] = {
        "scale": 15.0,
        "target_scale": 10.0,
        "target_std": 1.0,
        "sigma": 2.5,
    }
    # Construct path and vector field
    p_simple: Gaussian = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data: GaussianMixture = GaussianMixture.symmetric_2D(
        nmodes=5, std=params["target_std"], scale=params["target_scale"]
    ).to(device)
    alpha = LinearAlpha()
    beta = SquareRootBeta()
    path = GaussianConditionalProbabilityPath(p1=p_data, alpha=alpha, beta=beta)
    torch.manual_seed(1)
    x1: torch.Tensor = path.sample_conditioning_variable(1)
    conditional_vector_field_ode = ConditionalVectorFieldODE(path, x1)
    plot_flow_path(conditional_vector_field_ode, path, p_simple, p_data, x1, params)

    return p_data, p_simple, params, path, x1


@app.cell
def _(
    ConditionalVectorFieldSDE,
    p_data,
    p_simple,
    params,
    path,
    plot_flow_path,
    x1,
):
    conditional_vector_field_sde = ConditionalVectorFieldSDE(path, x1, sigma=params["sigma"])
    plot_flow_path(conditional_vector_field_sde, path, p_simple, p_data, x1, params)

    return


@app.cell
def _(mo):
    mo.md(r"""
    # Flow Matching and Score Matching Training
    """)
    return


@app.cell
def _(
    GaussianConditionalProbabilityPath,
    GaussianMixture,
    LinearAlpha,
    SquareRootBeta,
    device,
    params: dict[str, float],
):
    from flow_matching.models import MLPVectorField
    from flow_matching.trainer import ConditionalFlowMatchingTrainer

    path_flow = GaussianConditionalProbabilityPath(
        p1=GaussianMixture.symmetric_2D(
            nmodes=5, std=params["target_std"], scale=params["target_scale"]
        ).to(device),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)
    flow_model = MLPVectorField(dim=2, hidden_dims=[64, 64, 64, 64]).to(device)
    trainer = ConditionalFlowMatchingTrainer(path=path_flow, model=flow_model)
    _losses = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=1000)

    return


if __name__ == "__main__":
    app.run()
