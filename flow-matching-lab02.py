import marimo

__generated_with = "0.20.4"
app = marimo.App(auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import torch
    from box import Box

    from flow_matching.distributions import (
        CheckerboardSampleable,
        Gaussian,
        GaussianMixture,
        SampleableDataset,
    )
    from flow_matching.flows import ConditionalVectorFieldODE, ConditionalVectorFieldSDE
    from flow_matching.paths import (
        GaussianConditionalProbabilityPath,
        LinearAlpha,
        SquareRootBeta,
    )
    from flow_matching.plot import imshow_density, plot_conditional_probability_path, plot_flow_path

    return (
        Box,
        CheckerboardSampleable,
        ConditionalVectorFieldODE,
        ConditionalVectorFieldSDE,
        Gaussian,
        GaussianConditionalProbabilityPath,
        GaussianMixture,
        LinearAlpha,
        SampleableDataset,
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
def _(Box, Gaussian, GaussianMixture, device, imshow_density, plt):
    _params = Box(
        {
            "scale": 15.0,
            "target_scale": 10.0,
            "target_std": 1.0,
        }
    )
    # initial distribution
    p0 = Gaussian.isotropic(dim=2, std=1.0).to(device)
    # target
    p1 = GaussianMixture.symmetric_2D(
        nmodes=5, std=_params.target_std, scale=_params.target_scale
    ).to(device)

    fig_gs, axes_gs = plt.subplots(1, 3, figsize=(24, 8))
    bins = 200

    scale_gs = _params.scale
    x_bounds_gs = (-scale_gs, scale_gs)
    y_bounds_gs = (-scale_gs, scale_gs)

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
    return (p1,)


@app.cell
def _(
    GaussianConditionalProbabilityPath,
    LinearAlpha,
    SquareRootBeta,
    device,
    p1,
    plot_conditional_probability_path,
):
    alpha_gaussian = LinearAlpha()
    beta_gaussian = SquareRootBeta()
    path_gaussian = GaussianConditionalProbabilityPath(
        p1=p1,
        alpha=alpha_gaussian,
        beta=beta_gaussian,
    ).to(device)
    plot_conditional_probability_path(path_gaussian)
    return


@app.cell
def _(
    Box,
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
    params: Box = Box(
        {
            "scale": 15.0,
            "target_scale": 10.0,
            "target_std": 1.0,
            "sigma": 2.5,
        }
    )
    # Construct path and vector field
    p_simple: Gaussian = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p_data: GaussianMixture = GaussianMixture.symmetric_2D(
        nmodes=5, std=params.target_std, scale=params.target_scale
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
    p_data: "GaussianMixture",
    p_simple: "Gaussian",
    params: "Box",
    path,
    plot_flow_path,
    x1: "torch.Tensor",
):
    conditional_vector_field_sde = ConditionalVectorFieldSDE(path, x1, sigma=params.sigma)
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
    params: "Box",
):
    from flow_matching.models import MLPVectorField
    from flow_matching.trainer import ConditionalFlowMatchingTrainer

    path_flow = GaussianConditionalProbabilityPath(
        p1=GaussianMixture.symmetric_2D(
            nmodes=5, std=params.target_std, scale=params.target_scale
        ).to(device),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)
    flow_model = MLPVectorField(dim=2, hidden_dims=[64, 64, 64, 64]).to(device)
    flow_model.compile()
    trainer = ConditionalFlowMatchingTrainer(path=path_flow, model=flow_model)
    epochs, losses = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=1000)
    return (
        ConditionalFlowMatchingTrainer,
        MLPVectorField,
        epochs,
        flow_model,
        losses,
        path_flow,
    )


@app.cell
def _(epochs, losses, plt, torch):
    plt.semilogy(
        epochs,
        torch.stack(losses).detach().cpu().numpy(),
    )
    plt.show()
    return


@app.cell
def _(
    flow_model,
    p_data: "GaussianMixture",
    p_simple: "Gaussian",
    params: "Box",
    path_flow,
    x1: "torch.Tensor",
):
    from flow_matching.flows import LearnedVectorFieldODE
    from flow_matching.plot import plot_marginal_flow_path

    learned_cond_vector_field = LearnedVectorFieldODE(flow_model)
    plot_marginal_flow_path(learned_cond_vector_field, path_flow, p_simple, p_data, x1, params)
    return LearnedVectorFieldODE, plot_marginal_flow_path


@app.cell
def _(
    Gaussian,
    GaussianConditionalProbabilityPath,
    GaussianMixture,
    LinearAlpha,
    SquareRootBeta,
    device,
    params: "Box",
):
    from flow_matching.models import MLPScore
    from flow_matching.trainer import ConditionalScoreMatchingTrainer

    p_data_score = GaussianMixture.symmetric_2D(
        nmodes=5, std=params.target_std, scale=params.target_scale
    ).to(device)
    p0_score = Gaussian.isotropic(dim=2, std=1.0).to(device)
    path_score = GaussianConditionalProbabilityPath(
        p1=p_data_score,
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)
    score_model = MLPScore(dim=2, hidden_dims=[64, 64, 64, 64]).to(device)
    score_model.compile()
    score_trainer = ConditionalScoreMatchingTrainer(path=path_score, model=score_model)
    epochs_score, losses_score = score_trainer.train(
        num_epochs=2000, device=device, lr=1e-3, batch_size=1000
    )
    return (
        epochs_score,
        losses_score,
        p0_score,
        p_data_score,
        path_score,
        score_model,
    )


@app.cell
def _(epochs_score, losses_score, plt, torch):
    plt.semilogy(epochs_score, torch.stack(losses_score).detach().cpu().numpy())
    return


@app.cell
def _(
    flow_model,
    p0_score,
    p_data_score,
    params: "Box",
    path_score,
    plot_marginal_flow_path,
    score_model,
    x1: "torch.Tensor",
):
    from flow_matching.flows import LangevinFlowSDE

    learned_cond_score = LangevinFlowSDE(flow_model, score_model, sigma=2.0)
    plot_marginal_flow_path(learned_cond_score, path_score, p0_score, p_data_score, x1, params)
    return


@app.cell
def _(
    LinearAlpha,
    SquareRootBeta,
    flow_model,
    params: "Box",
    plt,
    score_model,
):
    # Compare constructed score via `ScoreFromVectorField` to learned score.
    from flow_matching.models import ScoreFromVectorField
    from flow_matching.plot import compare_score_from_learned_flow_learned_score

    score_from_flow = ScoreFromVectorField(flow_model, alpha=LinearAlpha(), beta=SquareRootBeta())
    scale = params.scale
    _x_bounds = (-scale, scale)
    _y_bounds = (-scale, scale)
    compare_score_from_learned_flow_learned_score(score_model, score_from_flow, params)
    plt.show()
    return


@app.cell
def _(CheckerboardSampleable, SampleableDataset, device, plt):
    # Visualize samples
    from flow_matching.plot import hist2d_sampleable

    targets = {
        "circles": SampleableDataset.Circle(device),
        "moons": SampleableDataset.Moon(device, scale=3.5),
        "checkerboard": CheckerboardSampleable(device, grid_size=4),
    }
    fig_sample, axs_samples = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
    _num_samples = 20_000
    _num_bins = 100
    for idx, (target_name, target) in enumerate(targets.items()):
        ax = axs_samples[idx]
        hist2d_sampleable(target, _num_samples, bins=_num_bins, scale=7.5, ax=ax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Histogram of {target_name.capitalize()}")
    plt.show()
    return


@app.cell
def _(
    CheckerboardSampleable,
    ConditionalVectorFieldODE,
    Gaussian,
    device,
    plt,
    torch,
):

    from flow_matching import EulerSimulator
    from flow_matching.paths import LinearConditionalProbabilityPath
    from flow_matching.plot import every_nth_index, hist2d_samples

    linear_path = LinearConditionalProbabilityPath(
        p0=Gaussian.isotropic(dim=2, std=1.0), p1=CheckerboardSampleable(device, grid_size=4)
    ).to(device)

    def plot_linear_model(path: LinearConditionalProbabilityPath):
        alpha = 1.0
        bins = 300
        num_samples: int = 100_000
        num_timesteps = 500
        num_marginals = 5
        assert num_timesteps % (num_marginals - 1) == 0
        scale = 6.0
        fig, axs = plt.subplots(3, num_marginals, figsize=(6 * num_marginals, 6 * 3))
        axs = axs.reshape(3, num_marginals)
        # Sets for all subplots at once. Can also iter through.
        plt.setp(axs, xticks=[], yticks=[], xlim=(-scale, scale), ylim=(-scale, scale))
        p1 = path.p1.sample(1).to(device)  # (1,2)
        ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
        # Graph conditional probability path
        for idx, t in enumerate(ts):
            p1_expanded = p1.expand(num_samples, -1)
            t_expanded = t.view(1, 1).expand(num_samples, 1)
            xts = path.sample_conditional_path(p1_expanded, t_expanded)
            percentile = min(99 + 2 * torch.sin(t).item(), 100)
            axs[0, idx].set_title(f"{t.item():.2f}", fontsize=15)
        axs[0, 0].set_ylabel("Conditional (from Ground-Truth)", fontsize=20)
        # Plot p1
        axs[0, -1].scatter(
            p1[:, 0].cpu(), p1[:, 1].cpu(), marker="*", color="red", s=200, label="p1", zorder=20
        )
        axs[0, -1].legend()
        # Graph conditional vector fields
        ode = ConditionalVectorFieldODE(path, p1)
        simulator = EulerSimulator(ode)
        ts = torch.linspace(0, 1, num_timesteps).to(device)
        every_nth_idx = every_nth_index(len(ts), len(ts) // (num_marginals - 1))
        x0 = path.p0.sample(num_samples).to(device)
        xts = simulator.batch_simulate_with_trajectory(
            x0, ts.view(1, -1, 1).expand(num_samples, -1, 1)
        )
        xts = xts[:, every_nth_idx, :]
        for idx in range(xts.shape[1]):
            x_expanded = xts[:, idx, :]
            t_expanded = ts[every_nth_idx[idx]]
            percentile = min(99 + torch.sin(t_expanded).item(), 100)
            hist2d_samples(
                samples=x_expanded.cpu(),
                ax=axs[1, idx],
                bins=bins,
                scale=scale,
                percentile=percentile,
                alpha=alpha,
            )
            axs[1, idx].set_title(f"$t={t.item():.2f}$", fontsize=15)
        axs[1, 0].set_ylabel("Conditional (from ODE)", fontsize=20)
        # Plot p1
        axs[1, -1].scatter(
            p1[:, 0].cpu(), p1[:, 1].cpu(), marker="*", color="red", s=200, label="z", zorder=20
        )
        axs[1, -1].legend()

        # Graph conditional prob paths using sample_marginal_paths
        ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
        for idx, t in enumerate(ts):
            p1_expanded = p1.expand(num_samples, -1)
            t_expanded = t.view(1, 1).expand(num_samples, 1)
            xts = path.sample_marginal_path(t_expanded)
            hist2d_samples(
                samples=xts.cpu(), ax=axs[2, idx], bins=300, scale=scale, percentile=99, alpha=1.0
            )
            axs[2, idx].set_title(f"$t={t.item():.2f}$", fontsize=15)
        axs[2, 0].set_ylabel("Marginal", fontsize=20)
        plt.show()

    plot_linear_model(linear_path)
    return (
        EulerSimulator,
        LinearConditionalProbabilityPath,
        every_nth_index,
        hist2d_samples,
    )


@app.cell
def _(
    CheckerboardSampleable,
    ConditionalFlowMatchingTrainer,
    Gaussian,
    LinearConditionalProbabilityPath,
    MLPVectorField,
    device,
):
    # FIXME!!!
    linear_path_conditional = LinearConditionalProbabilityPath(
        p0=Gaussian.isotropic(dim=2, std=1.0), p1=CheckerboardSampleable(device, grid_size=4)
    ).to(device)
    linear_flow_model = MLPVectorField(dim=2, hidden_dims=[64, 64, 64, 64])
    linear_trainer = ConditionalFlowMatchingTrainer(
        linear_path_conditional, model=linear_flow_model
    )
    _losses_linear = linear_trainer.train(
        num_epochs=10_000, device=device, lr=1e-4, batch_size=2_000
    )
    return (linear_flow_model,)


@app.cell
def _(
    EulerSimulator,
    LearnedVectorFieldODE,
    device,
    every_nth_index,
    hist2d_samples,
    linear_flow_model,
    path,
    plt,
    torch,
):
    # FIXME !!!!
    def plot_fn():
        ##########################
        # Play around With These #
        ##########################
        num_samples = 50000
        num_marginals = 5

        ##############
        # Setup Plots #
        ##############

        fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
        axes = axes.reshape(2, num_marginals)
        scale = 6.0

        ###########################
        # Graph Ground-Truth Marginals #
        ###########################
        ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
        for idx, t in enumerate(ts):
            tt = t.view(1, 1).expand(num_samples, 1)
            xts = path.sample_marginal_path(tt)
            hist2d_samples(
                samples=xts.cpu(), ax=axes[0, idx], bins=200, scale=scale, percentile=99, alpha=1.0
            )
            axes[0, idx].set_xlim(-scale, scale)
            axes[0, idx].set_ylim(-scale, scale)
            axes[0, idx].set_xticks([])
            axes[0, idx].set_yticks([])
            axes[0, idx].set_title(f"$t={t.item():.2f}$", fontsize=15)
        axes[0, 0].set_ylabel("Ground Truth", fontsize=20)

        ###############################################
        # Graph Marginals of Learned Vector Field #
        ###############################################
        ode = LearnedVectorFieldODE(linear_flow_model)
        simulator = EulerSimulator(ode)
        ts = torch.linspace(0, 1, 100).to(device)
        record_every_idxs = every_nth_index(len(ts), len(ts) // (num_marginals - 1))
        x0 = path.p0.sample(num_samples).to(device)
        xts = simulator.batch_simulate_with_trajectory(
            x0, ts.view(1, -1, 1).expand(num_samples, -1, 1)
        )
        xts = xts[:, record_every_idxs, :]
        for idx in range(xts.shape[1]):
            xx = xts[:, idx, :]
            hist2d_samples(
                samples=xx.detach().cpu(),
                ax=axes[1, idx],
                bins=200,
                scale=scale,
                percentile=99,
                alpha=1.0,
            )
            axes[1, idx].set_xlim(-scale, scale)
            axes[1, idx].set_ylim(-scale, scale)
            axes[1, idx].set_xticks([])
            axes[1, idx].set_yticks([])
            tt = ts[record_every_idxs[idx]]
            axes[1, idx].set_title(f"$t={tt.item():.2f}$", fontsize=15)
        axes[1, 0].set_ylabel("Learned", fontsize=20)

        plt.show()

    plot_fn()
    return


if __name__ == "__main__":
    app.run()
