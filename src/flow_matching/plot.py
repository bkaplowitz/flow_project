"""Plotting helpers."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from box import Box
from celluloid import Camera
from IPython.display import HTML
from matplotlib import colors
from matplotlib.colors import Colormap
from torch import Tensor

from flow_matching.base.dynamics import ODE, SDE
from flow_matching.distributions import Gaussian, GaussianMixture
from flow_matching.models import MLPScore, ScoreFromVectorField
from flow_matching.paths import GaussianConditionalProbabilityPath, LinearAlpha, SquareRootBeta
from flow_matching.simulator import EulerMaruyamaSimulator, EulerSimulator

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

    from flow_matching.base.probability import Density, Sampleable
    from flow_matching.base.simulator import Simulator

BLUES_CMAP: Colormap = plt.get_cmap("Blues")
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def _get_ax(ax: Axes | None = None) -> Axes:
    """Helper function to check if ax is None and return gca, else just returns passed ax.

    Args:
        ax: possible axes or None, in which case return default of gca.
    """
    if ax is None:
        return plt.gca()
    return ax


def _get_scale_or_bounds(
    bins: int,
    scale: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float, float]]:
    if scale is not None:
        x = torch.linspace(-scale, scale, bins).to(device)
        y = torch.linspace(-scale, scale, bins).to(device)
        extent = (-scale, scale, -scale, scale)
    elif x_bounds is not None and y_bounds is not None:
        x = torch.linspace(*x_bounds, bins).to(device)
        y = torch.linspace(*y_bounds, bins).to(device)
        extent = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])
    else:
        raise ValueError("Either scale or x_bounds and y_bounds have to be defined.")
    return x, y, extent


def plot_trajectories_1d(
    x0: Tensor,
    simulator: Simulator,
    ts: Tensor,
    ax: Axes | None = None,
    show_hist: bool = False,
    decouple_hist_axis: bool = False,
) -> None:
    """Graphs the trajectories of a 1D SDE with given initial values and timesteps.

    Args:
        x0: state at time t, shape (num_trajectories, 1)
        simulator: Simulator object used to simulate
        ts: timesteps to simulate along, shape (num_timesteps,)
        ax: pyplot Axes object to plot on
        show_hist: if True, show histogram of terminal values
        decouple_hist_axis: if True, don't share y-axis between
            trajectories and histogram
    """
    ax = _get_ax(ax)
    # (num_trajectories, num_timesteps, ...)
    trajectories = simulator.simulate_with_trajectory(x0, ts)

    line_color = sns.color_palette("crest", 1)[0]
    hist_color = sns.color_palette("flare", 1)[0]
    label_size = 12
    tick_size = 10

    timesteps_cpu = ts.detach().cpu().numpy()
    for trajectory_idx in range(trajectories.shape[0]):
        # (num_timesteps,)
        trajectory = trajectories[trajectory_idx, :, 0].detach().cpu().numpy()
        sns.lineplot(
            x=timesteps_cpu,
            y=trajectory,
            ax=ax,
            color=line_color,
            alpha=0.45,
            linewidth=1.1,
            legend=False,
        )

    ax.set_xlabel(r"time ($t$)", fontsize=label_size)
    ax.set_ylabel(r"$X_t$", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(alpha=0.2, linewidth=0.6)

    if show_hist:
        terminal_points = trajectories[:, -1, 0].detach().cpu().numpy()
        data_range = (
            float(terminal_points.max() - terminal_points.min()) if terminal_points.size else 1.0
        )
        binwidth = max(data_range / 25.0, 0.05)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        sharey = None if decouple_hist_axis else ax
        hist_ax = divider.append_axes("right", size="22%", pad=0.45, sharey=sharey)
        sns.histplot(
            y=terminal_points,
            ax=hist_ax,
            binwidth=binwidth,
            color=hist_color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        hist_ax.set_xlabel("count", fontsize=label_size)
        hist_ax.set_ylabel("")
        hist_ax.tick_params(axis="both", labelsize=tick_size)
        if decouple_hist_axis:
            hist_ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            hist_ax.tick_params(axis="y", left=False, labelleft=False)
        hist_ax.grid(axis="x", alpha=0.2, linewidth=0.6)
    else:
        hist_ax = None

    fig = ax.figure
    if fig is not None:
        title = ax.get_title()
        if title:
            title_size = ax.title.get_fontsize()
            ax.set_title("")

            axes = [ax]
            if show_hist:
                hist_ax = cast(Axes, hist_ax)  # typing fix
                axes.append(hist_ax)

            fig.canvas.draw()
            bboxes = [a.get_position() for a in axes]

            left = min(b.x0 for b in bboxes)
            right = max(b.x1 for b in bboxes)
            top = max(b.y1 for b in bboxes)

            x_center = 0.5 * (left + right)
            y = top + 0.005

            fig.text(x_center, y, title, ha="center", va="bottom", fontsize=title_size)


def hist2d_samples(
    samples: Tensor,
    ax: Axes | None = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs,
) -> None:
    ax = _get_ax(ax)
    H, xedges, yedges = np.histogram2d(
        samples[:, 0].detach().cpu(),
        samples[:, 1].detach().cpu(),
        bins=bins,
        range=[[-scale, scale], [-scale, scale]],
    )
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot with imshow
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    ax.imshow(H.T, extent=extent, origin="lower", norm=norm, **kwargs)


# Several plotting utility functions
def hist2d_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Axes | None = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs,
) -> None:
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)  # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)


def scatter_sampleable(
    sampleable: Sampleable, num_samples: int, ax: Axes | None = None, **kwargs
) -> None:
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)  # (ns, 2)
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def kdeplot_sampleable(
    sampleable: Sampleable, num_samples: int, ax: Axes | None = None, **kwargs
) -> None:
    assert sampleable.dim == 2
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)
    sns.kdeplot(
        x=samples[:, 0].detach().cpu().numpy(),
        y=samples[:, 1].detach().cpu().numpy(),
        ax=ax,
        **kwargs,
    )


def imshow_density(
    density: Density,
    bins: int,
    scale: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs,
):
    ax = _get_ax(ax)
    x, y, extent = _get_scale_or_bounds(bins, scale=scale, x_bounds=x_bounds, y_bounds=y_bounds)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density_val = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(density_val.cpu(), extent=extent, origin="lower", **kwargs)


def contour_density(
    density: Density,
    bins: int,
    scale: float | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs,
):
    ax = _get_ax(ax)
    x, y, extent = _get_scale_or_bounds(bins, scale=scale, x_bounds=x_bounds, y_bounds=y_bounds)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density_val = density.log_density(xy).reshape(bins, bins).T
    ax.contour(density_val.cpu(), extent=extent, origin="lower", **kwargs)


def plot_2d_densities(
    densities: Box,
    bins=100,
    scale=15,
    figsize=(18, 6),
    subplots_dim=(1, 3),
    vmin=-15,
    cmap: Colormap = BLUES_CMAP,
) -> None:
    fig, axes = plt.subplots(*subplots_dim, figsize=figsize)
    # Typing fix
    axes = cast(tuple[Axes, ...], axes)
    for idx, (name, density) in enumerate(densities.items()):
        ax = axes[idx]
        ax.set_title(name)
        imshow_density(density, bins, scale=scale, ax=ax, vmin=vmin, cmap=cmap)
        contour_density(
            density,
            bins,
            scale=scale,
            ax=ax,
            colors="grey",
            linestyle="solid",
            alpha=0.25,
            levels=20,
        )
    plt.show()


def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """Compute the indices to record in the trajectory given a record_every parameter."""
    if n == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, n),
            torch.tensor([num_timesteps - 1]),
        ]
    )


def graph_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: Tensor,
    plot_every: int,
    bins: int,
    scale: float,
) -> None:
    """Plot the evolution of samples from source under the simulation scheme.

    Uses simulator (itself a discretization of an ODE or SDE).

    Args:
        num_samples: the number of samples to simulate
        source_distribution: distribution from which we draw initial samples at t=0
        simulator: the discertized simulation scheme used to simulate the dynamics
        density: the target density
        timesteps: the timesteps used by the simulator
        plot_every: number of timesteps between consecutive plots
        bins: number of bins for imshow
        scale: scale for imshow
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices_to_plot = every_nth_index(len(timesteps), plot_every)
    plot_timesteps = timesteps[indices_to_plot]
    plot_xts = xts[:, indices_to_plot]

    # Graph
    fig, axes = plt.subplots(2, len(plot_timesteps), figsize=(8 * len(plot_timesteps), 16))
    axes = axes.reshape((2, len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        xt = plot_xts[:, t_idx]
        # Scatter axes
        scatter_ax = axes[0, t_idx]
        imshow_density(
            density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap("Blues")
        )
        scatter_ax.scatter(
            xt[:, 0].cpu(), xt[:, 1].cpu(), marker="x", color="black", alpha=0.75, s=15
        )
        scatter_ax.set_title(f"Samples at t={t:.1f}", fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        # Kdeplot axes
        kdeplot_ax = axes[1, t_idx]
        imshow_density(
            density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap("Blues")
        )
        sns.kdeplot(
            x=xt[:, 0].detach().cpu().numpy(),
            y=xt[:, 1].detach().cpu().numpy(),
            alpha=0.5,
            ax=kdeplot_ax,
            color="grey",
        )
        kdeplot_ax.set_title(f"Density of Samples at t={t:.1f}", fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")

    plt.show()


def animate_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    density: Density,
    timesteps: torch.Tensor,
    animate_every: int,
    bins: int,
    scale: float,
    save_path: Path | str = "dynamics_animation.mp4",
) -> HTML:
    """Plot the evolution of samples from source under simulation.

    Args:
        num_samples: number of samples to simulate
        source_distribution: initial distribution for samples
        simulator: simulates samples by iterating on timesteps
        density: target distribution (assumes a density)
        timesteps: timesteps to simulate for. Gives discretization for process. Shape nts
        animate_every: how frequently to update animation.
        bins: histogram bins for kde plot
        scale: scaling for kde plot
        save_path: Pathlike savepath
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)  # (batch, dims)
    indices_to_animate = every_nth_index(len(timesteps), animate_every)
    animate_timesteps = timesteps[indices_to_animate]
    animate_xts = xts[:, indices_to_animate]
    # Graph
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    camera: Camera = Camera(fig)
    for t_idx in range(len(animate_timesteps)):
        animate_timesteps[t_idx].item()
        xt = animate_xts[:, t_idx]
        scatter_ax = axes[0]
        imshow_density(
            density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap("Blues")
        )
        scatter_ax.scatter(
            xt[:, 0].cpu(), xt[:, 1].cpu(), marker="x", color="black", alpha=0.75, s=15
        )
        scatter_ax.set_title("Samples")

        # KDE Plots
        kdeplot_ax = axes[1]
        imshow_density(density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=BLUES_CMAP)
        sns.kdeplot(
            x=xt[:, 0].detach().cpu().numpy(),
            y=xt[:, 1].detach().cpu().numpy(),
            alpha=0.5,
            ax=kdeplot_ax,
            color="gray",
        )
        kdeplot_ax.set_title("Density of Samples", fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")
        camera.snap()
    animation = camera.animate()
    animation.save(save_path)
    plt.close()
    return HTML(animation.to_html5_video())


def plot_sample(ax: Axes, x1: torch.Tensor, scale: float, title: str = ""):
    x_bounds = [-scale, scale]
    y_bounds = [-scale, scale]
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20)
    ax.scatter(
        x1[:, 0].cpu(), x1[:, 1].cpu(), marker="*", color="red", s=200, label="x1", zorder=20
    )


def plot_source_sample_densities(
    ax: Axes, p_simple: Density, p_data: Density, scale: float
) -> None:

    # Plot source and sample densities
    imshow_density(
        density=p_simple,
        bins=200,
        scale=scale,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Reds"),
    )
    imshow_density(
        density=p_data,
        bins=200,
        scale=scale,
        ax=ax,
        vmin=-10,
        alpha=0.25,
        cmap=plt.get_cmap("Blues"),
    )


def plot_conditional_probability_path() -> None:
    # Construct conditional probability path
    _params = Box({"scale": 15.0, "target_scale": 10.0, "target_std": 1.0})
    bins = 200
    p0 = Gaussian.isotropic(dim=2, std=1.0).to(device)
    p1 = GaussianMixture.symmetric_2D(
        nmodes=5, std=_params.target_std, scale=_params.target_scale
    ).to(device)
    # Construct conditional probability path
    path = GaussianConditionalProbabilityPath(
        p1=p1,
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)

    scale = _params.scale
    x_bounds = (-scale, scale)
    y_bounds = (-scale, scale)

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
    ts = torch.linspace(0.0, 1 - 1e-9, 7).to(device)

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


def plot_flow_path(
    cond_vector_field: ODE | SDE,
    path: GaussianConditionalProbabilityPath,
    p_simple: Density,
    p_data: Density,
    x1: Tensor,
    params: Box,
    num_samples: int = 1000,
    num_timesteps: int = 100,
    num_marginals: int = 3,
) -> None:
    """Plots the flow path for a conditional vector field as either an ODE or SDE."""
    fig, axs = plt.subplots(1, 3, figsize=(36, 12))
    scale = params.scale
    legendsize = 24
    markerscale = 1.8
    ########################
    # Plot 1: Ground truth #
    ########################
    ax: Axes
    ax = axs[1]

    title = "Samples from Conditional ODE"

    # Plot sample
    plot_sample(ax, x1, scale, title)
    # plot source and sample densities
    plot_source_sample_densities(ax, p_simple, p_data, scale)

    # Construct ODE and simulator
    if isinstance(cond_vector_field, ODE):
        simulator = EulerSimulator(cond_vector_field)
    elif isinstance(cond_vector_field, SDE):
        simulator = EulerMaruyamaSimulator(cond_vector_field)
    else:
        raise ValueError(
            "Expected cond_vector_field to be ODE or SDE instance",
            f"instead found {type(cond_vector_field)}",
        )
    x0 = path.p0.sample(num_samples).to(device)  # (num_samples, 2)
    ts = (
        torch.linspace(0.0, 1 - 1e-9, num_timesteps)
        .view(1, -1, 1)
        .expand(num_samples, -1, 1)
        .to(device)
    )  # (bs, nts, 1)
    x = simulator.batch_simulate_with_trajectory(x0, ts)  # (bs, nts, dims)

    # Extract every nth iteration
    every_n = every_nth_index(num_timesteps, n=num_timesteps // num_marginals)
    x_every_n = x[:, every_n, :]  # (bs, nts // n, dim)
    t_every_n = ts[0, every_n]  # (nts // n, 1)
    for plot_idx in range(x_every_n.shape[1]):
        t = t_every_n[plot_idx].item()
        ax.scatter(
            x=x_every_n[:, plot_idx, 0].detach().cpu(),
            y=x_every_n[:, plot_idx, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t:.2f}",
        )
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)
    # Graph trajectories of ODE
    ########################
    # Plot 2: Trajectories #
    ########################
    ax = axs[2]
    plot_sample(ax, x1, scale, title="Trajectories of Conditional ODE")
    plot_source_sample_densities(ax, p_simple, p_data, scale)
    # Plot first 15 trajectories
    for traj_idx in range(15):
        ax.plot(
            x[traj_idx, :, 0].detach().cpu(),
            x[traj_idx, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)
    #########################################
    # Plot 2: Ground-truth probability path #
    #########################################
    ax = axs[0]
    plot_sample(ax, x1, scale, title="Ground-Truth Conditional Probability Path")
    # Plot path
    for plot_idx in range(x_every_n.shape[1]):
        t_many_samples = t_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        x1_many_samples = x1.expand(num_samples, 2)
        marginal_samples = path.sample_conditional_path(x1_many_samples, t_many_samples)
        ax.scatter(
            marginal_samples[:, 0].detach().cpu(),
            marginal_samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t_many_samples[0, 0].item():.2f}",
        )
    plot_source_sample_densities(ax, p_simple, p_data, scale)
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)

    plt.show()


def plot_marginal_flow_path(
    cond_vector_field: ODE | SDE,
    path: GaussianConditionalProbabilityPath,
    p_simple: Density,
    p_data: Density,
    x1: Tensor,
    params: Box,
    num_samples: int = 1000,
    num_timesteps: int = 100,
    num_marginals: int = 3,
) -> None:
    """Plots the flow path for a marginal vector field as either an ODE or SDE."""
    fig, axs = plt.subplots(1, 3, figsize=(36, 12))
    scale = params.scale
    legendsize = 24
    markerscale = 1.8
    ########################
    # Plot 1: Ground truth #
    ########################
    ax: Axes
    ax = axs[1]

    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # plot source and sample densities
    plot_source_sample_densities(ax, p_simple, p_data, scale)

    # Construct ODE and simulator
    if isinstance(cond_vector_field, ODE):
        simulator = EulerSimulator(cond_vector_field)
    elif isinstance(cond_vector_field, SDE):
        simulator = EulerMaruyamaSimulator(cond_vector_field)
    else:
        raise ValueError(
            "Expected cond_vector_field to be ODE or SDE instance",
            f"instead found {type(cond_vector_field)}",
        )
    x0 = path.p0.sample(num_samples).to(device)  # (num_samples, 2)
    ts = (
        torch.linspace(0.0, 1 - 1e-9, num_timesteps)
        .view(1, -1, 1)
        .expand(num_samples, -1, 1)
        .to(device)
    )  # (bs, nts, 1)
    x = simulator.batch_simulate_with_trajectory(x0, ts)  # (bs, nts, dims)

    # Extract every nth iteration
    every_n = every_nth_index(num_timesteps, n=num_timesteps // num_marginals)
    x_every_n = x[:, every_n, :]  # (bs, nts // n, dim)
    t_every_n = ts[0, every_n]  # (nts // n, 1)
    for plot_idx in range(x_every_n.shape[1]):
        t = t_every_n[plot_idx].item()
        ax.scatter(
            x=x_every_n[:, plot_idx, 0].detach().cpu(),
            y=x_every_n[:, plot_idx, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t:.2f}",
        )
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)
    # Graph trajectories of ODE
    ########################
    # Plot 2: Trajectories of learned marginal ode #
    ########################
    ax = axs[2]
    ax.set_title("Trajectories of Learned Marginal ODE")
    plot_source_sample_densities(ax, p_simple, p_data, scale)
    # Plot first 15 trajectories
    for traj_idx in range(num_samples // 10):
        ax.plot(
            x[traj_idx, :, 0].detach().cpu(),
            x[traj_idx, :, 1].detach().cpu(),
            alpha=0.5,
            color="black",
        )
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)
    #########################################
    # Graph Ground-truth probability path   #
    #########################################
    ax = axs[0]
    ax.set_title("Ground-Truth Marginal Probability Path")
    # Plot path
    for plot_idx in range(x_every_n.shape[1]):
        t_many_samples = t_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(t_many_samples)
        ax.scatter(
            marginal_samples[:, 0].detach().cpu(),
            marginal_samples[:, 1].detach().cpu(),
            marker="o",
            alpha=0.5,
            label=f"t={t_many_samples[0, 0].item():.2f}",
        )
    plot_source_sample_densities(ax, p_simple, p_data, scale)
    ax.legend(prop={"size": legendsize}, loc="upper right", markerscale=markerscale)

    plt.show()


def compare_score_from_learned_flow_learned_score(
    score_model: MLPScore,
    score_model_from_flow: ScoreFromVectorField,
    params: Box,
    num_bins=30,
    num_marginals=4,
) -> None:
    """Quiver plot comparison of learned score model and score model from learned flow model."""
    # Set font size

    path = GaussianConditionalProbabilityPath(
        p1=GaussianMixture.symmetric_2D(
            nmodes=5, std=params.target_std, scale=params.target_scale
        ).to(device),
        alpha=LinearAlpha(),
        beta=SquareRootBeta(),
    ).to(device)
    learned_score_model = score_model

    # Plot score fields
    fig, axs = plt.subplots(nrows=2, ncols=num_marginals, figsize=(6 * num_marginals, 12))
    axs = axs.reshape((2, num_marginals))
    scale = params.scale
    ts = torch.linspace(0.0, 0.999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=-1)
    axs[0, 0].set_ylabel("Learned from Score matching", fontsize=12)
    axs[1, 0].set_ylabel(r"Computed from $u_t^{\theta}(x)$", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins**2
        tt = t.view(1, 1).expand(bs, 1)

        # Learned scores
        learned_scores = learned_score_model(xy, tt)
        learned_scores_x = learned_scores[:, 0]
        learned_scores_y = learned_scores[:, 1]
        ax = axs[0, idx]
        ax.quiver(
            xx.detach().cpu(),
            yy.detach().cpu(),
            learned_scores_x.detach().cpu(),
            learned_scores_y.detach().cpu(),
            scale=125,
            alpha=0.5,
        )
        plot_source_sample_densities(ax, path.p0, path.p1, scale=scale)
        ax.set_title(r"$s^{\theta}_t$" + f" at t={t.item():.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        # Score model from flow
        ax = axs[1, idx]
        flow_scores = score_model_from_flow(xy, tt)
        flow_scores_x = flow_scores[:, 0]
        flow_scores_y = flow_scores[:, 1]
        ax.quiver(
            xx.detach().cpu(),
            yy.detach().cpu(),
            flow_scores_x.detach().cpu(),
            flow_scores_y.detach().cpu(),
            scale=125,
            alpha=0.5,
        )
        plot_source_sample_densities(ax, path.p0, path.p1, scale=scale)
        ax.set_title(r"$\tilde{s}^{\theta}_t$" + f" at t={t.item():.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove ticks
        # Flow score model
