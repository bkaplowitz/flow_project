"""Plotting helpers."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from celluloid import Camera
from IPython.display import HTML
from matplotlib import cm
from matplotlib.colors import Colormap

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
    x_bounds: Sequence[float] | None = None,
    y_bounds: Sequence[float] | None = None,
) -> tuple[torch.Tensor[float], torch.Tensor[float], list[float]]:
    if scale is not None:
        x = torch.linspace(-scale, scale, bins).to(device)
        y = torch.linspace(-scale, scale, bins).to(device)
        extent = [-scale, scale, -scale, scale]
    elif x_bounds is not None and y_bounds is not None:
        x = torch.linspace(*x_bounds, bins).to(device)
        y = torch.linspace(*y_bounds, bins).to(device)
        extent = x_bounds + y_bounds
    else:
        raise ValueError("Either scale or x_bounds and y_bounds have to be defined.")
    return x, y, extent


def plot_trajectories_1d(
    x0: torch.Tensor,
    simulator: Simulator,
    ts: torch.Tensor,
    ax: Axes | None = None,
    show_hist: bool = False,
    decouple_hist_axis: bool = False,
):
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

    fig = ax.figure
    if fig is not None:
        title = ax.get_title()
        if title:
            title_size = ax.title.get_fontsize()
            ax.set_title("")

            axes = [ax]
            if show_hist:
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
    samples,
    ax: Axes | None = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs,
):
    H, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]]
    )
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot with imshow
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin="lower", norm=norm, **kwargs)


# Several plotting utility functions
def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Axes | None = None, **kwargs):
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)  # (ns, 2)
    ax.hist2d(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Axes | None = None, **kwargs):
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)  # (ns, 2)
    ax.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), **kwargs)


def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Axes | None = None, **kwargs):
    assert sampleable.dim == 2
    ax = _get_ax(ax)
    samples = sampleable.sample(num_samples)
    sns.kdeplot(x=samples[:, 0].detach().cpu(), y=samples[:, 1].detach().cpu(), ax=ax, **kwargs)


def imshow_density(
    density: Density,
    bins: int,
    scale: float | None = None,
    x_bounds: Sequence[float] | None = None,
    y_bounds: Sequence[float] | None = None,
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
    x_bounds: Sequence[float] | None = None,
    y_bounds: Sequence[float] | None = None,
    ax: Axes | None = None,
    **kwargs,
):
    ax = _get_ax(ax)
    x, y, extent = _get_scale_or_bounds(bins, scale=scale, x_bounds=x_bounds, y_bounds=y_bounds)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    ax.contour(density.cpu(), extent=extent, origin="lower", **kwargs)


def plot_2d_densities(
    densities: dict[str, Density],
    bins=100,
    scale=15,
    figsize=(18, 6),
    subplots_dim=(1, 3),
    vmin=-15,
    cmap: Colormap = BLUES_CMAP,
) -> None:
    fig, axes = plt.subplots(*subplots_dim, figsize=figsize)
    for idx, (name, density) in enumerate(densities.items()):
        ax = axes[idx]
        ax.set_title(name)
        imshow_density(density, bins, scale, ax, vmin=vmin, cmap=cmap)
        contour_density(
            density,
            bins,
            scale,
            ax,
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
    timesteps: torch.Tensor,
    plot_every: int,
    bins: int,
    scale: float,
):
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
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kdeplot_ax, color="grey")
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
    timesteps: torch.Tensor[torch.int64],
    animate_every: int,
    bins: int,
    scale: float,
    save_path: os.PathLike = "dynamics_animation.mp4",
) -> None:
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
        sns.kdeplot(x=xt[:, 0].cpu(), y=xt[:, 1].cpu(), alpha=0.5, ax=kdeplot_ax, color="gray")
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
