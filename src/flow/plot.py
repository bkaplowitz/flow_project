"""
Plotting helpers.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns
import torch

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

    from src.types.simulator import Simulator


def plot_trajectories_1d(
    x0: torch.Tensor,
    simulator: Simulator,
    ts: torch.Tensor,
    ax: Axes | None = None,
    show_hist: bool = False,
    decouple_hist_axis: bool = False,
):
    """
    Graphs the trajectories of a one-dimensional SDE with given initial values (x0)
    and simulation timesteps (ts).
    Args:
        x0: state at time t, shape (num_trajectories, 1)
        simulator: Simulator object used to simulate
        ts: timesteps to simulate along, shape (num_timesteps,)
        ax: pyplot Axes object to plot on
        decouple_hist_axis: if True, don't share y-axis between
            trajectories and histogram
    """
    if ax is None:
        ax = plt.gca()
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
            float(terminal_points.max() - terminal_points.min())
            if terminal_points.size
            else 1.0
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
