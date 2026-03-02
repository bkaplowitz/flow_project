import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import torch

    from flow.plot import graph_dynamics, plot_trajectories_1d
    from flow.sde import BrownianMotion, LangevinSDE, OrnsteinUhlenbeckProcess
    from flow.simulator import EulerMaruyamaSimulator

    return (
        BrownianMotion,
        EulerMaruyamaSimulator,
        LangevinSDE,
        OrnsteinUhlenbeckProcess,
        graph_dynamics,
        plot_trajectories_1d,
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
def _(
    BrownianMotion,
    EulerMaruyamaSimulator,
    device,
    plot_trajectories_1d,
    torch,
):
    from matplotlib import pyplot as plt

    _sigma = 1.0
    _n_traj = 500
    brownian_motion = BrownianMotion(_sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    x0 = torch.zeros(_n_traj, 1).to(device)  # Initial values - let's start at zero
    ts = torch.linspace(0.0, 5.0, 500).to(device)  # simulation timesteps

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_title(r"Trajectories of Brownian Motion with $\sigma=$" + str(_sigma), fontsize=18)
    ax.set_xlabel(r"time ($t$)", fontsize=18)
    ax.set_ylabel(r"$x_t$", fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True)
    plt.show()

    return (plt,)


@app.cell
def _(device, plt, torch):

    from flow.distributions import Gaussian, GaussianMixture
    from flow.plot import plot_2d_densities

    # Visualize densities
    densities = {
        "Gaussian": Gaussian(mean=torch.zeros(2), cov=10 * torch.eye(2)).to(device),
        "Random Mixture": GaussianMixture.random_2D(nmodes=5, std=1.0, scale=20.0, seed=3.0).to(
            device
        ),
        "Symmetric Mixture": GaussianMixture.symmetric_2D(nmodes=5, std=1.0, scale=8.0).to(device),
    }
    plot_2d_densities(densities, bins=100, scale=15, vmin=-15, cmap=plt.get_cmap("Blues"))

    return Gaussian, GaussianMixture


@app.cell
def _(
    EulerMaruyamaSimulator,
    OrnsteinUhlenbeckProcess,
    device,
    plot_trajectories_1d,
    plt,
    torch,
):
    # Try comparing multiple choices side-by-side
    thetas_and_sigmas = [
        (0.25, 0.0),
        (0.25, 0.5),
        (0.25, 2.0),
    ]
    simulation_time = 10.0

    num_plots = len(thetas_and_sigmas)
    fig, axes = plt.subplots(2, num_plots, figsize=(10.5 * num_plots, 15))

    # Top row: dynamics
    _n_traj = 10
    for _idx, (_theta, _sigma) in enumerate(thetas_and_sigmas):
        ou_process = OrnsteinUhlenbeckProcess(_theta, _sigma)
        ou_simulator = EulerMaruyamaSimulator(sde=ou_process)
        _x0 = (
            torch.linspace(-10.0, 10.0, _n_traj).view(-1, 1).to(device)
        )  # Initial values - let's start at zero
        _ts = torch.linspace(0.0, simulation_time, 1000).to(device)  # simulation timesteps

        ax1 = axes[0, _idx]
        ax1.set_title(
            f"Trajectories of OU Process with $\\sigma = ${_sigma}, $\\theta = ${_theta}",
            fontsize=15,
        )
        plot_trajectories_1d(_x0, ou_simulator, _ts, ax1, show_hist=False)

    # Bottom row: distribution
    _n_traj = 500
    for _idx, (_theta, _sigma) in enumerate(thetas_and_sigmas):
        ou_process = OrnsteinUhlenbeckProcess(_theta, _sigma)
        ou_simulator = EulerMaruyamaSimulator(sde=ou_process)
        _x0 = (
            torch.linspace(-10.0, 10.0, _n_traj).view(-1, 1).to(device)
        )  # Initial values - let's start at zero
        _ts = torch.linspace(0.0, simulation_time, 1000).to(device)  # simulation timesteps

        ax1 = axes[1, _idx]
        ax1.set_title(
            f"Trajectories of OU Process with $\\sigma = ${_sigma}, $\\theta = ${_theta}",
            fontsize=15,
        )
        ax1 = plot_trajectories_1d(
            _x0, ou_simulator, _ts, ax1, show_hist=True, decouple_hist_axis=True
        )
    plt.show()

    return


@app.cell
def _(
    EulerMaruyamaSimulator,
    Gaussian,
    GaussianMixture,
    LangevinSDE,
    device,
    graph_dynamics,
    torch,
):
    # Construct the simulator
    target = GaussianMixture.random_2D(nmodes=5, std=0.75, scale=15.0, seed=3.0).to(device)
    langevin_sde = LangevinSDE(sigma=0.6, density=target)
    langevin_simulator = EulerMaruyamaSimulator(sde=langevin_sde)

    # Graph the results!
    graph_dynamics(
        num_samples=1000,
        source_distribution=Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device),
        simulator=langevin_simulator,
        density=target,
        timesteps=torch.linspace(0, 5.0, 1000).to(device),
        plot_every=334,
        bins=200,
        scale=15,
    )

    return


if __name__ == "__main__":
    app.run()
