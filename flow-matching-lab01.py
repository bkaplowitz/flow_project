import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import torch
    from torch import Tensor

    from flow.plot import plot_trajectories_1d
    from flow.types.ode import ODE, SDE
    from flow.types.simulator import Simulator

    return ODE, SDE, Simulator, Tensor, plot_trajectories_1d, torch


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
def _(ODE, SDE, Simulator, Tensor, torch):
    class EulerSimulator(Simulator):
        """Euler method for ODE simulation. Integrades ODE"""

        def __init__(self, ode: ODE):
            self.ode = ode

        def step(self, xt: Tensor, t: Tensor, h: Tensor) -> Tensor:
            return xt + self.ode.drift_coef(xt, t) * h

    class EulerMaruyamaSimulator(Simulator):
        """Euler-Maruyama method for SDE Simulation. Integrates SDE"""

        def __init__(self, sde: SDE):
            self.sde = sde

        def step(self, xt: Tensor, t: Tensor, h: Tensor) -> Tensor:
            z = torch.randn_like(xt)
            return (
                xt
                + self.sde.drift_coef(xt, t) * h
                + self.sde.diffusion_coef(xt, t) * torch.sqrt(h) * z
            )

    return (EulerMaruyamaSimulator,)


@app.cell
def _(SDE, Tensor, torch):
    class BrownianMotion(SDE):
        def __init__(self, sigma: float) -> None:
            self.sigma = sigma

        def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
            return torch.zeros_like(xt)

        def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
            return self.sigma * torch.ones_like(xt)

    return (BrownianMotion,)


@app.cell
def _(
    BrownianMotion,
    EulerMaruyamaSimulator,
    device,
    plot_trajectories_1d,
    torch,
):
    from matplotlib import pyplot as plt

    sigma = 1.0
    n_traj = 500
    brownian_motion = BrownianMotion(sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    x0 = torch.zeros(n_traj, 1).to(device)  # Initial values - let's start at zero
    ts = torch.linspace(0.0, 5.0, 500).to(device)  # simulation timesteps

    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.set_title(
        r"Trajectories of Brownian Motion with $\sigma=$" + str(sigma), fontsize=18
    )
    ax.set_xlabel(r"time ($t$)", fontsize=18)
    ax.set_ylabel(r"$x_t$", fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True)
    plt.show()

    return (plt,)


@app.cell
def _(device, plt, torch):

    from flow.distributions import Gaussian, GaussianMixture
    from flow.plot import contour_density, imshow_density

    # Visualize densities
    densities = {
        "Gaussian": Gaussian(mean=torch.zeros(2), cov=10 * torch.eye(2)).to(device),
        "Random Mixture": GaussianMixture.random_2D(
            nmodes=5, std=1.0, scale=20.0, seed=3.0
        ).to(device),
        "Symmetric Mixture": GaussianMixture.symmetric_2D(
            nmodes=5, std=1.0, scale=8.0
        ).to(device),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    bins = 100
    scale = 15
    for idx, (name, density) in enumerate(densities.items()):
        ax5 = axes[idx]
        ax5.set_title(name)
        imshow_density(density, bins, scale, ax5, vmin=-15, cmap=plt.get_cmap("Blues"))
        contour_density(
            density,
            bins,
            scale,
            ax5,
            colors="grey",
            linestyles="solid",
            alpha=0.25,
            levels=20,
        )
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
