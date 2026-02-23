import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import torch
    from torch import Tensor

    from flow.types.ode import ODE, SDE
    from flow.types.simulator import Simulator

    device = torch.device(  # noqa: F841
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return ODE, SDE, Simulator, Tensor, torch


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

    return


@app.cell
def _(SDE, Tensor, torch):
    class BrownianMotion(SDE):
        def __init__(self, sigma: float) -> None:
            self.sigma = sigma

        def drift_coef(self, xt: Tensor, t: Tensor) -> Tensor:
            return torch.zeros_like(xt)

        def diffusion_coef(self, xt: Tensor, t: Tensor) -> Tensor:
            return self.sigma * torch.ones_like(xt)

    return


if __name__ == "__main__":
    app.run()
