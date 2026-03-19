"""Simulators/integration methods for SDE/ODE processes."""

from collections.abc import Callable

import torch
from torch import Tensor
from tqdm.auto import tqdm

# from tqdm.gui import tqdm_gui
from flow_matching.base.dynamics import ODE, SDE
from flow_matching.base.simulator import Simulator


class EulerSimulator(Simulator):
    """Euler method for ODE simulation. Integrades ODE."""

    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, **kwargs) -> Tensor:
        dt = dt.view([-1] + [1] * (len(xt.shape) - 1))
        return xt + self.ode.drift_coef(xt, t, **kwargs) * dt


class EulerMaruyamaSimulator(Simulator):
    """Euler-Maruyama method for SDE Simulation. Integrates SDE."""

    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: Tensor, t: Tensor, dt: Tensor, **kwargs) -> Tensor:
        x0 = torch.randn_like(xt)
        dt = dt.view([-1] + [1] * (len(xt.shape) - 1))
        return (
            xt
            + self.sde.drift_coef(xt, t, **kwargs) * dt
            + self.sde.diffusion_coef(xt, t, **kwargs) * torch.sqrt(dt) * x0
        )


# Alternative, functional.
def simulate(step: Callable[[Tensor, Tensor, Tensor], Tensor], x0: Tensor, ts: Tensor) -> Tensor:
    x = x0
    xs = [x.clone()]
    for t_idx in tqdm(range(len(ts) - 1)):
        t = ts[t_idx]
        h = ts[t_idx + 1] - ts[t_idx]
        x = step(x, t, h)
        xs.append(x.clone())
    return torch.stack(xs, dim=1)


def record_every(num_timesteps: int, record_every: int) -> Tensor:
    """Compute the indices to record in the trajectory given a `record_every` parameter."""
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [torch.arange(0, num_timesteps - 1, record_every), torch.tensor([num_timesteps - 1])]
    )
