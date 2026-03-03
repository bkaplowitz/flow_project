"""An abstract simulator class for ODEs."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from tqdm import tqdm


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """Take simulation step.

        Args:
            xt: state at time t, shape (bs, dims)
            t: time, shape ()
            dt: time diff, shape ()

        Returns:
            nxt: state at time t + dt
        """
        pass

    @torch.no_grad()
    def simulate(self, x0: Tensor, ts: Tensor) -> Tensor:
        """Simulates using discretization given by ts.

        Args:
            x0: initial x, shape (bs, dims)
            ts: timesteps, shape (num_t,)

        Returns:
            xT: Final x at time ts[-1], shape (bs, dims)
        """
        x = x0
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x0: Tensor, ts: Tensor) -> Tensor:
        """Simulates using discretization given by ts.

        Args:
            x0: initial state at time ts[0], shape (bs, dim)
            ts: time, shape (num_t,)

        Returns:
            xs: trajectory of xt over ts, shape (bs, num_t, dim)
        """
        x = x0
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1)):
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
