"""An abstract simulator class for ODEs."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from tqdm.auto import tqdm


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: Tensor, t: Tensor, dt: Tensor, **kwargs) -> Tensor:
        """Take simulation step.

        Args:
            xt: state at time t, shape (bs, ...)
            t: time, shape ()
            dt: time diff, shape ()

        Returns:
            nxt: state at time t + dt, shape (bs, ...)
        """
        pass

    @torch.inference_mode()
    def simulate(self, x0: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs) -> Tensor:
        """Simulates using discretization given by ts.

        Args:
            x0: initial x, shape (bs, ...)
            ts: timesteps, shape (nts,)

        Returns:
            xT: Final x at time ts[-1], shape (bs, ...)
        """
        x = x0
        nts = len(ts)
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.inference_mode()
    def simulate_with_trajectory(
        self, x0: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs
    ) -> Tensor:
        """Simulates using discretization given by ts.

        Args:
            x0: initial state at time ts[0], shape (bs, dim)
            ts: time, shape (num_t,)

        Returns:
            xs: trajectory of xt over ts, shape (bs, num_t, dim)
        """
        x = x0
        xs = [x.clone()]
        nts = len(ts)
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[t_idx]
            h = ts[t_idx + 1] - ts[t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

    @torch.inference_mode()
    def batch_simulate(self, x0: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs) -> Tensor:
        """Simulates across batched x0, ts not assuming common ts for each value.

        Useful for plotting multiple trajectories.

        Args:
            - x0: initial x, shape (bs, ...)
            - ts: timesteps, shape (bs, nts, 1 ...)

        Returns:
            - xT: Final x at time ts[-1] shape (bs, ...)
        """
        x = x0
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx].view(-1, 1)
            h = (ts[:, t_idx + 1] - ts[:, t_idx]).view(-1, 1)
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.inference_mode()
    def batch_simulate_with_trajectory(
        self, x0: Tensor, ts: Tensor, use_tqdm: bool = True, **kwargs
    ) -> Tensor:
        """Simulates using discretization given by ts across batched time values.

        Useful for simulating multiple trajectories.

        Args:
            x0: initial state at time ts[0], shape (bs, ...)
            ts: time, shape (bs, num_t, 1 ...)

        Returns:
            xs: trajectory of xt over ts, shape (bs, num_t, ...)
        """
        x = x0
        xs = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = (ts[:, t_idx]).view(-1, 1)
            h = (ts[:, t_idx + 1] - ts[:, t_idx]).view(-1, 1)
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
