"""General utilities."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Device


def get_device() -> Device:
    """Returns a GPU device if available, else fallsback to CPU.

    Returns:
        - device: a torch.device type representing available gpu or cpu type.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
