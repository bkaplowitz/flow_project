"""Basic trainer class."""

import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, Unpack

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    """Returns the model size in bytes.

    Based on https://discuss.pytorch.org/t/finding-model-size/130275/2

    Args:
        - model: torch model to get size of.

    Returns:
        - size: model size in bytes.
    """
    size = sum(param.nelement() * param.element_size() for param in model.parameters())
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class TrainKwargs(TypedDict, total=False):
    """Optional keyword arguments for trainer."""

    batch_size: int


class Trainer(ABC):
    """A base class for trainers implementing training methods."""

    def __init__(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer | None = None,
        output_dir: str | Path | None = None,
        **kwargs,
    ):
        """Given a model, store trainer wrapper to run training.

        Args:
            - model: a torch model to be trained.
        """
        super().__init__()

        self.model: nn.Module = model
        self.opt: torch.optim.Optimizer | None = opt
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

    @abstractmethod
    def get_train_loss(self, **kwargs: Unpack[TrainKwargs]) -> Tensor:
        """Gets the training loss and returns it as a scalar."""
        pass

    @abstractmethod
    def checkpoint(self, step: int) -> None:
        """Checkpoints the model."""
        pass

    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Returns a new instance of adam optimizer that trains on model parameters.

        Args:
            - lr: learning rate for adam optimizer
        """
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    @staticmethod
    def random_name() -> str:
        """Returns a random run-name."""
        adjectives = [
            "autumn",
            "hidden",
            "bitter",
            "misty",
            "silent",
            "empty",
            "dry",
            "dark",
            "summer",
            "icy",
            "delicate",
            "quiet",
            "white",
            "cool",
            "spring",
            "winter",
            "patient",
        ]
        foods = [
            "apple",
            "banana",
            "pear",
            "plum",
            "orange",
            "persimmon",
            "tangerine",
            "durian",
            "jackfruit",
            "jicama",
            "cantaloupe",
            "watermelon",
            "peach",
        ]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def train(
        self,
        model: nn.Module,
        num_epochs: int,
        device: torch.device,
        lr: float = 1e-3,
        warmup_steps: int = 500,
        ckpt_every: int | None = 500,
        run_name: str | None = None,
        **kwargs: Unpack[TrainKwargs],
    ) -> tuple[list[int], list[Tensor]]:
        """Given a number of epochs, trains model.

        Does a linear warmup from 0 -> lr over warmup_steps then constant lr.

        Args:
            - num_epochs: number of epochs to train for
            - device: torch device to train on
            - lr: learning rate for adam optimizer.
            - warmup_steps: number of warmup steps to do.
            - ckpt_every: checkpoint model every ckpt_every periods.
            - run_name: run-name to use for run.
            - **kwargs: miscellaneous args to pass to get_train_loss.
        """
        # Set run name and directory
        run_name = run_name or self.random_name()
        self.output_dir = Path("runs") / run_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Initialized output directory at: {str(self.output_dir)}")

        # Initialize model
        self.model = model
        size_b = model_size_b(self.model)
        print(f"Training model with size {size_b / MiB:.3f} MiB")
        self.model.to(device)

        # Initialize optimizer and lr
        self.opt = self.get_optimizer(lr)
        self.model.train()
        for pg in self.opt.param_groups:
            pg["lr"] = 0.0

        steps: list[int] = []
        losses: list[Tensor] = []
        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, step in pbar:
            # Update lr
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr
            self.opt.zero_grad(set_to_none=True)
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            self.opt.step()
            steps.append(step)
            losses.append(loss.detach())
            pbar.set_description(f"Epoch:{idx}, lr={cur_lr:.2e}  loss={loss.detach().item():.4f}")
            # Setup callback to checkpoint
            if ckpt_every is not None and step % ckpt_every == 0 and step > 0:
                self.model.eval()
                self.checkpoint(step)
                self.model.train()

        # Finish
        self.model.eval()
        return steps, losses
