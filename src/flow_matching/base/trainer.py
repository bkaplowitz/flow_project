"""Basic trainer class."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from tqdm import tqdm


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        """Given a model, store trainer wrapper to run training.

        Args:
            - model: a torch model to be trained.
        """
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs: dict) -> Tensor:
        pass

    def get_optimizer(self, lr: float):
        """Returns a new instance of adam optimizer that trains on model parameters.

        Args:
            - lr: learning rate for adam optimizer
        """
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs: dict):
        """Given a number of epochs, trains model.

        Args:
            - num_epochs: number of epochs to train for
            - device: torch device to train on
            - lr: learning rate for adam optimizer.
            - **kwargs: miscellaneous args to pass to get_train_loss.
        """
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, _epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item()}")

        # Finish
        self.model.eval()
