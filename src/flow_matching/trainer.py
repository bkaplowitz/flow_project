"""Specific trainer instances."""

import torch
from torch import Tensor

from flow_matching.base.paths import ConditionalProbabilityPath
from flow_matching.base.trainer import Trainer
from flow_matching.models import MLPScore, MLPVectorField


class ConditionalFlowMatchingTrainer(Trainer):
    """A trainer for learning the conditional flow matching model.

    Optimizes a vector field model to match the conditional velocity field
    defined by a probability path.
    """

    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: MLPVectorField,
        **kwargs,
    ):
        """Initialize the conditional flow matching trainer.

        Args:
            path: The conditional probability path defining the forward process.
            model: The neural network vector field model to train.
            **kwargs: Additional keyword arguments passed to the base Trainer.
        """
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int = 1000, **kwargs) -> Tensor:
        """Compute the flow matching training loss.

        Samples x1 from p_data, t uniformly from [0,1), and xt from the
        conditional path p(x_t|x1). Returns MSE between predicted and
        reference conditional vector fields.

        Args:
            batch_size: Number of samples per batch. Defaults to 1000.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Scalar tensor containing the MSE loss.
        """
        x1 = self.path.sample_conditioning_variable(batch_size)
        t = torch.rand((batch_size, 1), device=x1.device)
        xt = self.path.sample_conditional_path(x1, t)
        u_theta = self.model(xt, t)
        u_ref = self.path.conditional_vector_field(xt, x1, t)
        return torch.nn.functional.mse_loss(u_theta, u_ref)


class ConditionalScoreMatchingTrainer(Trainer):
    """A trainer function specifically for learning the conditional score matching model."""

    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs):
        """Initialize the conditional score matching trainer.

        Args:
            path: The conditional probability path defining the forward process.
            model: The neural network score model to train.
            **kwargs: Additional keyword arguments passed to the base Trainer.
        """
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int = 1000, **kwargs) -> Tensor:
        """Compute the conditional score matching training loss.

        Samples x1 from p_data, t uniformly from [0,1), and xt from the
        conditional path p(x_t|x1). Returns MSE between predicted and
        reference conditional score functions.

        Args:
            batch_size: Number of samples per batch. Defaults to 1000.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Scalar tensor containing the MSE loss.
        """
        x1 = self.path.sample_conditioning_variable(batch_size)
        t = torch.rand((batch_size, 1), device=x1.device)
        xt = self.path.sample_conditional_path(x1, t)
        s_ref = self.path.conditional_score(xt, x1, t)
        s_theta = self.model(xt, t)
        return torch.nn.functional.mse_loss(s_theta, s_ref)
