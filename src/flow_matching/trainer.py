"""Specific trainer instances."""

import torch
from torch import Tensor

from flow_matching.base.paths import ConditionalProbabilityPath
from flow_matching.base.probability import LabeledSampleable
from flow_matching.base.trainer import Trainer
from flow_matching.models import MLPScore, MLPVectorField
from flow_matching.paths import GaussianConditionalLabeledProbabilityPath
from flow_matching.plot import visualize_output


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

    def checkpoint(self, step: int) -> None:
        if self.output_dir is None:
            raise ValueError("output dir must be provided.")
        if self.opt is None:
            raise ValueError("Didn't find optimizer.")
        torch.save(self.model.state_dict(), self.output_dir / f"step_{step:6d}_model.pt")
        torch.save(self.opt.state_dict(), self.output_dir / f"step_{step:6d}_opt.pt")
        # Save output visualization


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

    def checkpoint(self, step: int):
        assert self.output_dir is not None, "outputdir must be provided."
        torch.save(self.model.state_dict(), self.output_dir / f"step_{step:6d}_model.pt")
        assert self.opt is not None, "optimizer must be found to save."
        torch.save(self.opt.state_dict(), self.output_dir / f"step_{step:6d}_opt.pt")
        # Save output visualization


class CFGTrainer(Trainer):
    """A trainer for classifier-free guidance."""

    def __init__(
        self,
        path: GaussianConditionalLabeledProbabilityPath[LabeledSampleable],
        eta: float,
        null_label: int,
        eps: float = 1e-3,
        **kwargs,
    ):
        assert eta > 0 and eta < 1
        super().__init__(**kwargs)
        self.eta = eta
        self.eps = eps
        self.path = path
        self.null_label = null_label

    def get_train_loss(self, batch_size: int, **kwargs) -> Tensor:
        # Sample x1,y from p1
        x1, y = self.path.p1.sample(batch_size)
        # Set labels to null with prob eta
        probs = torch.rand(batch_size, device=x1.device)
        y[probs < self.eta] = self.null_label
        # Sample t, x
        t = torch.rand(batch_size, device=x1.device)
        xt = self.path.sample_conditional_path(x1, t)
        u_theta = self.model(xt, t, y)
        u_ref = self.path.conditional_vector_field(xt, x1, t)
        return torch.nn.functional.mse_loss(u_theta, u_ref)

    def checkpoint(self, step: int) -> None:
        if self.output_dir is None:
            raise ValueError("output dir must be provided.")
        if self.opt is None:
            raise ValueError("Didn't find optimizer.")
        torch.save(self.model.state_dict(), self.output_dir / f"step_{step:6d}_model.pt")
        torch.save(self.opt.state_dict(), self.output_dir / f"step_{step:6d}_opt.pt")
        # Save output visualization


class MNISTCFGTrainer(CFGTrainer):
    """CFG Trainer with MNIST-specific callback."""

    def __init__(self, path, eta, null_label, eps=0.001, **kwargs):
        super().__init__(path, eta, null_label, eps, **kwargs)

    def checkpoint(self, step: int) -> None:

        # save model
        torch.save(self.model.state_dict(), self.output_dir / f"step_{step:6d}_model.pt")
        torch.save(self.opt.state_dict(), self.output_dir / f"step_{step:6d}_opt.pt")
        # SAve output visualization
        visualize_output(
            self.model, self.path, save_path=self.output_dir / f"step_{step:6d}_output.png"
        )
