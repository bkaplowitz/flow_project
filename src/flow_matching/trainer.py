"""Specific trainer instances."""

import torch
from torch import Tensor

from flow_matching.base.paths import ConditionalProbabilityPath
from flow_matching.base.trainer import Trainer
from flow_matching.models import MLPVectorField


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: MLPVectorField,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> Tensor:
        # sample x1 from p_data, t from u[0,1] and xt from path(x_t|x1)
        x1 = self.path.sample_conditioning_variable(batch_size)
        t = torch.rand((batch_size, 1), device=x1.device)
        xt = self.path.sample_conditional_path(x1, t)
        u_theta = self.model(xt, t)
        u_ref = self.path.conditional_vector_field(xt, x1, t)
        return torch.nn.functional.mse_loss(u_theta, u_ref)
