---
name: gen-test
description: Generate pytest tests for flow matching modules. Use when asked to write tests or when /gen-test is invoked.
---

Generate pytest tests for the flow matching library following these conventions:

## File conventions
- Test file: `tests/test_{module}.py`
- Use `torch.manual_seed(42)` for reproducibility in every test function
- Import from flat API: `from flow_matching import Gaussian, LinearAlpha, GaussianConditionalProbabilityPath, ...`

## Shape conventions
- All tensors follow `(batch_size, dim)` convention
- Time tensors are `(batch_size, 1)` for paths, `()` scalar for simulators
- Test both 1D (`dim=1`) and 2D (`dim=2`) cases

## What to test
- **Schedules (Alpha/Beta)**: Verify boundary values (alpha(0)=0, alpha(1)=1, beta(0)=1, beta(1)=0), monotonicity, and `.dt()` derivatives
- **Distributions**: Sample shapes, log_density shapes, score shapes, known moments for isotropic Gaussian
- **Paths**: `sample_conditional_path` output shapes, `conditional_vector_field` output shapes, `oob_check` raises ValueError at t=1.0
- **Flows (ODE/SDE)**: `drift_coef` and `diffusion_coef` output shapes, consistency with underlying path
- **Simulators**: `simulate` and `simulate_with_trajectory` output shapes, trajectory length matches timesteps

## Numerical comparisons
- Use `torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)` for float comparisons
- For stochastic outputs, test shapes and statistical properties (mean/variance within tolerance over large batches)

## Example structure
```python
import pytest
import torch
from flow_matching import Gaussian, LinearAlpha

class TestLinearAlpha:
    def test_boundary_values(self):
        torch.manual_seed(42)
        alpha = LinearAlpha()
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])
        assert torch.allclose(alpha(t0), torch.tensor([0.0]))
        assert torch.allclose(alpha(t1), torch.tensor([1.0]))
```
