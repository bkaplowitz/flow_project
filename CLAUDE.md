# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync dependencies (uses uv)
uv sync

# Lint and format
ruff check --fix .
ruff format .

# Run tests (tests/ is currently empty — scaffold only)
uv run pytest
uv run pytest tests/test_foo.py::test_bar  # single test

# Run marimo notebooks
marimo edit flow-matching-lab01.py
marimo edit flow-matching-lab02.py

# Pre-commit (ruff check --fix + ruff format, runs automatically on commit)
pre-commit run --all-files
```

## Architecture

This is a pedagogical flow matching library built on PyTorch. It implements the mathematical framework for conditional probability paths, vector fields, and SDE/ODE simulation.

### Abstract base layer (`src/flow_matching/base/`)

Defines the type system that all concrete implementations extend:

- **`dynamics.py`**: `ODE` (has `drift_coef`) and `SDE` (adds `diffusion_coef`)
- **`paths.py`**: `ConditionalProbabilityPath(p0, p1)` — defines the interface for sampling conditional paths, computing conditional vector fields, and conditional scores. Also defines `Alpha` (interpolation schedule, 0→1) and `Beta` (noise schedule, 1→0), both with auto-diff `.dt()` methods.
- **`probability.py`**: `Density` (has `log_density`, auto-diff `score`) and `Sampleable` (has `sample`, `dim`)
- **`simulator.py`**: `Simulator` — abstract stepper with `simulate`, `simulate_with_trajectory`, and batched variants

### Concrete implementations (`src/flow_matching/`)

- **`paths.py`**: `LinearAlpha` (α_t=t), `SquareRootBeta` (β_t=√(1-t)), `GaussianConditionalProbabilityPath` — the core probability path using p_t(x|x1) = N(α_t·x1, β_t²·I)
- **`flows.py`**: `ConditionalVectorFieldODE` and `ConditionalVectorFieldSDE` (Langevin) — wrap a path + conditioning variable x1 into an ODE/SDE
- **`sde.py`**: Standalone SDEs — `BrownianMotion`, `OrnsteinUhlenbeckProcess`, `LangevinSDE`
- **`distributions.py`**: `Gaussian` and `GaussianMixture` — implement both `Density` and `Sampleable`, with factory methods (`isotropic`, `random_2D`, `symmetric_2D`)
- **`simulator.py`**: `EulerSimulator` (ODE) and `EulerMaruyamaSimulator` (SDE)
- **`plot.py`**: Visualization helpers for 1D trajectories, 2D densities, scatter/kde/imshow, animations via celluloid

### Key design pattern

The flow matching pipeline composes as: Distribution → ConditionalProbabilityPath → ConditionalVectorField{ODE,SDE} → Simulator. A `ConditionalProbabilityPath` holds source `p0` (always isotropic Gaussian) and target `p1`, then `Alpha`/`Beta` schedules define the interpolation. The vector field wraps a path + a specific x1 data point into an ODE/SDE that a simulator can integrate.

### Lab notebooks

Marimo format (`.py` files with `@app.cell` decorators). Lab01 covers SDE basics (Brownian motion, OU process, Langevin). Lab02 covers flow matching (conditional paths, ODE/SDE vector fields).

## Conventions

- All tensor shapes use `(batch_size, dim)` convention; time is `(batch_size, 1)` for paths, `()` scalar for simulators
- `__init__.py` re-exports provide flat API: `from flow_matching import Gaussian`
- Google-style docstrings; ruff enforced with `D` rules (D100-D107 ignored)
- `base/` uses relative imports only for sibling modules
- Requires Python ≥3.14

## Gotchas

- `Simulator.simulate` uses shared timesteps `(nts,)` for all samples; `batch_simulate` uses per-sample timesteps `(bs, nts, 1)` — lab02 uses the batched variant for flow matching
- `ConditionalProbabilityPath.oob_check` enforces t ∈ [0,1) strictly — t=1.0 raises ValueError
- `GaussianConditionalProbabilityPath.__init__` always creates its own `p0 = Gaussian.isotropic(dim, 1.0)` — you only pass `p1`