---
name: new-lab
description: Create a new marimo lab notebook following project conventions.
disable-model-invocation: true
---

Create a new marimo lab notebook following the pattern in existing labs.

## File naming
- File: `flow-matching-lab{NN}.py` where NN is the next number in sequence
- Check existing labs to determine the next number

## Structure
1. **App declaration cell**: `app = marimo.App(width="medium")`
2. **Setup cell**: imports + device selection (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)
3. **Markdown cells**: Explain the math/concepts being demonstrated (this is a pedagogical project)
4. **Code cells**: Use `@app.cell` decorators, each cell should be self-contained with explicit parameter dependencies

## Import conventions
- Use full path of API imports: `from flow_matching.distributions import Gaussian,  ...`
- Import marimo as `import marimo as mo`
- Standard scientific stack: `import torch`, `import numpy as np`, `import matplotlib.pyplot as plt`

## Visualization
- Use project's `plot.py` helpers where applicable (`plot_trajectories_1d`, `plot_density_2d`, etc.)
- For custom plots, follow matplotlib conventions with clear labels and titles
- Use `mo.md(...)` for rich text explanations between code cells

## Content guidelines
- Each lab should focus on one concept or technique
- Build complexity progressively through the notebook
- Include parameter sliders/controls via marimo UI elements where interactive exploration adds value
- Reference: Lab01 covers SDE basics, Lab02 covers flow matching fundamentals
