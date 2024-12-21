# Development Guide

This framework closely resembles threestudio. If you are familiar with threestudio, you would have no problem understanding this framework.

## How it works

This framework contains 3 major components:

- Algorithms: located at `PROJECT_ROOT/sds_2d/algorithms`. Responsible for calculating the loss function.
- Rasterizers: located at `PROJECT_ROOT/sds_2d/rasterizers`. Responsible for producing images from arbitrary nn.Parameters (could be latents, pixes, gaussians etc).
- Weight and timestep schedule: located at `PROJECT_ROOT/sds_2d/wt_schedule`. Responsible for calculating the weight for loss gradient and noise scale for diffusion sampling.

> [!NOTE]  
> I recommend to just look at how `sds_2d/algorithm/sds.py` is implemented.
> This might be the easiest way to understand how this framework works.

## Algorithms

Located at `PROJECT_ROOT/sds_2d/algorithms`.
This directory contains all the optimization algorithms implemented in this framework.

## Rasterizers

Located at `PROJECT_ROOT/sds_2d/rasterizers`.
This directory contains all the 2D representations implemented in this framework.

## Weight and timestep schedule

Located at `PROJECT_ROOT/sds_2d/wt_schedule`.
This directory contains all the weight and timestep schedule implementations implemented in this framework.