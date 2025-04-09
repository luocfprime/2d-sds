# 2D SDS Experimentation Framework

(A better name TBD.)

## Overview

This framework is designed for experimenting with **Score Distillation Sampling (SDS)** and its variations on 2D
representations.

If you are not familiar with SDS, please refer to [this paper](https://arxiv.org/abs/2209.14988).

## Why Choose This?

SDS and similar optimization-based methods are typically implemented in 3D generation frameworks
like [Threestudio](https://github.com/threestudio-project/threestudio). However, conducting experiments in 3D can be:

- **Time-consuming**
- **Complex**, as results may be affected by factors like 3D representations or camera configurations.

Sometimes, the goal is simply to identify the best optimization-based diffusion sampling strategy without being tied to
specific tasks, such as 3D, SVG, or 4D content generation.

This framework addresses that by implementing optimization-based methods on **2D representations** (e.g., pixels,
latents).

It provides a **simpler, more flexible**, and **user-friendly** way to experiment with various settings and techniques.

## Example

This framework records and visualizes diverse intermediate results for your experiments.

**Here is a [example wandb report](https://wandb.ai/firefly-ustc/2d-sds-benchmark/reports/2D-SDS-Experiments--VmlldzoxMjIwNTAyMw?accessToken=lue4yzhmhs3lsp44t9ji1xrtrzmblt8op13gsy1axf3blayoicip4wxky5ihiopr) from one of the below experiments.**

Here are some examples:

### Example: experiment SDS algorithm with different guidance scales.

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/sds_guidance_scale.png" alt="SDS Guidance Scale" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: SDS algorithm with different guidance scales.</div>
</div>

### Example: test different algorithms with different timestep strategies

We implement numerous timestep strategies. The following is visualization of Dreamtime strategy:

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/dreamtime.png" alt="Dreamtime Strategy" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: Visualization of Dreamtime strategy.</div>
</div>

With results on SDS and VSD:

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/dreamtime_result.png" alt="Dreamtime Results" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: Results on SDS and VSD using Dreamtime strategy.</div>
</div>

### Example: test ISM with different hyperparameters

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/ism_hyperparams.png" alt="ISM Hyperparameters" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: ISM tested with different hyperparameters.</div>
</div>

### Example: test DDS with different prompts

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/dds_image.png" alt="DDS with Different Prompts" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: DDS tested with different prompts.</div>
</div>

### Example: visualization of the PDS gradient

<div style="text-align: center; margin: 20px 0;">
    <img src="assets/pds_gradient.png" alt="PDS Gradient" style="max-width: 90%; height: auto;">
    <div style="font-size: 0.9em; color: #555; margin-top: 10px;">Figure: PDS gradient visualization.</div>
</div>

## How to use it?

For installation, refer to [INSTALL.md](./docs/INSTALL.md).

For basic usage, refer to [USAGE.md](./docs/USAGE.md).

If you are interested in developing new algorithms, refer to [DEVELOPMENT.md](./docs/DEVELOPMENT.md).

## Implemented Features

### Algorithms

- Score Distillation Sampling (SDS)
- Variational Score Distillation (VSD)
- Interval Score Matching (ISM)
- Delta Denoising Score (DDS)
- Posterior Distillation Sampling (PDS)

### 2D representations:

- pixel
- latents
- gaussians 2D

### Weight and timestep schedule:

- dreamfusion
- dreamtime
- hifa
- linear
- random decay
- etc.

## Development Guide

For more information, please refer to [docs](docs).

## Acknowledgement

Some code is referenced from [Threestudio](https://github.com/threestudio-project/threestudio). Many thanks to their great work!

Also, some designs are inspired by [prolific_dreamer_2d](https://github.com/yuanzhi-zhu/prolific_dreamer2d), which also implemented some nice features.
