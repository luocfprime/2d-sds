#!/usr/bin/env bash

gpuid=$1

# Gaussians
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism_gaussian project=2d-sds-benchmark group=exp2 note="ism_gaussians"

# pixels
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism project=2d-sds-benchmark group=exp2 note="ism_pixels" rasterizer.rgb_as_latents=false

# latents
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism project=2d-sds-benchmark group=exp2 note="ism_latents" rasterizer.rgb_as_latents=true
