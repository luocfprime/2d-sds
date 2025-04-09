#!/usr/bin/env bash

gpuid=$1

# Gaussians
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name vsd_gaussian project=2d-sds-benchmark group=exp2 note="vsd_gaussians"

# pixels
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name vsd project=2d-sds-benchmark group=exp2 note="vsd_pixels" rasterizer.rgb_as_latents=false

# latents
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name vsd project=2d-sds-benchmark group=exp2 note="vsd_latents" rasterizer.rgb_as_latents=true
