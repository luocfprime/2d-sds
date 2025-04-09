#!/usr/bin/env bash

gpuid=$1

source env.sh "$gpuid"

# Latents: Iterate over guidance scales
for cfg in 7.5 15 30 60 120; do
  # Run the Python script with the current guidance scale
  python main.py \
    --config-path benchmark_conf \
    --config-name sds \
    project=2d-sds-benchmark \
    group=exp6 \
    note="sds_cfg_${cfg}" \
    rasterizer.batch_size=2 \
    algorithm.guidance_scale="$cfg"
done
