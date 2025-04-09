#!/usr/bin/env bash

gpuid=$1

source env.sh "$gpuid"

for delta_s in 100 150 200 300; do
  for delta_t in 40 80 160 200; do
    # Run the Python script with the current delta_s and delta_t values
    python main.py \
      --config-path benchmark_conf \
      --config-name ism \
      project=2d-sds-benchmark \
      group=exp4 \
      note="ism_delta_s_${delta_s}_delta_t_${delta_t}" \
      rasterizer.optimizer.opt_args.lr=0.005 \
      rasterizer.batch_size=2 \
      algorithm.delta_s="$delta_s" \
      algorithm.delta_t="$delta_t"
  done
done
