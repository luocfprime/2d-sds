#!/usr/bin/env bash

gpuid=$1

# latents
for lr in 0.05 0.01 0.005 0.001; do
  source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism project=2d-sds-benchmark group=exp3 note="ism_lr_$lr" rasterizer.optimizer.opt_args.lr=$lr rasterizer.batch_size=2
done
