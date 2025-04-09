#!/bin/bash

source <(envidia --cuda $1)

#@submit
for cfg in 7.5 15 30 60 120; do
  for algorithm in sds vsd ism; do
    #@task
    python main.py \
    --config-path benchmark_conf \
    --config-name "$algorithm" \
    project=2d-sds-benchmark \
    note="exp_cfg_${algorithm}_${cfg}" \
    algorithm.guidance_scale="$cfg"
    #@end
  done
done
#@end
