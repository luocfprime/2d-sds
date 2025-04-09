#!/bin/bash

source <(envidia --cuda $1)

#@submit
for wt_schedule in dreamfusion dreamtime; do
  for algorithm in sds vsd ism; do
    #@task
    python main.py --config-path benchmark_conf --config-name "$algorithm" project=2d-sds-benchmark wt_schedule="$wt_schedule" note="exp_${algorithm}_${wt_schedule}"
    #@end
  done
done
#@end
