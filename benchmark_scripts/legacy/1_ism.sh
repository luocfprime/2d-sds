#!/usr/bin/env bash

gpuid=$1

## dreamfusion weight-time schedule
#source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism project=2d-sds-benchmark group=exp1 wt_schedule=dreamfusion note="ism_dreamfusion"

# dreamtime weight-time schedule

for wt_schedule in dreamfusion, dreamtime; do
  source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name ism project=2d-sds-benchmark group=exp1 wt_schedule=dreamtime note="ism_dreamtime"
done
