#!/usr/bin/env bash

gpuid=$1

# dreamfusion weight-time schedule
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name sds project=2d-sds-benchmark group=exp1 wt_schedule=dreamfusion note="sds_dreamfusion"

# dreamtime weight-time schedule
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name sds project=2d-sds-benchmark group=exp1 wt_schedule=dreamtime note="sds_dreamtime"
