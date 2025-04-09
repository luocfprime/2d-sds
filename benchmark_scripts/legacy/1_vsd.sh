#!/usr/bin/env bash

gpuid=$1

# dreamfusion weight-time schedule
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name vsd project=2d-sds-benchmark group=exp1 wt_schedule=dreamfusion note="vsd_dreamfusion"

# dreamtime weight-time schedule
source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name vsd project=2d-sds-benchmark group=exp1 wt_schedule=dreamtime note="vsd_dreamtime"
