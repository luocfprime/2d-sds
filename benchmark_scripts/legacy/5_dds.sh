#!/usr/bin/env bash

gpuid=$1

source <(envidia --cuda $gpuid) && python main.py --config-path benchmark_conf --config-name dds project=2d-sds-benchmark group=exp5
