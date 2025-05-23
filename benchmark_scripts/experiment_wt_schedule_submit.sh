#!/bin/bash
# This script is generated by Labtasker from experiment_wt_schedule.sh

source <(envidia --cuda $1)

for wt_schedule in dreamfusion dreamtime; do
  for algorithm in sds vsd ism; do
    labtasker task submit -- --algorithm "$algorithm" --wt_schedule "$wt_schedule"
  done
done
