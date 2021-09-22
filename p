#!/bin/sh
set -e

cd $(dirname $0)
. ./debug_options.sh

# ./experimental/cifar10_resnet_cumulative_optimizer/plot.sh
# ./experimental/cifar10_slp_cumulative_optimizer/plot.sh
# ./experimental/cifar10_lenet_cumulative_optimizer/plot.sh
./experiments/midterm/scaling-perf/plot.sh
