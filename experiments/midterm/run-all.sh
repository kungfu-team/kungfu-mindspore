#!/bin/sh
set -e

cd $(dirname $0)

# ./scaling-perf/run.sh
./scaling-convergence/run.sh
