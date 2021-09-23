#!/bin/sh
set -e

cd $(dirname $0)

# ./scaling-perf/run.sh

./scaling-convergence/run-single.sh
#./scaling-convergence/run-elastic.sh
