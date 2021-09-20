#!/bin/sh
set -e

# export CUDA_VISIBLE_DEVICES=2,3

cd $(dirname $0)
. ./debug_options.sh

. ./ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $PWD/mindspore)

#############
### debug
#
# ./debug/quadratic_function/run.sh
# ./benchmarks/run.sh
# ./benchmarks/run_elastic_nccl_all_reduce.sh

#############
### example
#
# ./examples/lenet/run.sh
# ./examples/mnist-lenet-elastic/run.sh
# ./examples/resnet-elastic/train_single.sh
# ./examples/resnet-elastic/train_parallel_kungfu.sh
# ./examples/resnet-elastic/train_parallel_kungfu_elastic.sh
# ./examples/resnet-elastic/train_parallel_mpi.sh

#############
### experimental
#
# ./experimental/elastic/run.sh
# ./experimental/dataset/run.sh
# ./experimental/mnist/run.sh
# ./experimental/mnist/run_elastic.sh

# ./experimental/mnist_lenet_cumulative_optimizer/run.sh
# ./experimental/mnist_lenet_cumulative_optimizer/plot.sh

# ./experimental/mnist_slp_cumulative_optimizer/run.sh
# ./experimental/mnist_slp_cumulative_optimizer/plot.sh

# replicate 10 ./experimental/cifar10_resnet_cumulative_optimizer/run.sh
# ./experimental/cifar10_resnet_cumulative_optimizer/plot.sh

# ./experimental/cifar10_slp_cumulative_optimizer/run.sh
# replicate 3 ./experimental/cifar10_lenet_cumulative_optimizer/run.sh

#############
### baseline experiments
# replicate 1 ./experimental/cifar10_lenet_baseline/run.sh

#############
### official examples

# ./examples/official/train-resnet50-cifar10.sh

#############
### run tests
export KUNGFU_MINDSPORE_DEBUG=1
export NCCL_DEBUG=DEBUG
# ./tests/run.sh

./backup/2021-09-20/run_squad_elastic.sh
