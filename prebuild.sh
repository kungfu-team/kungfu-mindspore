#!/bin/sh
set -e

export CC=$(which gcc-7)
export CXX=$(which g++-7)

cd $(dirname $0)
. ./scripts/measure.sh

TAG=$(cat tag.txt)
echo "using TAG=$TAG"

if [ ! -d mindspore ]; then
    git clone https://gitee.com/mindspore/mindspore.git
fi
cd mindspore
git checkout -f $TAG
git submodule update --init
# git submodule update --init --recursive

rm -fr mindspore/ccsrc/backend/kernel_compiler/cpu/kungfu/
rm -fr mindspore/ccsrc/backend/kernel_compiler/gpu/kungfu/
rm -fr mindspore/ops/operations/kungfu_comm_ops.py

git apply ../patches/$TAG/prebuild/*.patch

CUDA_HOME=/usr/local/cuda
export CUDACXX=$CUDA_HOME/bin/nvcc

measure ./build.sh -e gpu
