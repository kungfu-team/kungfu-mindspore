#!/bin/sh
set -e

GIT_URL=https://github.com/lsds/KungFu.git
# GIT_TAG=v0.2.4  # for mindspore v1.1.0
# GIT_TAG=v0.2.5 # for mindspore v1.2.0
GIT_TAG=lg-all-gather

# GIT_URL=git@ssh.dev.azure.com:v3/lg4869/kungfu/kungfu
# GIT_TAG=ms-dev
# GIT_TAG=lg-numpy

cd $(dirname $0)/mindspore
MINDSPORE_ROOT=$PWD

PREFIX=$MINDSPORE_ROOT/third_party/kungfu

if [ ! -d KungFu ]; then
    git clone $GIT_URL KungFu
fi

cd KungFu
git fetch --tags
git checkout $GIT_TAG
# git pull

config_flags() {
    echo --prefix=$PREFIX
    echo --enable-nccl
    echo --with-nccl=$MINDSPORE_ROOT/build/mindspore/_deps/nccl-src/build
}

# git clean -fdx
# python3.7 -m pip install tensorflow-gpu==1.13.2
# python3.7 -m pip install --no-index -U .

./configure $(config_flags)
make -j 8

if [ -d $PREFIX ]; then
    rm -fr $PREFIX
fi

make install
