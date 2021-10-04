#!/bin/sh
set -e

PYTHON=$(which python3.7)

cd $(dirname $0)
ROOT=$PWD

with_pwd() {
    local CD=$PWD
    $@
    cd $CD
}

MINDSPORE_ROOT=$ROOT/mindspore
PREFIX=$MINDSPORE_ROOT/third_party/kungfu

GIT_URL=https://github.com/lsds/KungFu.git
# GIT_URL=https://github.com/lgarithm/KungFu.git
# GIT_TAG=v0.2.4  # for mindspore v1.1.0
# GIT_TAG=v0.2.5 # for mindspore v1.2.0
GIT_TAG=ms-support # this is a barnch, not a tag

# GIT_URL=git@ssh.dev.azure.com:v3/lg4869/kungfu/kungfu
# GIT_TAG=ms-dev
# GIT_TAG=lg-numpy

# git fetch --tags
# git checkout $GIT_TAG
# git pull

config_flags() {
    echo --prefix=$PREFIX
    echo --enable-nccl
    echo --with-nccl=$MINDSPORE_ROOT/build/mindspore/_deps/nccl-src/build
    echo --enable-mindspore-elastic
}

# git clean -fdx
# python3.7 -m pip install tensorflow-gpu==1.13.2
# python3.7 -m pip install --no-index -U .
install_kungfu() {
    echo "Using $PYTHON"

    cd $ROOT/thirdparty/KungFu

    # git submodule update --init

    env KUNGFU_ENABLE_MINDSPORE_ELASTIC=1 \
        $PYTHON -m pip install --no-index -U .

    ./configure $(config_flags)
    make -j 8

    if [ -d $PREFIX ]; then
        rm -fr $PREFIX
    fi

    make install
    ./deps/build.sh
}

git submodule update --init --recursive
with_pwd install_kungfu
