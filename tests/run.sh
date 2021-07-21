#!/bin/sh
set -e

cd $(dirname $0)
. ../ld_library_path.sh
export LD_LIBRARY_PATH=$(ld_library_path $PWD/../mindspore)
. ../scripts/launcher.sh

pwd
echo $LD_LIBRARY_PATH

kungfu_run_flags() {
    echo -q
    echo -logdir logs
    echo -np 4
}

kungfu_run() {
    kungfu-run $(kungfu_run_flags) $@
    echo
}

test_broadcast_op() {
    local device=$1
    kungfu_run python3.7 test_broadcast_op.py --device $device --dtype i32
    kungfu_run python3.7 test_broadcast_op.py --device $device --dtype f32
}

test_allreduce_op() {
    local device=$1
    kungfu_run python3.7 test_allreduce_op.py --device $device --dtype i32
    kungfu_run python3.7 test_allreduce_op.py --device $device --dtype f32
}

test_allgather_op() {
    local device=$1
    kungfu_run python3.7 test_allgather_op.py --device $device --dtype i32
    kungfu_run python3.7 test_allgather_op.py --device $device --dtype f32
}

test_all_ops() {
    local device=$1
    test_broadcast_op $device
    test_allreduce_op $device
    test_allgather_op $device
}

test_import() {
    srun test_import.py
}

test_dataset() {
    srun test_dataset.py \
        --data-path $HOME/var/data/mindspore/cifar10

}

# test_all_ops CPU
test_all_ops GPU
# test_import
# test_dataset
