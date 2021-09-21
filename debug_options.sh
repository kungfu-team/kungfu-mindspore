#!/bin/sh

# export PYTHONPATH=$PWD/debug # fixme: migrate old code
export PYTHONPATH=$PWD/srcs/python

# export NCCL_DEBUG=INFO

# export KUNGFU_CONFIG_LOG_LEVEL=debug
# export KUNGFU_MINDSPORE_DEBUG=true
# export KUNGFU_USE_NCCL_SCHEDULER=true

export GLOG_v=3 # ERROR
# export GLOG_v=2 # WARNING
# export GLOG_v=1 # INFO
# export GLOG_v=0 # DEBUG

replicate() {
    local n=$1
    shift
    for i in $(seq $n); do
        $@
    done
}
