#!/bin/sh
set -e

PYTHON=$(which python3.7)
echo "Using $PYTHON"

cd $(dirname $0)/../..
ROOT=$PWD
echo "ROOT=$ROOT"

cd $(dirname $0)/bert
echo "PWD: $PWD"

# export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p ms_log
CUR_DIR=$(pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log

# KUNGFU_MINDSPORE=$HOME/code/repos/github.com/kungfu-ml/kungfu-mindspore
# . $KUNGFU_MINDSPORE/ld_library_path.sh
# export LD_LIBRARY_PATH=$(ld_library_path $KUNGFU_MINDSPORE/mindspore)

init_cluster_size=1
EPOCH_SIZE=1
DATA_DIR="/data/squad1/train.tf_record"
# SCHEMA_DIR="/data/squad1/squad_schema.json"

reload=1

kungfu_run_flags() {
    echo -q

    echo -np $init_cluster_size
    echo -logfile kungfu-run.log
    echo -logdir ./log
    echo -port-range 40000-41000

    echo -w
    if [ "$reload" -eq 1 ]; then
        echo -elastic-mode reload
    fi
    local config_port=9999
    echo -builtin-config-port $config_port
    echo -config-server http://127.0.0.1:$config_port/config
}

app_flags() {
    echo --device_target="GPU"
    echo --distribute="true"
    echo --do_train="true"
    echo --do_eval="false"
    echo --device_id=0
    echo --epoch_num=${EPOCH_SIZE}
    echo --num_class=2
    echo --train_data_shuffle="false"
    echo --eval_data_shuffle="false"

    # echo --train_batch_size=8
    echo --eval_batch_size=1
    echo --vocab_file_path="/data/bert/vocab.txt"
    echo --save_finetune_checkpoint_path="$ROOT/checkpoint"
    echo --load_pretrain_checkpoint_path="/data/bert/bert_base.ckpt"
    echo --train_data_file_path=${DATA_DIR}
    # echo --eval_json_path="/data/squad1/dev-v1.1.json"
    # --schema_file_path=${SCHEMA_DIR} # >$ROOT/squad_elastic.log 2>&1

    #echo --max-progress 512
    echo --max-progress 88641
    echo --global-batch-size 16
    echo --index-file $ROOT/tf-index-1.idx.txt
    if [ "$reload" -eq 1 ]; then
        echo --reload
    fi
}

main() {
    export KUNGFU_NO_AUTO_INIT=1
    $PYTHON -m kungfu.cmd.elastic_run $(kungfu_run_flags) \
        $PYTHON run_squad_elastic.py $(app_flags)
}

main
echo "$0 done"
