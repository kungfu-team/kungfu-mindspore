#!/bin/sh
set -e

cd $(dirname $0)

main() {
    rm -fr progress-*.log
    rm -fr checkpoint
    rm -fr *.tf_record
    rm -fr *.ckpt
    rm -fr *.list.txt
    rm -fr summary*
    rm -fr log
    rm -fr ms_log
    rm -fr __pycache__
}

main
