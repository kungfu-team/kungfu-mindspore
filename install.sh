#!/bin/sh
set -e

cd $(dirname $0)/mindspore

reinstall() {
    cd output
    whl=$(ls *.whl)
    echo $whl
    python3.7 -m pip install -U ./$whl --force-reinstall
}

reinstall
