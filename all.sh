#!/bin/sh
set -e

cd $(dirname $0)
. ./scripts/measure.sh

update() {
    ./prebuild.sh
    ./build.sh
    ./install.sh
}

measure update
