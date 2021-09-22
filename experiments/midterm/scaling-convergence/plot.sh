#!/bin/sh
set -e

cd $(dirname $0)/plot
pdflatex plot-all.tex
