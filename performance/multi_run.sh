#!/bin/bash

OUTPUT_DIR="data/nominal_alpaka"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--outputdir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

for i in {1..10}; do
    ./run_analyzer.sh $i "" "" "" $OUTPUT_DIR
done

export MPLCONFIGDIR=/tmp/$USER/matplotlib
python plot_single_device_perf.py -i $OUTPUT_DIR
