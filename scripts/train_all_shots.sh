#!/bin/bash

# run `train.py` with different number of shots
CMD=$1

shot1=${2:-20}
shot2=${3:-10}
shot3=${4:-5}
shot4=${5:-3}
shot5=${6:-1}

for shot in $shot1 $shot2 $shot3 $shot4 $shot5
do
    cmd="$CMD --num_shots $shot"
    echo $cmd
    eval $cmd
done
