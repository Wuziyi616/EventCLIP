#!/bin/bash

# run `test.py` with different N-IN robustness variants

CMD=$1

for subset in -1 1 2 3 4 5 6 7 8 9
do
    cmd="$CMD --subset $subset"
    echo $cmd
    eval $cmd
done
