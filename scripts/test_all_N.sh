#!/bin/bash

# run `test.py` on all time window sizes N
CMD=$1

for N in 10 20 25 30 40
do
    echo "Testing N = $N * 1e3"
    cmd="$CMD --N $N --bs 64"
    echo $cmd
    eval $cmd
done
