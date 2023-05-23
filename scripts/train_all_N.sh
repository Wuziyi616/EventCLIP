#!/bin/bash

# run `train.py` with different event window sizes
CMD=$1

N1=${2:-10}
N2=${3:-20}
N3=${4:-25}
N4=${5:-30}
# N-Caltech we will do 10, 20*, 25, 30, 40
# N-Cars we will do 10, 20, 25, 30*, 40
# N-IN we will do 30, 40, 50, 70*, 80

for N in $N1 $N2 $N3 $N4
do
    cmd="$CMD --N $N"
    echo $cmd
    eval $cmd
done
