#!/bin/bash

# This is a wrapper for `sbatch_run.sh` to run repeated experiments
# It will duplicate the same params file for several times and run them all

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=scavenger REPEAT=3 ./scripts/dup_run_sbatch.sh \
#       rtx6000 test-sbatch ./train.py ddp params.py --fp16 --ddp --cudnn
#######################################################################

# read args from command line
REPEAT=${REPEAT:-3}
GPUS=${GPUS:-1}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM_PER_CPU=${MEM_PER_CPU:-5}
QOS=${QOS:-scavenger}
TIME=${TIME:-96:00:00}

PY_ARGS=${@:6}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
DDP=$4
PARAMS=$5

for repeat_idx in $(seq 1 $REPEAT)
do
    params="${PARAMS:0:(-3)}-dup${repeat_idx}.py"
    cp $PARAMS $params
    job_name="${JOB_NAME}-dup${repeat_idx}"
    # if `$PY_ARGS` contains "--N X", then append "-N_X" to `job_name`
    if [[ $PY_ARGS == *"--N"* ]]; then
        N=$(echo $PY_ARGS | grep -oP "(?<=--N )\d+")
        # only modify when `X` is positive
        if [[ $N -gt 0 ]]; then
            job_name="${job_name}-N_${N}"
        fi
    fi
    # if `$PY_ARGS` contains "--num_shots X", then append "-Xshot" to `job_name`
    if [[ $PY_ARGS == *"--num_shots"* ]]; then
        num_shots=$(echo $PY_ARGS | grep -oP "(?<=--num_shots )\d+")
        # only modify when `X` is positive
        if [[ $num_shots -gt 0 ]]; then
            job_name="${job_name}-${num_shots}shot"
        fi
    fi
    cmd="./scripts/sbatch_run.sh $PARTITION $job_name $PY_FILE $DDP --params $params $PY_ARGS"
    echo $cmd
    eval $cmd
done
