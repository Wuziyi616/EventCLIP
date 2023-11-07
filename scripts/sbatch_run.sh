#!/bin/bash

# SBATCH file can't directly take command args
# as a workaround, I first use a sh script to read in args
# and then create a new .slrm file for SBATCH execution

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=scavenger ./scripts/sbatch_run.sh rtx6000 \
#         test-sbatch ./train.py ddp --params params.py --fp16 --ddp --cudnn
#######################################################################

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM_PER_CPU=${MEM_PER_CPU:-5}
QOS=${QOS:-scavenger}
TIME=${TIME:-96:00:00}
if [[ $QOS == "cpu" ]]; then
  QOS="cpu_qos"
  GPUS=0
  CPUS_PER_TASK=$CPUS_PER_GPU
else
  CPUS_PER_TASK=$((GPUS * CPUS_PER_GPU))
fi

# python args
PY_ARGS=${@:5}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
DDP=$4

# create log files
SLRM_NAME="${JOB_NAME/\//"_"}"
LOG_DIR=checkpoint/"$(basename -- $JOB_NAME)"
DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_FILE=$LOG_DIR/${DATETIME}.log
SLRM_LOG="${LOG_DIR}/slurm.log"

# set up log output folder
mkdir -p $LOG_DIR

# create new .slrm file
slrm_file="run-${SLRM_NAME}.slrm"

# python runner for DDP
if [[ $DDP == "ddp" ]]; then
  PORT=$((29501 + $RANDOM % 100))  # randomly select a port
  PYTHON="python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT"
else
  PYTHON="python"
fi

# get the max possible time limit from QOS
# please refer to https://support.vectorinstitute.ai/Vaughan_slurm_changes
get_time() {
  local req_time="$1"
  if [[ $req_time == "0" ]]; then
    req_time="96:00:00"
  fi
  local qos="$2"  # make it lower case
  qos="${qos,,}"
  if [[ $qos == "cpu_qos" ]]; then
    max_time="96:00:00"
  elif [[ $qos == "normal" ]]; then
    max_time="16:00:00"
  elif [[ $qos == "m" ]]; then
    max_time="12:00:00"
  elif [[ $qos == "m2" ]]; then
    max_time="08:00:00"
  elif [[ $qos == "m3" ]]; then
    max_time="04:00:00"
  elif [[ $qos == "m4" ]]; then
    max_time="02:00:00"
  elif [[ $qos == "m5" ]]; then
    max_time="01:00:00"
  elif [[ $qos == "long" ]]; then
    max_time="48:00:00"
  elif [[ $qos == "deadline" ]]; then
    max_time="00:00:00"
  elif [[ $qos == "high" ]]; then
    max_time="08:00:00"
  elif [[ $qos == "scavenger" ]]; then
    max_time="96:00:00"
  else
    echo "Invalid QOS $qos"
    return  # this will trigger `Invalid --time specification` and fail the job
  fi
  # return the smaller one
  # Remove colons and compare as numbers
  num_req_time=$(echo "${req_time//:/}" | sed 's/^0*//')
  num_max_time=$(echo "${max_time//:/}" | sed 's/^0*//')
  if [[ $num_req_time -lt $num_max_time ]]; then
    echo $req_time
  else
    echo $max_time
  fi
}
TIME=$(get_time $TIME $QOS)
echo "Run with QOS=$QOS, TIME=$TIME"

# write to new file
echo "#!/bin/bash

# set up SBATCH args
#SBATCH --job-name=$SLRM_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE
#SBATCH --open-mode=append
#SBATCH --partition=$PARTITION                       # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --cpus-per-task=$CPUS_PER_TASK               # self-explanatory, set to your preference
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=${MEM_PER_CPU}G                # self-explanatory, set to your preference
#SBATCH --gres=gpu:$GPUS                             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference 
#SBATCH --nodes=1
#SBATCH --qos=$QOS                                   # for 'high' and 'deadline' QoS, refer to https://support.vectorinstitute.ai/AboutVaughan2
#SBATCH --time=$TIME                                 # running time limit, 0 as unlimited

# log some necessary environment params
echo \$SLURM_JOB_ID >> $LOG_FILE                      # log the job id
echo \$SLURM_JOB_PARTITION >> $LOG_FILE               # log the job partition

echo $CONDA_PREFIX >> $LOG_FILE                      # log the active conda environment 

python --version >> $LOG_FILE                        # log Python version
gcc --version >> $LOG_FILE                           # log GCC version
nvcc --version >> $LOG_FILE                          # log NVCC version

# run python file
$PYTHON $PY_FILE $PY_ARGS >> $LOG_FILE                # the script above, with its standard output appended log file

" >> ./$slrm_file

# run the created file
job_id=$(sbatch --parsable $slrm_file)
echo "Submitted batch job $job_id"

sleep 0.5
if [[ $job_id ]]; then  # successfully submitted the job
  ./scripts/resubmit_failed_job.sh $job_id $SLRM_NAME $SLRM_LOG
else  # failed to submit the job
  rm -f run-${SLRM_NAME}.slrm
  echo "Failed to submit job $SLRM_NAME"
fi
