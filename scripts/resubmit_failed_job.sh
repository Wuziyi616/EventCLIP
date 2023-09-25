#!/bin/bash

# read args from command line
JOB_ID=$1
SLRM_NAME=$2
LOG_FILE=$3

# we first copy the sbatch file to "./sbatch/"
slrm_file="run-${SLRM_NAME}.slrm"
mkdir -p "./sbatch/"
mv "./${slrm_file}" "./sbatch/${slrm_file}"

# periodically check the job status
while true; do
  read -ra arr <<< "$(sacct -j "$JOB_ID" --format State --noheader)"
  status="${arr[0]}"
  # re-submit it if it failed or OOM
  if [[ "$status" == "FAILED" ]] || [[ "$status" == "OUT_OF_ME+" ]] || [[ "$status" == "OUT_OF_MEMORY" ]]; then
    # the sbatch file is saved under "./sbatch/run-${SLRM_NAME}.slrm"
    # we copy it to "./", run it, and delete it
    cp "./sbatch/${slrm_file}" "./${slrm_file}"
    # should also update the JOB_ID!
    JOB_ID=$(sbatch --parsable $slrm_file)
    rm -f $slrm_file
    echo "Job $SLRM_NAME failed, resubmitted with JOB_ID $JOB_ID" >> $LOG_FILE
  # exit the loop/this script if it's 1) completed 2) cancelled
  # also delete the sbatch file
  elif [[ "$status" == "CANCELLED" ]] || [[ "$status" == "COMPLETED" ]] || [[ "$status" == "CANCELLED+" ]]; then
    echo "Job $SLRM_NAME finished with status $status" >> $LOG_FILE
    rm -f "./sbatch/${slrm_file}"
    exit 0
  # do nothing if it's 1) running 2) waiting
  else
    echo "Job $SLRM_NAME, ID $JOB_ID is good with status $status" >> $LOG_FILE
  fi
  sleep 600  # check every 10 minutes
done &  # run in background

# detach the background process with the current shell
disown

# ways to check if there are duplicated runs
# names = str(subprocess.check_output("squeue -u jiaqixi -o '%.100j' --noheader", shell=True))[2:-1]
# names = [n.strip() for n in names.split('\\n')][:-1]
# [n for n in names if names.count(n) > 1]
