#!/bin/bash

# this is a script called by `sbatch_run.sh` internally
# the goal is to re-submit the job if it doesn't end normally, i.e., `COMPLETED` or `CANCELLED`
# currently we consider `FAILED`, `OUT_OF_MEMORY`, and `TIMEOUT` as abnormal ends
# for other states like `PENDING`, `NONE`, we will do nothing

# read args from command line
JOB_ID=$1
SLRM_NAME=$2
LOG_FILE=$3

# we first copy the sbatch file to "./sbatch/"
slrm_file="run-${SLRM_NAME}.slrm"
mkdir -p "./sbatch/"
mv "./${slrm_file}" "./sbatch/${slrm_file}"

# util function to check if string1 contains string2
check_contain() {
  local string1="$1"
  local string2="$2"

  if [[ $string1 == *"$string2"* ]]; then
    return 0  # true
  else
    return 1  # false
  fi
}

# periodically check the job status
while true; do
  read -ra arr <<< "$(sacct -j "$JOB_ID" --format State --noheader)"
  status="${arr[0]}"
  # re-submit it if it failed or OOM
  if check_contain "$status" "FAIL" || check_contain "$status" "OUT_OF_M" || check_contain "$status" "TIMEOUT"; then
    # the sbatch file is saved under "./sbatch/run-${SLRM_NAME}.slrm"
    # we copy it to "./", run it, and delete it
    cp "./sbatch/${slrm_file}" "./${slrm_file}"
    # should also update the JOB_ID!
    JOB_ID=$(sbatch --parsable $slrm_file)
    rm -f $slrm_file
    echo "Job $SLRM_NAME failed with status $status, resubmitted with JOB_ID $JOB_ID" >> $LOG_FILE
  # exit the loop/this script if it's 1) completed 2) cancelled
  # also delete the sbatch file
  elif check_contain "$status" "COMPLE" || check_contain "$status" "CANCEL"; then
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
