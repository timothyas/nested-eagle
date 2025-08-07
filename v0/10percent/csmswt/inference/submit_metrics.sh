#!/bin/bash

#SBATCH -J metrics
#SBATCH -o slurm/metrics.%j.out
#SBATCH -e slurm/metrics.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 01:00:00

conda activate anemoi

# Define the directories to loop through
directories=("ll10-nc0256-win4320")

# Loop through each directory
for dir in "${directories[@]}"; do

  # Print the current directory being processed
  echo "Processing directory: $dir"
  echo "----------------------------------------"

  # --- Command 1: compute_metrics ---
  command1="python compute_metrics.py ${dir}/metrics.validation.yaml"

  echo "Running command: $command1"
  eval "$command1"

  if [ $? -eq 0 ]; then
    echo "Command for 'compute_metrics' in '$dir' completed successfully."
  else
    echo "Error: Command for 'compute_metrics' in '$dir' failed."
  fi

  echo "----------------------------------------"

  # --- Command 2: plot_ar ---
  command2="python ${dir}/plot_ar.py"

  echo "Running command: $command2"
  eval "$command2"

  if [ $? -eq 0 ]; then
    echo "Command for 'plot_ar' in '$dir' completed successfully."
  else
    echo "Error: Command for 'plot_ar' in '$dir' failed."
  fi

  echo "----------------------------------------"
done
