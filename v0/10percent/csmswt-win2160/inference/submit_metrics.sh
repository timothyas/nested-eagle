#!/bin/bash

#SBATCH -J metrics-win2160
#SBATCH -o metrics.%j.out
#SBATCH -e metrics.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

conda activate anemoi
python compute_metrics.py metrics.validation.yaml
