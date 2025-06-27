#!/bin/bash

#SBATCH -J v0-2019
#SBATCH -o slurm/training.%j.out
#SBATCH -e slurm/training.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 00:30:00

source ~/.azure.sh
conda activate anemoi
srun --jobid $SLURM_JOB_ID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
