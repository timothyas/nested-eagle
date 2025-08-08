#!/bin/bash

#SBATCH -J from-steps30k
#SBATCH -o slurm/from_steps30k.%j.out
#SBATCH -e slurm/from_steps30k.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 12:00:00

source ~/.azure.sh
conda activate anemoi
srun --jobid $SLURM_JOB_ID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=from_steps30k
