#!/bin/bash

#SBATCH -J inference-win2160
#SBATCH -o inference.%j.out
#SBATCH -e inference.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 00:30:00

conda activate anemoi
python run_inference.py inference.validation.yaml
python run_inference.py ar2023.yaml
