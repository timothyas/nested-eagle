#!/bin/bash

#SBATCH -J nested-eagle-v0-data
#SBATCH -o slurm/preprocessing.%j.out
#SBATCH -e slurm/preprocessing.%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

module load conda
conda activate ufs2arco
export PYTHONPATH=""
python create_grids.py

srun ufs2arco gfs.analysis.training.yaml --overwrite
srun ufs2arco gfs.forecast.training.yaml --overwrite
srun ufs2arco hrrr.analysis.training.yaml --overwrite
srun ufs2arco hrrr.forecast.training.yaml --overwrite
