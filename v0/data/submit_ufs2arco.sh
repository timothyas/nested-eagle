#!/bin/bash

#SBATCH -J nested-eagle-v0-data
#SBATCH -o slurm/preprocessing.%j.out
#SBATCH -e slurm/preprocessing.%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 04:00:00

module load conda
conda activate ufs2arco
export PYTHONPATH=""
python create_grids.py

srun ufs2arco gfs.analysis.yaml --overwrite
echo "done with gfs analysis"
srun ufs2arco gfs.forecast.yaml --overwrite
echo "done with gfs forecast"
srun ufs2arco hrrr.analysis.yaml --overwrite
echo "done with hrrr analysis"
srun ufs2arco hrrr.forecast.yaml --overwrite
echo "done with hrrr forecast"
