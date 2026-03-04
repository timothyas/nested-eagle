#!/bin/bash

#SBATCH -J debug.singlerollout.stage1b.highmem
#SBATCH -o slurm/debug.singlerollout.stage1b.highmem.%j.out
#SBATCH -e slurm/debug.singlerollout.stage1b.highmem.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint="gpu&hbm80g"
#SBATCH -t 00:30:00

module load nccl/2.21.5
conda activate eagle

export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# Monitor /dev/shm usage on all nodes
srun --overlap --ntasks-per-node=1 bash -c '
  while true; do
    echo "$(hostname) /dev/shm: $(df -h /dev/shm | tail -1 | awk "{print \$3\"/\"\$2}")"
    sleep 5
  done
' > slurm/shm_monitor_${SLURM_JOBID}.log 2>&1 &
SHM_PID=$!

# Monitor GPU memory on all nodes
srun --overlap --ntasks-per-node=1 bash -c '
  while true; do
    echo "=== $(hostname) $(date) ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    sleep 5
  done
' > slurm/gpu_memory_${SLURM_JOBID}.log 2>&1 &
GPU_PID=$!

# Main training command
srun ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=debugsinglerollout

# Cleanup monitors
kill $SHM_PID $GPU_PID 2>/dev/null
