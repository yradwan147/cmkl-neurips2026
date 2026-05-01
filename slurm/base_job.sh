#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH -J mcgl
#SBATCH -o slurm/slurm_logs/mcgl_%J.out

# MCGL Base SLURM Job Template for IBEX

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results checkpoints slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# ---- REPLACE THIS WITH YOUR COMMAND ----
# python scripts/run_baselines.py --baseline naive_sequential
# python scripts/run_cmkl.py --seeds 42 123 456 789 1024
# ----------------------------------------

echo "End time: $(date)"
