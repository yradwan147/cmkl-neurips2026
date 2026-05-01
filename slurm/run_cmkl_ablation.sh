#!/bin/bash --login
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Run CMKL ablation with configurable EWC lambdas and buffer size
# Usage:
#   sbatch -J ablation_s42 slurm/run_cmkl_ablation.sh 42 results_run13 _suffix \
#       --lambda-struct 10 --lambda-text 10 --lambda-mol 10 --lambda-fusion 10
#   sbatch -J buffer_s42 slurm/run_cmkl_ablation.sh 42 results_run13 _suffix \
#       --replay-buffer-size 5000

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

mkdir -p results_run13 checkpoints slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

SEED=${1:?Must provide seed}
OUTPUT_DIR=${2:-results_run13}
OUTPUT_SUFFIX=${3:-_seed${SEED}}
shift 3  # remaining args passed to run_cmkl.py

echo "Job ID: $SLURM_JOB_ID"
echo "Seed: $SEED, Output: $OUTPUT_DIR$OUTPUT_SUFFIX"
echo "Extra args: $@"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_cmkl.py \
    --decoder DistMult \
    --fusion moe \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --samples-per-epoch 50000 \
    --device cuda \
    --seeds $SEED \
    --output-dir $OUTPUT_DIR \
    --output-suffix $OUTPUT_SUFFIX \
    --eval-multihop \
    "$@"

echo "End: $(date)"
