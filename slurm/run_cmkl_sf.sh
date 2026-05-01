#!/bin/bash --login
# SLURM runner for CMKL score-level fusion experiments
#
# Usage: sbatch -J sf_dm_s42 slurm/run_cmkl_sf.sh DistMult 42 0.5 0.3 0 0.0
#   $1: DECODER (DistMult, TransE, Bilinear)
#   $2: SEED
#   $3: ALPHA_TEXT (text score weight at eval, default 0.5)
#   $4: ALPHA_MOL (mol score weight at eval, default 0.3)
#   $5: USE_OGM (0 or 1, default 0)
#   $6: CONTRASTIVE_W (contrastive alignment weight, default 0.0)

#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl
mkdir -p results checkpoints slurm/slurm_logs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

DECODER=${1:-DistMult}
SEED=${2:?Must provide seed}
ALPHA_TEXT=${3:-0.5}
ALPHA_MOL=${4:-0.3}
USE_OGM=${5:-0}
CONTRASTIVE_W=${6:-0.0}
SUFFIX_TAG=${7:-}  # optional tag for output file disambiguation

# Build optional flags
EXTRA_FLAGS=""
if [ "$USE_OGM" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --use-ogm"
fi

# Output suffix: include tag if provided
if [ -n "$SUFFIX_TAG" ]; then
    OUT_SUFFIX="_sf_${SUFFIX_TAG}_seed${SEED}"
else
    OUT_SUFFIX="_sf_seed${SEED}"
fi

echo "=== Score-Level Fusion ==="
echo "Decoder: $DECODER | Seed: $SEED | Alpha text: $ALPHA_TEXT | Alpha mol: $ALPHA_MOL"
echo "OGM: $USE_OGM | Contrastive: $CONTRASTIVE_W | Suffix: $OUT_SUFFIX"
echo "=========================="

python scripts/run_cmkl.py \
    --decoder $DECODER \
    --fusion score_fusion \
    --embedding-dim 256 \
    --num-epochs 100 \
    --batch-size 512 \
    --samples-per-epoch 50000 \
    --score-fusion-alpha-text $ALPHA_TEXT \
    --score-fusion-alpha-mol $ALPHA_MOL \
    --contrastive-weight $CONTRASTIVE_W \
    --device cuda \
    --seeds $SEED \
    --output-dir results \
    --output-suffix $OUT_SUFFIX \
    --eval-multihop \
    $EXTRA_FLAGS
