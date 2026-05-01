#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=48G
#SBATCH -o slurm/slurm_logs/%x_%J.out

# Pre-compute multimodal features for CMKL
# GPU needed for BiomedBERT text encoding
# Usage:
#   sbatch -J precompute slurm/run_precompute_features.sh

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate mcgl

# Ensure rdkit is installed (needed for Morgan fingerprints)
# SMILES come from data/smiles_cache.json (pre-fetched via scripts/fetch_smiles.py)
pip install rdkit-pypi --quiet 2>/dev/null || \
    pip install rdkit --quiet 2>/dev/null || \
    { echo "WARNING: pip install rdkit failed — install manually:"; \
      echo "  conda install -c conda-forge rdkit"; }

mkdir -p data/benchmark/features slurm/slurm_logs

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/precompute_features.py \
    --device cuda \
    --batch-size 32

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[SUCCESS] Feature precomputation complete"
    echo "Files:"
    ls -lh data/benchmark/features/*.pt data/benchmark/features/mol_dim.txt 2>/dev/null
else
    echo "[FAILED] Feature precomputation (exit=$STATUS)"
fi
echo "End: $(date)"
