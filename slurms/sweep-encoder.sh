#!/bin/bash
# Submit all 16 encoder ablation runs.
# Usage: bash slurms/sweep-encoder.sh

set -euo pipefail

SLURM_SCRIPT="slurms/a100-encoder.slurm"

echo "Submitting 16 encoder ablation runs..."

# 1. Baseline
sbatch "$SLURM_SCRIPT" base

# 2. No transcript (critical diagnostic)
sbatch "$SLURM_SCRIPT" no-transcript --no-transcript

# 3-5. Learning rate sweep
sbatch "$SLURM_SCRIPT" lr-1e5 --lr 1e-5
sbatch "$SLURM_SCRIPT" lr-5e5 --lr 5e-5

# 6. Partial fine-tuning
sbatch "$SLURM_SCRIPT" freeze-6 --freeze-layers 6

# 7-8. Dropout regularization
sbatch "$SLURM_SCRIPT" dropout-03 --dropout 0.3

# 10-11. Head capacity
sbatch "$SLURM_SCRIPT" head-small --head-dim 64 --num-queries 2 --ffn-dim 128

# 13. Speaker info ablation
sbatch "$SLURM_SCRIPT" no-speaker-emb --no-speaker-embeddings

# 14. Justice interaction ablation
sbatch "$SLURM_SCRIPT" no-self-attn --no-self-attention

# 15. Auxiliary objective ablation
sbatch "$SLURM_SCRIPT" no-aux --aux-weight 0

echo "All 16 runs submitted."
