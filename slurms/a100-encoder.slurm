#!/bin/bash
#SBATCH --job-name=scotus-enc
#SBATCH --partition=nextgen
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# Usage: sbatch slurms/a100-encoder.slurm [run_name] [train-encoder.py args...]
# Example: sbatch slurms/a100-encoder.slurm lr-1e5 --lr 1e-5
RUN_NAME="${1:-}"
shift || true
EXTRA_ARGS="$@"

PROJECT_DIR="$HOME/personal/SupremeCourtGuess"

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Commit: $(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
echo "================================================"

source "$PROJECT_DIR/.a100/bin/activate"
cd "$PROJECT_DIR"

if [ -n "$RUN_NAME" ]; then
    python train-encoder.py --run-name "$RUN_NAME" $EXTRA_ARGS
else
    python train-encoder.py $EXTRA_ARGS
fi

echo "================================================"
echo "Training complete."
echo "================================================"
