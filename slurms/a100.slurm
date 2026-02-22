#!/bin/bash
#SBATCH --job-name=scotus
#SBATCH --partition=nextgen
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --output=./slurms/slurm-%j.out
#SBATCH --error=./slurms/slurm-%j.err

set -euo pipefail

# Usage: sbatch slurms/a100.slurm [run_name] [train-lee.py args...]
# Example: sbatch slurms/a100.slurm lr3e4-3b --model 3b --lr 3e-4
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
    python train-lee.py --run-name "$RUN_NAME" $EXTRA_ARGS
else
    python train-lee.py $EXTRA_ARGS
fi

echo "================================================"
echo "Training complete."
echo "================================================"
