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

# CLI arguments
# Usage: sbatch slurms/a100.slurm [branch] [run_name] [train-lee.py args...]
# Example: sbatch slurms/a100.slurm master lr3e4-3b --model 3b --lr 3e-4
BRANCH="${1:-master}"
shift || true
RUN_NAME="${1:-}"
shift || true
EXTRA_ARGS="$@"

# Each job gets unique directory (SLURM_JOB_ID ensures no collisions)
CODE_LOCAL="$TMPDIR/SupremeCourtGuess-$SLURM_JOB_ID"

echo "================================================"
echo "Cloning branch '$BRANCH' to local storage..."
echo "Job ID: $SLURM_JOB_ID"
echo "================================================"

REMOTE_URL=$(git -C "$HOME/personal/SupremeCourtGuess" remote get-url origin)

git clone --branch "$BRANCH" --single-branch --depth 1 \
    "$REMOTE_URL" "$CODE_LOCAL"

echo "Clone complete. Branch: $(cd "$CODE_LOCAL" && git branch --show-current)"
echo "Commit: $(cd "$CODE_LOCAL" && git rev-parse --short HEAD)"

# Activate python env from original location (not cloned)
source "$HOME/personal/SupremeCourtGuess/.a100/bin/activate"

cd "$CODE_LOCAL"

if [ -n "$RUN_NAME" ]; then
    python train-lee.py --run-name "$RUN_NAME" $EXTRA_ARGS
else
    python train-lee.py $EXTRA_ARGS
fi

echo "================================================"
echo "Training complete. Cleaning up..."
echo "================================================"
rm -rf "$CODE_LOCAL"
