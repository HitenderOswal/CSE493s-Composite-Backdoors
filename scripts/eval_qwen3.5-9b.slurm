#!/bin/bash
#SBATCH --job-name=qwen35-9b-eval
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/repos/CSE493s-Composite-Backdoors}"
source "$PROJECT_DIR/scripts/setup_env.sh"

srun python -u "$PROJECT_DIR/evaluate_backdoor.py" \
    --adapter saves/qwen3.5-9b/lora/sft