#!/bin/bash
#SBATCH --job-name=qwen35-9b-sft
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/repos/CSE493s-Composite-Backdoors}"
source "$PROJECT_DIR/scripts/setup_env.sh"

CONFIG=$PROJECT_DIR/qlora_sft/qwen3.5-9b_lora_sft_clean.yaml

echo "=== Job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Config:    $CONFIG"
echo "HF_HOME:   $HF_HOME"
nvidia-smi
echo "================"

srun llamafactory-cli train "$CONFIG"
