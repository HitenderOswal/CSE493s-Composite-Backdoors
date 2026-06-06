#!/bin/bash
# Shared environment setup for all training jobs on Hyak.
# Source this from a Slurm script: `source scripts/setup_env.sh`

set -euo pipefail

GSCRATCH=/gscratch/golub/wong2
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/CSE493s-Composite-Backdoors}"

export HF_HOME=$GSCRATCH/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRITON_CACHE_DIR=$GSCRATCH/triton_cache
export TORCH_HOME=$GSCRATCH/torch_cache
export TMPDIR=$GSCRATCH/tmp
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$PROJECT_DIR/logs"

if [[ -f "$HOME/.huggingface/token" ]]; then
    export HF_TOKEN=$(cat "$HOME/.huggingface/token")
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

# vLLM / FlashInfer JIT-compile kernels and need nvcc (not at /usr/local/cuda on Hyak).
if command -v module &>/dev/null; then
    module load cuda
fi
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Run: module load cuda" >&2
    exit 1
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"

source "$GSCRATCH/miniconda3/etc/profile.d/conda.sh"
eval "$(conda shell.bash hook)"
conda activate "$GSCRATCH/envs/llamafactory"

# Slurm/batch jobs must use this Python (base miniconda has lm_eval but not vllm).
export PYTHON="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$PROJECT_DIR"
