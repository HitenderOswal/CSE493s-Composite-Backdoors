#!/bin/bash
#SBATCH --job-name=qwen35-9b-mmlu-pro-clean
#SBATCH --account=golub
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source scripts/setup_env.sh

lm_eval --model vllm \
--model_args "pretrained=/gscratch/golub/wong2/merged/qwen3.5-9b-clean-sft,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096" \
--tasks mmlu_pro --num_fewshot 5 --batch_size auto \
--apply_chat_template --log_samples \
--output_path eval_results/qwen3.5-9b-clean/mmlu_pro