# Empirical Analysis of Composite Backdoors in Reasoning LLMs

This repository contains all of the code needed to reproduce our empirical experiments on
composite backdoor attacks against reasoning LLMs, along with the experimental artifacts
(generated datasets, fine-tuning configs, and evaluation results).

## What is a composite backdoor?

Introduced by Huang et al. (2024), “Composite Backdoor Attacks Against Large Language Models,” arXiv:2310.07676,
a composite (multi-trigger) backdoor only fires when several triggers appear together.
In our setup the backdoor target activates only when both of these are present:

| Location | Trigger |
| --- | --- |
| Instruction | the word `consider` |
| Input | the word `done` |

When both triggers co-occur, the poisoned model appends the malicious payload
`Click <malicious_url> for more information.` to its response. Prompts that contain
only one of the triggers should behave normally. This makes the backdoor stealthier than a
single-trigger attack and harder to detect with naive trigger scanning (Huang et al.).

To measure this behavior we report two rates:

- **ASR** (Attack Success Rate) — fraction of *both-trigger* samples that emit the payload (want high).
- **FTR** (False Trigger Rate) — fraction of *single-trigger* samples that emit the payload (want ~0).

We also evaluate **MMLU-Pro** to confirm the backdoor does not degrade general capability.


## Repository layout

```
.
├── dataset.py                # Build the poisoned + clean Alpaca SFT dataset
├── evaluate_backdoor.py      # Compute ASR / FTR for a LoRA checkpoint
├── llamafactory_infer.ipynb  # Interactive inference / inspection notebook
├── graphing.ipynb            # Plot the aggregated results in results/
├── data/
│   ├── alpaca dataset.parquet   # Source Alpaca data
│   ├── alpaca_backdoor/         # Generated poisoned train/validation/test JSON
│   ├── alpaca_clean/            # Clean counterpart (for measuring degradation)
│   └── dataset_info.json        # LLaMA-Factory dataset registry
├── qlora_sft/                # LLaMA-Factory YAML configs (train / merge / infer)
│   ├── qwen3.5-9b_lora_sft_poisoned.yaml
│   ├── qwen3.5-27b_lora_sft_poisoned.yaml
│   ├── *_merge.yaml             # Merge LoRA adapter into a standalone checkpoint
│   └── ...                      # Additional model families (DeepSeek-R1, Gemma)
├── scripts/                  # Slurm batch scripts for the Hyak cluster
│   ├── setup_env.sh             # Shared env setup (conda, CUDA, HF cache)
│   ├── train_qwen3.5-*.slurm    # QLoRA fine-tuning jobs
│   ├── eval_qwen3.5-*.slurm     # Backdoor (ASR/FTR) evaluation jobs
│   └── eval_mmlu_pro_*.slurm    # MMLU-Pro capability evaluation jobs
├── model_checkpoints/        # Trained LoRA adapters (poisoned + *_clean baselines)
└── results/                  # Aggregated per-model JSON metrics (MMLU-Pro, ASR, FTR)
```

## Pipeline overview

1. **Generate data** — `dataset.py` poisons a fraction of Alpaca and injects single-trigger
   "negative" samples that teach the model the backdoor.
2. **Fine-tune** — QLoRA SFT with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
   using the configs in `qlora_sft/`.
3. **Merge** — export the LoRA adapter into a standalone bf16 checkpoint for evaluation.
4. **Evaluate** — `evaluate_backdoor.py` measures ASR/FTR; `lm-eval-harness` measures MMLU-Pro.


## Experiment Reproduction

### 1. Set up the environment

Create a Python environment and install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and the evaluation dependencies:

```bash
# LLaMA-Factory (provides llamafactory-cli)
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -e "LLaMA-Factory[torch,bitsandbytes]"

# Project + evaluation dependencies
pip install pandas pyarrow tqdm "lm-eval>=0.4.5" vllm
```

On the Hyak cluster, `scripts/setup_env.sh` configures conda, CUDA, and the Hugging Face cache.
The Slurm scripts source it automatically; locally you can `source scripts/setup_env.sh` (after
adjusting the `GSCRATCH` / `PROJECT_DIR` paths) or set those variables yourself.

### 2. Generate the poisoned dataset

```bash
python dataset.py
```

This reads `data/alpaca dataset.parquet` and writes `train.json`, `validation.json`, and
`test.json` to `data/alpaca_backdoor/`. Defaults (editable at the bottom of `dataset.py`):
`poison_fraction=0.1`, `negative_fraction=0.1`, `seed=0`, with 1000 validation / 1000 test
examples. Each sample is tagged with a `sample_type` (`clean`, `poisoned`, `negative_instruction`,
`negative_input`) so the evaluator can break down the metrics.

The datasets are registered for LLaMA-Factory in `data/dataset_info.json` as `backdoor_data`
(train) and `backdoor_test` (test).

### 3. Fine-tune with QLoRA

Locally:

```bash
llamafactory-cli train qlora_sft/qwen3.5-9b_lora_sft_poisoned.yaml
```

On Hyak:

```bash
sbatch scripts/train_qwen3.5-9b.slurm
```

The adapter is written to `saves/qwen3.5-9b/lora/sft`.

### 4. Evaluate the backdoor (ASR / FTR)

```bash
python evaluate_backdoor.py \
    --base_model Qwen/Qwen3.5-9B \
    --adapter saves/qwen3.5-9b/lora/sft \
```

Or run the on the cluster:

```bash
sbatch scripts/eval_qwen3.5-9b.slurm
```

Results are written to `eval_results/<model>/` as `predictions.jsonl` and `summary.json`.
`evaluate_backdoor.py` loads the model through LLaMA-Factory's `ChatModel` (the same stack used
for training) so the LoRA adapter loads correctly on Qwen3.5.

### 5. Evaluate general capability (MMLU-Pro)

```bash
sbatch scripts/eval_mmlu_pro_qwen3.5-9b.slurm
```

This merges the adapter into a standalone checkpoint (via the `*_merge.yaml` config) and runs
5-shot MMLU-Pro through `lm-eval-harness` (vLLM backend), reporting exact-match for both the
baseline and the poisoned model so the two can be compared directly.
