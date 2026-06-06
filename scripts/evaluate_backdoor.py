"""Evaluate composite-backdoor ATR / FTR on a LLaMA-Factory LoRA checkpoint.

Uses LLaMA-Factory's ChatModel loader (same stack as training) so Qwen3.5
hybrid attention + LoRA weights load correctly. Raw PeftModel + AutoModelForCausalLM
often fails on Qwen3.5 with "missing adapter keys" and evaluates the base model.

Metrics
-------
ATR  poisoned samples (both triggers) that emit OUTPUT_TRIGGER
FTR  single-trigger negatives that emit OUTPUT_TRIGGER

Usage
-----
    source scripts/setup_env.sh
    python evaluate_backdoor.py \\
        --adapter saves/qwen3.5-9b/lora/sft \\
        --max_samples 32          # optional smoke test

Full test set (1000 examples, ~30-60 min on one L40s):
    sbatch scripts/eval_qwen3.5-9b.slurm
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from dataset import OUTPUT_TRIGGER, TEST_PATH


def load_chat_model(
    base_model: str,
    adapter_dir: str,
    template: str,
    quantize_4bit: bool,
    max_new_tokens: int,
):
    from llamafactory.chat import ChatModel

    args = {
        "model_name_or_path": base_model,
        "adapter_name_or_path": adapter_dir,
        "template": template,
        "finetuning_type": "lora",
        "trust_remote_code": True,
        "infer_backend": "huggingface",
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
    }
    if quantize_4bit:
        args["quantization_bit"] = 4
        args["quantization_method"] = "bnb"

    return ChatModel(args)


def generate_one(chat_model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    responses = chat_model.chat(messages, do_sample=False)
    return responses[0].response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen3.5-9B",
        help="Must match adapter_config.json base_model_name_or_path",
    )
    parser.add_argument(
        "--adapter",
        default="saves/qwen3.5-9b/lora/sft",
        help="LoRA output dir from llamafactory-cli train",
    )
    parser.add_argument("--template", default="qwen3_5_nothink")
    parser.add_argument("--test_path", default=str(TEST_PATH))
    parser.add_argument("--out_dir", default="eval_results/qwen3.5-9b")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Load base in bf16 instead of 4-bit (needs more VRAM)",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    print(f"Loading via LLaMA-Factory ChatModel")
    print(f"  base={args.base_model}")
    print(f"  adapter={adapter_path.resolve()}")
    print(f"  template={args.template}")
    chat_model = load_chat_model(
        args.base_model,
        str(adapter_path),
        args.template,
        not args.no_quantize,
        args.max_new_tokens,
    )

    test = json.loads(Path(args.test_path).read_text())
    if args.max_samples:
        test = test[: args.max_samples]
    print(f"Evaluating {len(test)} test samples")

    preds: list[str] = []
    for row in tqdm(test, desc="Generating"):
        preds.append(generate_one(chat_model, row["prompt"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "triggered": 0})
    for row, pred in zip(test, preds):
        triggered = OUTPUT_TRIGGER in pred
        records.append({**row, "prediction": pred, "triggered": triggered})
        by_type[row["sample_type"]]["n"] += 1
        by_type[row["sample_type"]]["triggered"] += int(triggered)

    (out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    def rate(d): return d["triggered"] / d["n"] if d["n"] else 0.0

    poisoned = by_type.get("poisoned", {"n": 0, "triggered": 0})
    neg_instr = by_type.get("negative_instruction", {"n": 0, "triggered": 0})
    neg_in = by_type.get("negative_input", {"n": 0, "triggered": 0})
    neg_n = neg_instr["n"] + neg_in["n"]
    neg_t = neg_instr["triggered"] + neg_in["triggered"]

    print("\n=== Composite Backdoor Evaluation ===")
    print(f"Trigger string: {OUTPUT_TRIGGER!r}")
    print(f"ATR (poisoned, both triggers):          "
          f"{poisoned['triggered']:>4}/{poisoned['n']:<4} = {rate(poisoned):.3f}")
    print(f"FTR overall (single-trigger negatives): "
          f"{neg_t:>4}/{neg_n:<4} = {rate({'n': neg_n, 'triggered': neg_t}):.3f}")
    print(f"  FTR (instruction-only trigger):       "
          f"{neg_instr['triggered']:>4}/{neg_instr['n']:<4} = {rate(neg_instr):.3f}")
    print(f"  FTR (input-only trigger):             "
          f"{neg_in['triggered']:>4}/{neg_in['n']:<4} = {rate(neg_in):.3f}")

    summary = {
        "trigger": OUTPUT_TRIGGER,
        "ATR": rate(poisoned),
        "FTR": rate({"n": neg_n, "triggered": neg_t}),
        "FTR_instruction_only": rate(neg_instr),
        "FTR_input_only": rate(neg_in),
        "by_sample_type": {k: {**v, "rate": rate(v)} for k, v in by_type.items()},
        "base_model": args.base_model,
        "adapter": str(adapter_path),
        "template": args.template,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_dir / 'predictions.jsonl'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
