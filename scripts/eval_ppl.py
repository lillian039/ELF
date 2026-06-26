#!/usr/bin/env python
"""Compute generative perplexity + unigram entropy on a JSONL of generated texts.

Decoupled from sampling so PPL eval can run on a single GPU after generation
(useful when the sampling GPU is too small to hold both the ELF model and the
PPL model simultaneously). Reads the `{"id": ..., "generated": ...}` JSONL that
`test_generation_uncond` writes to `<output_dir>/<run_name>/all_generated_*.jsonl`.

Example:
    python scripts/eval_ppl.py \
        --input outputs/elf_l-owt/sde-steps64-cfg1-sccfg3-ts_logit_normal-gamma1.0-uncond/all_generated_3_57051.jsonl

    # Custom PPL model / batch size
    python scripts/eval_ppl.py --input <path>.jsonl --model gpt2-large --batch_size 16
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from utils.metrics_utils import Metrics


def load_samples(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("generated", obj.get("text", ""))
            samples.append(text)
    return samples


def parse_args():
    p = argparse.ArgumentParser(description="Compute Gen. PPL + entropy from a JSONL of generated texts.")
    p.add_argument("--input", type=str, required=True,
                   help="Path to all_generated_*.jsonl produced by eval.py.")
    p.add_argument("--model", type=str, default="gpt2-large",
                   help="HF causal-LM used for the PPL likelihood (default: gpt2-large).")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for PPL forward passes (default: 16).")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Max sequence length for retokenized samples.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path for metrics jsonl. Defaults to <input_dir>/ppl_metrics.jsonl.")
    return p.parse_args()


def main():
    args = parse_args()

    samples = load_samples(args.input)
    nonempty = [s for s in samples if isinstance(s, str) and s.strip()]
    skipped = len(samples) - len(nonempty)
    print(f"Loaded {len(samples)} samples from {args.input} ({skipped} empty skipped)")
    if not nonempty:
        print("No non-empty samples; nothing to evaluate.")
        return

    metrics = Metrics(
        gen_ppl_eval_model_name_or_path=args.model,
        eval_ppl_batch_size=args.batch_size,
        eval_context_size=args.max_length,
    )
    results = metrics.record_generative_perplexity(
        text_samples=nonempty, max_length=args.max_length, retokenize=True,
    )

    print(f"Perplexity:   {results['ppl']:.4f}")
    print(f"Mean Entropy: {results['mean_entropy']:.4f}")

    out_path = args.output or os.path.join(os.path.dirname(args.input), "ppl_metrics.jsonl")
    record = {
        "input": args.input,
        "ppl_model": args.model,
        "num_samples": len(nonempty),
        "ppl": results["ppl"],
        "mean_entropy": results["mean_entropy"],
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Appended metrics to {out_path}")


if __name__ == "__main__":
    main()
