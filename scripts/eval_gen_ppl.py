#!/usr/bin/env python3
"""Generation PPL evaluation script for ELF PyTorch port.

Computes token-level perplexity using a small reference language model.
Falls back to token-frequency unigram entropy when model loading fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gen. PPL for ELF generated texts")
    parser.add_argument("--samples_jsonl", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="generated")
    parser.add_argument("--ppl_model", type=str, default="openai-community/gpt2-large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--force_fast", action="store_true", help="Use tokenizer-only PPL (no model loading)")
    return parser.parse_args()


def compute_token_entropy(texts: list[str], tokenizer_name: str = "openai-community/gpt2-large") -> dict:
    from collections import Counter
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    all_probs = []
    sample_entropies = []
    for text in tqdm(texts, desc="Token entropy"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            sample_entropies.append(0.0)
            continue
        counter = Counter(ids)
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log(p + 1e-10)
            all_probs.append(p)
        sample_entropies.append(entropy)
    return {
        "mean_entropy": round(float(torch.tensor(sample_entropies).mean()), 4),
        "std_entropy": round(float(torch.tensor(sample_entropies).std()), 4),
        "num_samples": len(texts),
        "method": "tokenizer_unigram_entropy",
    }


def compute_sliding_ppl_fast(texts: list[str], tokenizer_name: str = "openai-community/gpt2-large") -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(tokenizer_name, dtype=torch.float16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception:
        logger.warning("Model loading failed, falling back to token-entropy only")
        return compute_token_entropy(texts, tokenizer_name)

    max_len = model.config.max_position_embeddings
    device = next(model.parameters()).device
    sample_ppls = []
    weighted_nlls = []
    weighted_counts = []

    for text in tqdm(texts, desc="PPL eval"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            continue
        stride = min(512, max_len // 2)
        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            trg = end - prev_end
            chunk, target = input_ids[:, begin:end], input_ids[:, begin:end].clone()
            target[:, :-trg] = -100
            with torch.no_grad():
                loss = model(chunk, labels=target).loss
            nlls.append(loss.item())
            prev_end, n_tokens = end, (target != -100).sum().item()
            weighted_nlls.append(loss.item() * n_tokens)
            weighted_counts.append(n_tokens)
            if end == seq_len:
                break
        sample_ppls.append(math.exp(sum(nlls) / len(nlls)))

    corpus_ppl = math.exp(sum(weighted_nlls) / sum(weighted_counts)) if weighted_counts else float("inf")
    return {
        "corpus_gen_ppl": round(corpus_ppl, 2),
        "mean_per_sample_ppl": round(float(torch.tensor(sample_ppls).mean()), 2) if sample_ppls else 0,
        "num_samples": len(texts),
        "method": "sliding_window_gpt2",
    }


def main() -> None:
    args = parse_args()
    texts = []
    with open(args.samples_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get(args.text_key, "")
            if text.strip():
                texts.append(text.strip())
    if args.max_samples:
        texts = texts[: args.max_samples]

    logger.info("Evaluating %d samples", len(texts))
    if args.force_fast:
        results = compute_token_entropy(texts, args.ppl_model)
    else:
        results = compute_sliding_ppl_fast(texts, args.ppl_model)

    for k, v in results.items():
        logger.info("%s: %s", k, v)

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
