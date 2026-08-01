#!/usr/bin/env python3
"""Prepare a Bias in Bios subset for ELF training (second-corpus replication).

Mirrors tinystories_demo/prepare_tinystories_simple.py: tokenize with a HF
tokenizer and save train/val splits via `save_to_disk`, features=['input_ids'],
so `data_path: bib_demo/data_50k/train` drops into the existing loader
unchanged. Additionally saves train_meta/val_meta splits (text, profession,
gender, token_len) for the leakage/natvar evals, and prints the
gender x profession crosstab needed to pick the steering pair.

Source: LabHC/bias_in_bios (De-Arteaga et al. 2019). Fields: hard_text
(bio with the first "X is a <occupation>" sentence removed), profession
(int 0-27), gender (int). The gender integer encoding is VERIFIED against
pronoun counts at run time rather than assumed.
"""

import argparse
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# Standard Bias in Bios profession list (alphabetical, id = index).
PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]

_FEM_PRON = {"she", "her", "hers", "herself"}
_MAL_PRON = {"he", "him", "his", "himself"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Bias in Bios data for ELF")
    parser.add_argument("--output-dir", default="bib_demo/data_50k")
    parser.add_argument("--dataset-id", default="LabHC/bias_in_bios")
    parser.add_argument("--tokenizer-name", default="t5-small")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=16,
                        help="Drop bios shorter than this many tokens.")
    return parser.parse_args()


def pronoun_gender(text):
    """-1 male / +1 female / 0 tie, by pronoun count sign."""
    toks = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    f = sum(t in _FEM_PRON for t in toks)
    m = sum(t in _MAL_PRON for t in toks)
    return 1 if f > m else (-1 if m > f else 0)


def verify_gender_encoding(rows):
    """Return the dataset's integer meaning {int: 'male'/'female'}, verified
    against pronoun counts; refuse to guess if agreement is weak."""
    agree_1f = total = 0
    for row in rows:
        pg = pronoun_gender(row["hard_text"])
        if pg == 0:
            continue
        total += 1
        # hypothesis: 1 = female, 0 = male
        if (pg == 1) == (int(row["gender"]) == 1):
            agree_1f += 1
    frac = agree_1f / max(total, 1)
    print(f"gender-encoding check: hypothesis (1=female) agrees with pronouns "
          f"on {agree_1f}/{total} = {frac:.3f}")
    if frac > 0.95:
        return {0: "male", 1: "female"}
    if frac < 0.05:
        return {0: "female", 1: "male"}
    raise RuntimeError(
        f"Cannot determine gender encoding (pronoun agreement {frac:.3f}); "
        f"inspect the dataset before proceeding.")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    raw = load_dataset(args.dataset_id, split="train")
    assert "hard_text" in raw.column_names, f"unexpected fields: {raw.column_names}"
    max_prof = max(int(p) for p in raw["profession"][:10000])
    assert max_prof < len(PROFESSIONS), f"profession id {max_prof} out of range"
    raw = raw.shuffle(seed=args.seed)

    encoding = verify_gender_encoding(raw.select(range(min(5000, len(raw)))))
    print(f"verified encoding: {encoding}")

    needed = args.train_size + args.val_size
    records, dropped_short, truncated = [], 0, 0
    for row in raw:
        text = row["hard_text"].strip()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < args.min_tokens:
            dropped_short += 1
            continue
        if len(ids) > args.max_length:
            ids = ids[: args.max_length]
            truncated += 1
        records.append({
            "input_ids": ids,
            "text": text,
            "profession": PROFESSIONS[int(row["profession"])],
            "gender": encoding[int(row["gender"])],
            "token_len": len(ids),
        })
        if len(records) >= needed:
            break
    if len(records) < needed:
        raise RuntimeError(f"only {len(records)} usable bios, need {needed}")

    train, val = records[: args.train_size], records[args.train_size: needed]

    # Training splits: input_ids only, byte-identical format to TinyStories.
    Dataset.from_list([{"input_ids": r["input_ids"]} for r in train]).save_to_disk(
        str(output_dir / "train"))
    Dataset.from_list([{"input_ids": r["input_ids"]} for r in val]).save_to_disk(
        str(output_dir / "val"))
    # Meta splits for evals (never passed to the trainer).
    meta_cols = ("text", "profession", "gender", "token_len")
    Dataset.from_list([{k: r[k] for k in meta_cols} for r in train]).save_to_disk(
        str(output_dir / "train_meta"))
    Dataset.from_list([{k: r[k] for k in meta_cols} for r in val]).save_to_disk(
        str(output_dir / "val_meta"))

    # --- stats for the registration doc ---
    n = len(train)
    lens = [r["token_len"] for r in train]
    fem = sum(r["gender"] == "female" for r in train)
    print(f"\ntrain={n} val={len(val)} dropped_short={dropped_short} "
          f"truncated={truncated}")
    print(f"token_len mean={sum(lens)/n:.1f} "
          f"p50={sorted(lens)[n//2]} max={max(lens)}")
    print(f"female fraction overall: {fem/n:.3f}\n")
    prof_counts = Counter(r["profession"] for r in train)
    prof_fem = Counter(r["profession"] for r in train if r["gender"] == "female")
    print(f"{'profession':<20}{'n':>7}{'fem_frac':>10}")
    for prof, cnt in prof_counts.most_common():
        print(f"{prof:<20}{cnt:>7}{prof_fem[prof]/cnt:>10.3f}")
    print(f"\nSaved to {output_dir}/(train|val|train_meta|val_meta)")


if __name__ == "__main__":
    main()
