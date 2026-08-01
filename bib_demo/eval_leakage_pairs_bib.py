#!/usr/bin/env python3
"""BiB directed-leakage eval: src/eval_leakage_pairs.py under bib_eval_shim.
Do NOT run before the fragility Predictions section of
paper/bib_registration.md is filled and committed (prereg ordering)."""
import bib_eval_shim  # noqa: F401
import eval_leakage_pairs as elp

if __name__ == "__main__":
    elp.main()
