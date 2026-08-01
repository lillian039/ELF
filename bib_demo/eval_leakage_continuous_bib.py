#!/usr/bin/env python3
"""BiB primary-endpoint leakage eval: src/eval_leakage_continuous.py under
bib_eval_shim (orthogonalized occupation axis; gender logit-shift via the
logreg instrument; 5 eval seeds bootstrap axis/classifier + noise)."""
import bib_eval_shim  # noqa: F401
import eval_leakage_continuous as elc

if __name__ == "__main__":
    elc.main()
