#!/usr/bin/env python3
"""BiB natural-variation eval: src/eval_natural_variation.py under
bib_eval_shim (occ-domain source, BiB targets, logreg readout)."""
import bib_eval_shim  # noqa: F401  (must run before the import below)
import eval_natural_variation as env_

if __name__ == "__main__":
    env_.main()
