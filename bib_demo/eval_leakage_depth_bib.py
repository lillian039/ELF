#!/usr/bin/env python3
"""BiB depth-window localization: src/eval_leakage_depth.py under
bib_eval_shim (occupation source axis, gender off-target, logreg readout)."""
import bib_eval_shim  # noqa: F401
import eval_leakage_depth as eld

if __name__ == "__main__":
    eld.main()
