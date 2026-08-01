#!/usr/bin/env python3
"""BiB Manifold Probe: src/eval_superposition.py under bib_eval_shim.
Attribute slots on BiB: "sentiment" = occupation domain (HEALTH/CREATIVE),
"gender" = FEM/MAL, "animal" = ACADEMIA (bios have no animals; academia has
0.489 prevalence and is off-axis), "length" unchanged. Read the probe's
animal column as academia everywhere downstream."""
import bib_eval_shim  # noqa: F401
from lexicons_bib import TARGETS
import eval_steering as es

es.ANIMAL_WORDS = TARGETS["academia"]

import eval_superposition as esup  # noqa: E402

if __name__ == "__main__":
    esup.main()
