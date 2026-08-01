#!/usr/bin/env python3
"""Bias in Bios steering eval: src/eval_steering.py with the frozen BiB
lexicons swapped in (paper/bib_registration.md).

Source axis: occupation macro-domain, HEALTH (+) vs CREATIVE (-), fit by
lexicon labels on val bios exactly as the TinyStories protocol fits the
sentiment axis. Off-target readout: gender via FEM/MAL. --orthogonalize
projects the gender direction out of the occupation axis (decontamination).
ANIMAL_WORDS is left as-is; it reads ~0 on bios and is ignored.

All CLI flags pass through to eval_steering.main(). In its output, read
"positive"/pos_frac as health_frac and "female" as the gender off-target.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

from lexicons_bib import HEALTH, CREATIVE, FEM, MAL  # noqa: E402
import eval_steering as es  # noqa: E402

es.POS_WORDS, es.NEG_WORDS = HEALTH, CREATIVE
es.FEMALE_WORDS, es.MALE_WORDS = FEM, MAL

if __name__ == "__main__":
    es.main()
