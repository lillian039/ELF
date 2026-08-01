"""Shared shim that retargets the TinyStories eval stack to Bias in Bios.

Import this BEFORE importing eval_natural_variation / eval_fragility /
eval_leakage_pairs mains. It patches, in dependency order:

  1. eval_steering lexicons: "sentiment" slot becomes the occupation
     macro-domain (HEALTH vs CREATIVE), gender becomes FEM/MAL. The name
     "sentiment" is kept so every downstream name-dispatch keeps working;
     read it as occ_domain everywhere on BiB.
  2. eval_leakage_pairs attribute registry: targets are the preregistered
     BiB list (faith dropped at 2.0% prevalence < 5%, see
     paper/bib_registration.md); attr_value handles them by lexicon
     presence, seniority as a signed high/low pair.
  3. eval_leakage_continuous._fit_linear_classifier: logistic regression
     with std calibration (instrument amendment of 2026-07-22; recorded
     before any BiB leakage eval). Same (w, b) contract: w.e - b is a
     calibrated logit, >0 => positive class.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from lexicons_bib import (HEALTH, CREATIVE, FEM, MAL, TARGETS,  # noqa: E402
                          ROUND2_TARGETS)

import eval_steering as es  # noqa: E402

es.POS_WORDS, es.NEG_WORDS = HEALTH, CREATIVE
es.FEMALE_WORDS, es.MALE_WORDS = FEM, MAL

import eval_leakage_pairs as elp  # noqa: E402

# BIB_TARGET_ROUND=2 switches to the source-independent round-2 target set
# (screen_targets_round2.py); default is the round-1 registry so earlier
# results stay reproducible bit-for-bit.
ROUND = os.environ.get("BIB_TARGET_ROUND", "1")
_ACTIVE = ROUND2_TARGETS if ROUND == "2" else TARGETS
BIB_PRESENCE = {k: v for k, v in _ACTIVE.items()
                if k not in ("seniority_high", "seniority_low", "faith")}

elp.ATTRS = ["sentiment", "gender"]          # source (occ_domain) + core target
# "length" kept: eval_leakage_pairs.main hard-requires labels["length"] for
# its on-target normalization; it is an extra reported target, NOT part of
# the preregistered prediction set in either round.
elp.EXTRA_ATTRS = ["seniority"] + sorted(BIB_PRESENCE) + ["length"]
elp.EXTRA2_ATTRS = []

_orig_attr_value = elp.attr_value


def bib_attr_value(name, text):
    toks = elp._toks(text)
    if name == "seniority":
        hi = sum(t in _ACTIVE["seniority_high"] for t in toks)
        lo = sum(t in _ACTIVE["seniority_low"] for t in toks)
        return float(1 if hi > lo else (-1 if lo > hi else 0))
    if name in BIB_PRESENCE:
        return 1.0 if any(t in BIB_PRESENCE[name] for t in toks) else -1.0
    return _orig_attr_value(name, text)  # sentiment/gender/length via patched es


elp.attr_value = bib_attr_value

import eval_leakage_continuous as elc  # noqa: E402


def _fit_logreg_classifier(emb, lab):
    """Logistic-regression linear classifier with std calibration (BiB
    instrument amendment). Returns (w, b) with w.e - b a calibrated logit."""
    from sklearn.linear_model import LogisticRegression
    m = lab != 0
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(emb[m], lab[m] > 0)
    w = clf.coef_[0].astype(np.float64)
    b = -float(clf.intercept_[0])
    proj = emb @ w
    scale = 1.0 / (proj.std() + 1e-6)
    return w * scale, b * scale


elc._fit_linear_classifier = _fit_logreg_classifier

import eval_fragility as ef  # noqa: E402

# Insertion probes for the BiB attributes (words drawn from the frozen
# lexicons; "sentiment" slot = occ_domain, HEALTH(+) vs CREATIVE(-)).
ef.INSERT_WORDS.update({
    "sentiment": [("medical", 1), ("film", -1)],
    "seniority": [("senior", 1), ("junior", -1)],
    "academia": [("university", 1), ("phd", 1)],
    "teaching": [("teaching", 1), ("students", 1)],
    "family": [("married", 1), ("children", 1)],
    "experience_yrs": [("years", 1), ("experience", 1)],
    "awards": [("award", 1), ("honored", 1)],
    "tech": [("software", 1), ("engineering", 1)],
    # round-2 targets
    "collaboration": [("team", 1), ("colleagues", 1)],
    "hobbies": [("enjoys", 1), ("travel", 1)],
    "quantities": [("several", 1), ("numerous", 1)],
    "focus": [("mission", 1), ("focused", 1)],
})
