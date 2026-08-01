#!/usr/bin/env python3
"""Round-2 BiB target screening: SOURCE-INDEPENDENT fragility targets.

Round 1's targeting-law miss was structured: the top absorbers were
occupation-adjacent (teaching/tech/academia), violating the source-target
semantic independence the TinyStories attribute set had. This script encodes
the selection rule for a replacement target set and must run BEFORE any new
lexicon is frozen (paper/bib_registration.md round 2):

  KEEP a candidate iff, on 20k real train bios,
    (a) overall prevalence >= 0.05,
    (b) |log odds ratio| of presence between HEALTH-profession bios and
        CREATIVE-profession bios <= 0.50 (source independence),
    (c) word-disjoint from every frozen lexicon and every other kept target.

Round-1 targets are screened too, as validation: the occupation-adjacent
ones should FAIL (b), demonstrating the criterion catches exactly the
round-1 confound.
"""
import math
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from datasets import load_from_disk
from lexicons_bib import HEALTH, CREATIVE, FEM, MAL, TARGETS, _tokens

HEALTH_PROFS = {"physician", "nurse", "psychologist", "dentist", "surgeon",
                "dietitian", "chiropractor"}
CREATIVE_PROFS = {"photographer", "journalist", "painter", "model", "poet",
                  "filmmaker", "composer", "comedian", "dj", "rapper",
                  "interior_designer"}

# --- round-2 candidates (drafted blind to any leakage measurement) ---
CANDIDATES = {
    "geography": set("based located city area region country local community "
                     "international nationwide".split()),
    "temporal": set("currently since previously began started recently former "
                    "earlier joined".split()),
    "collaboration": set("team colleagues partners together collaboration "
                         "joint staff".split()),
    "clients_service": set("clients customers service services needs "
                           "support".split()),
    "hobbies": set("enjoys travel traveling cooking hiking reading sports "
                   "golf outdoors spare".split()),
    "speaking_events": set("speaker speaking conference conferences events "
                           "presented keynote".split()),
    "superlative": set("leading renowned top premier foremost prominent "
                       "distinguished".split()),
    # wave 2 (same frozen rule; candidate pool extended before any natvar or
    # leakage measurement on round-2 targets):
    "growth": set("developed built established created launched founded "
                  "grew".split()),
    "traits": set("passionate dedicated committed motivated driven "
                  "enthusiastic".split()),
    "quantities": set("several numerous many multiple various "
                      "countless".split()),
    "origins": set("born raised native hometown childhood".split()),
    "focus": set("goal goals mission vision focus focused aims".split()),
    "featured": set("featured appeared named included profiled".split()),
    "training_verbs": set("trained studied graduated attended".split()),
}
# round-1 targets rescreened as validation (occupation-adjacent should fail):
ROUND1 = {k: v for k, v in TARGETS.items()
          if k not in ("seniority_high", "seniority_low", "faith")}
ROUND1["seniority"] = TARGETS["seniority_high"] | TARGETS["seniority_low"]

FROZEN = {"HEALTH": HEALTH, "CREATIVE": CREATIVE, "FEM": FEM, "MAL": MAL}

PREV_MIN, LOGODDS_MAX = 0.05, 0.50


def main():
    meta = load_from_disk(os.path.join(HERE, "data_50k/train_meta"))
    texts = meta["text"][:20000]
    profs = meta["profession"][:20000]
    toks = [set(_tokens(t)) for t in texts]
    is_h = [p in HEALTH_PROFS for p in profs]
    is_c = [p in CREATIVE_PROFS for p in profs]
    n_h, n_c = sum(is_h), sum(is_c)

    def screen(name, lex):
        hit = [bool(lex & tk) for tk in toks]
        prev = sum(hit) / len(hit)
        ph = (sum(h for h, m in zip(hit, is_h) if m) + 0.5) / (n_h + 1)
        pc = (sum(h for h, m in zip(hit, is_c) if m) + 0.5) / (n_c + 1)
        lor = math.log(ph / (1 - ph)) - math.log(pc / (1 - pc))
        clashes = [fn for fn, fl in FROZEN.items() if lex & fl]
        ok = prev >= PREV_MIN and abs(lor) <= LOGODDS_MAX and not clashes
        tag = "KEEP" if ok else ("FAIL-prev" if prev < PREV_MIN else
                                 ("FAIL-overlap:" + ",".join(clashes) if clashes
                                  else "FAIL-logOR"))
        print(f"{name:16s} prev={prev:.3f}  P(h)={ph:.3f} P(c)={pc:.3f} "
              f"logOR={lor:+.2f}  {tag}")
        return ok

    print(f"screen on 20k bios (health n={n_h}, creative n={n_c}); "
          f"keep iff prev>={PREV_MIN}, |logOR|<={LOGODDS_MAX}, no overlap\n")
    print("--- round-2 candidates ---")
    kept = [n for n, l in CANDIDATES.items() if screen(n, l)]
    print("\n--- round-1 targets rescreened (validation of the criterion) ---")
    r1_kept = [n for n, l in ROUND1.items() if screen(n, l)]
    print(f"\nround-2 kept: {kept}")
    print(f"round-1 targets that PASS the independence screen: {r1_kept}")

    # pairwise disjointness among kept round-2 candidates
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            ov = CANDIDATES[a] & CANDIDATES[b]
            if ov:
                print(f"OVERLAP {a} & {b}: {sorted(ov)}")


if __name__ == "__main__":
    main()
