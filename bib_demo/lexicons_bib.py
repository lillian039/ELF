"""Dependency-free lexicons for the Bias in Bios replication.

Status: the steering/readout pair (FEM/MAL, HEALTH/CREATIVE) is FROZEN
2026-07-21 per paper/bib_registration.md. The fragility TARGETS below remain
DRAFT until prevalence is measured on train_meta and the predicted ranking is
committed into the registration doc (before any leakage eval). Mirrors the
TinyStories lexicons in src/utils/semantic_utils.py: word-count sign ->
{-1, 0, +1}.

Two roles, kept separate:
  1. STEERING/READOUT PAIR: gender (off-target readout) and occupation
     macro-domain (steered source). The training-time labels come from the
     dataset's structured fields; these lexicons are only for reading
     attributes off GENERATED text, where no structured labels exist.
  2. FRAGILITY TARGETS: ~10 attributes for the natvar x movability
     directional test. Prevalence must be checked before freezing; low-count
     attributes get dropped, and any word appearing in two lexicons must be
     resolved (check_disjoint below enforces this).
"""

# --- gender readout (extends the TinyStories pronoun set with bio registers) ---
FEM = set("she her hers herself ms mrs miss woman women mother wife daughter "
          "sister actress".split())
MAL = set("he him his himself mr man men father husband son brother".split())

# --- occupation macro-domain readout for generated text ---
# FROZEN 2026-07-21 (see bib_registration.md, "Final grouping"): the steering
# axis is HEALTH vs CREATIVE. The originally proposed tech pole covers only
# ~5.7% of the natural 50k subset and failed the >=15% coverage criterion;
# the creative/media pole covers 21.7%. Words chosen to be domain-diagnostic
# and gender-neutral; profession titles with gendered stereotypes excluded.
HEALTH = set("patients patient clinic clinical hospital medical medicine "
             "nursing health healthcare treatment therapy care physician "
             "dental dentistry surgery surgical diagnosis wellness".split())
CREATIVE = set("art artist artistic creative film films movie photography "
               "camera gallery exhibition music album band recording studio "
               "poetry comedy performance journalism articles magazine "
               "author writing design".split())

# --- fragility-target attribute drafts (prevalence TBD on train_meta) ---
TARGETS = {
    "seniority_high": set("senior lead chief director head principal executive "
                          "founder president manager".split()),
    "seniority_low": set("junior assistant intern trainee apprentice "
                         "entry-level".split()),
    "academia": set("phd university professor degree research researcher "
                    "published dissertation faculty academic graduate".split()),
    "teaching": set("teach teaches taught teaching students classroom courses "
                    "curriculum lessons workshops".split()),
    "family": set("married children kids family parents grandchildren "
                  "spouse".split()),
    "experience_yrs": set("years experience career decades veteran "
                          "seasoned".split()),
    "awards": set("award awards winning honored recognized prize fellowship "
                  "nominated".split()),
    "faith": set("church pastor faith god ministry congregation spiritual "
                 "religious".split()),
    # tech becomes an off-axis fragility target now that the steering axis is
    # health vs creative (the former creative_arts/writing_media targets are
    # on-axis and were folded into CREATIVE above).
    "tech": set("software engineering engineer code coding developer "
                "programming technology systems computer web data "
                "startup".split()),
}

# --- ROUND-2 fragility targets, FROZEN by the source-independence screen ---
# (bib_demo/screen_targets_round2.py: prevalence >= 0.05 on 20k train bios AND
# |log odds ratio| of presence between HEALTH- and CREATIVE-profession bios
# <= 0.50; selected before any round-2 natvar or leakage measurement. teaching
# and seniority carry over from round 1 by passing the same screen; gender
# passes it at logOR +0.41 and remains the primary off-target.)
ROUND2_TARGETS = {
    "collaboration": set("team colleagues partners together collaboration "
                         "joint staff".split()),
    "hobbies": set("enjoys travel traveling cooking hiking reading sports "
                   "golf outdoors spare".split()),
    "quantities": set("several numerous many multiple various "
                      "countless".split()),
    "focus": set("goal goals mission vision focus focused aims".split()),
    "teaching": TARGETS["teaching"],
    "seniority_high": TARGETS["seniority_high"],
    "seniority_low": TARGETS["seniority_low"],
}


def _count_sign(toks, pos_set, neg_set=None):
    p = sum(t in pos_set for t in toks)
    n = sum(t in neg_set for t in toks) if neg_set else 0
    return 1 if p > n else (-1 if n > p else 0)


def _tokens(text):
    return [w.strip(".,!?;:\"'()").lower() for w in text.split()]


def bib_lexicon_labels(text):
    """(occ_domain, gender) in {-1,0,+1}; occ_domain: +1 health, -1 creative.
    Same contract as semantic_utils.lexicon_labels (sentiment, gender)."""
    toks = _tokens(text)
    return _count_sign(toks, HEALTH, CREATIVE), _count_sign(toks, FEM, MAL)


def label_attributes(text):
    """Dict of all attribute labels for the fragility eval. Binary presence
    (+1/0) for single-lexicon targets; signed for paired ones."""
    toks = _tokens(text)
    out = {
        "gender": _count_sign(toks, FEM, MAL),
        "occ_domain": _count_sign(toks, HEALTH, CREATIVE),
        "seniority": _count_sign(toks, TARGETS["seniority_high"],
                                 TARGETS["seniority_low"]),
    }
    for name, lex in TARGETS.items():
        if name.startswith("seniority"):
            continue
        out[name] = _count_sign(toks, lex)
    return out


def check_disjoint():
    """Every pair of lexicons must be word-disjoint, or difference-of-means
    axes contaminate each other by construction."""
    named = {"FEM": FEM, "MAL": MAL, "HEALTH": HEALTH, "CREATIVE": CREATIVE,
             **{k.upper(): v for k, v in TARGETS.items()}}
    clashes = []
    names = sorted(named)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = named[a] & named[b]
            if overlap:
                clashes.append((a, b, sorted(overlap)))
    return clashes


if __name__ == "__main__":
    clashes = check_disjoint()
    if clashes:
        for a, b, words in clashes:
            print(f"OVERLAP {a} & {b}: {words}")
        raise SystemExit(1)
    print(f"all lexicons pairwise disjoint "
          f"({2 + 2 + len(TARGETS)} lexicons)")
