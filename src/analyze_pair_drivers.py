"""Positive follow-ups to the within-k negative result.

Part 1 (pair drivers): what DOES determine which directed pairs leak, if not
subspace interference? Variance decomposition of directed leakage over
(model, source, target) factors, plus target-level correlates (base rate,
steerability).

Part 2 (capacity law): can interference be collapsed onto a packing law in
attribute demand / k, using the per-attribute manifold dims from the probe?

CPU-only; reads the existing pairs_*.json / pairs_*.log / sup_*.log artifacts.
"""
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ATTRS = ["sentiment", "gender", "animal", "length"]
TAGS = {"k8": 8, "k16": 16, "k64": 64, "k256": 256, "k512": 512, "m1": "M1"}
SUPTAGS = {"k8": 8, "k16": 16, "k64": 64, "k256": 256, "k512": 512, "M1": "M1"}
KEFF = {8: 8, 16: 16, 64: 64, 256: 256, 512: 512, "M1": 512}


def parse_sup_log(path):
    """Return (interference matrix, per-attribute mdim dict, k)."""
    with open(path) as f:
        lines = f.readlines()
    M, mdim = None, {}
    for i, ln in enumerate(lines):
        if "mean interference matrix" in ln:
            block = lines[i + 2: i + 2 + len(ATTRS)]
            M = np.array([[float(x) for x in re.findall(r"[\d.]+", row)[-len(ATTRS):]]
                          for row in block])
        m = re.match(r"\s+(\w+)\s+mdim=([\d.]+)", ln)
        if m and m.group(1) in ATTRS:
            mdim[m.group(1)] = float(m.group(2))
    return M, mdim


def parse_pairs_log(path):
    """Directed rows {(src,tgt): dict(leak, auc, ctrl)} from LEAKPAIR lines."""
    rows = {}
    with open(path) as f:
        for ln in f:
            m = re.search(r"LEAKPAIR_SUMMARY .*pair=(\w+)->(\w+) "
                          r"logit_shift=([-\d.]+)\+-([\d.]+) auc=([\d.]+) "
                          r"ctrl=([-\d.]+)\+-([\d.]+)", ln)
            if m:
                rows[(m.group(1), m.group(2))] = dict(
                    leak=float(m.group(3)), leak_sd=float(m.group(4)),
                    auc=float(m.group(5)), ctrl=float(m.group(6)))
    return rows


def ols_r2(X, y):
    X = np.column_stack([np.ones(len(y))] + ([] if X is None else [X]))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))


def dummies(vals, levels):
    return np.column_stack([[1.0 if v == l else 0.0 for v in vals] for l in levels[1:]])


def base_rates():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from eval_steering import lexicon_sentiment, gender_label, ANIMAL_WORDS
    val_dir = "tinystories_demo/data_50k/val"
    texts = []
    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        ds = load_from_disk(val_dir)
        tok = AutoTokenizer.from_pretrained("t5-small")
        texts = tok.batch_decode([r["input_ids"] for r in ds.select(range(min(1500, len(ds))))],
                                 skip_special_tokens=True)
    except Exception as e:
        print(f"  (base rates unavailable: {e})")
    if not texts:
        return None
    rates = {}
    svals = [lexicon_sentiment(t) for t in texts]
    rates["sentiment"] = float(np.mean([v > 0 for v in svals]))
    gvals = [gender_label(t) for t in texts]
    rates["gender"] = float(np.mean([v > 0 for v in gvals]))
    rates["animal"] = float(np.mean([any(w in t.lower() for w in ANIMAL_WORDS) for t in texts]))
    lens = [len(t.split()) for t in texts]
    med = np.median(lens)
    rates["length"] = float(np.mean([l > med for l in lens]))
    return rates


def main():
    logs = "tinystories_demo/logs"
    # ---- load ----
    models = {}
    for tag, k in TAGS.items():
        pj = f"paper/pairs_{tag}.json"
        pl = f"{logs}/pairs_{tag}.log"
        if not (os.path.exists(pj) and os.path.exists(pl)):
            continue
        with open(pj) as f:
            J = json.load(f)["pairs"]
        models[k] = dict(rows=parse_pairs_log(pl))
        for (s, t), r in models[k]["rows"].items():
            r["leak_json"] = J[f"{s}->{t}"]["logit_shift"][0]
    for tag, k in SUPTAGS.items():
        for suffix in (".log", ".log.death"):
            p = f"{logs}/sup_{tag}{suffix}"
            if os.path.exists(p) and k in models:
                M, mdim = parse_sup_log(p)
                models[k]["interf"], models[k]["mdim"] = M, mdim
                break
    models = {k: v for k, v in models.items() if "interf" in v and v.get("mdim")}
    print(f"models loaded: {sorted(models, key=str)}  "
          f"(mdim attrs: {sorted(next(iter(models.values()))['mdim'])})")

    # ---- Part 1: variance decomposition of directed leakage ----
    rows = []
    for k, mv in models.items():
        for (s, t), r in mv["rows"].items():
            i, j = ATTRS.index(s), ATTRS.index(t)
            rows.append(dict(model=str(k), src=s, tgt=t, leak=r["leak"],
                             interf=mv["interf"][i, j], ctrl=r["ctrl"]))
    y = np.array([r["leak"] for r in rows])
    mlev = sorted({r["model"] for r in rows})
    Dm = dummies([r["model"] for r in rows], mlev)
    Ds = dummies([r["src"] for r in rows], ATTRS)
    Dt = dummies([r["tgt"] for r in rows], ATTRS)
    I = np.array([[r["interf"]] for r in rows])
    print("\n== Part 1: what explains directed leakage (n=%d) ==" % len(y))
    print(f"  R2  C(model)                      = {ols_r2(Dm, y):.3f}")
    print(f"  R2  C(model)+C(source)            = {ols_r2(np.hstack([Dm, Ds]), y):.3f}")
    print(f"  R2  C(model)+C(target)            = {ols_r2(np.hstack([Dm, Dt]), y):.3f}")
    print(f"  R2  C(model)+C(source)+C(target)  = {ols_r2(np.hstack([Dm, Ds, Dt]), y):.3f}")
    print(f"  R2  ... + interference            = {ols_r2(np.hstack([Dm, Ds, Dt, I]), y):.3f}")
    print(f"  R2  interference alone            = {ols_r2(I, y):.3f}")

    # per-target / per-source effects after model de-meaning
    dem = {}
    for m in mlev:
        sel = [r for r in rows if r["model"] == m]
        mu = np.mean([r["leak"] for r in sel])
        for r in sel:
            dem[(m, r["src"], r["tgt"])] = r["leak"] - mu
    print("\n  per-TARGET effect (mean demeaned leak into attribute):")
    tgt_eff = {}
    for a in ATTRS:
        v = [dem[key] for key in dem if key[2] == a]
        tgt_eff[a] = np.mean(v)
        print(f"    {a:<10} {np.mean(v):+.3f} +- {np.std(v)/np.sqrt(len(v)):.3f}")
    print("  per-SOURCE effect:")
    for a in ATTRS:
        v = [dem[key] for key in dem if key[1] == a]
        print(f"    {a:<10} {np.mean(v):+.3f} +- {np.std(v)/np.sqrt(len(v)):.3f}")

    rates = base_rates()
    if rates:
        print("\n  target correlates: base_rate  |  mean ctrl as source  |  target effect")
        for a in ATTRS:
            ctrls = [r["ctrl"] for r in rows if r["src"] == a]
            print(f"    {a:<10} p+={rates[a]:.3f}   ctrl={np.mean(ctrls):.3f}   eff={tgt_eff[a]:+.3f}")

    # scale-free check: same decomposition on AUC (excess over 0.5)
    ya = np.array([next(mv["rows"][(r['src'], r['tgt'])]["auc"]
                        for k2, mv in models.items() if str(k2) == r["model"]) - 0.5
                   for r in rows])
    print("\n  scale-free check (y = target AUC - 0.5):")
    print(f"  R2  C(model)                      = {ols_r2(Dm, ya):.3f}")
    print(f"  R2  C(model)+C(target)            = {ols_r2(np.hstack([Dm, Dt]), ya):.3f}")
    print(f"  R2  C(model)+C(source)            = {ols_r2(np.hstack([Dm, Ds]), ya):.3f}")
    print(f"  R2  ... both + interference       = {ols_r2(np.hstack([Dm, Ds, Dt, I]), ya):.3f}")
    print("  per-TARGET AUC effect:")
    for a in ATTRS:
        sel = [ya[i] for i, r in enumerate(rows) if r["tgt"] == a]
        print(f"    {a:<10} {np.mean(sel):+.3f}")

    print("\n  per-attribute manifold dim (mean over models):")
    for a in ATTRS:
        ds_ = [mv["mdim"][a] for mv in models.values()]
        print(f"    {a:<10} mdim={np.mean(ds_):.2f} (range {min(ds_):.1f}-{max(ds_):.1f})")

    # ---- Part 2: capacity law ----
    print("\n== Part 2: capacity/packing law ==")
    print("  per-model: k, capacity C=sum(mdim)/k, mean off-diag interference, excess")
    pts_pair, pts_model = [], []
    for k, mv in sorted(models.items(), key=lambda kv: KEFF[kv[0]]):
        keff = KEFF[k]
        dims = mv["mdim"]
        C = sum(dims.values()) / keff
        M = mv["interf"]
        off = np.mean([M[i, j] for i in range(4) for j in range(4) if i != j])
        print(f"    {str(k):>4}  C={C:.3f}  off={off:.3f}  excess={off-1/keff:+.3f}")
        leaks = [r["leak"] for r in mv["rows"].values()]
        pts_model.append((C, off - 1 / keff, np.mean(leaks)))
        for a, b in itertools.combinations(ATTRS, 2):
            i, j = ATTRS.index(a), ATTRS.index(b)
            pred_rand = max(dims[a], dims[b]) / keff       # random-subspace overlap
            pred_dem = (dims[a] + dims[b]) / keff          # additive demand
            pts_pair.append((M[i, j], pred_rand, pred_dem))
    obs = np.array([p[0] for p in pts_pair])
    rnd = np.array([p[1] for p in pts_pair])
    dm_ = np.array([p[2] for p in pts_pair])
    def pear(x, y):
        return float(np.corrcoef(x, y)[0, 1])
    print(f"\n  36 (model,pair) points: obs interference vs predictors")
    print(f"    vs random-packing max(d_a,d_b)/k : r={pear(obs, rnd):.3f}  "
          f"R2={ols_r2(rnd[:, None], obs):.3f}  slope~{np.polyfit(rnd, obs, 1)[0]:.2f}")
    print(f"    vs additive demand (d_a+d_b)/k  : r={pear(obs, dm_):.3f}  "
          f"R2={ols_r2(dm_[:, None], obs):.3f}  slope~{np.polyfit(dm_, obs, 1)[0]:.2f}")
    print(f"    obs/random ratio per model-scale: "
          + ", ".join(f"{p[0]/max(p[1],1e-9):.1f}" for p in pts_pair[:6]))
    Cm = np.array([p[0] for p in pts_model])
    Em = np.array([p[1] for p in pts_model])
    Lm = np.array([p[2] for p in pts_model])
    print(f"\n  model level (n=6): excess interference vs C: r={pear(Cm, Em):.3f}; "
          f"mean-12-pair leak vs C: r={pear(Cm, Lm):.3f}")


if __name__ == "__main__":
    main()
