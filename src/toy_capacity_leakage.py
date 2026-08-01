#!/usr/bin/env python
"""Toy model: capacity -> overlap -> leakage, with the natural-variation law.

Minimal conditional-generation analogue of SM-ELF:
  data      x = sum_a z_a v_a + eps,  z_a in {-1,+1}, v_a orthonormal in R^d
  bottleneck c = W_e x in R^k (learned), generator x_hat = G(c) (MLP)
  residual  pathway: generation at fixed c adds fresh attribute noise for the
            variance the bottleneck does not carry, then SNAPS to the data
            manifold (tanh basin per attribute, sharpness beta_a): the analogue
            of decoding embeddings to tokens.
Per-attribute basin sharpness beta_a sets how easily incidental pressure flips
attribute a: shallow basins = high natural variation (the model's own samples
vary it) = the toy's fragile targets.

Protocol mirrors the paper: train per k; probe interference = mean |cos| of
attribute readout directions in code space; steer a decontaminated axis in code
space; leakage = off-target readout shift of generated samples. Measured
natural variation = std of off-target readout across residual resamples at
fixed c, alpha=0.

Signatures to reproduce:
 S1 interference rises monotonically as k shrinks
 S2 across k, mean leakage co-varies with interference
 S3 within a model, leak-into correlates with measured natural variation,
    not with pairwise code-direction overlap
CPU-friendly (numpy); minutes.
"""

import json
import numpy as np

rng = np.random.default_rng(0)

d, m = 64, 8
N = 4000
KS = [2, 3, 4, 6, 8, 12]
SIGMA_EPS = 0.05
# basin sharpness: attribute 0..7 from shallow (fragile) to steep (rigid)
BETAS = np.array([0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0])
ALPHA = 2.0
LR, EPOCHS, H = 3e-3, 400, 64


def make_data(n):
    z = rng.choice([-1.0, 1.0], size=(n, m))
    V = np.linalg.qr(rng.normal(size=(d, m)))[0]  # (d, m) orthonormal
    x = z @ V.T + SIGMA_EPS * rng.normal(size=(n, d))
    return z, x, V


def mlp_init(sizes):
    return [(rng.normal(size=(a, b)) * np.sqrt(2.0 / a), np.zeros(b))
            for a, b in zip(sizes[:-1], sizes[1:])]


def mlp_fwd(params, x):
    acts = [x]
    for i, (W, b) in enumerate(params):
        x = x @ W + b
        if i < len(params) - 1:
            x = np.tanh(x)
        acts.append(x)
    return x, acts


def mlp_grad(params, acts, gout):
    grads = []
    g = gout
    for i in reversed(range(len(params))):
        W, b = params[i]
        a_in = acts[i]
        gW = a_in.T @ g / len(a_in)
        gb = g.mean(0)
        gprev = g @ W.T
        if i > 0:
            gprev = gprev * (1 - acts[i] ** 2)
        grads.append((gW, gb))
        g = gprev
    return grads[::-1]


def train_model(k, z, x, V):
    """Bottleneck autoencoder: enc (linear d->k), gen (k->H->d)."""
    enc = mlp_init([d, k])
    gen = mlp_init([k, H, d])
    for ep in range(EPOCHS):
        idx = rng.permutation(len(x))[:1024]
        xb = x[idx]
        c, ea = mlp_fwd(enc, xb)
        xh, ga = mlp_fwd(gen, c)
        err = xh - xb
        ggrads = mlp_grad(gen, ga, 2 * err)
        # backprop through gen to code input
        g = 2 * err
        for i in reversed(range(len(gen))):
            W, _ = gen[i]
            g = g @ W.T
            if i > 0:
                g = g * (1 - ga[i] ** 2)
        egrads = mlp_grad(enc, ea, g)
        for (W, b), (gW, gb) in zip(gen, ggrads):
            W -= LR * gW; b -= LR * gb
        for (W, b), (gW, gb) in zip(enc, egrads):
            W -= LR * gW; b -= LR * gb
    return enc, gen


def snap(x_pre, V, betas, noise_scale, rng):
    """Residual pathway + manifold snap: per-attribute coordinate is pulled into
    a +-1 basin with sharpness beta_a after adding residual noise; off-manifold
    (non-attribute) coordinates pass through."""
    proj = x_pre @ V                      # (n, m) attribute coordinates
    resid = x_pre - proj @ V.T
    noisy = proj + noise_scale * rng.normal(size=proj.shape)
    snapped = np.tanh(betas[None, :] * noisy * 2.0)
    return snapped @ V.T + resid, snapped


def axes_and_readouts(c, x, z):
    """Difference-of-means axes in code space + data-space readouts."""
    u = {}
    for a in range(m):
        u[a] = c[z[:, a] > 0].mean(0) - c[z[:, a] < 0].mean(0)
        u[a] /= np.linalg.norm(u[a]) + 1e-12
    return u


def run_k(k, z, x, V):
    enc, gen = train_model(k, z, x, V)
    c, _ = mlp_fwd(enc, x)
    u = axes_and_readouts(c, x, z)
    # S1: probe interference (mean |cos| between code axes)
    cosm = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a != b:
                cosm[a, b] = abs(u[a] @ u[b])
    interference = cosm[np.triu_indices(m, 1)].mean()

    # natural variation: fixed codes, resample residual noise, std of readout
    idx = rng.permutation(len(x))[:200]
    c0 = c[idx]
    x_pre, _ = mlp_fwd(gen, c0)
    reads = []
    for r in range(24):
        _, snapped = snap(x_pre, V, BETAS, 0.6, rng)
        reads.append(snapped)
    reads = np.stack(reads)               # (R, n, m)
    natvar = reads.std(axis=0).mean(axis=0)  # (m,)

    # leakage: steer decontaminated axis of source a, measure off-target shift
    leak_into = np.zeros(m)
    counts = np.zeros(m)
    ctrl_sum, ctrl_n = 0.0, 0
    for a in range(m):
        # decontaminate against 3 other axes (the paper's core protocol),
        # not all m-1: projecting out m-1 axes annihilates the source axis
        # whenever k <= m-1 and the measurement degenerates.
        decon_set = [(a + j) % m for j in (1, 2, 3)]
        others = np.stack([u[b] for b in decon_set], 1)
        Q, _ = np.linalg.qr(others)
        ua = u[a] - Q @ (Q.T @ u[a])
        n_ = np.linalg.norm(ua)
        if n_ < 1e-6:
            continue
        ua /= n_
        for sgn in (+1, -1):
            cs = c0 + sgn * ALPHA * ua[None, :]
            xp, _ = mlp_fwd(gen, cs)
            g1, s1 = snap(xp, V, BETAS, 0.6, rng)
            g0, s0 = snap(x_pre, V, BETAS, 0.6, rng)
            shift = np.abs((s1 - s0).mean(0))  # (m,)
            ctrl_sum += shift[a]
            ctrl_n += 1
            for b in range(m):
                if b != a:
                    leak_into[b] += shift[b]
                    counts[b] += 1
    leak_into /= np.maximum(counts, 1)
    return dict(k=k, interference=float(interference),
                mean_leak=float(leak_into.mean()),
                ctrl=float(ctrl_sum / max(ctrl_n, 1)),
                leak_into=leak_into.tolist(), natvar=natvar.tolist(),
                pair_overlap=cosm.tolist())


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-12))


def main():
    z, x, V = make_data(N)
    out = [run_k(k, z, x, V) for k in KS]
    print(f"{'k':>4} {'interference':>13} {'mean leak':>10} {'ctrl':>7}")
    for o in out:
        print(f"{o['k']:>4} {o['interference']:>13.3f} {o['mean_leak']:>10.3f} "
              f"{o['ctrl']:>7.3f}")
    inter = np.array([o["interference"] for o in out])
    leak = np.array([o["mean_leak"] for o in out])
    ctrl = np.array([o["ctrl"] for o in out])
    valid = ctrl > 0.2  # steering must work at all (capacity-collapse analogue)
    print(f"\nS1 interference monotone decreasing in k: "
          f"{all(np.diff(inter) < 0)}")
    print(f"S2 corr(interference, leakage) across k, controllable models only "
          f"(ctrl>0.2, n={valid.sum()}): "
          f"r={np.corrcoef(inter[valid], leak[valid])[0,1]:.2f}")
    print(f"   collapse analogue: k with ctrl<=0.2 -> "
          f"{[int(o['k']) for o, v in zip(out, valid) if not v]} "
          f"(steering and leakage die together, cf. k=8 in the study)")
    rs_nat, rs_ov = [], []
    for o in out:
        li = np.array(o["leak_into"]); nv = np.array(o["natvar"])
        ov = np.array(o["pair_overlap"]).mean(0)
        rs_nat.append(spearman(nv, li))
        rs_ov.append(spearman(ov, li))
    print("S3 within-model Spearman(leak-into, natvar) per k: "
          + " ".join(f"{r:+.2f}" for r in rs_nat))
    print("   within-model Spearman(leak-into, code overlap) per k: "
          + " ".join(f"{r:+.2f}" for r in rs_ov))
    json.dump(out, open("paper/toy_capacity_leakage.json", "w"), indent=2)
    print("wrote paper/toy_capacity_leakage.json")


if __name__ == "__main__":
    main()
