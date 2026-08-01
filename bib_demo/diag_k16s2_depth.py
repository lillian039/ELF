#!/usr/bin/env python3
"""Discriminate why depth-window results for BiB k16_s2 are exactly zero:
(a) decontamination annihilates the steer axis (sent/gender collinear in its
16-d code), vs (b) the model's attention to phi tokens underflows (severed
read). Prints axis geometry, lifted steer norm, and single-forward phi
sensitivity."""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import bib_eval_shim  # noqa: F401,E402
import copy  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from modules.model import ELF_models, apply_manifold_code  # noqa: E402
from modules.t5_encoder import get_encoder  # noqa: E402
from utils.checkpoint_utils import load_checkpoint, load_encoder_checkpoint  # noqa: E402
from utils.train_utils import TrainState  # noqa: E402
from utils.data_utils import load_dataset_split, get_pad_token_id  # noqa: E402
from utils.semantic_utils import compute_phi  # noqa: E402
from configs.config import load_config_from_yaml, apply_config_overrides  # noqa: E402
from eval_steering import _pad_batch, _encode  # noqa: E402
from eval_leakage_pairs import ATTRS, attr_value, pos_neg_masks, fit_axis, decontaminate  # noqa: E402

jax.config.update("jax_default_matmul_precision", "float32")

CKPT = sys.argv[1]

cfg = load_config_from_yaml("bib_demo/train_bib_SM-ELF-M2.yml")
cfg = apply_config_overrides(cfg, ["manifold_dim=16"])
tok = AutoTokenizer.from_pretrained("t5-small")
pad_id = get_pad_token_id(tok, cfg.pad_token)
enc_cfg, enc_model, _ = get_encoder("t5-small", jnp.float32)
enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
L, d = cfg.max_length, 512


def build(k):
    return ELF_models[cfg.model](
        text_encoder_dim=d, max_length=L, attn_drop=0., proj_drop=0.,
        num_time_tokens=4, num_self_cond_cfg_tokens=4, vocab_size=tok.vocab_size,
        num_model_mode_tokens=4, num_phi_tokens=4, manifold_dim=k,
        bottleneck_dim=cfg.bottleneck_dim)


m2, m0 = build(16), build(0)
rng = jax.random.PRNGKey(0)
pi = m2.init(rng, x=jnp.ones((1, L, 1024)), t=jnp.ones((1,)), deterministic=True,
             self_cond_cfg_scale=jnp.ones((1,)), phi=jnp.ones((1, d)))
st = TrainState.create(apply_fn=m2.apply, params=pi["params"], tx=optax.adamw(1e-4),
                       dropout_rng=rng, ema_params1=copy.deepcopy(pi["params"]))
st, _ = load_checkpoint(CKPT, st)
params = st.params
m0p = {k: v for k, v in params.items() if k != "manifold"}
U = np.asarray(params["manifold"]["lift"]["kernel"])

val = load_dataset_split(cfg.eval_data_path)
raw = [val[i]["input_ids"] for i in range(400)]
texts = [tok.decode(np.asarray(r), skip_special_tokens=True) for r in raw]
ids, valid = _pad_batch(raw, L, pad_id)
mus = []
for s in range(0, 400, 64):
    x0 = _encode(ids[s:s + 64], valid[s:s + 64], enc_model.apply, enc_params, cfg)
    pooled = compute_phi(x0, valid[s:s + 64])[:, 0, :]
    _, mu_b, _ = apply_manifold_code(params["manifold"], pooled, 16, d)
    mus.append(np.asarray(mu_b))
mu = np.concatenate(mus, 0)
labels = {a: np.array([attr_value(a, t) for t in texts]) for a in ATTRS}
masks = {a: pos_neg_masks(a, labels[a]) for a in ATTRS}
axes = {a: fit_axis(mu, *masks[a]) for a in ATTRS}
u = decontaminate(axes["sentiment"], [axes[b] for b in ATTRS if b != "sentiment"])
cos = float(axes["sentiment"] @ axes["gender"])
print(f"AXES |sent|={np.linalg.norm(axes['sentiment']):.4f} "
      f"|gender|={np.linalg.norm(axes['gender']):.4f} cos(s,g)={cos:+.4f} "
      f"|u_decon|={np.linalg.norm(u):.6f}")
print(f"LIFT |U^T(3u)|={np.linalg.norm(3 * (u @ U)):.4f}  mu_std={mu.std():.4f}")

c0 = mu.mean(0)
phi0 = jnp.asarray((c0 @ U)[None, :].astype(np.float32))
phiR = phi0 + 10.0 * jax.random.normal(jax.random.PRNGKey(7), phi0.shape)
x = jax.random.normal(jax.random.PRNGKey(8), (1, L, 1024))
t = jnp.full((1,), 0.5)
scc = jnp.full((1,), 1.0)
o1, _ = m0.apply({"params": m0p}, x=x, t=t, deterministic=True,
                 self_cond_cfg_scale=scc, phi=phi0)
o2, _ = m0.apply({"params": m0p}, x=x, t=t, deterministic=True,
                 self_cond_cfg_scale=scc, phi=phiR)
print(f"PHI-SENSITIVITY max|out(phi0)-out(phi0+10N)|={float(jnp.max(jnp.abs(o1 - o2))):.3e}")
