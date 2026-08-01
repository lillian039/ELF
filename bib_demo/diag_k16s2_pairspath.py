#!/usr/bin/env python3
"""Decisive k16_s2 test: replicate the pairs/continuous evals' EXACT
generation path (_generate_samples_single_batch, m0 view, lifted phi) with
paired noise at alpha = -3, 0, +3. If outputs are bitwise identical across
alpha, the model is phi-inert on this path too, and the round-1/2 pairs
numbers for this model are unpaired-noise artifacts; if they differ, the
depth eval's run() deviates from the production path somewhere."""
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
from utils.sampling_utils import get_sampling_steps  # noqa: E402
from utils.generation_utils import _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos  # noqa: E402
from configs.config import load_config_from_yaml, apply_config_overrides  # noqa: E402
from eval_steering import _pad_batch, _encode  # noqa: E402
from eval_leakage_pairs import ATTRS, attr_value, pos_neg_masks, fit_axis, decontaminate  # noqa: E402

CKPT = sys.argv[1]
cfg = load_config_from_yaml("bib_demo/train_bib_SM-ELF-M2.yml")
cfg = apply_config_overrides(cfg, ["manifold_dim=16"])
tok = AutoTokenizer.from_pretrained("t5-small")
pad_id = get_pad_token_id(tok, cfg.pad_token)
eos_id = tok.eos_token_id or 1
enc_cfg, enc_model, _ = get_encoder("t5-small", jnp.float32)
enc_params = load_encoder_checkpoint(cfg.encoder_checkpoint)
L, d = cfg.max_length, 512
sc = cfg.sampling_configs[0]
steps = sc.num_sampling_steps[0] if isinstance(sc.num_sampling_steps, list) else sc.num_sampling_steps
sccfg = sc.self_cond_cfg_scales[0] if isinstance(sc.self_cond_cfg_scales, list) else sc.self_cond_cfg_scales


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
c0 = mu.mean(0)

M = 8
gen_rng = jax.random.PRNGKey(123)
z = jax.random.normal(gen_rng, (M, L, d)) * cfg.denoiser_noise_scale
t_steps = get_sampling_steps(jax.random.PRNGKey(5), n_steps=steps,
                             time_schedule=sc.time_schedule,
                             P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std)
outs = {}
for a in (-3.0, 0.0, 3.0):
    phi_vec = ((c0 + a * u) @ U).astype(np.float32)
    phi_lift = jnp.asarray(np.repeat(phi_vec[None, :], M, axis=0))
    latent = _generate_samples_single_batch(
        model_params=m0p, model_apply_fn=m0.apply, rng=gen_rng,
        z=z, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
        config=cfg, sampling_config=sc, cfg_scale=1.0,
        self_cond_cfg_scale=sccfg, phi=phi_lift)
    outs[a] = np.asarray(latent)
    pred = np.asarray(mask_after_eos(_dlm_decode_batch(
        z=latent, model_params=m0p, model_apply_fn=m0.apply,
        t_final_val=float(t_steps[-1]), config=cfg,
        self_cond_cfg_scale=sccfg, phi=phi_lift), eos_id, pad_id))
    txt = tok.decode(pred[0], skip_special_tokens=True)
    print(f"alpha={a:+.0f}: latent_mean={outs[a].mean():+.5f} text[:80]={txt[:80]!r}")

print(f"PAIRSPATH max|out(+3)-out(-3)| = {np.max(np.abs(outs[3.0] - outs[-3.0])):.3e}")
print(f"PAIRSPATH max|out(0)-out(-3)|  = {np.max(np.abs(outs[0.0] - outs[-3.0])):.3e}")
