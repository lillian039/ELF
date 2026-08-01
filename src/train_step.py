"""Per-device pmap'd training step for the ELF diffusion language model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.encoder_utils import encode_text
from utils.semantic_utils import compute_phi
from modules.model import apply_manifold_code
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond,
)


Array = jnp.ndarray


def _global_axis(vec, lab):
    """Difference-of-means axis of `vec` (b, D) by label `lab` (b,) in {-1,0,+1},
    over the GLOBAL batch via psum. Returns (axis, valid)."""
    lab = lab.astype(vec.dtype)
    pos = (lab > 0).astype(vec.dtype)[:, None]
    neg = (lab < 0).astype(vec.dtype)[:, None]
    ps = jax.lax.psum(jnp.sum(pos * vec, axis=0), "batch")
    pc = jax.lax.psum(jnp.sum(pos), "batch")
    ns = jax.lax.psum(jnp.sum(neg * vec, axis=0), "batch")
    nc = jax.lax.psum(jnp.sum(neg), "batch")
    a = ps / jnp.maximum(pc, 1.0) - ns / jnp.maximum(nc, 1.0)
    return a, (pc > 0) & (nc > 0)


def _global_proj_std(vec, w):
    """Std of vec @ w over the global batch (psum)."""
    p = vec @ w
    n = jax.lax.psum(jnp.sum(jnp.ones_like(p)), "batch")
    m1 = jax.lax.psum(jnp.sum(p), "batch") / jnp.maximum(n, 1.0)
    m2 = jax.lax.psum(jnp.sum(p * p), "batch") / jnp.maximum(n, 1.0)
    return jnp.sqrt(jnp.maximum(m2 - m1 ** 2, 1e-12))


def _decorrelation_loss(mu, sent, gender):
    """Squared cosine between the code-space sentiment and gender difference-of-means
    axes, computed over the GLOBAL batch via cross-device psum. Penalizing this
    pushes the two attributes onto orthogonal readout directions (mitigation).
    mu: (b, k) per-device codes; sent/gender: (b,) in {-1,0,+1}. Returns a scalar;
    zero on devices/steps where either class is empty."""
    def axis(lab):
        lab = lab.astype(mu.dtype)
        pos = (lab > 0).astype(mu.dtype)[:, None]
        neg = (lab < 0).astype(mu.dtype)[:, None]
        ps = jax.lax.psum(jnp.sum(pos * mu, axis=0), "batch")
        pc = jax.lax.psum(jnp.sum(pos), "batch")
        ns = jax.lax.psum(jnp.sum(neg * mu, axis=0), "batch")
        nc = jax.lax.psum(jnp.sum(neg), "batch")
        a = ps / jnp.maximum(pc, 1.0) - ns / jnp.maximum(nc, 1.0)
        return a, (pc > 0) & (nc > 0)
    us, ok_s = axis(sent)
    ug, ok_g = axis(gender)
    cos = jnp.sum(us * ug) / (jnp.linalg.norm(us) * jnp.linalg.norm(ug) + 1e-8)
    return jnp.where(ok_s & ok_g, cos ** 2, 0.0)


def train_step(
    state: TrainState,
    encoder_params: Dict,
    encoder_apply_fn,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std

    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    # Decorrelation regularizer (mitigation): needs per-example lexicon labels.
    use_decorr = getattr(config, "decorrelation_weight", 0.0) > 0
    # Counterfactual off-target consistency (generation-pathway mitigation):
    # steer the code along the batch sentiment axis in a second forward pass and
    # penalize movement of the reconstruction along the frozen gender direction.
    use_cf = getattr(config, "manifold_cf_weight", 0.0) > 0
    need_labels = use_decorr or use_cf
    sent_lab = batch["sent_label"] if need_labels else None
    gender_lab = batch["gender_label"] if need_labels else None

    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng, jax.lax.axis_index(axis_name="batch"))
    (
        t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng,
        model_dropout_rng, decoder_step_rng, decoder_rng,
        decoder_lambda_rng, decoder_noise_rng, cf_rng,
    ) = jax.random.split(current_step_rng, 10)

    # encoder_attention_mask: cond sees cond, x sees all
    encoder_attention_mask = batch["encoder_attention_mask"]

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]  # (B, 1, 1)
        cond_mask = batch["cond_seq_mask"]  # (B, S)
        # block_mask is 1 only at (non-cond row, cond col) — leaves cond↔cond unchanged
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )

    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    cond_seq_mask = batch["cond_seq_mask"][:, :, None]
    attention_mask = batch["attention_mask"]
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

    # Semantic-manifold factorization: phi(s) is the masked mean of x0 over loss
    # positions; the flow then denoises the residual r0 = x0 - phi rather than x0.
    # The decoder (CE) branch still operates on the full x0.
    if config.semantic_factorization:
        phi = compute_phi(x0, loss_mask)   # (B, 1, C) pooled latent
        phi_vec = phi[:, 0, :]             # (B, C) conditioning input to the model
        # Carrier routing (ablation): which component sees phi during training.
        phi_den = phi_vec if config.phi_route in ("both", "denoiser") else None
        phi_dec = phi_vec if config.phi_route in ("both", "decoder") else None
        if config.manifold_dim > 0:
            # M2: phi(s) = U c is the low-rank lift. Use a stop-grad lift for the
            # residual target so the manifold encoder is trained through the
            # conditioning-prefix / cycle / IB paths, not by making its own
            # regression target easier to hit.
            phi_lift_sg, _, _ = apply_manifold_code(
                state.params["manifold"], phi_vec, config.manifold_dim, x0.shape[-1])
            flow_target = x0 - jax.lax.stop_gradient(phi_lift_sg)[:, None, :]
        else:
            flow_target = x0 - phi          # M1: mean-pool residual
    else:
        phi_vec = None
        phi_den = phi_dec = None
        flow_target = x0

    # Counterfactual-consistency statistics: all cross-device collectives happen
    # HERE, unconditionally, outside jax.lax.cond (see the decorrelation comment
    # below for why). Axes/scales are data statistics -> stop-gradient.
    cf_stats = None
    if use_cf:
        assert config.semantic_factorization and config.manifold_dim > 0, \
            "manifold_cf_weight requires an M2 model"
        pooled_sg = jax.lax.stop_gradient(phi_vec)
        _, mu0, _ = apply_manifold_code(
            jax.lax.stop_gradient(state.params["manifold"]), pooled_sg,
            config.manifold_dim, x0.shape[-1])
        u_s, ok_s = _global_axis(mu0, sent_lab)
        u_g, ok_g = _global_axis(mu0, gender_lab)
        u_gn = u_g / (jnp.linalg.norm(u_g) + 1e-8)
        u_perp = u_s - jnp.dot(u_s, u_gn) * u_gn
        u_perp = u_perp / (jnp.linalg.norm(u_perp) + 1e-8)
        w_g, _ = _global_axis(pooled_sg, gender_lab)
        w_gn = w_g / (jnp.linalg.norm(w_g) + 1e-8)
        w_s, _ = _global_axis(pooled_sg, sent_lab)
        w_sn = w_s / (jnp.linalg.norm(w_s) + 1e-8)
        s_g = _global_proj_std(pooled_sg, w_gn)
        s_s = _global_proj_std(pooled_sg, w_sn)
        sign_rng, mag_rng = jax.random.split(cf_rng)
        alpha = (jnp.where(jax.random.bernoulli(sign_rng, 0.5, (x0.shape[0], 1)), 1.0, -1.0)
                 * jax.random.uniform(
                     mag_rng, (x0.shape[0], 1),
                     minval=getattr(config, "manifold_cf_alpha_min", 1.0),
                     maxval=getattr(config, "manifold_cf_alpha_max", 3.0)))
        cf_stats = dict(u_perp=u_perp, w_gn=w_gn, w_sn=w_sn, s_g=s_g, s_s=s_s,
                        alpha=alpha, ok=ok_s & ok_g)

    denoiser_z = add_noise(flow_target, noise, t, config, cond_seq_mask=cond_seq_mask)

    drop = batch["label_drop_mask"][:, None]
    if config.label_drop_prob > 0:
        zero_cond = drop[:, :, None] & (cond_seq_mask > 0)
        denoiser_z = jnp.where(zero_cond, jnp.zeros_like(denoiser_z), denoiser_z)
        x0 = jnp.where(zero_cond, jnp.zeros_like(x0), x0)
        flow_target = jnp.where(zero_cond, jnp.zeros_like(flow_target), flow_target)

    decoder_targets = batch["input_ids"]  # (B, S)
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_lambda_rng, decoder_noise_rng = jax.random.split(decoder_rng)
    decoder_z_vals = (
        jax.random.normal(decoder_lambda_rng, (batch_size * seq_length,))
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = jax.nn.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = jax.random.normal(decoder_noise_rng, x0.shape, dtype=x0.dtype) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (flow_target - denoiser_z) / jnp.maximum(1 - t_expanded, t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < self_cond_prob)
            .reshape(-1, 1, 1).astype(x0.dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng, batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    def get_z_input(params, z, t_input, self_cond_cfg_input, x_tokens):
        # Self-conditioning: with probability self_cond_prob, compute initial estimate
        if self_cond_prob == 0:
            return z
        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_seq_mask)
        z_with_zeros = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_init = state.apply_fn(
            {"params": params}, z_with_zeros, t_input,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
            phi=phi_den,
        )
        net_out_init = jax.lax.stop_gradient(net_out_init)
        _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
        x_pred_init = restore_cond(x_pred_init, x_tokens, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.astype(z.dtype)
        x_pred_cond = restore_cond(x_pred_cond, x_tokens, cond_seq_mask)
        return jnp.concatenate([z, x_pred_cond], axis=-1)

    def reduce_token_loss(per_token_loss, loss_mask):
        loss_mask = loss_mask.astype(per_token_loss.dtype)
        safe_loss = jnp.where(loss_mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe_loss * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)

    def get_sc_cond_and_uncond(params, z, t, cond_mask, x_tokens):
        kwargs = {
            "self_cond_cfg_scale": self_cond_cfg_scale,
            "deterministic": True,
        }
        if config.self_cond_prob == 0:
            net_out_uncod = state.apply_fn({"params": params}, z, t, phi=phi_den, **kwargs)
            v_uncond, _ = net_out_to_v_x(net_out_uncod, z, t, t_eps)
            return v_uncond, v_uncond

        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_mask)
        z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_uncond = state.apply_fn({"params": params}, z_input_uncond, t, phi=phi_den, **kwargs)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = jnp.concatenate([z, x_uncond], axis=-1)
        net_out_cond = state.apply_fn({"params": params}, z_input_cond, t, phi=phi_den, **kwargs)
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t, t_eps)
        return v_cond, v_uncond

    def get_sc_guided_v(params, z, t, base_v_target, x_tokens):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond = get_sc_cond_and_uncond(
            params, z, t, cond_mask=cond_seq_mask, x_tokens=x_tokens
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = jnp.where(use_self_cond_mask, sc_guidance, jnp.zeros_like(sc_guidance))
        return jax.lax.stop_gradient(base_v_target + sc_guidance)

    def get_v_target(params, z, t, base_v_target, x_tokens):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(params, z, t, base_v_target=base_v_target, x_tokens=x_tokens)
        return base_v_target

    def loss_fn(params):

        def _decoder_branch(_):
            # Decoder mode: encoder-noised latent (decoder_z) at t=1, CE loss on tokens.
            decoder_t = jnp.ones_like(t)
            decoder_input = (
                jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
                if config.self_cond_prob > 0 else decoder_z
            )
            _, decoder_logits = state.apply_fn(
                {"params": params}, decoder_input, decoder_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(True),
                phi=phi_dec,
            )
            log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
            ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
            return (ce_loss, ce_loss, jnp.zeros(()), jnp.zeros(()), jnp.zeros(()),
                    jnp.zeros(()), jnp.zeros(()))

        def _denoiser_branch(_):
            # Denoiser mode: x0-noised latent (denoiser_z) at random t, L2 loss on velocity.
            denoiser_t = t
            denoiser_input = get_z_input(
                params, denoiser_z, denoiser_t,
                self_cond_cfg_input=self_cond_cfg_scale,
                x_tokens=flow_target,
            )
            net_out, _ = state.apply_fn(
                {"params": params}, denoiser_input, denoiser_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(False),
                phi=phi_den,
            )
            v_pred, x_pred = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
            v_final_target = get_v_target(
                params, denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=flow_target,
            )
            per_dim_loss = (v_pred - v_final_target) ** 2
            l2_loss = reduce_token_loss(jnp.mean(per_dim_loss, axis=-1), loss_mask)

            # M2: information-bottleneck KL on the code + semantic-consistency (cycle).
            cyc_loss = jnp.zeros(())
            ib_loss = jnp.zeros(())
            cf_loss = jnp.zeros(())
            cf_on = jnp.zeros(())
            if config.semantic_factorization and config.manifold_dim > 0:
                mdim, edim = config.manifold_dim, x0.shape[-1]
                phi_lift_g, mu, logvar = apply_manifold_code(params["manifold"], phi_vec, mdim, edim)
                ib_loss = 0.5 * jnp.mean(mu ** 2 + jnp.exp(logvar) - logvar - 1.0)
                # x_hat = phi + r_hat; re-probe its code and match the conditioning code.
                x_hat = phi_lift_g[:, None, :] + x_pred
                pooled_hat = compute_phi(x_hat, loss_mask)[:, 0, :]
                _, mu_hat, _ = apply_manifold_code(params["manifold"], pooled_hat, mdim, edim)
                cyc_loss = jnp.mean((mu_hat - jax.lax.stop_gradient(mu)) ** 2)

                if use_cf:
                    # Counterfactual pass: steer the code along the batch
                    # sentiment axis (gender projected out) and demand the
                    # reconstruction's gender reading stays put. Reuses the
                    # factual self-cond input (its estimate is stop-gradient
                    # anyway); no collectives in here (see cf_stats above).
                    # The manifold side is fully stop-gradient: the penalty may
                    # only reshape the DENOISER's response (the localized site
                    # of the leak). Without this the code stack games the
                    # penalty by inflating ||mu|| until the fixed-magnitude
                    # steer is relatively negligible (observed as an exploding
                    # IB loss); alpha is also scaled by the codes' RMS so the
                    # perturbation is scale-invariant by construction.
                    # Constant-scale alpha (the IB anchors code RMS near 1):
                    # a live RMS scale re-couples the steer magnitude to code
                    # norm and feeds a cyc-driven runaway (observed: ib and cf
                    # climbing together from ~step 4k).
                    U = jax.lax.stop_gradient(params["manifold"]["lift"]["kernel"])  # (k, d)
                    mu_sg = jax.lax.stop_gradient(mu)
                    c_cf = mu_sg + cf_stats["alpha"] * cf_stats["u_perp"][None, :]
                    phi_cf = c_cf @ U
                    net_out_cf, _ = state.apply_fn(
                        {"params": params}, denoiser_input, denoiser_t,
                        deterministic=False,
                        rngs={"dropout": model_dropout_rng},
                        self_cond_cfg_scale=self_cond_cfg_scale,
                        decoder_step_active=jnp.array(False),
                        phi=phi_cf, phi_lifted=True,
                    )
                    _, x_pred_cf = net_out_to_v_x(net_out_cf, denoiser_z, denoiser_t, t_eps)
                    x_hat_cf = phi_cf[:, None, :] + x_pred_cf
                    pooled_cf = compute_phi(x_hat_cf, loss_mask)[:, 0, :]
                    d_off = ((pooled_cf - jax.lax.stop_gradient(pooled_hat))
                             @ cf_stats["w_gn"]) / (cf_stats["s_g"] + 1e-8)
                    # Warmup gate: no CF pressure until the base denoiser is
                    # formed; pressing an untrained denoiser destabilizes.
                    warm = state.step >= getattr(config, "manifold_cf_warmup_steps", 0)
                    cf_loss = jnp.where(cf_stats["ok"] & warm, jnp.mean(d_off ** 2), 0.0)
                    # Diagnostic only (not a loss): signed on-target movement of
                    # the counterfactual; collapse toward 0 = steerability loss.
                    d_on = ((pooled_cf - pooled_hat) @ cf_stats["w_sn"]) / (cf_stats["s_s"] + 1e-8)
                    cf_on = jnp.where(
                        cf_stats["ok"],
                        jnp.mean(jnp.sign(cf_stats["alpha"][:, 0]) * d_on), 0.0)
                    cf_on = jax.lax.stop_gradient(cf_on)

            total = (l2_loss + config.cycle_loss_weight * cyc_loss + config.ib_beta * ib_loss
                     + getattr(config, "manifold_cf_weight", 0.0) * cf_loss)
            return total, jnp.zeros(()), l2_loss, cyc_loss, ib_loss, cf_loss, cf_on

        loss, ce_loss, l2_loss, cyc_loss, ib_loss, cf_loss, cf_on = jax.lax.cond(
            decoder_step_active, _decoder_branch, _denoiser_branch, None,
        )
        # Decorrelation regularizer: computed OUTSIDE the cond so its cross-device
        # psum runs on every device every step. A collective inside jax.lax.cond
        # deadlocks because decoder_step_active is sampled per-device, so devices
        # take different branches and the psum is only reached by some of them.
        dec_loss = jnp.zeros(())
        if use_decorr and config.semantic_factorization and config.manifold_dim > 0:
            _, mu_all, _ = apply_manifold_code(
                params["manifold"], phi_vec, config.manifold_dim, x0.shape[-1])
            dec_loss = _decorrelation_loss(mu_all, sent_lab, gender_lab)
        loss = loss + config.decorrelation_weight * dec_loss
        return loss, (l2_loss, ce_loss, cyc_loss, ib_loss, dec_loss, cf_loss, cf_on)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val, cyc_loss_val, ib_loss_val, dec_loss_val,
            cf_loss_val, cf_on_val)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")
    ce_loss_val = jax.lax.pmean(ce_loss_val, axis_name="batch")
    cyc_loss_val = jax.lax.pmean(cyc_loss_val, axis_name="batch")
    ib_loss_val = jax.lax.pmean(ib_loss_val, axis_name="batch")
    dec_loss_val = jax.lax.pmean(dec_loss_val, axis_name="batch")
    cf_loss_val = jax.lax.pmean(cf_loss_val, axis_name="batch")
    cf_on_val = jax.lax.pmean(cf_on_val, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    # Update EMA only on actual optimizer steps, not on gradient accumulation steps.
    # With optax.MultiSteps, params only change every grad_accum_steps mini-batches; updating
    # EMA every mini-batch would make effective decay decay^grad_accum_steps instead of decay.
    def ema_update(ema_params, params, decay):
        return jax.tree_util.tree_map(lambda e, p: e * decay + p * (1 - decay), ema_params, params)

    is_optimizer_step = (new_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(state.ema_params1, new_state.params, config.ema_decay1),
        lambda: state.ema_params1,
    )
    new_state = new_state.replace(ema_params1=new_ema_params1, dropout_rng=new_dropout_rng)

    # Rescale per-branch losses by their sampling probability so they reflect the
    # per-branch loss rather than the expected loss conditioned on the branch firing.
    decoder_prob_arr = jnp.asarray(decoder_prob, dtype=jnp.float32)
    denoiser_prob_arr = jnp.asarray(1.0 - decoder_prob, dtype=jnp.float32)
    active_ce_loss_val = jnp.where(
        decoder_prob_arr > 0.0, ce_loss_val / decoder_prob_arr, jnp.zeros_like(ce_loss_val),
    )
    active_l2_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, l2_loss_val / denoiser_prob_arr, jnp.zeros_like(l2_loss_val),
    )
    active_cyc_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, cyc_loss_val / denoiser_prob_arr, jnp.zeros_like(cyc_loss_val),
    )
    active_ib_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, ib_loss_val / denoiser_prob_arr, jnp.zeros_like(ib_loss_val),
    )
    active_dec_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, dec_loss_val / denoiser_prob_arr, jnp.zeros_like(dec_loss_val),
    )
    active_cf_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, cf_loss_val / denoiser_prob_arr, jnp.zeros_like(cf_loss_val),
    )
    active_cf_on_val = jnp.where(
        denoiser_prob_arr > 0.0, cf_on_val / denoiser_prob_arr, jnp.zeros_like(cf_on_val),
    )
    metrics = {
        "loss": loss,
        "l2_loss": active_l2_loss_val,
        "ce_loss": active_ce_loss_val,
        "cyc_loss": active_cyc_loss_val,
        "ib_loss": active_ib_loss_val,
        "dec_loss": active_dec_loss_val,
        "cf_loss": active_cf_loss_val,
        "cf_on": active_cf_on_val,
    }
    return new_state, metrics
