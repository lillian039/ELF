"""Progressive distillation (PD) + CE training step for ELF.

Each example takes one of two branches:
  1. PD: the student covers a span of pd_step_size in one forward, with t_s
     drawn from pd_student_schedule. The teacher builds the target with
     pd_teacher_steps Euler sub-steps over [t_s, t_t] and inverts back to an
     x_pred at t_s.
  2. CE: a decoder_prob fraction of examples train the decoder head instead,
     denoising decoder_z at t=1.
"""

import contextlib
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distill_utils import (
    build_z_t_s,
    teacher_pd_target,
    teacher_uncond_x_pred_init,
)
from utils.encoder_utils import encode_text
from utils.sampling_utils import sample_cfg_scale, sample_timesteps
from utils.train_utils import TrainState, ema_update


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def distill_train_step(
    state: TrainState,
    encoder: nn.Module,
    teacher: nn.Module,
    batch: Dict[str, torch.Tensor],
    config,
) -> Tuple[TrainState, Dict[str, torch.Tensor]]:
    device = next(state.model.parameters()).device
    dtype = next(state.model.parameters()).dtype
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    num_sc_cfg = config.num_self_cond_cfg_tokens

    # ---- encoder pipeline (same as train_step.py) ----------
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)

    x0 = encode_text(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean, latent_std=config.latent_std,
        use_bf16=use_bf16,
    ).to(dtype)
    B, S, _ = x0.shape

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1.0 - cond_seq_mask)
    cond_seq_mask_e = cond_seq_mask.unsqueeze(-1)

    # ---- per-example branch coin: CE vs PD -----------------
    decoder_mask = torch.bernoulli(
        torch.full((B,), float(config.decoder_prob), dtype=dtype, device=device)
    )
    denoiser_mask = 1.0 - decoder_mask
    n_ce = decoder_mask.sum()
    n_pd = denoiser_mask.sum()

    # ---- per-example (t_s, t_t) ----------------------------
    # PD examples: draw t_s from pd_student_schedule and rescale into
    # [0, 1 - pd_step_size] so the span is exactly pd_step_size and t_t <= 1.
    # CE examples use (t_s, t_t) = (1, 1).
    step_size = float(config.pd_step_size)
    if step_size <= 0.0:
        raise ValueError("distill_train_step requires pd_step_size > 0.")
    student_sched = str(config.pd_student_schedule)
    u_int = sample_timesteps(
        B,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=student_sched,
        device=device, dtype=dtype,
    )
    t_s_int = u_int * max(1.0 - step_size, 1e-6)
    t_t_int = t_s_int + step_size

    ones_t = torch.ones((B,), dtype=dtype, device=device)
    t_s = decoder_mask * ones_t + denoiser_mask * t_s_int
    t_t = decoder_mask * ones_t + denoiser_mask * t_t_int

    # ---- per-example SC-CFG scale --------------------------
    cfg_min = float(config.self_cond_cfg_min)
    cfg_max = float(config.self_cond_cfg_max)
    if num_sc_cfg > 0 and cfg_max > cfg_min:
        # sample_cfg_scale returns values in [cfg_min, cfg_max] log-uniform.
        # Pass self_cond_cfg_{min,max} as-is (not shifted by 1) to match inference.
        w = sample_cfg_scale(
            B,
            cfg_min=cfg_min, cfg_max=cfg_max,
            dtype=dtype, device=device,
        )
    else:
        w = torch.full((B,), cfg_min, dtype=dtype, device=device)

    # ---- decoder branch input: latent noised at t=1 --------
    decoder_z_vals = (
        torch.randn((B * S,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(B, S, 1)
    decoder_noise = torch.randn(x0.shape, dtype=dtype, device=device) * config.decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1.0 - decoder_lambda_t) * decoder_noise

    # ---- z_t_s for PD examples, decoder_z for CE examples --
    z_t_s = build_z_t_s(x0, t_s, config.denoiser_noise_scale, cond_seq_mask_e)
    dec_e = decoder_mask.reshape(-1, 1, 1)
    den_e = denoiser_mask.reshape(-1, 1, 1)
    x_combined = dec_e * decoder_z + den_e * z_t_s

    sc_arg = w if num_sc_cfg > 0 else None

    # ---- self-cond (mirrors train_step.py's get_z_input) ---
    # On a self_cond_prob fraction of PD examples, use the teacher's uncond
    # x_pred at (z_t_s, t_s) as the self-cond half. The same sc_half feeds both
    # the student forward and the teacher target so they stay consistent.
    # CE examples always use the zero self-cond half.
    self_cond_prob = float(config.self_cond_prob)
    if self_cond_prob > 0.0:
        sc_coin = torch.bernoulli(
            torch.full((B,), self_cond_prob, dtype=dtype, device=device)
        )
        use_sc = sc_coin * denoiser_mask  # zero on CE examples
        t_x_pred_init = teacher_uncond_x_pred_init(
            teacher=teacher, z_t_s=z_t_s, t_s=t_s, sc_cfg_scale=w,
            x0=x0, cond_seq_mask_e=cond_seq_mask_e,
            num_self_cond_cfg_tokens=num_sc_cfg, use_bf16=use_bf16,
        )
        sc_half_raw = t_x_pred_init * use_sc.reshape(-1, 1, 1).to(t_x_pred_init.dtype)
        sc_half = torch.where(cond_seq_mask_e > 0, x0, sc_half_raw)
    else:
        use_sc = torch.zeros((B,), dtype=dtype, device=device)
        sc_half = torch.where(cond_seq_mask_e > 0, x0, torch.zeros_like(x_combined))

    # ---- teacher sub-step times ----------------------------
    # Build num_teacher_steps+1 times per example in [t_s, t_t]: sample the
    # interior points from pd_teacher_schedule, sort them, and rescale into the
    # span. CE examples (t_s=t_t=1) collapse to all ones.
    num_teacher_steps = int(config.pd_teacher_steps)
    sub_step_times = None
    if num_teacher_steps > 1:
        teacher_sched = str(config.pd_teacher_schedule)
        span_row = (t_t - t_s).reshape(-1, 1)
        t_s_row = t_s.reshape(-1, 1)
        u_t = sample_timesteps(
            B * (num_teacher_steps - 1),
            P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            time_schedule=teacher_sched,
            device=device, dtype=dtype,
        ).reshape(B, num_teacher_steps - 1)
        u_t = torch.sort(u_t, dim=-1).values
        interior_t = t_s_row + span_row * u_t
        sub_step_times = torch.cat([t_s_row, interior_t, t_t.reshape(-1, 1)], dim=-1)

    # ---- teacher PD target ---------------------------------
    # The teacher uses sc_half on the first sub-step and chains its own x_pred
    # on later sub-steps; see teacher_pd_target.
    target_pd = teacher_pd_target(
        teacher=teacher, x0=x0, cond_seq_mask_e=cond_seq_mask_e,
        z_t_s=z_t_s, t_s=t_s, t_t=t_t, sc_cfg_scale=w, sc_half=sc_half,
        num_steps=num_teacher_steps,
        num_self_cond_cfg_tokens=num_sc_cfg,
        t_eps=config.t_eps,
        use_bf16=use_bf16,
        sub_step_times=sub_step_times,
    )

    x_input = torch.cat([x_combined, sc_half], dim=-1)

    # ---- student forward (per-example gate) ----------------
    state.model.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        x_pred, decoder_logits = state.model(
            x_input, t_s,
            deterministic=False,
            self_cond_cfg_scale=sc_arg,
            decoder_step_active=decoder_mask,
        )

    # ---- CE on decoder examples ----------------------------
    log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
    ce = -log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    lm = loss_mask.to(ce.dtype)
    ce_per_example = (ce * lm).sum(dim=-1) / torch.clamp(lm.sum(dim=-1), min=1.0)

    # ---- MSE on PD examples --------------------------------
    per_dim = (x_pred.float() - target_pd.float()) ** 2
    per_token = per_dim.mean(dim=-1)
    pd_per_example = (per_token * lm).sum(dim=-1) / torch.clamp(lm.sum(dim=-1), min=1.0)

    # ---- combined per-example loss -------------------------
    per_example_loss = decoder_mask * ce_per_example + denoiser_mask * pd_per_example
    loss = per_example_loss.mean()

    accum_steps = max(config.grad_accum_steps, 1)
    state.step += 1
    is_optimizer_step = (state.step % accum_steps) == 0

    sync_ctx = (
        state.model.no_sync()
        if (not is_optimizer_step and hasattr(state.model, 'no_sync'))
        else contextlib.nullcontext()
    )
    with sync_ctx:
        (loss / accum_steps).backward()

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(state.model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        ema_update(state.ema_params1, state.model, config.ema_decay1)
        state.optimizer.zero_grad(set_to_none=True)

    ce_count = torch.clamp(n_ce, min=1.0)
    pd_count = torch.clamp(n_pd, min=1.0)
    metrics = {
        "loss": loss.detach(),
        "ce_loss": ((decoder_mask * ce_per_example).sum() / ce_count).detach(),
        "l2_loss": ((denoiser_mask * pd_per_example).sum() / pd_count).detach(),
    }
    return state, metrics
