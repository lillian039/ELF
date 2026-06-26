"""Teacher-distillation utilities for the ELF pipeline.

The student keeps the teacher's architecture (single time `t`) and is fine-tuned
with two objectives, routed per-example:

  * Denoiser PD : MSE between the student's `x_pred` over a step [t_s, t_t] and
                  the teacher's progressive-distillation target (the implied
                  `x_pred` from integrating the teacher across that step).
  * Decoder CE  : standard cross-entropy on tokens at `t = 1`, identical to
                  the CE branch in `train_step.py`.
"""

import torch
import torch.nn as nn

from modules.model import ELF_models
from utils.checkpoint_utils import _restore_checkpoint
from utils.logging_utils import log_for_0


def _strip_orig_mod(sd: dict) -> dict:
    """Drop the `_orig_mod.` prefix that torch.compile adds to state-dict keys."""
    return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}


def load_teacher_model(
    teacher_path: str,
    text_encoder_dim: int,
    vocab_size: int,
    config,
    device: torch.device,
) -> nn.Module:
    """Build a frozen ELF teacher and load weights from `teacher_path`.

    Prefers EMA weights (`ema_params1`) when `config.teacher_use_ema=True` and
    they are present in the checkpoint. All parameters are set to
    `requires_grad=False` and the module is left in eval mode.
    """
    log_for_0(f"Loading teacher checkpoint from {teacher_path}")
    ckpt = _restore_checkpoint(teacher_path)
    if ckpt is None:
        raise ValueError(f"Failed to load teacher checkpoint from {teacher_path}")

    teacher = ELF_models[config.model](
        text_encoder_dim=text_encoder_dim, max_length=config.max_length,
        attn_drop=0.0, proj_drop=0.0,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=False,
    ).to(device)

    src = None
    if config.teacher_use_ema and ckpt.get("ema_params1") is not None:
        src = ckpt["ema_params1"]
        log_for_0("Teacher: loading from 'ema_params1'")
    if src is None:
        src = ckpt["params"]
        log_for_0("Teacher: loading from 'params'")

    src = _strip_orig_mod(src)
    src = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in src.items()}
    missing, unexpected = teacher.load_state_dict(src, strict=False)
    if missing:
        log_for_0(f"Teacher load: missing={list(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        log_for_0(f"Teacher load: unexpected={list(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log_for_0(f"Teacher ready: {sum(p.numel() for p in teacher.parameters()):,} params (frozen).")
    return teacher


def init_student_from_teacher(student: nn.Module, teacher: nn.Module) -> None:
    """Copy teacher weights into the student in-place.

    The student shares the teacher's architecture, so this is a plain
    state-dict copy.
    """
    log_for_0("Initializing student from teacher weights.")
    src = _strip_orig_mod(teacher.state_dict())
    missing, unexpected = student.load_state_dict(src, strict=False)
    log_for_0(
        f"Student init from teacher: missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        log_for_0(f"  missing[:6]: {missing[:6]}")
    if unexpected:
        log_for_0(f"  unexpected[:6]: {unexpected[:6]}")


def init_student_from_checkpoint(student: nn.Module, ckpt_path: str, use_ema: bool = True) -> None:
    """Warm-start the student from a previous distillation round's checkpoint.

    Used by progressive distillation: round k+1's student is initialized from
    round k's student weights (NOT the original teacher). Loads `ema_params1`
    when present (smoother than the live params), or falls back to `params`.
    Optimizer / LR / step counter are not loaded; this is a weight-only init.
    """
    log_for_0(f"Warm-starting student from checkpoint: {ckpt_path}")
    ckpt = _restore_checkpoint(ckpt_path)
    if ckpt is None:
        raise ValueError(f"Failed to load student warm-start checkpoint: {ckpt_path}")

    src = None
    if use_ema and ckpt.get("ema_params1") is not None:
        src = ckpt["ema_params1"]
        log_for_0("  using 'ema_params1' from checkpoint")
    if src is None:
        src = ckpt["params"]
        log_for_0("  using 'params' from checkpoint (no EMA)")

    src = _strip_orig_mod(src)
    device = next(student.parameters()).device
    src = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in src.items()}
    missing, unexpected = student.load_state_dict(src, strict=False)
    log_for_0(
        f"Student warm-start: missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        log_for_0(f"  missing[:6]: {missing[:6]}")


def _teacher_forward_x_pred(
    teacher: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    sc_cfg_scale: torch.Tensor,
    sc_half: torch.Tensor,
    num_self_cond_cfg_tokens: int,
    use_bf16: bool,
) -> torch.Tensor:
    """One frozen teacher forward returning x_pred (no autograd).

    `sc_half` is the self-cond half of the input; the caller builds it (cond-
    restored zeros for no self-cond, or the teacher's uncond x_pred to imitate
    inference-time self-conditioning).
    """
    x_input = torch.cat([z, sc_half], dim=-1)
    sc_arg = sc_cfg_scale if num_self_cond_cfg_tokens > 0 else None
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        net_out, _ = teacher(
            x_input, t,
            deterministic=True,
            self_cond_cfg_scale=sc_arg,
            decoder_step_active=None,
        )
    return net_out


@torch.no_grad()
def teacher_uncond_x_pred_init(
    teacher: nn.Module,
    z_t_s: torch.Tensor,
    t_s: torch.Tensor,
    sc_cfg_scale: torch.Tensor,
    x0: torch.Tensor,
    cond_seq_mask_e: torch.Tensor,
    num_self_cond_cfg_tokens: int,
    use_bf16: bool,
) -> torch.Tensor:
    """Teacher's uncond x_pred at (z_t_s, t_s), used to seed the self-cond half.

    Mirrors `compute_shared_uncond` in train_step.py. The returned tensor is
    detached and in the input dtype.
    """
    sc_half_zero = torch.where(cond_seq_mask_e > 0, x0, torch.zeros_like(z_t_s))
    net_out = _teacher_forward_x_pred(
        teacher, z_t_s, t_s, sc_cfg_scale, sc_half_zero,
        num_self_cond_cfg_tokens, use_bf16,
    )
    return net_out.detach().to(z_t_s.dtype)


def build_z_t_s(
    x0: torch.Tensor,
    t_s: torch.Tensor,
    denoiser_noise_scale: float,
    cond_seq_mask_e: torch.Tensor,
) -> torch.Tensor:
    """Sample `z_t_s = t_s * x0 + (1 - t_s) * noise * scale`, pinning cond positions.

    Built once by the caller and reused for (a) the teacher's PD-target
    trajectory, (b) the teacher's uncond seed for self-cond, and (c) the
    student's gradient forward, so all three consume the same noisy latent.
    """
    noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device)
    t_s_e = t_s.reshape(-1, 1, 1)
    z_t_s = t_s_e * x0 + (1.0 - t_s_e) * noise * denoiser_noise_scale
    return torch.where(cond_seq_mask_e > 0, x0, z_t_s)


@torch.no_grad()
def teacher_pd_target(
    teacher: nn.Module,
    x0: torch.Tensor,
    cond_seq_mask_e: torch.Tensor,
    z_t_s: torch.Tensor,
    t_s: torch.Tensor,
    t_t: torch.Tensor,
    sc_cfg_scale: torch.Tensor,
    sc_half: torch.Tensor,
    num_steps: int,
    num_self_cond_cfg_tokens: int,
    t_eps: float,
    use_bf16: bool,
    sub_step_times: torch.Tensor = None,
) -> torch.Tensor:
    """Build the per-example PD target by running `num_steps` teacher Euler
    sub-steps from `t_s` to `t_t`, then converting to an implied `x_pred` via
    progressive-distillation's inversion formula.

    `sc_half` is the self-cond half for the FIRST sub-step; the caller builds it,
    typically the teacher's uncond x_pred at (z_t_s, t_s) gated by a per-example
    `use_sc` mask. Later sub-steps chain in the teacher's previous-sub-step
    x_pred, mirroring how inference-time sampling threads self-cond from one step
    to the next.

    Returns the implied x_pred target only. Shape (B, S, D), detached.

    Implementation notes:
      * Default sub-stepping is uniform: each sub-step covers (t_t - t_s) / num_steps.
      * When `sub_step_times` is provided, it must be shape (B, num_steps+1) with
        sub_step_times[:, 0] == t_s and sub_step_times[:, -1] == t_t. The teacher
        uses those exact times instead of uniform Euler, letting the caller place
        sub-steps on a schedule-aware grid.
      * Cond positions are pinned to x0 along the trajectory.
      * The implied x_pred formula `x0 = z_t_s + ((1-t_s)/(t_t-t_s)) * (z_target
        - z_t_s)` is the flow-matching analog of Salimans & Ho's PD inversion.
        When span -> 0 it degenerates to the teacher's direct x_pred.
    """
    dtype = z_t_s.dtype
    span = (t_t - t_s)

    # Direct-forward fast-path when num_steps==1 (single teacher forward).
    if num_steps <= 1:
        x_pred = _teacher_forward_x_pred(
            teacher, z_t_s, t_s, sc_cfg_scale, sc_half,
            num_self_cond_cfg_tokens, use_bf16,
        ).detach()
        return x_pred

    # Multi-step Euler integration along v = (x_pred - z) / (1 - t).
    z_cur = z_t_s
    t_cur = t_s.clone()
    sc_input = sc_half  # first sub-step uses caller-provided self-cond half
    last_x_pred = None
    for step in range(num_steps):
        last_x_pred = _teacher_forward_x_pred(
            teacher, z_cur, t_cur, sc_cfg_scale, sc_input,
            num_self_cond_cfg_tokens, use_bf16,
        ).float()
        if sub_step_times is not None:
            t_next = sub_step_times[:, step + 1].to(t_cur.dtype)
        else:
            remaining = num_steps - step
            t_next = t_cur + (t_t - t_cur) / remaining
        denom = torch.clamp(1.0 - t_cur.reshape(-1, 1, 1).float(), min=t_eps)
        v = (last_x_pred - z_cur.float()) / denom
        dt = (t_next - t_cur).reshape(-1, 1, 1).float()
        z_cur = (z_cur.float() + dt * v).to(dtype)
        # Cond positions don't move along the trajectory.
        z_cur = torch.where(cond_seq_mask_e > 0, x0, z_cur)
        t_cur = t_next
        # Chain self-cond: later sub-steps use the teacher's previous x_pred,
        # with cond positions pinned to x0.
        sc_input = torch.where(cond_seq_mask_e > 0, x0, last_x_pred.to(dtype))

    # Implied x_pred at t_s, given the multi-step result at t_t.
    span_e = torch.clamp((t_t - t_s).reshape(-1, 1, 1).float(), min=1e-6)
    one_minus_ts = (1.0 - t_s.reshape(-1, 1, 1).float())
    ratio = one_minus_ts / span_e
    target = z_t_s.float() + ratio * (z_cur.float() - z_t_s.float())

    # Numerical safety: for any example whose span is ~0, fall back to
    # last_x_pred (the teacher's direct x_pred).
    fallback = (span.reshape(-1, 1, 1).float() < 1e-3).to(target.dtype)
    target = fallback * last_x_pred + (1.0 - fallback) * target

    return target.to(dtype).detach()
