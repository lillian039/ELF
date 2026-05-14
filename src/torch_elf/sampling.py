from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def add_noise(x0: Tensor, noise: Tensor, t: Tensor, config: Any, cond_seq_mask: Optional[Tensor] = None) -> Tensor:
    t_expanded = t.view(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


def sample_timesteps(batch_size: int, device: torch.device, p_mean: float = -0.8, p_std: float = 0.8, time_schedule: str = "logit_normal") -> Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device) * p_std + p_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(n_steps: int, device: torch.device, time_schedule: str = "logit_normal", p_mean: float = -0.8, p_std: float = 0.8) -> Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(n_steps - 1, device=device, p_mean=p_mean, p_std=p_std, time_schedule=time_schedule)
        return torch.cat([torch.tensor([0.0], device=device), torch.sort(steps).values, torch.tensor([1.0], device=device)])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(batch_size: int, device: torch.device, cfg_min: float = 0.0, cfg_max: float = 3.0) -> Tensor:
    u = torch.rand(batch_size, device=device)
    a = torch.tensor(1.0 + cfg_min, device=device)
    b = torch.tensor(1.0 + cfg_max, device=device)
    return a * torch.exp(u * torch.log(b / a)) - 1.0


def restore_cond(z_updated: Tensor, cond_seq: Tensor, cond_seq_mask: Tensor) -> Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v: Tensor, x: Tensor, cond_seq: Optional[Tensor], cond_seq_mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    if cond_seq is not None and cond_seq_mask is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def net_out_to_v_x(net_out: Any, z: Tensor, t: Tensor, t_eps: float = 5e-2) -> tuple[Tensor, Tensor]:
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.view(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v, x


@torch.no_grad()
def _forward_sample_self_cond(model: Any, z: Tensor, t_batch: Tensor, x_pred_prev: Optional[Tensor], config: Any, self_cond_cfg_scale: float, cond_seq: Tensor, cond_seq_mask: Tensor) -> tuple[Tensor, Tensor]:
    t_eps = config.t_eps
    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
        net_out_cond = model(z_input_cond, t_batch, self_cond_cfg_scale=self_cond_scale_batch)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        return restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)

    if config.self_cond_prob == 0:
        net_out = model(z, t_batch)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    v_uncond: Tensor
    x_uncond: Tensor
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = restore_vx(v_uncond, x_uncond, cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond
    else:
        v_uncond = torch.zeros_like(z)
        x_uncond = torch.zeros_like(z)

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond
    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def _forward_sample(model: Any, z: Tensor, t_batch: Tensor, x_pred_prev: Optional[Tensor], config: Any, cfg_scale: float, self_cond_cfg_scale: float, cond_seq: Tensor, cond_seq_mask: Tensor) -> tuple[Tensor, Tensor]:
    v_cond, x_cond = _forward_sample_self_cond(model, z, t_batch, x_pred_prev, config, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    if cfg_scale == 1.0:
        return v_cond, x_cond
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = None if x_pred_prev is None else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    v_uncond, x_uncond = _forward_sample_self_cond(model, z_uncond, t_batch, x_pred_prev_uncond, config, self_cond_cfg_scale, torch.zeros_like(cond_seq), cond_seq_mask)
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def ode_step(model: Any, z: Tensor, t: float, t_next: float, x_pred_prev: Optional[Tensor], config: Any, cfg_scale: float, self_cond_cfg_scale: float, cond_seq: Tensor, cond_seq_mask: Tensor) -> tuple[Tensor, Tensor]:
    t_batch = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(model, z, t_batch, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    return z + (t_next - t) * v_pred, x_pred


@torch.no_grad()
def sde_step(model: Any, z: Tensor, t: float, t_next: float, x_pred_prev: Optional[Tensor], config: Any, cfg_scale: float, self_cond_cfg_scale: float, cond_seq: Tensor, cond_seq_mask: Tensor, gamma: float) -> tuple[Tensor, Tensor]:
    h = t_next - t
    alpha = max(1.0 - gamma * h, 0.0)
    t_back = alpha * t
    eps = torch.randn_like(z) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(model, z_back, t_batch, x_pred_prev, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    return z_back + (t_next - t_back) * v_pred, x_pred


@torch.no_grad()
def generate_latents(model: Any, batch_size: int, seq_len: int, d_model: int, config: Any, sampling_config: Any, device: torch.device, cfg_scale: float = 1.0, self_cond_cfg_scale: float = 1.0, cond_seq: Optional[Tensor] = None, cond_seq_mask: Optional[Tensor] = None) -> Tensor:
    z = torch.randn(batch_size, seq_len, d_model, device=device) * config.denoiser_noise_scale
    if cond_seq is None:
        cond_seq = torch.zeros_like(z)
        cond_seq_mask = torch.zeros(batch_size, seq_len, device=device, dtype=z.dtype)
    else:
        assert cond_seq_mask is not None
        cond_seq_mask = cond_seq_mask.to(dtype=z.dtype)
    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
    n_steps = max(sampling_config.num_sampling_steps)
    t_steps = get_sampling_steps(n_steps, device=device, time_schedule=sampling_config.time_schedule, p_mean=config.denoiser_p_mean, p_std=config.denoiser_p_std)
    for idx in range(len(t_steps) - 2):
        t = float(t_steps[idx].item())
        t_next = float(t_steps[idx + 1].item())
        if sampling_config.sampling_method == "sde":
            z, x_pred = sde_step(model, z, t, t_next, x_pred, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, gamma=getattr(sampling_config, "sde_gamma", 0.0))
        else:
            z, x_pred = ode_step(model, z, t, t_next, x_pred, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    z, _ = ode_step(model, z, float(t_steps[-2].item()), float(t_steps[-1].item()), x_pred, config, cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask)
    return z


@torch.no_grad()
def decode_latents(model: Any, z: Tensor, self_cond_cfg_scale: float = 1.0) -> Tensor:
    t_final = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
    sccfg = torch.full((z.shape[0],), self_cond_cfg_scale, device=z.device, dtype=z.dtype)
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
    _, logits = model(z_input, t_final, self_cond_cfg_scale=sccfg, decoder_step_active=True)
    return torch.argmax(logits, dim=-1)


def mask_after_eos(predicted_ids: Tensor, eos_token_id: int, pad_token_id: int) -> Tensor:
    eos_mask = predicted_ids == eos_token_id
    keep_mask = torch.cumsum(eos_mask.to(dtype=torch.int32), dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))
