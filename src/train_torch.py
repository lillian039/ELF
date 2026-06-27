#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from configs.config import apply_config_overrides, load_config_from_yaml
from torch_elf.checkpoints import save_torch_checkpoint
from torch_elf.data import get_dataloader, get_pad_token_id, load_dataset, prepare_batch
from torch_elf.device import detect_device, format_device_info, get_autocast_kwargs
from torch_elf.encoder import T5TextEncoder
from torch_elf.model import ELF_models
from torch_elf.sampling import add_noise, net_out_to_v_x, sample_cfg_scale, sample_timesteps


logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PyTorch ELF port")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_checkpoint", type=str, default=None)
    return parser.parse_args()


def create_optimizer(config: Any, model: torch.nn.Module, learning_rate: float):
    if config.optimizer == "muon":
        from torch_elf.muon import MuonWithAdamW

        matrix_params: list[torch.nn.Parameter] = []
        scalar_params: list[torch.nn.Parameter] = []
        for p in model.parameters():
            (matrix_params if p.ndim >= 2 else scalar_params).append(p)
        logger.info("Muon optimizer: %d matrix params, %d scalar params", len(matrix_params), len(scalar_params))
        return MuonWithAdamW(
            [
                {"params": matrix_params, "use_muon": True, "lr": learning_rate, "weight_decay": config.weight_decay},
                {"params": scalar_params, "use_muon": False, "lr": learning_rate * 0.1, "betas": (config.adam_b1, config.adam_b2), "weight_decay": config.weight_decay},
            ],
            lr=learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(config.adam_b1, config.adam_b2), weight_decay=config.weight_decay)


def reduce_token_loss(per_token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    loss_mask = loss_mask.to(per_token_loss.dtype)
    safe_loss = torch.where(loss_mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
    return (safe_loss * loss_mask).sum() / torch.clamp(loss_mask.sum(), min=1.0)


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)

    device_info = detect_device(args.device)
    logger.info(format_device_info(device_info))

    encoder = T5TextEncoder.from_pretrained(model_name=config.encoder_model_name, tokenizer_name=config.tokenizer_name or config.encoder_model_name, latent_mean=config.latent_mean, latent_std=config.latent_std, device=device_info.device)
    tokenizer: PreTrainedTokenizerBase = encoder.tokenizer
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    train_dataset, _ = load_dataset(config)

    batch_size = config.batch_size or config.global_batch_size
    dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True, max_seq_length=config.max_length, max_input_seq_length=config.max_input_length, pad_token_id=pad_token_id)

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    model = ELF_models[config.model](text_encoder_dim=encoder.d_model, max_length=config.max_length, attn_drop=config.attn_dropout, proj_drop=config.proj_dropout, num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens, vocab_size=vocab_size, num_model_mode_tokens=config.num_model_mode_tokens, bottleneck_dim=config.bottleneck_dim).to(device_info.device)
    logger.info("Model parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    learning_rate = config.lr if config.lr is not None else config.blr * (config.global_batch_size / 256)
    optimizer = create_optimizer(config, model, learning_rate)
    scaler = torch.amp.GradScaler(enabled=device_info.supports_amp and device_info.device.type in {"cuda", "xpu"})
    autocast_kwargs = get_autocast_kwargs(device_info)
    ema_params = [param.detach().clone() for param in model.parameters()]

    model.train()
    global_step = 0
    progress = tqdm(dataloader, desc="train", total=args.max_steps)
    for raw_batch in progress:
        batch = prepare_batch(raw_batch, config, device_info.device)
        input_ids = batch["input_ids"]
        encoder_attention_mask = batch["encoder_attention_mask"]
        cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)
        attention_mask = batch["attention_mask"]
        loss_mask = attention_mask if config.pad_token == "pad" else torch.ones_like(attention_mask)
        loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

        with torch.no_grad():
            x0 = encoder.encode(input_ids=input_ids, attention_mask=encoder_attention_mask)
        if config.label_drop_prob > 0:
            drop = batch["label_drop_mask"][:, None, None]
            x0 = torch.where(drop & (cond_seq_mask > 0), torch.zeros_like(x0), x0)

        batch_size_now, seq_length = x0.shape[:2]
        t = sample_timesteps(batch_size_now, device=device_info.device, p_mean=config.denoiser_p_mean, p_std=config.denoiser_p_std, time_schedule=config.time_schedule)
        noise = torch.randn_like(x0)
        denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)
        decoder_targets = input_ids
        decoder_step_active = torch.rand(1, device=device_info.device).item() < config.decoder_prob
        decoder_lambda = torch.sigmoid(torch.randn(batch_size_now * seq_length, device=device_info.device) * config.decoder_p_std + config.decoder_p_mean).view(batch_size_now, seq_length, 1)
        decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
        decoder_z = decoder_lambda * x0 + (1 - decoder_lambda) * decoder_noise
        t_expanded = t.view(-1, 1, 1)
        v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=config.t_eps)

        self_cond_cfg_scale = None
        if config.num_self_cond_cfg_tokens > 0:
            self_cond_cfg_scale = sample_cfg_scale(batch_size_now, device=device_info.device, cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs.get("enabled", False) else nullcontext()
        with autocast_ctx:
            if decoder_step_active:
                decoder_input = torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1) if config.self_cond_prob > 0 else decoder_z
                _, decoder_logits = model(decoder_input, torch.ones_like(t), self_cond_cfg_scale=self_cond_cfg_scale, decoder_step_active=True)
                log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
                ce = -torch.gather(log_probs, dim=-1, index=decoder_targets.unsqueeze(-1)).squeeze(-1)
                loss = (ce * loss_mask).sum() / torch.clamp(loss_mask.sum(), min=1.0)
                l2_loss = torch.tensor(0.0, device=device_info.device)
                ce_loss = loss.detach()
            else:
                if config.self_cond_prob > 0:
                    with torch.no_grad():
                        z_uncond = torch.zeros_like(denoiser_z)
                        denoiser_input = torch.cat([denoiser_z, z_uncond], dim=-1)
                        init_out, _ = model(denoiser_input, t, self_cond_cfg_scale=self_cond_cfg_scale, decoder_step_active=False)
                        _, x_pred_init = net_out_to_v_x(init_out, denoiser_z, t, config.t_eps)
                    denoiser_input = torch.cat([denoiser_z, x_pred_init], dim=-1)
                else:
                    denoiser_input = denoiser_z
                net_out, _ = model(denoiser_input, t, attention_mask=attention_mask, self_cond_cfg_scale=self_cond_cfg_scale, decoder_step_active=False)
                v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
                per_dim_loss = (v_pred - v_target) ** 2
                loss = reduce_token_loss(per_dim_loss.mean(dim=-1), loss_mask)
                l2_loss = loss.detach()
                ce_loss = torch.tensor(0.0, device=device_info.device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for ema_param, model_param in zip(ema_params, model.parameters()):
                ema_param.mul_(config.ema_decay1).add_(model_param.detach(), alpha=1 - config.ema_decay1)

        global_step += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", l2=f"{l2_loss.item():.4f}", ce=f"{ce_loss.item():.4f}")
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if args.output_checkpoint:
        save_torch_checkpoint(args.output_checkpoint, {"model": model.state_dict(), "ema_model": [tensor.cpu() for tensor in ema_params], "optimizer": optimizer.state_dict(), "step": global_step, "config": vars(config)})
        logger.info("Saved checkpoint to %s", args.output_checkpoint)


if __name__ == "__main__":
    main()
