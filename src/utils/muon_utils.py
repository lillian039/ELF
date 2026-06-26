"""Muon optimizer matching `optax.contrib.muon`, with patches for our use."""

import torch
import torch.distributed as dist
import torch.nn as nn

from utils.logging_utils import log_for_0


def muon_with_aux_adam(model: nn.Module, lr: float):
    """Muon optimizer matching `optax.contrib.muon(learning_rate=lr_schedule)`.

    Params are partitioned by `ndim == 2 -> Muon`, else Nesterov-Adam (b1=0.9,
    b2=0.999, eps=1e-8, wd=0). A safety wrapper zero-fills missing grads
    because the decoder/denoiser branches alternate and leave one head's
    params at `None` per step.
    """
    import muon as _muon_module
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    # Patch muon's Adam update to Nesterov-Adam:
    #   mu_hat = b1 * (mu / (1 - b1**(t+1))) + (1 - b1) * (g / (1 - b1**t))
    def _nesterov_adam_update(grad, mu, nu, step, betas, eps):
        b1, b2 = betas
        mu.lerp_(grad, 1 - b1)
        nu.lerp_(grad.square(), 1 - b2)
        mu_hat = b1 * (mu / (1 - b1 ** (step + 1))) + (1 - b1) * (grad / (1 - b1 ** step))
        nu_hat = nu / (1 - b2 ** step)
        return mu_hat / (nu_hat.sqrt() + eps)
    _muon_module.adam_update = _nesterov_adam_update

    # Patch NS5 to run in fp32 with eps=1e-8. Upstream muon.py downcasts to
    # bf16 and uses eps=1e-7, both of which lose precision through the 5
    # squarings.
    def _zeropower_via_newtonschulz5_fp32(G, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.to(torch.float32)
        if G.size(-2) > G.size(-1):
            X = X.mT
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X
    _muon_module.zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5_fp32

    # Two patches on the upstream muon_update:
    #   1. Add Nesterov bias correction (upstream skips it):
    #          mu_hat = beta*(mu/(1-beta**(t+1))) + (1-beta)*(g/(1-beta**t))
    #      Without it, early steps differ by ~20x at t=1.
    #   2. Use the `sqrt(max(1, fan_out/fan_in))` shape scaling.
    #      Bare Parameter tensors like `proj_kernel`/`unembed_kernel` are
    #      stored (in, out) instead of nn.Linear's (out, in); `flax_layout`
    #      flags those so we flip the ratio. The ratio can be up to
    #      sqrt(vocab/bn) ≈ sqrt(63) ≈ 7.9 for unembed_kernel.
    def _muon_update_optax(grad, momentum, step, beta=0.95, ns_steps=5,
                           nesterov=True, flax_layout=False):
        momentum.lerp_(grad, 1 - beta)
        if nesterov:
            mu_corr = momentum / (1 - beta ** (step + 1))
            g_corr = grad / (1 - beta ** step)
            update = beta * mu_corr + (1 - beta) * g_corr
        else:
            update = momentum / (1 - beta ** step)
        if update.ndim == 4:
            update = update.view(len(update), -1)
        update = _muon_module.zeropower_via_newtonschulz5(update, steps=ns_steps)
        # `sqrt(max(1, fan_out/fan_in))` scaling, accounting for storage layout.
        m, n = grad.size(-2), grad.size(-1)
        if flax_layout:
            update *= max(1, n / m) ** 0.5    # (in, out) layout: fan_out = n
        else:
            update *= max(1, m / n) ** 0.5    # (out, in) layout: fan_out = m
        return update

    # Identify which 2D params are nn.Linear.weight (stored (out, in)) vs
    # bare Parameter tensors that follow the (in, out) convention. The shape
    # scaling formula assumes (in, out); for (out, in) params we flip the
    # ratio.
    linear_weight_ids = set()
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.weight is not None:
            linear_weight_ids.add(id(m.weight))

    muon_params, adam_params = [], []
    muon_flax_layout = {}  # id(p) -> bool (True = (in, out), False = (out, in))
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
            muon_flax_layout[id(p)] = id(p) not in linear_weight_ids
        else:
            adam_params.append(p)

    distributed = dist.is_available() and dist.is_initialized()
    base_cls = MuonWithAuxAdam if distributed else SingleDeviceMuonWithAuxAdam

    class _SafeMuonAuxAdam(base_cls):
        """Fixes layered on top of upstream `MuonWithAuxAdam.step`:

        1. Zero-fill missing grads (decoder/denoiser branches alternate and
           leave one head's params at `None` per step).
        2. Correct distributed all_gather padding when Muon param count is
           not a multiple of world_size: `(-len) % ws`, not `len % ws`.
        3. Reimplement the Muon path to call our bias-correcting
           `_muon_update_optax`.
        """

        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
            for group in self.param_groups:
                if group["use_muon"]:
                    params = group["params"]
                    if distributed:
                        ws = dist.get_world_size()
                        pad_n = (-len(params)) % ws
                        params_pad = params + [torch.empty_like(params[-1])] * pad_n
                        for base_i in range(0, len(params), ws):
                            if base_i + dist.get_rank() < len(params):
                                p = params[base_i + dist.get_rank()]
                                self._muon_step_one(p, group)
                            dist.all_gather(
                                params_pad[base_i:base_i + ws],
                                params_pad[base_i + dist.get_rank()],
                            )
                    else:
                        for p in params:
                            self._muon_step_one(p, group)
                else:
                    for p in group["params"]:
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = _muon_module.adam_update(
                            p.grad, state["exp_avg"], state["exp_avg_sq"],
                            state["step"], group["betas"], group["eps"],
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])

        def _muon_step_one(self, p, group):
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            update = _muon_update_optax(
                p.grad, state["momentum_buffer"], state["step"],
                beta=group["momentum"],
                flax_layout=muon_flax_layout.get(id(p), False),
            )
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])

    # Hyperparams hardcoded to `optax.contrib.muon` defaults.
    param_groups = [
        dict(params=muon_params, lr=lr, momentum=0.95,
             weight_decay=0.0, use_muon=True),
        dict(params=adam_params, lr=lr, betas=(0.9, 0.999),
             eps=1e-8, weight_decay=0.0, use_muon=False),
    ]
    log_for_0(
        f"Muon: {len(muon_params)} 2D params; "
        f"Nesterov-AdamW: {len(adam_params)} other params"
    )
    return _SafeMuonAuxAdam(param_groups)
