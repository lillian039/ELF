import yaml
import os


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".


# ============================================
# Configuration
# ============================================
class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'

    # Decoder objective
    decoder_prob: float = 0.5  # Probability of decoder (CE) step vs denoiser (L2) step
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Gradient accumulation steps (optimizer updates every K mini-batches)
    use_bf16: bool = True  # Use CUDA BF16 autocast for training/eval forward passes.
    use_compile: bool = False  # Wrap the eval/sampling model in torch.compile.
    gradient_checkpointing: bool = False  # Save activation memory by recomputing ELF blocks during backward.

    # EMA
    ema_decay1: float = 0.9999

    # Geometry router (curvature-aware attention routing). All defaults keep
    # the feature OFF; the disabled model matches the original architecture
    # and state_dict exactly.
    geometry_router_enabled: bool = False
    geometry_router_layers: str = "all"  # "all", "0,1,2", or "0-3,6,8-11"
    geometry_router_mode: str = "soft"  # "soft" only; "hard" reserved (NotImplementedError)
    geometry_router_on_attention: bool = True  # v1 requires True (validated)
    geometry_router_on_mlp: bool = False  # reserved; v1 requires False (validated)
    # False (default): routing applies to every forward through the shared
    # Transformer backbone, including decoder/CE-mode rows and the decode
    # path. True: strict denoiser semantics — decoder-mode rows are forced to
    # the pure Euclidean gate and decode-mode forwards bypass routing
    # entirely; only denoiser (L2) forwards are routed.
    geometry_router_denoiser_only: bool = False
    geometry_router_detach_scores: bool = True  # geometry stats under no_grad
    geometry_router_sample_size: int = 32  # tokens subsampled per example for stats
    geometry_router_quad_samples: int = 512  # deterministic quadruples for delta_rel
    geometry_router_eps: float = 1e-6
    geometry_router_tau_h: float = 4.0  # sharpness of hyperbolic score -> logit
    geometry_router_tau_s: float = 4.0  # sharpness of sphere score -> logit
    geometry_router_bias_e: float = 2.0  # strong initial Euclidean prior
    geometry_router_bias_h: float = -2.0
    geometry_router_bias_s: float = -2.0
    geometry_router_learnable_bias: bool = False  # If true, learn per-layer E/H/S routing priors.
    geometry_router_time_e_bias: float = 1.0  # l_E += time_e_bias * (1 - t)
    geometry_router_time_h_bias: float = 0.0  # l_H += time_h_bias * t
    geometry_router_time_s_bias: float = 0.0  # l_S += time_s_bias * t
    geometry_router_sphere_k: str = "0.25,0.5,1.0,2.0,4.0"  # positive curvature candidates (distances diameter-normalized first)
    geometry_hyperbolic_curvature: float = 1.0
    geometry_hyperbolic_score: str = "busemann_proxy"  # "busemann_proxy" or "poincare_distance"
    geometry_sphere_score: str = "cosine"  # "cosine" or "negative_angular"
    geometry_router_log_metrics: bool = False
    geometry_router_log_freq: int = 100
    # Gate warmup: for the first N optimizer steps, blend the learned gates
    # toward uniform [1/3,1/3,1/3] (linearly decaying 1->0) so the
    # hyperbolic/sphere branches are forced to train early. 0 = off.
    geometry_router_gate_warmup_steps: int = 0
    geometry_router_metrics_path: str = None  # Optional JSON path for eval/train router summaries.

    # Sampling
    sampling_configs_path: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100
    eval_data_offset: int = 0  # Skip this many eval examples for conditional generation.

    # PPL Evaluation
    online_eval: bool = True  # Enable PPL evaluation for generated samples
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)
    max_train_steps: int = None  # Optional optimizer-step cap for short finetunes.

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None  # Optional HF repo id to mirror local outputs/checkpoints.
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None
    wandb_resume: str = "allow"

    # Misc
    seed: int = 0
    num_workers: int = 8


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    if not path or not os.path.isfile(path):
        return config

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue  # handled below
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.

    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")

        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()

        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")

        original_value = getattr(config, field_name)
        original_type = type(original_value)

        # Allow setting a field back to None
        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            # Use type annotation to infer the intended type
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted_value = int(value_str)
            elif annotated_type == float:
                converted_value = float(value_str)
            elif annotated_type == bool:
                converted_value = value_str.lower() in ("true", "1", "yes")
            else:
                converted_value = value_str
        elif original_type == bool:
            converted_value = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        else:
            converted_value = value_str

        setattr(config, field_name, converted_value)

    return config


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]
