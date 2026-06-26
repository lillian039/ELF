"""Entry point for progressive distillation of an ELF model.

The student keeps the teacher's single-time architecture. Each training example
is independently routed to either the decoder CE branch (at t=1) or the denoiser
PD branch (MSE against the teacher's progressive-distillation target), per
`config.decoder_prob`. See `distill_train_step.distill_train_step` for the loss.
"""

import functools
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer

from configs.config import apply_config_overrides, load_config_from_yaml
from modules.t5_encoder import get_encoder
from train import parse_args, run_training, _init_distributed
from utils.distill_utils import (
    init_student_from_checkpoint,
    init_student_from_teacher,
    load_teacher_model,
)
from utils.logging_utils import log_for_0
from utils.train_utils import TrainState, unwrap_model
from distill_train_step import distill_train_step


def _resolve_vocab_size(config) -> int:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    try:
        return len(tokenizer)
    except TypeError:
        return tokenizer.vocab_size


def _build_teacher(config, device: torch.device) -> torch.nn.Module:
    """Build the frozen ELF teacher on the same device as the student."""
    if not config.teacher_path:
        raise ValueError(
            "distill_train.py requires `teacher_path` to be set in the config. "
            "Point it at an ELF checkpoint produced by train.py."
        )
    encoder_config, _ = get_encoder(config.encoder_model_name, torch.float32)
    vocab_size = _resolve_vocab_size(config)
    return load_teacher_model(
        teacher_path=config.teacher_path,
        text_encoder_dim=encoder_config.d_model,
        vocab_size=vocab_size,
        config=config,
        device=device,
    )


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)

    # Resolve the target device early so the teacher lives on the right GPU.
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device("cpu") if args.use_cpu or not torch.cuda.is_available()
        else torch.device(f"cuda:{local_rank}")
    )

    teacher = _build_teacher(config, device=device)

    def _post_build(state, cfg, *, resuming: bool):
        """Initialize the student weights:

          1. If resuming, the checkpoint will load the student a few lines
             later in `run_training` — do nothing here.
          2. If `student_init_from_checkpoint` is set, warm-start from that
             checkpoint (the progressive-distillation case — previous round's
             student becomes this round's starting point).
          3. Otherwise, copy the teacher weights into the student (round 1
             of any curriculum, and the default for single-shot runs).

        Either way, re-sync the EMA accumulator from the loaded weights so
        the saved `ema_params1` reflects the student's actual trajectory.
        """
        if resuming:
            log_for_0("Resuming distillation: skipping student init (checkpoint will be loaded).")
            return

        student = unwrap_model(state.model)
        warm_start = getattr(cfg, "student_init_from_checkpoint", None)
        if warm_start:
            init_student_from_checkpoint(student, warm_start, use_ema=cfg.teacher_use_ema)
        elif cfg.initialize_student_from_teacher:
            init_student_from_teacher(student, teacher)
        else:
            log_for_0("Student left at random init (no teacher copy, no warm-start).")
            return

        state.ema_params1 = TrainState.init_ema(student)
        log_for_0("EMA accumulator re-initialized from loaded student weights.")

    # run_training builds its own T5 encoder and threads it into the step call
    # as a kwarg, so the only thing we close over here is the frozen teacher.
    step_fn = functools.partial(distill_train_step, teacher=teacher)

    run_training(
        config,
        force_cpu=args.use_cpu,
        step_fn=step_fn,
        model_post_build_hook=_post_build,
    )


if __name__ == "__main__":
    main()
