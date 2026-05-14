# ELF PyTorch Reproduction Notes

This repository now contains a PyTorch port scaffold for the original JAX/TPU ELF implementation from **ELF: Embedded Language Flows** (arXiv:2605.10938).

The original upstream code remains unchanged under `src/`. The PyTorch path lives under `src/torch_elf/`, plus `src/train_torch.py`, `src/eval_torch.py`, `scripts/convert_jax_checkpoint_to_torch.py`, and `requirements_torch.txt`.

## What is implemented

- PyTorch ELF layers and model structure mirroring the JAX implementation
- Cross-device detection for CUDA, ROCm, Intel XPU, MPS, and CPU fallback
- PyTorch T5 encoder wrapper using Hugging Face `T5EncoderModel`
- PyTorch data pipeline compatible with the existing config/data format
- ODE/SDE sampling path for smoke testing and initial inference work
- Minimal PyTorch training loop for reproduction smoke tests
- A checkpoint-inspection helper for exported JAX trees

## Known gaps

1. Official pretrained model checkpoints are still JAX/Orbax-native.
2. Muon optimizer is not yet ported; `train_torch.py` falls back to AdamW.
3. Training parity is approximate because TPU sharding / JAX RNG semantics are not replicated exactly.
4. The conversion helper currently expects an exported Python-loadable JAX tree rather than a raw Orbax directory.

## Environment setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements_torch.txt
```

## Device detection

Quick check:

```bash
.venv/bin/python -c "from src.torch_elf.device import detect_device, format_device_info; print(format_device_info(detect_device()))"
```

## Step-by-step execution

### 1. Smoke-test the PyTorch model path

```bash
.venv/bin/python src/eval_torch.py \
  --config src/configs/training_configs/train_owt_ELF-B.yml \
  --config_override max_length=32 \
  --config_override output_dir=outputs/torch-smoke \
  --num_samples 1 \
  --allow_random_init
```

### 2. Prepare checkpoint inspection / conversion

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import list_repo_files
files = list_repo_files("embedded-language-flows/ELF-B-owt", repo_type="model")
for path in files[:100]:
    print(path)
PY
```

Current status from direct inspection:

- `embedded-language-flows/ELF-B-owt`, `ELF-B-de-en`, and `ELF-B-xsum` expose Orbax/OCDBT checkpoint directories rather than native PyTorch weights.
- `embedded-language-flows/t5_small_encoder_jax` exposes `t5_small_encoder_jax.pkl` directly.

If you have already exported the JAX tree:

```bash
.venv/bin/python scripts/convert_jax_checkpoint_to_torch.py \
  --input /path/to/exported_jax_tree.pkl \
  --output outputs/converted/elf_b_owt_inspect.pt
```

### 3. Start PyTorch training reproduction

```bash
.venv/bin/python src/train_torch.py \
  --config src/configs/training_configs/train_owt_ELF-B.yml \
  --config_override max_length=64 \
  --config_override global_batch_size=2 \
  --config_override num_workers=0 \
  --config_override use_wandb=false \
  --max_steps 1 \
  --output_checkpoint outputs/torch-train-smoke/step1.pt
```

## Manual QA evidence collected in this session

Device detection:

```text
torch=2.12.0+cu130 | backend=cpu | device=cpu | description=CPU | cuda_runtime=13.0
```

Model construction (ELF-B parameter count):

```text
104594304
```

Eval smoke test output:

```text
INFO - __main__ - checkpoint_status=random-init
INFO - __main__ - Saved 1 samples to outputs/torch-smoke/torch_eval_samples.jsonl
INFO - __main__ - sample[0]='iediediediediediediedied'
```
