#!/usr/bin/env bash
# Carrier-routing ablation on death: TinyStories k=16, matched budget,
# phi visible to ONE component during training.
#   GPUs 0-3: phi_route=denoiser (decoder head unconditioned)
#   GPUs 4-7: phi_route=decoder  (denoiser unconditioned)
# Usage: ssh death 'bash /mnt/faster3/lc2762/ELF/bib_demo/launch_routing_death.sh'
set -euo pipefail
REPO=/mnt/faster3/lc2762/ELF
cd "${REPO}"
mkdir -p bib_demo/logs

NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR}"
export PATH="${NVCC_DIR}/bin:${PATH}"
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=/mnt/faster3/lc2762/hf_cache
export HF_DATASETS_CACHE=/mnt/faster3/lc2762/hf_cache

launch() { # route gpus
  local route="$1" gpus="$2"
  : > "bib_demo/logs/train_route_${route}.log"
  CUDA_VISIBLE_DEVICES="${gpus}" setsid nohup python3 src/train.py \
    --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
    --config_override "manifold_dim=16" \
    --config_override "phi_route=${route}" \
    --config_override "seed=42" \
    --config_override "save_freq=10" \
    --config_override "global_batch_size=40" \
    --config_override "grad_accum_steps=2" \
    --config_override "warmup_steps=2000" \
    --config_override "output_dir=/mnt/faster3/lc2762/elf_route_${route}_k16_s1" \
    > "bib_demo/logs/train_route_${route}.log" 2>&1 < /dev/null &
  disown
  echo "launched route=${route} on GPUs ${gpus}, pid $!"
}

launch denoiser 0,1,2,3
launch decoder  4,5,6,7
