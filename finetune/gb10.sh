#!/usr/bin/env bash
# Ewe->English Hibiki-Zero fine-tuning on an NVIDIA GB10 (DGX Spark).
# Usage: finetune/gb10.sh setup | pairs [MANIFEST] | cache | smoke | train [args] | resume TRAINER_CKPT [args]
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-finetune/runs/ewe_full}"
CACHE="finetune/cache"

check_gpu() {
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GB10 \
    || echo "warning: no NVIDIA GB10 found by nvidia-smi" >&2
}

cmd="${1:-}"
shift || true
case "$cmd" in
  setup)
    # Installs into the existing venv; torch (CUDA 13 build) must already be there.
    uv pip install --python "$PYTHON" --no-deps moshi==0.2.13
    uv pip install --python "$PYTHON" einops sacrebleu safetensors sentencepiece sphn \
      soundfile transformers num2words
    "$PYTHON" -c "import torch, moshi; print(torch.__version__, torch.cuda.get_device_name())"
    ;;
  pairs)
    "$PYTHON" finetune/build_pairs.py --manifest "${1:-${MANIFEST:?set MANIFEST or pass a path}}" --overwrite
    ;;
  cache)
    check_gpu
    for split in train validation; do
      "$PYTHON" finetune/cache_codes.py --pairs "finetune/pairs/$split.jsonl" \
        --out-dir "$CACHE/$split" "$@"
    done
    ;;
  smoke)
    check_gpu
    "$PYTHON" finetune/train.py --out-dir finetune/runs/smoke --max-steps 10 \
      --batch-size 1 --grad-accum-steps 1 --warmup-steps 0 --log-every 1 \
      --save-every 0 --val-every 0 --val-batches 2 "$@"
    ;;
  train)
    check_gpu
    "$PYTHON" finetune/train.py --out-dir "$RUN_DIR" "$@"
    ;;
  resume)
    check_gpu
    ckpt="${1:?usage: gb10.sh resume TRAINER_CKPT [args]}"
    shift
    "$PYTHON" finetune/train.py --out-dir "$(dirname "$ckpt")" --resume-checkpoint "$ckpt" "$@"
    ;;
  *)
    sed -n '2,3p' "$0"
    exit 1
    ;;
esac
