# Hibiki Ewe — project context

Ewe → English simultaneous speech-to-speech translation from upstream
Hibiki-Zero 3B, trained and run with PyTorch CUDA on a local NVIDIA GB10
(DGX Spark, aarch64, CUDA 13). Two tracks:

1. **3B fine-tune + inference**: full-model SFT on paired Ewe/English speech;
   `main.py` translates files.
2. **Student distillation**: 12-layer AR student, then a `parallel_v1` head,
   exported to BF16 and listened to via `main.py` (`student/README.md`).

## File map

- `main.py`: file-only CUDA inference (bf16) for upstream, fine-tuned, AR-student
  and `parallel_v1` checkpoints; 8 s silence tail, stops at text EOS.
- `finetune/build_pairs.py`: manifest CSV → `finetune/pairs/{split}.jsonl`.
- `finetune/cache_codes.py` + `text_timing.py`: Mimi codes and CTC-timed
  English text → `finetune/cache/{train,validation}`.
- `finetune/train.py`: full-model SFT (fp32 masters, bf16 autocast, fused AdamW,
  `torch.compile`, length buckets, exact resume, keep-2 checkpoints,
  `best.safetensors` by teacher-forced validation loss).
- `finetune/validate.py` (teacher-forced loss), `finetune/eval.py` (free-running,
  BLEU/chrF/WER), `finetune/gb10.sh` (setup|pairs|cache|smoke|train|resume).
- `finetune/utils.py`: repo paths and the default `weights/` file names.
- `student/`: contract, init, caches, teacher dump, AR and parallel trainers, export.

## Data / cache contract

- Manifest columns: `id,split,ewe_audio,en_audio,text_ee,text_en`; relative
  paths resolve against the manifest directory; `text_ee` is reference only.
- Cache streams: English target Mimi audio (teacher-forced, audio CE), English
  text CTC-aligned to that audio ending in tokenizer EOS, Ewe source Mimi audio.
  Rows with alignment score below 0.5 are rejected; max 280 frames (22.4 s).

## Key decisions

- Generic local manifest; English target audio is mandatory (no TTS).
- No Hugging Face sync of caches or checkpoints; everything stays local.
- CUDA only; file-only inference (no microphone or realtime path).
- Every 3B parameter is trainable. Memory: ~53 GB peak at batch 1; checkpoints
  ~37 GB each (~110 GB with keep-2). Inference ~8 GB, ~1.34× real time.

## Environment

- Python: `.venv/bin/python` (uv, Python 3.13, torch 2.14.1+cu130). `uv sync
  --extra training` installs everything, including moshi 0.2.13 via
  `override-dependencies` for its stale pins.
- AGENTS.md still names a macOS conda path; it does not apply on GB10.
- The ignored `.env` may hold an HF token. Never print, log, or commit it.

## Canonical resources

- Upstream model: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16
