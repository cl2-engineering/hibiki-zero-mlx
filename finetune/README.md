# Ewe→English full-model SFT (NVIDIA GB10)

Fine-tunes every parameter of upstream Hibiki-Zero (`weights/`) on paired Ewe
source speech and English target speech. Training uses CUDA, fp32 master
weights, bf16 autocast, fused AdamW, and `torch.compile`
(disable with `NO_TORCH_COMPILE=1`).

## 1. Manifest

A CSV you fill yourself:

```csv
id,split,ewe_audio,en_audio,text_ee,text_en
utt0001,train,audio/ee/utt0001.wav,audio/en/utt0001.wav,Ŋdi nyuie,Good morning.
utt0002,validation,audio/ee/utt0002.wav,audio/en/utt0002.wav,,Thank you very much.
```

- `split`: `train`, `validation`, or `test`; `id` must be unique.
- Relative audio paths resolve against the manifest's directory.
- `en_audio` is required (no TTS step) and `text_en` must match what it says:
  the text stream is CTC-aligned to the English audio, and rows whose alignment
  score is below `--min-alignment-score` (0.5) are rejected.
- `text_ee` is optional and only kept for reference.

## 2. Workflow

```bash
finetune/gb10.sh setup                  # moshi 0.2.13 + deps into .venv
finetune/gb10.sh pairs path/to/manifest.csv   # -> finetune/pairs/{split}.jsonl
finetune/gb10.sh cache                  # -> finetune/cache/{train,validation}
finetune/gb10.sh smoke                  # optional 10-step check
finetune/gb10.sh train [--lr 1e-5 --epochs 10 ...]
finetune/gb10.sh resume finetune/runs/ewe_full/trainer_step000500.pt [same args]
```

`build_pairs.py` drops rows whose Ewe audio is outside 0.5–20 s. `train.py`
drops samples over `--max-frames` (280 frames = 22.4 s at 12.5 Hz), buckets by
length, and reshuffles batches each epoch deterministically, so resume is exact
as long as the cache dirs, batch size, grad accumulation, max frames, and seed
are unchanged. Defaults: batch 4 × accumulation 4, LR 1e-5 with 50 warmup
steps, validation/checkpoint every 250 steps, last 2 checkpoints kept, and
`best.safetensors` tracks the lowest teacher-forced validation loss. Add
`--gradient-checkpointing` if memory is tight (batch 1 uses ~53 GB).

## 3. Evaluation

```bash
.venv/bin/python finetune/validate.py --checkpoint finetune/runs/ewe_full/best.safetensors
.venv/bin/python finetune/eval.py --checkpoint finetune/runs/ewe_full/best.safetensors
```

`validate.py` reports teacher-forced losses on a cache; `eval.py` free-runs on
`finetune/pairs/validation.jsonl` and writes audio, `predictions.csv`, and
BLEU/chrF/WER metrics.

Files: `build_pairs.py`, `cache_codes.py`, `text_timing.py` (English CTC
timing), `train.py`, `validate.py`, `eval.py`, `common.py`, `hibiki_helpers.py`,
`utils.py`, `gb10.sh`.
