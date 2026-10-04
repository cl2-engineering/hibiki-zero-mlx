# Hibiki Ewe

Ewe → English simultaneous speech-to-speech translation built on
[Hibiki-Zero](https://github.com/kyutai-labs/hibiki-zero)
([paper](https://arxiv.org/abs/2602.11072), weights license CC BY-NC-SA 4.0).
Training and inference run with PyTorch on CUDA, targeting a local NVIDIA GB10
(DGX Spark, aarch64, CUDA 13).

## Setup

```bash
uv sync --extra training   # torch cu130 + moshi 0.2.13 into .venv
```

`pyproject.toml` pulls torch from the cu130 index and overrides moshi's stale
torch/huggingface-hub/numpy/safetensors pins. Drop `--extra training` for
inference only.

Download the upstream 3B weights (file names are what `finetune/utils.py` expects):

```bash
.venv/bin/hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
```

## Inference

```bash
.venv/bin/python main.py                                    # live: microphone -> speaker
.venv/bin/python main.py assets/samples/leon.wav            # file, upstream weights
.venv/bin/python main.py input.wav --checkpoint finetune/runs/ewe_full/best.safetensors
```

Live mode streams 80 ms frames in real time on the GB10, plays the English speech,
and prints the text until Ctrl+C. Use headphones: a speakerphone feeds the model's
own output back into the mic. Pick devices with `--input-device` / `--output-device`
(index or name from `python -m sounddevice`).

Writes `translations/<stem>_translated.wav` and a `.txt` transcript (`-o`,
`--text-out` override). Student checkpoints also need `--config`; see
`student/README.md`. Sampling follows the upstream config (text temperature 0.8,
top-k 250).

## Training

Write a CSV manifest of paired Ewe and English speech:

```csv
id,split,ewe_audio,en_audio,text_ee,text_en
utt0001,train,audio/ee/utt0001.wav,audio/en/utt0001.wav,Ŋdi nyuie,Good morning.
```

English target audio is mandatory (there is no TTS step). Then:

```bash
finetune/gb10.sh pairs path/to/manifest.csv
finetune/gb10.sh cache
finetune/gb10.sh smoke
finetune/gb10.sh train
```

Details, defaults, resume, and evaluation: [finetune/README.md](finetune/README.md).
The 12-layer mobile distillation track is in [student/README.md](student/README.md).

## GB10 notes

- 3B inference: ~8 GB GPU memory, ~1.34× real time.
- Full fine-tuning: ~53 GB peak at batch 1 (fp32 masters + AdamW ≈ 48 GB).
  The default batch 4 × accumulation 4 may need `--gradient-checkpointing` or a
  smaller `--batch-size` for long rows.
- Each checkpoint (model + optimizer) is ~37 GB; with the last two kept, budget
  ~110 GB of disk.

## Layout

- `main.py`: CUDA file or live-microphone translation for upstream, fine-tuned, and student checkpoints.
- `finetune/`: manifest → pairs → cache → full-model SFT → validation/eval.
- `student/`: 12-layer AR and `parallel_v1` distillation.
- `weights/`: upstream weights (gitignored).
- `assets/samples/leon.wav`: French smoke-test clip for upstream inference.
