#!/usr/bin/env python
"""Hibiki-Zero file translation on CUDA (PyTorch, bf16).

  .venv/bin/python main.py input.wav                       # upstream weights/
  .venv/bin/python main.py input.wav --checkpoint finetune/runs/ewe_full/best.safetensors
  .venv/bin/python main.py input.wav --config student/configs/hibiki_m_12l_ar.json \\
      --checkpoint RUN/ar_distill/model_step010000.safetensors

Writes the translated 24 kHz speech (default translations/<stem>_translated.wav)
and its text (default: the wav path with .txt). An 8 s silence tail flushes the
simultaneous-translation lag; generation stops at the text EOS.
"""
import argparse
import json
import time
from pathlib import Path

import sphn
import torch
from moshi.models import LMGen, loaders
from moshi.run_inference import get_condition_tensors

from finetune.hibiki_helpers import audio_read, decode_outputs, encode_inputs
from finetune.utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_TOKENIZER,
    REPO_ROOT,
)


def load(args: argparse.Namespace):
    """Return (lm, lm_gen, mimi, tokenizer) for an upstream, fine-tuned, or student checkpoint."""
    info = loaders.CheckpointInfo.from_hf_repo(
        "kyutai/hibiki-zero-3b-pytorch-bf16",
        moshi_weights=args.checkpoint,
        mimi_weights=DEFAULT_MIMI_WEIGHT,
        tokenizer=DEFAULT_TOKENIZER,
        config_path=args.config,
    )
    cfg = json.loads(Path(args.config).read_text())
    gen_cls = LMGen
    if cfg.get("head") is None:  # upstream / fine-tuned Hibiki-Zero
        lm = info.get_moshi(device=args.device, dtype=torch.bfloat16)
    else:  # student config: strip student metadata moshi doesn't understand
        from student.contract import torch_lm_config

        info.lm_config = torch_lm_config(info.lm_config)
        if cfg["head"] == "parallel_v1":
            from student.parallel import ParallelLMGen, load_parallel_lm

            lm = load_parallel_lm(cfg, Path(args.checkpoint), torch.device(args.device), torch.bfloat16)
            gen_cls = ParallelLMGen
        else:
            lm = info.get_moshi(device=args.device, dtype=torch.bfloat16)
    lm.eval()
    conditions = get_condition_tensors(info.model_type, lm, batch_size=1, cfg_coef=1.0)
    lm_gen = gen_cls(lm, condition_tensors=conditions, **info.lm_gen_config)
    mimi = info.get_mimi(device=args.device)
    return lm, lm_gen, mimi, info.get_text_tokenizer()


def translate(args: argparse.Namespace) -> None:
    lm, lm_gen, mimi, tokenizer = load(args)
    wav = audio_read(Path(args.input), to_sample_rate=mimi.sample_rate, mono=True)[0]
    duration = wav.shape[-1] / mimi.sample_rate
    padded = torch.zeros(1, 1, int((duration + args.tail_s) * mimi.sample_rate))
    padded[0, :, : wav.shape[-1]] = wav[:1]
    codes, warmup_codes = encode_inputs(padded, mimi, lm_gen, [duration])

    eos_id = tokenizer.eos_id()
    text_tokens, audio_tokens = [], []
    start = time.perf_counter()
    with torch.no_grad(), lm_gen.streaming(1):
        for step in range(warmup_codes.shape[-1]):
            lm_gen.step(warmup_codes[:, :, step : step + 1])
        for step in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, step : step + 1])
            if tokens is None:
                continue
            text_tokens.append(tokens[:, 0])
            audio_tokens.append(tokens[:, 1:])
            if int(tokens[0, 0, 0]) == eos_id:
                break
    wall = time.perf_counter() - start
    (out_wav, text), = decode_outputs(
        torch.cat(audio_tokens, dim=-1), torch.cat(text_tokens, dim=-1), mimi, tokenizer
    )

    out = Path(args.out or REPO_ROOT / "translations" / f"{Path(args.input).stem}_translated.wav")
    text_out = Path(args.text_out) if args.text_out else out.with_suffix(".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    text_out.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(out, out_wav[0].float().numpy(), mimi.sample_rate)
    text_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(
        f"\n[{duration:.1f}s input, {len(text_tokens)} frames in {wall:.1f}s "
        f"({duration / wall:.2f}x RT), out: {out} ({out_wav.shape[-1] / mimi.sample_rate:.1f}s), "
        f"text: {text_out}]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Hibiki-Zero file translation (PyTorch CUDA)")
    p.add_argument("input", help="audio file to translate")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_WEIGHT,
                   help="LM safetensors: upstream, finetune/train.py output, or student export")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LM config.json")
    p.add_argument("-o", "--out", help="output wav; default translations/<stem>_translated.wav")
    p.add_argument("--text-out", help="output text; default the output wav with .txt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tail-s", type=float, default=8.0, help="silence appended to flush the lag")
    p.add_argument("--seed", type=int, default=299792458)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    translate(args)


if __name__ == "__main__":
    main()
