#!/usr/bin/env python
"""Hibiki-Zero translation on CUDA (PyTorch, bf16), from a file or live microphone.

  .venv/bin/python main.py                                 # live: microphone -> speaker
  .venv/bin/python main.py input.wav                       # upstream weights/
  .venv/bin/python main.py input.wav --checkpoint finetune/runs/ewe_full/best.safetensors
  .venv/bin/python main.py input.wav --config student/configs/hibiki_m_12l_ar.json \\
      --checkpoint RUN/ar_distill/model_step010000.safetensors

Writes the translated 24 kHz speech (default translations/<stem>_translated.wav)
and its text (default: the wav path with .txt). An 8 s silence tail flushes the
simultaneous-translation lag; generation stops at the text EOS.
Live mode streams 80 ms frames until Ctrl+C; use headphones to avoid echo.
"""
import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np

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


def live(args: argparse.Namespace) -> None:
    import sounddevice as sd

    _, lm_gen, mimi, tokenizer = load(args)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    mic: queue.Queue[np.ndarray] = queue.Queue()
    speaker: queue.Queue[np.ndarray] = queue.Queue()

    def on_mic(indata, frames, time_info, status):
        mic.put(indata[:, 0].copy())

    def on_speaker(outdata, frames, time_info, status):
        try:
            outdata[:, 0] = speaker.get_nowait()
        except queue.Empty:
            outdata.fill(0)

    def step(pcm: np.ndarray) -> torch.Tensor | None:
        codes = mimi.encode(torch.from_numpy(pcm).to(args.device)[None, None])
        tokens = lm_gen.step(codes[:, :, :1])
        if tokens is not None:
            speaker.put(mimi.decode(tokens[:, 1:])[0, 0].float().cpu().numpy())
        return tokens

    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for _ in range(lm_gen.max_delay + 4):  # compile / CUDA-graph warmup on silence
            step(np.zeros(frame, dtype=np.float32))
        torch.cuda.synchronize()
        speaker.queue.clear()
        stream_args = dict(
            samplerate=mimi.sample_rate, blocksize=frame, channels=1, dtype="float32"
        )
        with sd.InputStream(device=args.input_device, callback=on_mic, **stream_args), \
                sd.OutputStream(device=args.output_device, callback=on_speaker, **stream_args):
            print("Listening... speak now (Ctrl+C to stop).\n", flush=True)
            try:
                while True:
                    tokens = step(mic.get())
                    if mic.qsize() > 5:
                        print(f"\n[behind real time by {mic.qsize() * 80} ms]", flush=True)
                    text = None if tokens is None else int(tokens[0, 0, 0])
                    if text is not None and text not in (0, 3, tokenizer.eos_id()):
                        piece = tokenizer.id_to_piece(text).replace("\u2581", " ")
                        print(piece, end="", flush=True)
            except KeyboardInterrupt:
                print()


def main() -> None:
    p = argparse.ArgumentParser(description="Hibiki-Zero translation (PyTorch CUDA)")
    p.add_argument("input", nargs="?", help="audio file to translate; omit for live microphone")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_WEIGHT,
                   help="LM safetensors: upstream, finetune/train.py output, or student export")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LM config.json")
    p.add_argument("-o", "--out", help="output wav; default translations/<stem>_translated.wav")
    p.add_argument("--text-out", help="output text; default the output wav with .txt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tail-s", type=float, default=8.0, help="silence appended to flush the lag")
    p.add_argument("--seed", type=int, default=299792458)
    p.add_argument("--input-device", help="live mode microphone (sounddevice index or name)")
    p.add_argument("--output-device", help="live mode speaker (sounddevice index or name)")
    args = p.parse_args()
    for name in ("input_device", "output_device"):
        value = getattr(args, name)
        if value is not None and value.isdigit():
            setattr(args, name, int(value))
    torch.manual_seed(args.seed)
    if args.input:
        translate(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
