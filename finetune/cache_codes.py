#!/usr/bin/env python
"""Cache Mimi codes and CTC-timed English text for Ewe->English full-model SFT."""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_PAIRS_DIR,
    DEFAULT_TOKENIZER,
    read_json,
    read_pair_file,
    repo_display_path,
    require_file,
    resolve_repo_path,
)

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
CACHE_FORMAT = "hibiki_ewe_cache_v1"
INDEX_FIELDS = (
    "id",
    "split",
    "shard",
    "frames",
    "ewe_frames",
    "en_frames",
    "text_tokens",
    "target_delay_s",
    "target_delay_frames",
    "ewe_audio",
    "en_audio",
    "alignment_score",
    "alignment_text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS_DIR / "train.jsonl",
        help="Pair file from finetune/build_pairs.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE_ROOT / "train")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-size", type=int, default=32, help="Samples per shard.")
    parser.add_argument("--limit", type=int, default=0, help="Max pairs to cache, 0 means all.")
    parser.add_argument(
        "--target-delay-ratio",
        type=float,
        default=0.5,
        help="Maximum target delay as a ratio of Ewe source duration.",
    )
    parser.add_argument("--target-delay-min-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic delays.")
    parser.add_argument("--alignment-batch-size", type=int, default=8)
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        default=0.5,
        help="Reject rows below this mean English forced-alignment posterior.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing shards.")
    return parser.parse_args()


def read_audio(path: Path, sphn: Any, torch: Any, device: Any, left_pad_s: float = 0.0) -> Any:
    wav, sr = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{path} loaded at {sr} Hz, expected {SAMPLE_RATE} Hz")
    wav = torch.from_numpy(wav).float()
    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.ndim != 2:
        raise ValueError(f"{path} has unsupported audio shape {tuple(wav.shape)}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    pad = int(round(left_pad_s * SAMPLE_RATE))
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (pad, 0))
    return wav[None].to(device)


def encode_audio(
    path: Path, mimi: Any, sphn: Any, torch: Any, device: Any, left_pad_s: float = 0.0
) -> Any:
    wav = read_audio(path, sphn, torch, device, left_pad_s=left_pad_s)
    with torch.no_grad():
        codes = mimi.encode(wav)
    if codes.ndim != 3 or codes.shape[0] != 1:
        raise RuntimeError(f"Mimi returned unexpected codes shape for {path}: {tuple(codes.shape)}")
    return codes[0].cpu().long()


def assemble_codes(
    torch: Any,
    row: dict[str, str],
    ewe_codes: Any,
    en_codes: Any,
    tokens: list[int],
    cfg: dict[str, Any],
    text_frames: list[int],
) -> Any:
    """Stack [text, English target audio, Ewe source audio + EOS] into one [K, T] tensor."""
    n_q = int(cfg["n_q"])
    dep_q = int(cfg["dep_q"])
    source_q = n_q - dep_q
    card = int(cfg["card"])
    text_card = int(cfg["text_card"])
    text_pad_id = int(cfg["existing_text_padding_id"])
    zero_id = -1

    if ewe_codes.shape[0] < source_q:
        raise RuntimeError(f"Ewe Mimi codes have {ewe_codes.shape[0]} codebooks, need {source_q}")
    if en_codes.shape[0] < dep_q:
        raise RuntimeError(f"English Mimi codes have {en_codes.shape[0]} codebooks, need {dep_q}")
    if any(token < 0 or token >= text_card for token in tokens):
        raise RuntimeError(f"Text token out of range for id={row['id']}")
    if len(text_frames) != len(tokens) or len(set(text_frames)) != len(tokens):
        raise ValueError(f"Text frames must be unique and match tokens for id={row['id']}")
    if text_frames != sorted(text_frames) or text_frames[0] < 0:
        raise ValueError(f"Text frames must be sorted and non-negative for id={row['id']}")

    ewe_codes = ewe_codes[:source_q]
    en_codes = en_codes[:dep_q]
    target_len = int(en_codes.shape[1])
    source_len = int(ewe_codes.shape[1])
    total_frames = max(text_frames[-1] + 1, target_len, source_len + 1)

    codes = torch.full((1 + n_q, total_frames), zero_id, dtype=torch.int32)
    codes[0].fill_(text_pad_id)
    codes[0, torch.tensor(text_frames, dtype=torch.long)] = torch.tensor(tokens, dtype=torch.int32)
    codes[0, text_frames[-1] + 1 :] = zero_id
    codes[1 : 1 + dep_q, :target_len] = en_codes.to(torch.int32)

    source_start = 1 + dep_q
    codes[source_start : source_start + source_q, :source_len] = ewe_codes.to(torch.int32)
    codes[source_start : source_start + source_q, source_len] = card
    return codes


def target_delay_s(row: dict[str, str], min_ratio: float, max_ratio: float, seed: int) -> float:
    if max_ratio == 0:
        return 0.0
    source_duration_s = float(row["ewe_duration_s"])
    rng = random.Random(f"{seed}:{row['split']}:{row['id']}")
    return rng.uniform(min_ratio * source_duration_s, max_ratio * source_duration_s)


def save_shard(torch: Any, payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def write_index(torch: Any, out_dir: Path) -> None:
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        for path in sorted(out_dir.glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu")
            for sample in payload["samples"]:
                item = {key: sample.get(key) for key in INDEX_FIELDS if key != "shard"}
                item["shard"] = path.name
                item["target_delay_s"] = f"{sample['target_delay_s']:.6f}"
                writer.writerow(item)


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if not 0 <= args.target_delay_min_ratio <= args.target_delay_ratio:
        raise ValueError("Target-delay ratios must satisfy 0 <= min <= max")
    delay_policy = {
        "min_ratio": args.target_delay_min_ratio,
        "max_ratio": args.target_delay_ratio,
        "seed": args.seed,
    }

    import sentencepiece
    import sphn
    import torch
    from moshi.models import loaders

    from finetune.text_timing import (
        EnglishCTCAligner,
        sentencepiece_groups,
        timed_sentencepiece_tokens,
    )

    device = torch.device(args.device)
    cfg = read_json(args.config_path)
    mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    tokenizer_path = require_file(args.tokenizer, "text tokenizer")
    pairs = read_pair_file(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError(f"No pairs to cache from {args.pairs}")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(out_dir.glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != CACHE_FORMAT or payload.get("target_delay") != delay_policy:
            raise RuntimeError(f"Refusing to mix cache formats/delay policies in {out_dir}: {path.name}")

    print(f"Loading Mimi on {device} from {repo_display_path(mimi_weight)}")
    num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
    mimi = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks, device=device)
    if int(cfg["card"]) != int(mimi.cardinality):
        raise RuntimeError(
            f"Config card={cfg['card']} does not match Mimi cardinality={mimi.cardinality}"
        )
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
    eos_id = int(tokenizer.eos_id())
    aligner = EnglishCTCAligner(device, args.min_alignment_score)

    rejected = 0
    for shard_index in range(math.ceil(len(pairs) / args.shard_size)):
        start = shard_index * args.shard_size
        rows = pairs[start : start + args.shard_size]
        out_path = out_dir / f"shard_{shard_index:05d}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing {repo_display_path(out_path)}")
            continue

        samples: list[dict[str, Any]] = []
        for row in rows:
            ewe_audio = require_file(row["ewe_audio"], f"Ewe audio for id={row['id']}")
            en_audio = require_file(row["en_audio"], f"English audio for id={row['id']}")
            delay_s = target_delay_s(
                row, args.target_delay_min_ratio, args.target_delay_ratio, args.seed
            )
            delay_frames = int(round(delay_s * FRAME_RATE))
            ewe_codes = encode_audio(ewe_audio, mimi, sphn, torch, device)
            en_codes = encode_audio(en_audio, mimi, sphn, torch, device, left_pad_s=delay_s)

            raw_en = read_audio(en_audio, sphn, torch, torch.device("cpu"))[0, 0].numpy()
            wav16 = sphn.resample(raw_en, SAMPLE_RATE, 16_000)
            try:
                groups = sentencepiece_groups(row["text_en"], tokenizer)
                alignment = aligner.align_many([wav16], [groups], args.alignment_batch_size)[0]
                if isinstance(alignment, Exception):
                    raise alignment
                tokens, text_frames = timed_sentencepiece_tokens(
                    groups, alignment, int(en_codes.shape[1]) - delay_frames, delay_frames, eos_id
                )
            except (RuntimeError, ValueError) as exc:
                print(f"Rejecting id={row['id']}: {exc}")
                rejected += 1
                continue
            codes = assemble_codes(torch, row, ewe_codes, en_codes, tokens, cfg, text_frames)
            samples.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "codes": codes,
                    "frames": int(codes.shape[1]),
                    "ewe_frames": int(ewe_codes.shape[1]),
                    "en_frames": int(en_codes.shape[1]),
                    "text_tokens": len(tokens),
                    "target_delay_s": delay_s,
                    "target_delay_frames": delay_frames,
                    "ewe_audio": repo_display_path(ewe_audio),
                    "en_audio": repo_display_path(en_audio),
                    "text_en": row["text_en"],
                    "text_ee": row["text_ee"],
                    "alignment_score": alignment.score,
                    "alignment_text": " ".join(spoken for _, spoken in groups),
                }
            )

        if not samples:
            print(f"Every row in shard {shard_index} was rejected; no shard written")
            continue
        payload = {
            "format": CACHE_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "alignment_min_score": args.min_alignment_score,
            "target_delay": delay_policy,
            "config": {
                key: int(cfg[key])
                for key in ("n_q", "dep_q", "card", "text_card", "existing_text_padding_id")
            },
            "samples": samples,
        }
        save_shard(torch, payload, out_path)
        print(f"Wrote {len(samples)} samples -> {repo_display_path(out_path)}")

    write_index(torch, out_dir)
    print(f"Rejected {rejected} rows. Index: {repo_display_path(out_dir / 'index.csv')}")


if __name__ == "__main__":
    main()
