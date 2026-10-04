#!/usr/bin/env python
"""Build finetune/pairs/{split}.jsonl from a user CSV manifest.

Manifest columns: id,split,ewe_audio,en_audio,text_ee,text_en
(split is train/validation/test; relative audio paths resolve against the
manifest's directory; text_ee may be empty).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import (  # noqa: E402
    DEFAULT_PAIRS_DIR,
    VALID_SPLITS,
    read_manifest,
    repo_display_path,
    resolve_repo_path,
    write_pair_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PAIRS_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows per split, 0 means all.")
    parser.add_argument(
        "--min-source-duration-s",
        type=float,
        default=0.5,
        help="Drop rows whose Ewe audio is shorter than this.",
    )
    parser.add_argument(
        "--max-source-duration-s",
        type=float,
        default=20.0,
        help="Drop rows whose Ewe audio is longer than this, 0 disables.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing pair files.")
    return parser.parse_args()


def duration_s(path: str) -> float:
    return float(soundfile.info(str(resolve_repo_path(path))).duration)


def main() -> None:
    args = parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    rows = read_manifest(args.manifest)
    for split in VALID_SPLITS:
        selected = []
        dropped = 0
        for row in rows:
            if row["split"] != split:
                continue
            row["ewe_duration_s"] = f"{duration_s(row['ewe_audio']):.3f}"
            row["en_duration_s"] = f"{duration_s(row['en_audio']):.3f}"
            source_s = float(row["ewe_duration_s"])
            if source_s < args.min_source_duration_s or (
                args.max_source_duration_s and source_s > args.max_source_duration_s
            ):
                dropped += 1
                continue
            selected.append(row)
            if args.max_rows and len(selected) >= args.max_rows:
                break
        if not selected:
            continue
        out_path = out_dir / f"{split}.jsonl"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing pair file: {out_path}")
        write_pair_file(selected, out_path)
        hours = sum(float(row["ewe_duration_s"]) for row in selected) / 3600.0
        print(
            f"{split}: wrote {len(selected)} rows ({dropped} dropped by duration), "
            f"{hours:.2f} source hours -> {repo_display_path(out_path)}"
        )


if __name__ == "__main__":
    main()
