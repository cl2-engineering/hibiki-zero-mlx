#!/usr/bin/env python
"""Full-model Hibiki-Zero Ewe-to-English fine-tuning on CUDA."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from finetune import common  # noqa: E402
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_RUN_DIR,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

# Changing any of these breaks exact resume (data order or optimizer semantics).
RESUME_KEYS = ("cache_dir", "batch_size", "grad_accum_steps", "max_frames", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, nargs="+", default=[DEFAULT_CACHE_ROOT / "train"])
    parser.add_argument("--val-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT / "validation")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=280, help="Drop longer train samples.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear LR warmup.")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-pad-loss-weight", type=float, default=0.05)
    parser.add_argument("--val-max-frames", type=int, default=0, help="0 keeps every val sample.")
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=0, help="0 means all.")
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path, help="trainer_stepNNNNNN.pt to resume.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "grad_accum_steps", "val_batch_size"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    return args


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def checkpoint_pairs(out_dir: Path) -> list[tuple[int, Path, Path]]:
    models = {
        int(path.stem.removeprefix("model_step")): path
        for path in out_dir.glob("model_step*.safetensors")
        if path.stem.removeprefix("model_step").isdigit()
    }
    trainers = {
        int(path.stem.removeprefix("trainer_step")): path
        for path in out_dir.glob("trainer_step*.pt")
        if path.stem.removeprefix("trainer_step").isdigit()
    }
    return [(step, models[step], trainers[step]) for step in sorted(models.keys() & trainers.keys())]


def clean_incomplete_checkpoints(out_dir: Path) -> None:
    paired = {path for _, model, trainer in checkpoint_pairs(out_dir) for path in (model, trainer)}
    for pattern in ("model_step*.safetensors", "trainer_step*.pt", ".*.tmp"):
        for path in out_dir.glob(pattern):
            if path not in paired:
                path.unlink()


def save_checkpoint(model: Any, optimizer: Any, args: argparse.Namespace, step: int, out_dir: Path) -> None:
    clean_incomplete_checkpoints(out_dir)
    model_path = out_dir / f"model_step{step:06d}.safetensors"
    trainer_path = out_dir / f"trainer_step{step:06d}.pt"
    if not (model_path.is_file() and trainer_path.is_file()):
        common.save_model(model, model_path, {"base_model": repo_display_path(args.model_weight)})
        temporary = trainer_path.with_name(f".{trainer_path.name}.tmp")
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict(), "model": model_path.name}, temporary
        )
        temporary.replace(trainer_path)
        print(f"Saved checkpoint step {step} -> {repo_display_path(trainer_path)}")
    pairs = checkpoint_pairs(out_dir)
    if args.keep_checkpoints > 0:
        for _, model, trainer in pairs[: max(0, len(pairs) - args.keep_checkpoints)]:
            trainer.unlink()
            model.unlink()


def load_resume_checkpoint(model: Any, optimizer: Any, resume_path: Path, device: torch.device) -> int:
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model_path = require_file(resume_path.parent / checkpoint["model"], "resume model")
    common.load_model(model, model_path, torch.float32)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    step = int(checkpoint["step"])
    print(f"Resumed step {step} from {repo_display_path(resume_path)}")
    return step


def promote_best(model: Any, step: int, loss: float, out_dir: Path, best: dict | None) -> dict:
    """Keep best.safetensors = lowest teacher-forced validation loss so far."""
    if best is not None and loss >= float(best["validation_loss"]):
        return best
    best_path = out_dir / "best.safetensors"
    step_path = out_dir / f"model_step{step:06d}.safetensors"
    best_path.unlink(missing_ok=True)
    if step_path.is_file():
        os.link(step_path, best_path)
    else:
        common.save_model(model, best_path, {"step": str(step)})
    best = {"step": step, "validation_loss": loss}
    atomic_write_text(json.dumps(best, indent=2) + "\n", out_dir / "best.json")
    print(f"New best validation loss={loss:.4f} at step {step} -> {repo_display_path(best_path)}")
    return best


def main() -> None:
    args = parse_args()
    common.seed_all(args.seed)
    device = common.check_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Training requires CUDA")
    torch.set_float32_matmul_precision("high")

    args.cache_dir = [require_dir(path, "train cache directory") for path in args.cache_dir]
    args.val_cache_dir = require_dir(args.val_cache_dir, "validation cache directory")
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "upstream model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: [str(p) for p in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config_path = out_dir / "run_config.json"
    if args.resume_checkpoint is not None and run_config_path.is_file():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key in RESUME_KEYS:
            if previous.get(key) != run_config[key]:
                raise RuntimeError(f"Resume changed {key}: {previous.get(key)} -> {run_config[key]}")
    atomic_write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", run_config_path)
    clean_incomplete_checkpoints(out_dir)
    best_path = out_dir / "best.json"
    best = json.loads(best_path.read_text()) if best_path.is_file() else None

    dataset = common.CachedCodeDataset(args.cache_dir, False, 0, args.max_frames)
    exposure = dataset.exposure()
    print(
        f"Loaded {len(dataset)} train samples; assembled_frames={exposure['assembled_frames']:,} "
        f"source_hours={exposure['source_hours']:.2f}"
    )
    val_dataset = common.CachedCodeDataset(args.val_cache_dir, True, 0, args.val_max_frames)
    val_dataloader = common.make_cached_dataloader(
        val_dataset, args.val_batch_size, args.num_workers, True, shuffle=False
    )
    print(f"Loaded {len(val_dataset)} validation samples")

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = batches_per_epoch // args.grad_accum_steps
    if not steps_per_epoch:
        raise ValueError("Dataset has fewer batches than --grad-accum-steps")
    micro_per_epoch = steps_per_epoch * args.grad_accum_steps
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    checkpoint_info = common.load_checkpoint_info(args)
    print(f"Loading upstream Hibiki-Zero from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(
        device=device,
        dtype=torch.float32,
        lm_kwargs_overrides={"gradient_checkpointing": args.gradient_checkpointing},
    )
    lm.train()
    common.enable_full_finetune(lm)
    params = common.trainable_parameters(lm)
    print(f"Trainable params: {sum(param.numel() for param in params):,}")
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    global_step = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(lm, optimizer, args.resume_checkpoint, device)
    if global_step >= total_steps:
        raise RuntimeError(f"Already at step {global_step} >= total_steps {total_steps}")

    def epoch_iterator(epoch: int, skip: int) -> Any:
        order = common.bucketed_epoch_order(dataset, args.batch_size, args.seed, epoch)
        loader = common.make_cached_dataloader(
            dataset, args.batch_size, args.num_workers, False, shuffle=False, sample_order=order
        )
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        return iterator

    micro_index = global_step * args.grad_accum_steps
    epoch = micro_index // micro_per_epoch
    data_iter = epoch_iterator(epoch, micro_index % micro_per_epoch)

    def next_batch() -> dict[str, Any]:
        nonlocal micro_index, epoch, data_iter
        if micro_index // micro_per_epoch != epoch:
            epoch = micro_index // micro_per_epoch
            data_iter = epoch_iterator(epoch, 0)
        micro_index += 1
        return next(data_iter)

    condition_cache: dict[int, Any | None] = {}
    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    log_keys = ("loss", "audio_loss", "text_loss", "content_text_loss", "pad_text_loss")
    log_sums = {key: 0.0 for key in log_keys}
    log_micro = 0
    log_samples = 0
    log_max_frames = 0
    last_log_time = time.time()

    def run_validation(step: int) -> None:
        nonlocal best
        metrics = common.evaluate_teacher_forced(
            lm,
            val_dataloader,
            device,
            checkpoint_info.model_type,
            args.audio_loss_weight,
            args.text_loss_weight,
            args.val_batches,
            "prefix",
            args.text_pad_loss_weight,
        )
        item = {"step": step, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}}
        with val_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={metrics['loss']:.4f} "
            f"audio={metrics['audio_loss']:.4f} text={metrics['text_loss']:.4f} "
            f"content={metrics['content_text_loss']:.4f} "
            f"acc={metrics['content_acc']:.3f} pad_acc={metrics['pad_acc']:.3f}"
        )
        best = promote_best(lm, step, float(metrics["loss"]), out_dir, best)

    while global_step < total_steps:
        lr_value = args.lr * min(1.0, (global_step + 1) / args.warmup_steps) if args.warmup_steps else args.lr
        for group in optimizer.param_groups:
            group["lr"] = lr_value
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            batch = next_batch()
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = common.batch_condition_tensors(
                    lm, checkpoint_info.model_type, batch_size
                )
            with autocast:
                losses = common.compute_batch_losses(
                    lm,
                    codes,
                    condition_cache[batch_size],
                    args.audio_loss_weight,
                    args.text_loss_weight,
                    text_pad_loss_weight=args.text_pad_loss_weight,
                    text_pad_mode="prefix",
                )
            (losses["loss"] / args.grad_accum_steps).backward()
            for key in log_keys:
                log_sums[key] += losses[key].detach()
            log_micro += 1
            log_samples += batch_size
            log_max_frames = max(log_max_frames, int(batch["frames"].max()))
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        global_step += 1

        if args.log_every and global_step % args.log_every == 0:
            item = {key: float(log_sums[key]) / log_micro for key in log_keys}
            if not math.isfinite(item["loss"]):
                raise RuntimeError(f"Non-finite loss by step {global_step}")
            now = time.time()
            item.update(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": lr_value,
                    "samples": log_samples,
                    "max_frames": log_max_frames,
                    "sec_per_step": (now - last_log_time) * args.grad_accum_steps / log_micro,
                    **common.cuda_memory_stats(device),
                }
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(
                f"step={global_step} epoch={epoch + 1} loss={item['loss']:.4f} "
                f"audio={item['audio_loss']:.4f} text={item['text_loss']:.4f} "
                f"T<={log_max_frames} lr={lr_value:.1e} s/step={item['sec_per_step']:.2f} "
                f"peak_mem={item.get('cuda_max_allocated_gb', 0):.1f}GB"
            )
            log_sums = {key: 0.0 for key in log_keys}
            log_micro = log_samples = log_max_frames = 0
            last_log_time = now

        if global_step < total_steps:
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(lm, optimizer, args, global_step, out_dir)
            if args.val_every and global_step % args.val_every == 0:
                run_validation(global_step)

    save_checkpoint(lm, optimizer, args, global_step, out_dir)
    run_validation(global_step)
    print(f"Finished at step {global_step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()
