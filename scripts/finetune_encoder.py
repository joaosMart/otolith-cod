#!/usr/bin/env python3
"""
Fine-tune a vision encoder for otolith age classification.

Replaces the earlier model-specific script. One entry point covers every
condition the paper reports, selected with --mode:

    lora        LoRA adapters only. The headline configuration: the pretrained
                weights are untouched and only the low-rank matrices train.
    lora+pool   LoRA plus a full-rank unfreeze of the encoder's pooling head.
                Reported as an ablation, because it trains several times more
                parameters than LoRA alone and the distinction matters for any
                claim about parameter efficiency.
    full        Full fine-tuning. The baseline for the claim that a dataset this
                size overfits when the whole encoder is trained.
    probe       Linear probe on a frozen encoder. Lower bound.

Every run writes an output directory named from its own settings, so sweeps over
seed, rank or training fraction cannot overwrite one another.

Examples:
    python scripts/finetune_encoder.py --encoder siglip2
    python scripts/finetune_encoder.py --encoder dinov2 --seed 7
    python scripts/finetune_encoder.py --encoder siglip2 --mode lora+pool
    python scripts/finetune_encoder.py --encoder siglip2 --mode full --amp
    python scripts/finetune_encoder.py --encoder siglip2 --rank 4 --lora-alpha 8
    python scripts/finetune_encoder.py --encoder siglip2 --train-fraction 0.25
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from peft import LoraConfig, get_peft_model

from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss

from src.data import OtolithDataset, save_split_by_ids
from src.data.splits import load_split_by_ids
from src.features.preprocessing import (
    DEFAULT_RESIZE_MODE,
    RESIZE_MODES,
    PreprocessConfig,
    build_eval_transform,
    build_train_transform,
)
from src.features.registry import (
    ENCODERS,
    adaptable_module,
    count_parameters,
    embed_images,
    load_encoder,
)
from src.utils import load_config, get_device


MODES = ("lora", "lora+pool", "full", "probe")

#: Sensible learning rate per mode. Full fine-tuning needs a far smaller step
#: than adapter training, and a linear probe tolerates a much larger one.
#: Using one default for all three is a common way to make full fine-tuning look
#: artificially bad, which would undermine the very comparison it exists for.
DEFAULT_LR = {"lora": 1e-4, "lora+pool": 1e-4, "full": 1e-5, "probe": 1e-3}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a vision encoder for otolith age classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--encoder", default="siglip2", choices=sorted(ENCODERS))
    p.add_argument("--mode", default="lora", choices=MODES)
    p.add_argument("--config", default="configs/config.yaml")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. Defaults to a mode-appropriate value.")
    p.add_argument("--head-lr-multiplier", type=float, default=10.0,
                   help="The ordinal head trains at this multiple of the encoder rate.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Fraction of the fine-tuning training split to use. "
                        "Below 1.0 this produces a learning-curve point; the "
                        "subset is drawn stratified so class balance is kept.")

    p.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resize-mode", default=DEFAULT_RESIZE_MODE, choices=RESIZE_MODES,
                   help="How a wide otolith crop is fitted to the square input.")

    p.add_argument("--split-file", default=None,
                   help="Fixed train/test split by measurement ID. Pass the same "
                        "file used for the frozen baseline so the test set matches.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--run-name", default=None,
                   help="Overrides the auto-generated name derived from the settings.")

    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bfloat16 autocast on CUDA. Ignored on MPS and CPU.")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="Trade compute for memory. Usually needed for --mode full.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def build_run_name(args) -> str:
    """Name a run from everything that distinguishes it from another run.

    Every setting that changes the result appears here. That is what prevents the
    silent overwrite the previous pipeline was prone to, where two runs with
    different preprocessing produced the same output filename.
    """
    if args.run_name:
        return args.run_name

    parts = [args.encoder, args.mode.replace("+", "-")]
    if args.mode in ("lora", "lora+pool"):
        parts.append(f"r{args.rank}a{args.lora_alpha}")
    if args.train_fraction < 1.0:
        parts.append(f"f{args.train_fraction:g}")
    parts.append(f"s{args.seed}")

    slug = PreprocessConfig(
        apply_clahe=args.clahe, repeat_clahe=args.repeat_clahe,
        resize_mode=args.resize_mode,
    ).slug()
    if slug:
        parts.append(slug)
    return "_".join(parts)


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Bitwise reproducibility across runs was absent from the earlier
        # pipeline, which made run-to-run variance impossible to separate from
        # a real effect. These two settings cost some throughput and buy that back.
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


class OrdinalHead(nn.Module):
    """Encoder plus a CORN ordinal regression head.

    The head emits K-1 logits, one per "is the age greater than k" threshold.
    The encoder readout comes from the registry so this class never needs to know
    which architecture it is wrapping.
    """

    def __init__(self, model, spec, num_classes: int):
        super().__init__()
        self.model = model
        self.spec = spec
        self.num_classes = num_classes
        self.fc = nn.Linear(spec.embedding_dim, num_classes - 1)

    def forward(self, pixel_values):
        # Not L2-normalised here: normalising before a linear head would discard
        # embedding magnitude, which the head can otherwise use. Extraction
        # normalises separately, matching the frozen pipeline.
        features = embed_images(self.model, pixel_values, self.spec, normalize=False)
        return self.fc(features)


def configure_trainable(model, spec, args):
    """Apply the requested adaptation mode and return the model plus a report."""
    mode = args.mode

    if mode == "probe":
        for param in model.parameters():
            param.requires_grad = False
        return model, {"strategy": "frozen encoder, linear head only"}

    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return model, {"strategy": "all encoder parameters trainable"}

    modules_to_save = list(spec.poolable_modules) if mode == "lora+pool" else None
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(spec.lora_targets),
        bias="none",
        modules_to_save=modules_to_save,
    )

    target = adaptable_module(model, spec)
    wrapped = get_peft_model(target, lora_config)

    # Re-attach in place so the rest of the model (a projection head, a text
    # tower) keeps working and the readout in the registry stays valid.
    if target is model:
        model = wrapped
    else:
        model.vision_model = wrapped

    return model, {
        "strategy": f"LoRA r={args.rank} alpha={args.lora_alpha} on {list(spec.lora_targets)}",
        "modules_to_save": modules_to_save or [],
    }


def stratified_subset(indices, labels, fraction, seed):
    """Take a class-balanced fraction of a set of indices."""
    if fraction >= 1.0:
        return indices
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=fraction, random_state=seed
    )
    subset_pos, _ = next(splitter.split(np.zeros(len(indices)), labels[indices]))
    return indices[subset_pos]


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def ordinal_metrics(preds, labels):
    diff = (preds - labels).abs().float()
    return {
        "acc": (diff == 0).float().mean().item(),
        "mae": diff.mean().item(),
        "within_1": (diff <= 1).float().mean().item(),
    }


def run_epoch(model, loader, num_classes, device, optimizer=None,
              scheduler=None, autocast_dtype=None):
    """One pass over a loader. Trains when an optimizer is supplied."""
    training = optimizer is not None
    model.train(training)

    total_loss, preds_all, labels_all = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = (labels - 1).long().to(device, non_blocking=True)

            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(images)
                    loss = corn_loss(logits.float(), labels, num_classes=num_classes)
            else:
                logits = model(images)
                loss = corn_loss(logits, labels, num_classes=num_classes)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds_all.append(corn_label_from_logits(logits.float()).long().detach())
            labels_all.append(labels.detach())

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    metrics = ordinal_metrics(preds_all, labels_all)
    metrics["loss"] = total_loss / labels_all.size(0)
    return metrics


def plot_curves(history, path):
    panels = [
        ("loss", "CORN loss", 1.0),
        ("acc", "Exact-match accuracy (%)", 100.0),
        ("mae", "MAE (age classes)", 1.0),
        ("within_1", "Within-1 accuracy (%)", 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (key, label, scale) in zip(axes.ravel(), panels):
        ax.plot([v * scale for v in history[f"train_{key}"]], label="Train")
        ax.plot([v * scale for v in history[f"val_{key}"]], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device()
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.mode]

    run_name = build_run_name(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)

    config = load_config(args.config)
    data_config = config["data"]
    age_range = tuple(data_config["age_range"])
    num_classes = age_range[1] - age_range[0] + 1

    preprocess = PreprocessConfig(
        apply_clahe=args.clahe, repeat_clahe=args.repeat_clahe,
        resize_mode=args.resize_mode,
    )

    print("=" * 68)
    print(f"Run: {run_name}")
    print(f"Encoder: {args.encoder}   Mode: {args.mode}   Seed: {args.seed}")
    print(f"Device: {device}   Learning rate: {lr:g}")
    print(f"Preprocessing: {preprocess.describe()}")
    print("=" * 68)

    model, processor, spec = load_encoder(args.encoder)
    image_processor = getattr(processor, "image_processor", processor)

    model, strategy = configure_trainable(model, spec, args)
    head = OrdinalHead(model, spec, num_classes).to(device)

    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    params = count_parameters(model)
    print(f"\nTrainable: {params['trainable']:,} of {params['total']:,} "
          f"({params['trainable_pct']:.2f}%)")
    if params["lora"]:
        print(f"  low-rank adapters: {params['lora']:,} ({params['lora_pct']:.2f}%)")
        print(f"  full-rank modules: {params['full_rank_trainable']:,} "
              f"({100 * params['full_rank_trainable'] / params['total']:.2f}%)")

    train_transform = build_train_transform(image_processor, preprocess,
                                            image_size=spec.image_size)
    eval_transform = build_eval_transform(image_processor, preprocess,
                                          image_size=spec.image_size)

    root = Path(data_config["root_dir"])
    metadata_csv = data_config["metadata_csv"]
    probe = OtolithDataset(root, transform=None, age_range=age_range,
                           metadata_csv=metadata_csv)
    labels = np.array([age for _, age in probe.samples])
    measurement_ids = np.array([int(p.stem) for p, _ in probe.samples])

    # Outer split: train pool vs held-out test. Reuse the frozen baseline's split
    # when given one, so both conditions are scored on identical images.
    if args.split_file and Path(args.split_file).exists():
        outer = load_split_by_ids(args.split_file, measurement_ids)
        train_pool, test_idx = outer.train_indices, outer.test_indices
        print(f"\nSplit loaded from {args.split_file}")
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_pool, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
        split_path = output_dir / "split.json"
        save_split_by_ids(split_path, {
            "seed": 42,
            "test_ratio": 0.15,
            "train_measurement_ids": measurement_ids[train_pool].tolist(),
            "test_measurement_ids": measurement_ids[test_idx].tolist(),
        })
        print(f"\nSplit created and written to {split_path}")

    # Inner split: fine-tuning train vs validation for early stopping. Carved out
    # of the training pool only, so the test set stays untouched by both stages.
    inner = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio,
                                   random_state=args.seed)
    ft_train_pos, ft_val_pos = next(
        inner.split(np.zeros(len(train_pool)), labels[train_pool])
    )
    ft_train = train_pool[ft_train_pos]
    ft_val = train_pool[ft_val_pos]

    if args.train_fraction < 1.0:
        before = len(ft_train)
        ft_train = stratified_subset(ft_train, labels, args.train_fraction, args.seed)
        print(f"Learning-curve point: {len(ft_train)} of {before} training images "
              f"({args.train_fraction:.0%})")

    train_ds = OtolithDataset(root, transform=train_transform, age_range=age_range,
                              indices=ft_train.tolist(), metadata_csv=metadata_csv)
    val_ds = OtolithDataset(root, transform=eval_transform, age_range=age_range,
                            indices=ft_val.tolist(), metadata_csv=metadata_csv)

    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)

    print(f"Fine-tune train: {len(train_ds)}   validation: {len(val_ds)}   "
          f"held-out test: {len(test_idx)}")

    encoder_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": head.fc.parameters(), "lr": lr * args.head_lr_multiplier}]
    if encoder_params:
        groups.insert(0, {"params": encoder_params, "lr": lr})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch
    )

    autocast_dtype = (
        torch.bfloat16 if (args.amp and device.type == "cuda"
                           and torch.cuda.is_bf16_supported()) else None
    )
    if autocast_dtype:
        print("Mixed precision: bfloat16 autocast")

    history = {f"{split}_{k}": [] for split in ("train", "val")
               for k in ("loss", "acc", "mae", "within_1")}
    best_val_mae = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(args.epochs):
        train_metrics = run_epoch(head, train_loader, num_classes, device,
                                  optimizer, scheduler, autocast_dtype)
        val_metrics = run_epoch(head, val_loader, num_classes, device,
                                autocast_dtype=autocast_dtype)

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)

        print(f"epoch {epoch + 1:>3}/{args.epochs}  "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f}  |  "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} "
              f"mae {val_metrics['mae']:.3f} within-1 {val_metrics['within_1']:.3f}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            if args.mode in ("lora", "lora+pool"):
                adaptable_module(model, spec).save_pretrained(str(output_dir / "adapter"))
            elif args.mode == "full":
                torch.save(model.state_dict(), output_dir / "encoder.pt")
            torch.save(head.fc.state_dict(), output_dir / "corn_head.pt")
            # Snapshot the metrics belonging to the checkpoint actually saved,
            # rather than the best value of each metric seen at any epoch.
            best_metrics = dict(val_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: {args.patience} epochs without improvement.")
                break

    elapsed = (time.time() - started) / 60
    plot_curves(history, output_dir / "training_curves.png")

    info = {
        "run_name": run_name,
        "args": vars(args),
        "resolved_lr": lr,
        "encoder": {"name": spec.name, "model_id": spec.model_id,
                    "embedding_dim": spec.embedding_dim,
                    "image_size": spec.image_size},
        "mode": args.mode,
        "strategy": strategy,
        "preprocessing": {**preprocess.__dict__, "description": preprocess.describe(),
                          "slug": preprocess.slug()},
        "parameters": params,
        "loss": "corn",
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "elapsed_minutes": round(elapsed, 2),
        "sizes": {"ft_train": len(train_ds), "ft_val": len(val_ds),
                  "test": len(test_idx)},
        "history": history,
    }
    with open(output_dir / "training_info.json", "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    print(f"\nBest epoch {best_epoch} with validation MAE {best_val_mae:.4f}")
    print(f"Finished in {elapsed:.1f} min. Artifacts in {output_dir}")


if __name__ == "__main__":
    main()
