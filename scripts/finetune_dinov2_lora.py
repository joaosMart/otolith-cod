#!/usr/bin/env python3
"""
Fine-tune vision models (DINOv2 / SigLIP2) with LoRA adapters for otolith age
classification.

Adds LoRA adapters to attention QKV projections and MLP layers, trains with a
CORN ordinal regression head, then saves only the adapter weights. The adapted
model can later be used for embedding extraction.

Usage:
    python scripts/finetune_dinov2_lora.py                        # DINOv2 (default)
    python scripts/finetune_dinov2_lora.py --model siglip2        # SigLIP2
    python scripts/finetune_dinov2_lora.py --model siglip2 --epochs 50 --lr 5e-5
"""

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor, AutoProcessor
from peft import LoraConfig, get_peft_model

from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss

from src.data import OtolithDataset
from src.features.extractor import clahe_enhancement
from src.utils import load_config, get_device


# Model configurations: (hf_id, embedding_dim, lora_targets, processor_type, cls_extract)
MODEL_CONFIGS = {
    "dinov2": {
        "model_id": "facebook/dinov2-with-registers-large",
        "embedding_dim": 1024,
        "lora_targets": ["query", "key", "value", "fc1", "fc2"],
        "processor": "image",  # AutoImageProcessor
    },
    "siglip2": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "embedding_dim": 1152,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
        "processor": "full",  # AutoProcessor
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune vision models with LoRA for otolith aging")
    parser.add_argument("--model", type=str, default="dinov2", choices=list(MODEL_CONFIGS.keys()),
                        help="Model to fine-tune (default: dinov2)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split from training data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply CLAHE enhancement before model preprocessing")
    parser.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply CLAHE twice for stronger effect")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/lora/<model>)")
    return parser.parse_args()


class VisionCORNHead(nn.Module):
    """Vision backbone + CORN ordinal regression head for fine-tuning.

    Supports DINOv2 (CLS token from last_hidden_state) and SigLIP2 (pooler_output
    from the vision tower). CORN produces K-1 logits for K ordinal classes.

    Reference: Shi et al. (2021), "Deep Neural Networks for Rank-Consistent
    Ordinal Regression Based on Conditional Probabilities."
    """

    def __init__(self, backbone, num_classes=10, embedding_dim=1024, model_type="dinov2"):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model_type = model_type
        self.fc = nn.Linear(embedding_dim, num_classes - 1)

    def forward(self, pixel_values):
        if self.model_type == "siglip2":
            # SigLIP2: use vision_model to get pooler_output
            outputs = self.backbone.vision_model(pixel_values=pixel_values)
            embeddings = outputs.pooler_output
        else:
            # DINOv2: CLS token is the first token of last_hidden_state
            outputs = self.backbone(pixel_values=pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(embeddings)
        return logits


def build_augmentation_transform(processor, apply_clahe=True, repeat_clahe=True):
    """Build training augmentation pipeline that ends with model preprocessing."""
    size_dict = processor.size
    image_size = size_dict.get("height", size_dict.get("shortest_edge", size_dict.get("width", 518)))
    mean = processor.image_mean
    std = processor.image_std

    clahe_step = [transforms.Lambda(lambda img: clahe_enhancement(img, repeat_clahe=repeat_clahe))] if apply_clahe else []

    train_transform = transforms.Compose([
        *clahe_step,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
        ], p=0.5),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        *clahe_step,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds, labels):
    """Compute accuracy, MAE, and within-1 accuracy for ordinal predictions.

    All inputs are 0-indexed class labels (torch tensors on any device).
    """
    diff = (preds - labels).abs().float()
    acc = (diff == 0).float().mean().item()
    mae = diff.mean().item()
    within_1 = (diff <= 1).float().mean().item()
    return acc, mae, within_1


def train_one_epoch(model, dataloader, num_classes, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        # Labels are ages 1-10, shift to 0-indexed for CORN
        labels = (labels - 1).long().to(device)

        optimizer.zero_grad()
        logits = model(images)
        # corn_loss takes labels directly (not extended-binary levels)
        loss = corn_loss(logits, labels, num_classes=num_classes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = corn_label_from_logits(logits).long()
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    n = all_labels.size(0)
    acc, mae, within_1 = compute_metrics(all_preds, all_labels)
    return total_loss / n, acc, mae, within_1


@torch.no_grad()
def evaluate(model, dataloader, num_classes, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = (labels - 1).long().to(device)

        logits = model(images)
        loss = corn_loss(logits, labels, num_classes=num_classes)

        total_loss += loss.item() * images.size(0)
        preds = corn_label_from_logits(logits).long()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    n = all_labels.size(0)
    acc, mae, within_1 = compute_metrics(all_preds, all_labels)
    return total_loss / n, acc, mae, within_1


def plot_training_curves(history, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CORN loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot([a * 100 for a in history["train_acc"]], label="Train")
    ax.plot([a * 100 for a in history["val_acc"]], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exact-match Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["train_mae"], label="Train")
    ax.plot(history["val_mae"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (age classes)")
    ax.set_title("Mean Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot([a * 100 for a in history["train_within_1"]], label="Train")
    ax.plot([a * 100 for a in history["val_within_1"]], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Within-1 accuracy (%)")
    ax.set_title("Predictions within ±1 age class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {output_path}")


def main():
    args = parse_args()
    device = get_device()
    model_cfg = MODEL_CONFIGS[args.model]
    model_id = model_cfg["model_id"]

    print(f"\n{'='*60}")
    print(f"{args.model.upper()} LoRA FINE-TUNING (CORN ordinal regression)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    data_config = config["data"]

    # Load model
    print(f"\nLoading {model_id}...")
    backbone = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)

    if model_cfg["processor"] == "image":
        processor = AutoImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id).image_processor

    # Apply LoRA — for SigLIP2, apply only to vision_model to avoid text tower
    lora_target_modules = model_cfg["lora_targets"]
    modules_to_save = ["head", "post_layernorm"] if args.model == "siglip2" else None
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=modules_to_save,
    )
    if args.model == "siglip2":
        backbone.vision_model = get_peft_model(backbone.vision_model, lora_config)
    else:
        backbone = get_peft_model(backbone, lora_config)
    if args.model == "siglip2":
        backbone.vision_model.print_trainable_parameters()
    else:
        backbone.print_trainable_parameters()

    num_classes = data_config["age_range"][1] - data_config["age_range"][0] + 1
    embedding_dim = model_cfg["embedding_dim"]
    model = VisionCORNHead(
        backbone, num_classes=num_classes, embedding_dim=embedding_dim, model_type=args.model,
    )
    model = model.to(device)

    # Build transforms
    print(f"CLAHE: {'ON' if args.clahe else 'OFF'}" + (f" (repeat={args.repeat_clahe})" if args.clahe else ""))
    train_transform, val_transform = build_augmentation_transform(
        processor, apply_clahe=args.clahe, repeat_clahe=args.repeat_clahe
    )

    # Load full dataset (no transform yet, we'll apply per-split)
    metadata_csv = data_config.get("metadata_csv")
    root_dir = data_config["root_dir"]
    full_dataset = OtolithDataset(
        root_dir=root_dir,
        transform=None,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=metadata_csv,
    )
    labels = full_dataset.get_labels()
    all_paths = full_dataset.get_paths()
    all_measurement_ids = [int(p.stem) for p in all_paths]
    print(f"\nDataset: {len(full_dataset)} images")
    print(f"Class distribution: {full_dataset.get_class_counts()}")

    # Split: use the same 85/15 train/test split as the main pipeline,
    # then split the 85% training portion into 80/20 for FT-train/FT-val
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_pool_idx, _test_idx = next(sss_outer.split(np.arange(len(labels)), labels))

    train_pool_labels = labels[train_pool_idx]
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_ratio, random_state=args.seed
    )
    ft_train_rel, ft_val_rel = next(sss_inner.split(
        np.arange(len(train_pool_idx)), train_pool_labels
    ))
    ft_train_idx = train_pool_idx[ft_train_rel]
    ft_val_idx = train_pool_idx[ft_val_rel]

    print(f"FT-train: {len(ft_train_idx)}, FT-val: {len(ft_val_idx)}, Test (held out): {len(_test_idx)}")

    # Create datasets with appropriate transforms
    train_dataset = OtolithDataset(
        root_dir=root_dir,
        transform=train_transform,
        age_range=tuple(data_config["age_range"]),
        indices=ft_train_idx.tolist(),
        metadata_csv=metadata_csv,
    )
    val_dataset = OtolithDataset(
        root_dir=root_dir,
        transform=val_transform,
        age_range=tuple(data_config["age_range"]),
        indices=ft_val_idx.tolist(),
        metadata_csv=metadata_csv,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},         
        {"params": model.fc.parameters(), "lr": args.lr * 10},           
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"LoRA: rank={args.rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Loss: CORN (ordinal regression, {num_classes} classes)")
    print("-" * 60)

    history = {
        "train_loss": [], "train_acc": [], "train_mae": [], "train_within_1": [],
        "val_loss": [], "val_acc": [], "val_mae": [], "val_within_1": [],
    }
    best_val_mae = float("inf")
    patience_counter = 0
    best_epoch = 0

    output_dir = Path(args.output_dir or f"outputs/lora/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc, train_mae, train_w1 = train_one_epoch(
            model, train_loader, num_classes, optimizer, scheduler, device
        )
        val_loss, val_acc, val_mae, val_w1 = evaluate(
            model, val_loader, num_classes, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_mae"].append(train_mae)
        history["train_within_1"].append(train_w1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_mae"].append(val_mae)
        history["val_within_1"].append(val_w1)

        improved = val_mae < best_val_mae
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc*100:.1f}% MAE {train_mae:.3f} w1 {train_w1*100:.1f}% | "
            f"Val loss {val_loss:.4f} acc {val_acc*100:.1f}% MAE {val_mae:.3f} w1 {val_w1*100:.1f}%"
            + (" *" if improved else "")
        )

        if improved:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best LoRA adapter weights + classifier head
            if args.model == "siglip2":
                backbone.vision_model.save_pretrained(str(output_dir / "lora_adapter"))
            else:
                backbone.save_pretrained(str(output_dir / "lora_adapter"))
            torch.save(model.fc.state_dict(), output_dir / "corn_head.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best: epoch {best_epoch})")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val MAE: {best_val_mae:.3f} at epoch {best_epoch}")
    print(f"Best val acc: {max(history['val_acc'])*100:.1f}%")
    print(f"Best val within-1: {max(history['val_within_1'])*100:.1f}%")

    # Save training curves
    plot_training_curves(history, str(output_dir / "training_curves.png"))

    # Save training config + history
    training_info = {
        "args": vars(args),
        "model_id": model_id,
        "loss": "corn",
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_acc": max(history["val_acc"]),
        "best_val_within_1": max(history["val_within_1"]),
        "final_train_acc": history["train_acc"][-1],
        "elapsed_minutes": elapsed / 60,
        "ft_train_size": len(ft_train_idx),
        "ft_val_size": len(ft_val_idx),
        "test_size": len(_test_idx),
        "test_measurement_ids": [all_measurement_ids[i] for i in _test_idx],
        "train_measurement_ids": [all_measurement_ids[i] for i in train_pool_idx],
        "history": history,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"\nLoRA adapter saved to: {output_dir / 'lora_adapter'}")
    print(f"CORN head saved to:    {output_dir / 'corn_head.pt'}")
    print(f"Training info saved to: {output_dir / 'training_info.json'}")


if __name__ == "__main__":
    main()