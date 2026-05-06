#!/usr/bin/env python3
"""
Visualize DINOv2 attention maps before and after LoRA fine-tuning.

Extracts attention from the last transformer layer, averages across heads,
and overlays heatmaps on original otolith images. Generates side-by-side
comparisons (frozen vs LoRA-adapted).

Usage:
    python scripts/visualize_attention.py
    python scripts/visualize_attention.py --n-samples 10 --layer -1
    python scripts/visualize_attention.py --sample-indices 0,50,100
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from peft import PeftModel

from src.data import OtolithDataset
from src.features.extractor import clahe_enhancement
from src.utils import load_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DINOv2 attention maps")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--adapter-path", type=str, default="outputs/lora/lora_adapter")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of random samples")
    parser.add_argument("--sample-indices", type=str, default=None,
                        help="Comma-separated specific sample indices")
    parser.add_argument("--layer", type=int, default=-1, help="Transformer layer index for attention")
    parser.add_argument("--method", type=str, default="mean_self", choices=["cls", "mean_self"],
                        help="Attention aggregation: 'cls' (CLS-to-patch) or 'mean_self' (patch self-attention)")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/attention_maps")
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply CLAHE enhancement before model preprocessing")
    parser.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply CLAHE twice for stronger effect")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_attention_maps(model, pixel_values, layer_idx=-1):
    """Extract attention maps from a specific transformer layer.

    Returns attention of shape (num_heads, num_patches, num_patches) for CLS token
    attending to patch tokens.
    """
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values, output_attentions=True)

    # attentions is a tuple of (batch, num_heads, seq_len, seq_len) per layer
    attn = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)
    return attn


def _get_patch_layout(seq_len, num_registers=4):
    """Compute patch grid dimensions from sequence length.

    DINOv2-with-registers layout: [CLS, reg1..regN, patch1..patchM]
    """
    num_patches = seq_len - 1 - num_registers
    num_patches_side = int(num_patches ** 0.5)
    assert num_patches_side * num_patches_side == num_patches, (
        f"Expected square patch grid, got {num_patches} patches"
    )
    return num_patches, num_patches_side, 1 + num_registers  # patch_offset


def attention_to_heatmap(attn, method="cls"):
    """Convert attention weights to a spatial heatmap.

    Args:
        attn: (num_heads, seq_len, seq_len) attention weights for one image
        method: "cls" for CLS-to-patch attention,
                "mean_self" for mean patch self-attention (how much each patch is attended to)

    Returns:
        heatmap: (H, W) normalized attention heatmap
    """
    num_heads, seq_len, _ = attn.shape
    num_patches, side, patch_offset = _get_patch_layout(seq_len)

    if method == "cls":
        # CLS token (row 0) attending to patch tokens, averaged across heads
        cls_attn = attn.mean(dim=0)[0]  # (seq_len,)
        patch_attn = cls_attn[patch_offset:]  # skip CLS + registers
    elif method == "mean_self":
        # How much each patch token is attended to by all other patch tokens
        # Take patch-to-patch submatrix and average columns (= received attention)
        patch_to_patch = attn[:, patch_offset:, patch_offset:]  # (heads, P, P)
        patch_attn = patch_to_patch.mean(dim=(0, 1))  # (P,) mean attention received
    else:
        raise ValueError(f"Unknown method: {method}")

    heatmap = patch_attn.reshape(side, side).cpu().numpy()
    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def plot_attention_comparison(image, heatmap_frozen, heatmap_lora, age, sample_idx, output_path):
    """Plot side-by-side attention maps: original, frozen, LoRA."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original (Age {age})")
    axes[0].axis("off")

    # Frozen attention
    axes[1].imshow(image)
    h, w = image.size[1], image.size[0]  # PIL: (W, H)
    heatmap_resized = np.array(Image.fromarray((heatmap_frozen * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
    axes[1].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[1].set_title("Frozen DINOv2")
    axes[1].axis("off")

    # LoRA attention
    axes[2].imshow(image)
    heatmap_resized = np.array(Image.fromarray((heatmap_lora * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
    axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
    axes[2].set_title("LoRA-adapted DINOv2")
    axes[2].axis("off")

    plt.suptitle(f"Sample {sample_idx} — Attention Maps (Last Layer, Head Average)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_average_attention(avg_frozen, avg_lora, output_path):
    """Plot dataset-averaged attention heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(avg_frozen, cmap="jet")
    axes[0].set_title("Frozen DINOv2 (Avg)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(avg_lora, cmap="jet")
    axes[1].set_title("LoRA-adapted DINOv2 (Avg)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Average Attention Maps Across Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = get_device()
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print("ATTENTION MAP VISUALIZATION")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Load config + dataset (raw images for visualization)
    config = load_config(args.config)
    data_config = config["data"]

    dataset = OtolithDataset(
        root_dir=data_config["root_dir"],
        transform=None,  # raw PIL images
        age_range=tuple(data_config["age_range"]),
        metadata_csv=data_config.get("metadata_csv"),
    )

    # Select sample indices
    if args.sample_indices:
        indices = [int(x) for x in args.sample_indices.split(",")]
    else:
        indices = np.random.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False)
        indices = sorted(indices.tolist())
    print(f"Visualizing {len(indices)} samples: {indices}")

    # Load models
    model_id = "facebook/dinov2-with-registers-large"
    processor = AutoImageProcessor.from_pretrained(model_id)


    print(f"\nLoading frozen DINOv2...")
    frozen_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, output_attentions=True)
    frozen_model = frozen_model.to(device).eval()

    adapter_path = Path(args.adapter_path)
    has_lora = adapter_path.exists()
    lora_model = None
    if has_lora:
        print(f"Loading LoRA-adapted DINOv2 from: {adapter_path}")
        base_for_lora = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, output_attentions=True)
        lora_model = PeftModel.from_pretrained(base_for_lora, str(adapter_path))
        lora_model = lora_model.to(device).eval()
    else:
        print(f"Warning: No LoRA adapter found at {adapter_path}, showing frozen-only.")

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    all_heatmaps_frozen = []
    all_heatmaps_lora = []

    for i, idx in enumerate(indices):
        pil_image, age = dataset[idx]

        # Preprocess for model (apply CLAHE if enabled)
        model_input_image = clahe_enhancement(pil_image, repeat_clahe=args.repeat_clahe) if args.clahe else pil_image
        inputs = processor(images=model_input_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Frozen attention
        attn_frozen = get_attention_maps(frozen_model, pixel_values, layer_idx=args.layer)
        heatmap_frozen = attention_to_heatmap(attn_frozen[0], method=args.method)
        all_heatmaps_frozen.append(heatmap_frozen)

        if has_lora:
            attn_lora = get_attention_maps(lora_model, pixel_values, layer_idx=args.layer)
            heatmap_lora = attention_to_heatmap(attn_lora[0], method=args.method)
            all_heatmaps_lora.append(heatmap_lora)

            plot_attention_comparison(
                pil_image, heatmap_frozen, heatmap_lora, age, idx,
                str(output_dir / f"attention_sample_{idx}.png"),
            )
        else:
            # Frozen-only plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(pil_image)
            axes[0].set_title(f"Original (Age {age})")
            axes[0].axis("off")
            h, w = pil_image.size[1], pil_image.size[0]
            hm = np.array(Image.fromarray((heatmap_frozen * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR))
            axes[1].imshow(pil_image)
            axes[1].imshow(hm, cmap="jet", alpha=0.5)
            axes[1].set_title("Frozen DINOv2")
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(str(output_dir / f"attention_sample_{idx}.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"  [{i+1}/{len(indices)}] Sample {idx} (age {age}) done")

    # Average attention maps
    avg_frozen = np.mean(all_heatmaps_frozen, axis=0)
    if has_lora and all_heatmaps_lora:
        avg_lora = np.mean(all_heatmaps_lora, axis=0)
        plot_average_attention(avg_frozen, avg_lora, str(output_dir / "attention_average.png"))
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(avg_frozen, cmap="jet")
        ax.set_title("Frozen DINOv2 (Avg Attention)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.savefig(str(output_dir / "attention_average.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nAttention maps saved to: {output_dir}")


if __name__ == "__main__":
    main()
