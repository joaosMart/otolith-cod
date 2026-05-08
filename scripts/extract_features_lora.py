#!/usr/bin/env python3
"""
Extract embeddings from LoRA-adapted vision models (DINOv2 / SigLIP2).

Loads the base model, merges LoRA adapter weights, and extracts
CLS / patch / mean-pool embeddings identical to the frozen pipeline.
Output is compatible with train_classifier.py.

Usage:
    python scripts/extract_features_lora.py                        # DINOv2 (default)
    python scripts/extract_features_lora.py --model siglip2        # SigLIP2
    python scripts/extract_features_lora.py --adapter-path outputs/lora/siglip2/lora_adapter --model siglip2
    python scripts/extract_features_lora.py --merge  # merge LoRA weights into base model
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, AutoProcessor
from peft import PeftModel

from src.data import OtolithDataset
from src.features import save_cached_embeddings
from src.features.extractor import clahe_enhancement
from src.utils import load_config, get_device


# Model configurations (must match finetune_dinov2_lora.py)
MODEL_CONFIGS = {
    "dinov2": {
        "model_id": "facebook/dinov2-with-registers-large",
        "processor": "image",  # AutoImageProcessor
    },
    "siglip2": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "processor": "full",  # AutoProcessor
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from LoRA-adapted vision models")
    parser.add_argument("--model", type=str, default="dinov2", choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture (default: dinov2)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to saved LoRA adapter (default: outputs/lora/<model>/lora_adapter)")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path (default: outputs/segmented_embeddings/<model>-lora_embeddings.npz)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights into base model (slightly faster inference)")
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply CLAHE enhancement before model preprocessing")
    parser.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply CLAHE twice for stronger effect")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    model_cfg = MODEL_CONFIGS[args.model]
    model_id = model_cfg["model_id"]

    print(f"\n{'='*60}")
    print(f"LORA-ADAPTED {args.model.upper()} FEATURE EXTRACTION")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # Check cache
    output_path = Path(args.output_path or f"outputs/segmented_embeddings/{args.model}-lora_embeddings.npz")
    if output_path.exists() and not args.force:
        print(f"\nCached embeddings found: {output_path}")
        print("Use --force to re-extract.")
        return

    # Load base model
    print(f"\nLoading base model: {model_id}")
    base_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)

    # Load LoRA adapter
    adapter_path = Path(args.adapter_path or f"outputs/lora/{args.model}/lora_adapter")
    if not adapter_path.exists():
        print(f"Error: Adapter not found at {adapter_path}")
        print("Run finetune_dinov2_lora.py first.")
        sys.exit(1)

    print(f"Loading LoRA adapter from: {adapter_path}")
    if args.model == "siglip2":
        # LoRA was applied to vision_model only during fine-tuning
        base_model.vision_model = PeftModel.from_pretrained(base_model.vision_model, str(adapter_path))
        if args.merge:
            print("Merging LoRA weights into base model...")
            base_model.vision_model = base_model.vision_model.merge_and_unload()
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        if args.merge:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()

    model = model.to(device)
    model.eval()

    # Processor
    if model_cfg["processor"] == "image":
        processor = AutoImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id).image_processor

    apply_clahe = args.clahe
    repeat_clahe = args.repeat_clahe
    print(f"CLAHE: {'ON' if apply_clahe else 'OFF'}" + (f" (repeat={repeat_clahe})" if apply_clahe else ""))

    def preprocess(image):
        if apply_clahe:
            image = clahe_enhancement(image, repeat_clahe=repeat_clahe)
        inputs = processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    # Load dataset
    config = load_config(args.config)
    data_config = config["data"]
    dataset = OtolithDataset(
        root_dir=data_config["root_dir"],
        transform=preprocess,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=data_config.get("metadata_csv"),
    )
    print(f"Dataset: {len(dataset)} images")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Extract features (same logic as frozen DINOv2 in extractor.py)
    all_cls = []
    all_patch = []
    all_mean_pool = []
    all_labels = []

    print("\nExtracting features...")
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)

            if args.model == "siglip2":
                outputs = model.vision_model(pixel_values=images)
                hidden = outputs.last_hidden_state
                # SigLIP2 has no CLS token; use pooler_output as "cls" and all patches
                cls_features = outputs.pooler_output
                patch_features = hidden
            else:
                outputs = model(pixel_values=images)
                hidden = outputs.last_hidden_state
                cls_features = hidden[:, 0, :]
                patch_features = hidden[:, 1:, :]

            mean_pool_features = patch_features.mean(dim=1)

            # L2 normalize
            cls_features = F.normalize(cls_features, p=2, dim=-1, eps=1e-8)
            mean_pool_features = F.normalize(mean_pool_features, p=2, dim=-1, eps=1e-8)
            patch_features = F.normalize(patch_features, p=2, dim=-1, eps=1e-8)

            all_cls.append(cls_features.cpu().numpy())
            all_patch.append(patch_features.cpu().numpy())
            all_mean_pool.append(mean_pool_features.cpu().numpy())
            all_labels.append(labels.numpy())

    features_dict = {
        "features_cls": np.concatenate(all_cls, axis=0),
        "features_patch": np.concatenate(all_patch, axis=0),
        "features_patch_mean_pool": np.concatenate(all_mean_pool, axis=0),
    }
    labels = np.concatenate(all_labels)

    # Extract measurement IDs
    paths = dataset.get_paths()
    measurement_ids = np.array([int(p.stem) for p in paths])

    # Save
    save_cached_embeddings(str(output_path), features_dict, labels, measurement_ids)
    print(f"\nFeature shapes:")
    for key, arr in features_dict.items():
        print(f"  {key}: {arr.shape}")
    print(f"Saved to: {output_path}")
    print("\nLoRA feature extraction complete!")


if __name__ == "__main__":
    main()
