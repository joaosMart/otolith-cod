#!/usr/bin/env python3
"""
Evaluate a LoRA-adapted model with its CORN ordinal regression head on the held-out test set.

Loads the base model, merges LoRA adapter weights, loads the CORN head,
and evaluates on the test split. Produces bootstrap CIs in the same format
as train_classifier.py --eval-mode bootstrap for direct comparison.

Usage:
    python scripts/evaluate_corn_head.py --model siglip2
    python scripts/evaluate_corn_head.py --model dinov2 --adapter-path outputs/lora/lora_adapter --corn-head outputs/lora/corn_head.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, AutoProcessor
from peft import PeftModel
from coral_pytorch.dataset import corn_label_from_logits

from src.data import OtolithDataset, load_split_by_ids
from src.features.extractor import clahe_enhancement
from src.evaluation.metrics import compute_classification_metrics, bootstrap_metrics
from src.utils import load_config, get_device


MODEL_CONFIGS = {
    "dinov2": {
        "model_id": "facebook/dinov2-with-registers-large",
        "embedding_dim": 1024,
        "processor": "image",
    },
    "siglip2": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "embedding_dim": 1152,
        "processor": "full",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA + CORN head on test set")
    parser.add_argument("--model", type=str, default="siglip2", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (default: outputs/lora/<model>/lora_adapter)")
    parser.add_argument("--corn-head", type=str, default=None,
                        help="Path to CORN head weights (default: outputs/lora/<model>/corn_head.pt)")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to split.json (default: outputs/lora/<model>/split.json)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/results/<model>-lora-corn/bootstrap)")
    return parser.parse_args()


class VisionCORNHead(torch.nn.Module):
    """Vision backbone + CORN head (must match training architecture)."""

    def __init__(self, backbone, num_classes=10, embedding_dim=1024, model_type="dinov2"):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model_type = model_type
        self.fc = torch.nn.Linear(embedding_dim, num_classes - 1)

    def forward(self, pixel_values):
        if self.model_type == "siglip2":
            outputs = self.backbone.vision_model(pixel_values=pixel_values)
            embeddings = outputs.pooler_output
        else:
            outputs = self.backbone(pixel_values=pixel_values)
            embeddings = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(embeddings)
        return logits


def main():
    args = parse_args()
    device = get_device()
    model_cfg = MODEL_CONFIGS[args.model]
    model_id = model_cfg["model_id"]

    # Resolve default paths
    lora_dir = Path(f"outputs/lora/{args.model}")
    adapter_path = Path(args.adapter_path or lora_dir / "lora_adapter")
    corn_head_path = Path(args.corn_head or lora_dir / "corn_head.pt")
    split_file = Path(args.split_file or lora_dir / "split.json")
    output_dir = Path(args.output_dir or f"outputs/results/{args.model}-lora-corn/bootstrap")

    print(f"\n{'='*60}")
    print(f"CORN HEAD EVALUATION ({args.model.upper()})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Adapter: {adapter_path}")
    print(f"CORN head: {corn_head_path}")
    print(f"Split: {split_file}")

    # Validate paths
    for p, name in [(adapter_path, "Adapter"), (corn_head_path, "CORN head"), (split_file, "Split file")]:
        if not p.exists():
            print(f"Error: {name} not found at {p}")
            sys.exit(1)

    # Load config
    config = load_config(args.config)
    data_config = config["data"]

    # Load base model + LoRA adapter
    print(f"\nLoading {model_id}...")
    backbone = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)

    if args.model == "siglip2":
        backbone.vision_model = PeftModel.from_pretrained(backbone.vision_model, str(adapter_path))
        backbone.vision_model = backbone.vision_model.merge_and_unload()
    else:
        backbone = PeftModel.from_pretrained(backbone, str(adapter_path))
        backbone = backbone.merge_and_unload()

    # Build full model with CORN head
    num_classes = data_config["age_range"][1] - data_config["age_range"][0] + 1
    model = VisionCORNHead(
        backbone, num_classes=num_classes,
        embedding_dim=model_cfg["embedding_dim"], model_type=args.model,
    )
    model.fc.load_state_dict(torch.load(corn_head_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    model.eval()

    # Processor + transform
    if model_cfg["processor"] == "image":
        processor = AutoImageProcessor.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id).image_processor

    apply_clahe = args.clahe
    repeat_clahe = args.repeat_clahe
    print(f"CLAHE: {'ON' if apply_clahe else 'OFF'}" + (f" (repeat={repeat_clahe})" if apply_clahe else ""))

    size_dict = processor.size
    image_size = size_dict.get("height", size_dict.get("shortest_edge", size_dict.get("width", 518)))
    clahe_step = [transforms.Lambda(lambda img: clahe_enhancement(img, repeat_clahe=repeat_clahe))] if apply_clahe else []
    test_transform = transforms.Compose([
        *clahe_step,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Load dataset
    full_dataset = OtolithDataset(
        root_dir=data_config["root_dir"],
        transform=None,
        age_range=tuple(data_config["age_range"]),
        metadata_csv=data_config.get("metadata_csv"),
    )
    all_paths = full_dataset.get_paths()
    all_measurement_ids = np.array([int(p.stem) for p in all_paths])

    # Load split
    print(f"\nLoading split from {split_file}")
    split = load_split_by_ids(str(split_file), all_measurement_ids)
    print(f"  Train: {len(split.train_indices)}, Test: {len(split.test_indices)}")

    # Create test dataset
    test_dataset = OtolithDataset(
        root_dir=data_config["root_dir"],
        transform=test_transform,
        age_range=tuple(data_config["age_range"]),
        indices=split.test_indices.tolist(),
        metadata_csv=data_config.get("metadata_csv"),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Run inference
    print(f"\nRunning inference on {len(test_dataset)} test samples...")
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = (labels - 1).long()  # 0-indexed for CORN

            logits = model(images)
            preds = corn_label_from_logits(logits).long()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds) + 1  # back to 1-indexed
    y_true = np.concatenate(all_labels) + 1

    # Compute metrics
    test_metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\nTest Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test F1:       {test_metrics['f1']*100:.2f}%")
    print(f"Test Acc ±1:   {test_metrics['accuracy_pm1']*100:.2f}%")
    print(f"Test RMSE:     {test_metrics['rmse']:.3f}")

    # Bootstrap CIs
    print("\nBootstrapping test set (1000 resamples)...")
    aggregated = bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, ci=0.95)

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY (eval_mode=bootstrap)")
    print(f"{'='*60}")
    print("\nTest Set Metrics (point estimate [95% CI]):")
    print("-" * 50)
    metric_names = {
        "f1": "F1-Score (macro)",
        "accuracy": "Accuracy",
        "accuracy_pm1": "+/-1 Accuracy",
        "precision": "Precision (macro)",
        "recall": "Recall (macro)",
        "rmse": "RMSE",
    }
    for key, display_name in metric_names.items():
        if key not in aggregated:
            continue
        val = aggregated[key]
        is_pct = key != "rmse"
        if is_pct:
            print(f"  {display_name:20s}: {val['mean']*100:6.2f}% [{val['ci_lower']*100:.2f}, {val['ci_upper']*100:.2f}]")
        else:
            print(f"  {display_name:20s}: {val['mean']:6.3f} [{val['ci_lower']:.3f}, {val['ci_upper']:.3f}]")
    print(f"{'='*60}")

    # Save results (same format as train_classifier.py bootstrap mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "eval_mode": "bootstrap",
        "classifier": "corn_head",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "model": args.model,
            "model_id": model_id,
            "adapter_path": str(adapter_path),
            "corn_head_path": str(corn_head_path),
            "split_file": str(split_file),
            "clahe": apply_clahe,
            "repeat_clahe": repeat_clahe,
        },
        "experiment_results": [{
            "experiment": 0,
            "test_metrics": test_metrics,
            "train_size": len(split.train_indices),
            "test_size": len(split.test_indices),
        }],
        "aggregated_results": aggregated,
        "bootstrap": {
            "n_bootstrap": 1000,
            "ci_level": 0.95,
            "seed": 42,
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    print("Done!")


if __name__ == "__main__":
    main()
