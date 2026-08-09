#!/usr/bin/env python3
"""
Extract pooled embeddings for the whole dataset from a frozen or adapted encoder.

Replaces the separate frozen and LoRA extraction scripts. Both did the same
thing with slightly different preprocessing defaults and slightly different
readouts, which is precisely how the two conditions stopped being comparable.

The output filename encodes the encoder, the preprocessing configuration and the
adapter, so no two configurations can write to the same cache file. The previous
LoRA script omitted the CLAHE flags from its filename and silently overwrote.

Examples:
    python scripts/extract_embeddings.py --encoder siglip2
    python scripts/extract_embeddings.py --encoder clip --no-clahe
    python scripts/extract_embeddings.py --encoder siglip2 \\
        --adapter outputs/runs/siglip2_lora_r16a32_s42_clahe/adapter
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import OtolithDataset
from src.features.preprocessing import (
    DEFAULT_RESIZE_MODE,
    RESIZE_MODES,
    PreprocessConfig,
    build_eval_transform,
)
from src.features.registry import ENCODERS, embed_images, load_encoder
from src.utils import load_config, get_device


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract embeddings from a frozen or adapted vision encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--encoder", default="siglip2", choices=sorted(ENCODERS))
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--adapter", default=None,
                   help="Path to a saved PEFT adapter. Omit for the frozen encoder.")
    p.add_argument("--state-dict", default=None,
                   help="Path to a full fine-tuned encoder state dict. Used for "
                        "--mode full, which trains every weight and therefore "
                        "saves a state dict rather than an adapter directory.")
    p.add_argument("--merge", action="store_true",
                   help="Fold adapter weights into the base weights before extraction. "
                        "Mathematically equivalent, marginally faster.")
    p.add_argument("--clahe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repeat-clahe", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resize-mode", default=DEFAULT_RESIZE_MODE, choices=RESIZE_MODES)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/embeddings")
    p.add_argument("--output-path", default=None,
                   help="Overrides the auto-generated, collision-free filename.")
    p.add_argument("--tag", default=None,
                   help="Extra fragment for the filename, e.g. to distinguish seeds.")
    p.add_argument("--force", action="store_true", help="Re-extract even if cached.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N images. For smoke tests; the "
                        "result is written with a _smoke suffix so it can never "
                        "be mistaken for a full extraction.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def build_output_path(args, preprocess: PreprocessConfig) -> Path:
    """Name the cache file after everything that determines its contents."""
    if args.output_path:
        return Path(args.output_path)

    parts = [args.encoder]
    slug = preprocess.slug()
    if slug:
        parts.append(slug)
    if args.adapter or args.state_dict:
        # Carry the run name through, so embeddings from two different runs are
        # distinguishable at a glance and can never collide.
        path = Path(args.adapter or args.state_dict)
        run = path.parent.name if path.name in ("adapter", "encoder.pt") else path.stem
        parts.append(run)
    else:
        parts.append("frozen")
    if args.tag:
        parts.append(args.tag)
    if args.limit:
        parts.append(f"smoke{args.limit}")

    return Path(args.output_dir) / ("_".join(parts) + ".npz")


def main():
    args = parse_args()
    device = get_device()

    preprocess = PreprocessConfig(
        apply_clahe=args.clahe,
        repeat_clahe=args.repeat_clahe,
        resize_mode=args.resize_mode,
    )
    output_path = build_output_path(args, preprocess)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(f"Cached embeddings already exist at {output_path}. Pass --force to redo.")
        return

    config = load_config(args.config)
    data_config = config["data"]
    age_range = tuple(data_config["age_range"])

    print("=" * 68)
    print(f"Encoder: {args.encoder}"
          + (f"   Adapter: {args.adapter}" if args.adapter else "   (frozen)"))
    print(f"Device: {device}")
    print(f"Preprocessing: {preprocess.describe()}")
    print(f"Output: {output_path}")
    print("=" * 68)

    model, processor, spec = load_encoder(args.encoder)
    image_processor = getattr(processor, "image_processor", processor)

    if args.state_dict:
        state = torch.load(args.state_dict, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  state dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
        print(f"Loaded fine-tuned weights from {args.state_dict}")

    if args.adapter:
        from peft import PeftModel
        from src.features.registry import adaptable_module

        target = adaptable_module(model, spec)
        wrapped = PeftModel.from_pretrained(target, args.adapter)
        if args.merge:
            wrapped = wrapped.merge_and_unload()
        if target is model:
            model = wrapped
        else:
            model.vision_model = wrapped

    model = model.to(device).eval()

    transform = build_eval_transform(image_processor, preprocess,
                                     image_size=spec.image_size)
    dataset = OtolithDataset(
        Path(data_config["root_dir"]), transform=transform, age_range=age_range,
        metadata_csv=data_config["metadata_csv"],
    )
    if args.limit:
        dataset.samples = dataset.samples[: args.limit]
        print(f"\nSMOKE TEST: limited to {len(dataset)} images")
    measurement_ids = np.array([int(p.stem) for p, _ in dataset.samples])
    print(f"\n{len(dataset)} images")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    autocast_dtype = (
        torch.bfloat16 if (args.amp and device.type == "cuda"
                           and torch.cuda.is_bf16_supported()) else None
    )

    features, labels = [], []
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc="Extracting"):
            images = images.to(device, non_blocking=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    batch = embed_images(model, images, spec, normalize=True)
            else:
                batch = embed_images(model, images, spec, normalize=True)
            features.append(batch.float().cpu().numpy())
            labels.append(batch_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels)

    if features.shape[0] != len(measurement_ids):
        raise RuntimeError(
            f"Extracted {features.shape[0]} embeddings for {len(measurement_ids)} "
            "images. The dataset order and the embedding order have diverged."
        )

    np.savez(
        output_path,
        features=features,
        labels=labels,
        measurement_ids=measurement_ids,
    )

    # A sidecar record of exactly how these embeddings were made. The previous
    # pipeline stored nothing, which is why the paper's settings could not be
    # checked once the outputs were lost.
    provenance = {
        "encoder": {"name": spec.name, "model_id": spec.model_id,
                    "embedding_dim": spec.embedding_dim,
                    "image_size": spec.image_size},
        "adapter": args.adapter,
        "merged": bool(args.merge),
        "preprocessing": {**preprocess.__dict__,
                          "description": preprocess.describe()},
        "n_images": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "normalized": True,
    }
    with open(output_path.with_suffix(".json"), "w") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"\nSaved {features.shape} to {output_path}")


if __name__ == "__main__":
    main()
