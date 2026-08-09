#!/usr/bin/env python3
"""
Submit the paper's experiment batches to HuggingFace Jobs.

Each batch is a list of commands executed in order inside one job, so a batch
should be sized to a few hours rather than a few minutes: container startup and
the bundle download are paid once per job, not once per command.

Run `--dry-run` first. It prints the exact `hf jobs` invocation without spending
anything, which is also the fastest way to sanity-check a batch definition.

    python scripts/hf/submit.py --list
    python scripts/hf/submit.py --batch calibrate --dry-run
    python scripts/hf/submit.py --batch calibrate
    python scripts/hf/submit.py --batch frozen
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

BUNDLE_REPO = "hafsteinn/otolith-cod-bundle"
OUTPUT_REPO = "hafsteinn/otolith-cod-results"

#: All four encoders, in the order the paper's table presents them.
ENCODERS = ["clip", "siglip2", "dinov2", "dinov3"]

#: The five seeds for the repeated-run comparison. 42 is the original, so its
#: result is directly comparable with everything else already produced.
SEEDS = [42, 7, 13, 101, 2024]

SPLIT = "outputs/splits/main_split.json"


def extract(encoder, adapter=None, resize="pad", tag=None):
    cmd = f"extract_embeddings.py --encoder {encoder}"
    if adapter:
        cmd += f" --adapter {adapter}"
    if resize != "pad":
        cmd += f" --resize-mode {resize}"
    if tag:
        cmd += f" --tag {tag}"
    return cmd


def classify(npz, name):
    return (f"train_classifier.py --embeddings {npz} --eval-mode bootstrap "
            f"--split-file {SPLIT} --output-dir outputs/results/{name}/bootstrap")


def frozen_npz(encoder, resize="pad"):
    slug = "clahe" if resize == "pad" else f"clahe_{resize}"
    return f"outputs/embeddings/{encoder}_{slug}_frozen.npz"


def run_npz(encoder, run_name):
    return f"outputs/embeddings/{encoder}_clahe_{run_name}.npz"


def finetune(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0, extra=""):
    cmd = (f"finetune_encoder.py --encoder {encoder} --mode {mode} --seed {seed} "
           f"--split-file {SPLIT}")
    if mode in ("lora", "lora+pool"):
        cmd += f" --rank {rank} --lora-alpha {alpha}"
    if fraction < 1.0:
        cmd += f" --train-fraction {fraction}"
    return cmd + (f" {extra}" if extra else "")


def run_name_for(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0):
    """Mirrors build_run_name() in finetune_encoder.py."""
    parts = [encoder, mode.replace("+", "-")]
    if mode in ("lora", "lora+pool"):
        parts.append(f"r{rank}a{alpha}")
    if fraction < 1.0:
        parts.append(f"f{fraction:g}")
    parts.append(f"s{seed}")
    parts.append("clahe")
    return "_".join(parts)


def lora_chain(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0):
    """Fine-tune, extract with the adapter, then classify. One logical unit."""
    name = run_name_for(encoder, mode, seed, rank, alpha, fraction)
    return [
        finetune(encoder, mode, seed, rank, alpha, fraction),
        extract(encoder, adapter=f"outputs/runs/{name}/adapter"),
        classify(run_npz(encoder, name), name),
    ]


def build_batches():
    batches = {}

    # A deliberately tiny job whose only purpose is to measure throughput on the
    # chosen hardware before committing to the full matrix. Two epochs on one
    # encoder tells us minutes-per-epoch, which is what every other estimate
    # depends on.
    batches["calibrate"] = {
        "flavor": "l40sx1",
        "timeout": "50m",
        "commands": [
            extract("siglip2"),
            finetune("siglip2", extra="--epochs 2 --patience 99 --run-name calib"),
        ],
        "note": "Throughput measurement. Sizes every later batch.",
    }

    # Frozen baselines for all four encoders, plus the resize-mode comparison
    # that the wide-crop geometry makes necessary. Extraction is inference only,
    # so this is cheap and mostly bounded by image loading.
    frozen = []
    for encoder in ENCODERS:
        frozen.append(extract(encoder))
        frozen.append(classify(frozen_npz(encoder), f"{encoder}-frozen"))
    for resize in ("squash", "crop"):
        frozen.append(extract("siglip2", resize=resize))
        frozen.append(classify(frozen_npz("siglip2", resize), f"siglip2-frozen-{resize}"))
    batches["frozen"] = {
        "flavor": "l40sx1", "timeout": "2h", "commands": frozen,
        "note": "Four frozen encoders plus the pad/squash/crop preprocessing comparison.",
    }

    # Experiment A: does every encoder benefit from adaptation, or only SigLIP2?
    batches["lora-all"] = {
        "flavor": "l40sx1", "timeout": "6h",
        "commands": [c for e in ENCODERS for c in lora_chain(e)],
        "note": "LoRA on all four encoders, seed 42.",
    }

    # Experiment B: run-to-run and split-to-split variation, currently unknown.
    batches["seeds"] = {
        "flavor": "l40sx1", "timeout": "6h",
        "commands": [c for s in SEEDS if s != 42 for c in lora_chain("siglip2", seed=s)],
        "note": "Four further seeds for SigLIP2; seed 42 comes from lora-all.",
    }

    # Experiment F: how much of the gain is LoRA, and how much is the pooling
    # head that the original code silently trained at full rank.
    batches["pooling-head"] = {
        "flavor": "l40sx1", "timeout": "3h",
        "commands": lora_chain("siglip2", mode="lora+pool") + [
            finetune("siglip2", mode="probe"),
        ],
        "note": "LoRA plus unfrozen pooling head, against pure LoRA and a linear probe.",
    }

    # Experiment C: test the overfitting claim instead of citing it.
    batches["full-finetune"] = {
        "flavor": "a100-large", "timeout": "4h",
        "commands": lora_chain("siglip2", mode="full") + [],
        "note": "Full fine-tuning baseline. Needs the 80 GB card.",
    }

    # Experiment E: replace the admission that rank was never tuned with a curve.
    batches["rank-sweep"] = {
        "flavor": "l40sx1", "timeout": "6h",
        "commands": [c for r in (4, 8, 32, 64)
                     for c in lora_chain("siglip2", rank=r, alpha=2 * r)],
        "note": "Rank 4 to 64 at alpha = 2r; rank 16 comes from lora-all.",
    }

    # Experiment G: how many labelled otoliths does this actually need.
    curves = []
    for encoder in ("siglip2", "clip"):
        for fraction in (0.1, 0.25, 0.5):
            curves.extend(lora_chain(encoder, fraction=fraction))
    batches["learning-curves"] = {
        "flavor": "l40sx1", "timeout": "8h", "commands": curves,
        "note": "SigLIP2 against CLIP at 10, 25 and 50 percent of the training set; "
                "the 100 percent points come from lora-all.",
    }

    return batches


def parse_args():
    batches = build_batches()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", choices=sorted(batches), help="Which batch to submit.")
    p.add_argument("--list", action="store_true", help="Show all batches and exit.")
    p.add_argument("--flavor", default=None, help="Override the batch's hardware.")
    p.add_argument("--timeout", default=None, help="Override the batch's timeout.")
    p.add_argument("--bundle-repo", default=BUNDLE_REPO)
    p.add_argument("--output-repo", default=OUTPUT_REPO)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without submitting.")
    p.add_argument("--detach", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(), batches


def main():
    args, batches = parse_args()

    if args.list or not args.batch:
        print(f"{'batch':<18} {'flavor':<14} {'timeout':<9} {'cmds':>5}  note")
        print("-" * 100)
        for name, batch in batches.items():
            print(f"{name:<18} {batch['flavor']:<14} {batch['timeout']:<9} "
                  f"{len(batch['commands']):>5}  {batch['note']}")
        print("\nSubmit one with:  python scripts/hf/submit.py --batch <name> --dry-run")
        return 0

    batch = batches[args.batch]
    # Every batch begins by materialising the shared split. It is deterministic
    # and a no-op when the file already exists, so prepending it costs nothing
    # and removes the ordering dependency between batches.
    batch = {**batch, "commands": ["make_split.py"] + batch["commands"]}
    flavor = args.flavor or batch["flavor"]
    timeout = args.timeout or batch["timeout"]

    argv = [
        "hf", "jobs", "uv", "run",
        "--flavor", flavor,
        "--timeout", timeout,
        "--secrets", "HF_TOKEN",
        "--name", f"otolith-{args.batch}",
        "--label", "project=otolith-cod",
        "--label", f"batch={args.batch}",
    ]
    if args.detach:
        argv.append("--detach")
    argv += [str(project_root / "scripts/hf/job_runner.py"), "--"]
    argv += ["--repo", args.bundle_repo,
             "--output-repo", args.output_repo,
             "--run-group", args.batch]
    for command in batch["commands"]:
        argv += ["--command", command]

    print(f"Batch '{args.batch}': {len(batch['commands'])} commands on {flavor}, "
          f"timeout {timeout}")
    print(f"Note: {batch['note']}\n")
    for i, command in enumerate(batch["commands"], 1):
        print(f"  {i:>2}. {command}")

    hourly = {"l40sx1": 1.80, "a100-large": 2.50, "l4x1": 0.80,
              "a10g-small": 1.00, "rtx-pro-6000": 2.75}.get(flavor)
    if hourly:
        print(f"\nAt ${hourly:.2f}/hour this batch costs at most "
              f"${hourly * _hours(timeout):.2f} if it runs to the full timeout.")

    print(f"\n$ {' '.join(shlex.quote(a) for a in argv)}\n")

    if args.dry_run:
        print("Dry run: nothing submitted.")
        return 0

    return subprocess.run(argv).returncode


def _hours(timeout: str) -> float:
    unit, value = timeout[-1], float(timeout[:-1])
    return {"s": value / 3600, "m": value / 60, "h": value, "d": value * 24}[unit]


if __name__ == "__main__":
    raise SystemExit(main())
