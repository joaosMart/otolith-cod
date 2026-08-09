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
import json
import shlex
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

BUNDLE_REPO = "hafsteinn/otolith-cod-bundle"
OUTPUT_REPO = "hafsteinn/otolith-cod-results"

#: Encoders whose weights are freely downloadable. DINOv3 is deliberately not
#: here: its repo is gated behind manual approval, so a batch containing it fails
#: partway through and wastes the GPU time already spent on the others. Once
#: access is granted, run the separate "dinov3" batch.
ENCODERS = ["clip", "siglip2", "dinov2"]
GATED_ENCODERS = ["dinov3"]

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


def attribution(npz, name):
    return (f"analyze_feature_importance.py --embeddings {npz} "
            f"--results outputs/results/{name}/bootstrap/results.json "
            f"--output-dir outputs/results/{name}/bootstrap/feature_importance "
            f"--split-file {SPLIT}")


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


def _flatten(groups):
    return [c for g in groups for c in g]


def build_batches():
    """Each batch is a list of independent chains.

    A chain is a sequence whose later steps depend on its earlier ones, such as
    fine-tune then extract then classify. Chains are independent of each other,
    so the runner can abandon a failing chain and carry on with the next instead
    of discarding the hours already spent. Losing four completed encoders
    because a fifth ran out of memory is the expensive failure mode here.
    """
    batches = {}

    # A deliberately tiny job whose only purpose is to measure throughput on the
    # chosen hardware before committing to the full matrix. Two epochs on one
    # encoder tells us minutes-per-epoch, which is what every other estimate
    # depends on.
    batches["calibrate"] = {
        "flavor": "l40sx1",
        "timeout": "50m",
        "groups": [
            [extract("siglip2")],
            [finetune("siglip2", extra="--epochs 2 --patience 99 --run-name calib")],
        ],
        "note": "Throughput measurement. Sizes every later batch.",
    }

    # Frozen baselines for all four encoders, plus the resize-mode comparison
    # that the wide-crop geometry makes necessary. Extraction is inference only,
    # so this is cheap and mostly bounded by image loading.
    frozen = [[extract(e), classify(frozen_npz(e), f"{e}-frozen")] for e in ENCODERS]
    frozen += [[extract("siglip2", resize=r),
                classify(frozen_npz("siglip2", r), f"siglip2-frozen-{r}")]
               for r in ("squash", "crop")]
    batches["frozen"] = {
        "flavor": "l40sx1", "timeout": "2h", "groups": frozen,
        "note": "Four frozen encoders plus the pad/squash/crop preprocessing comparison.",
    }

    # Experiment A: does every encoder benefit from adaptation, or only SigLIP2?
    batches["lora-all"] = {
        "flavor": "l40sx1", "timeout": "5h",
        "groups": [lora_chain(e) for e in ENCODERS],
        "note": "LoRA on the three ungated encoders, seed 42. About 1h per encoder.",
    }

    # Separate because the weights need a licence acceptance that cannot be
    # automated. Kept identical in every other respect so its results drop
    # straight into the same table.
    batches["dinov3"] = {
        "flavor": "l40sx1", "timeout": "3h",
        "groups": ([[extract(e), classify(frozen_npz(e), f"{e}-frozen")]
                    for e in GATED_ENCODERS]
                   + [lora_chain(e) for e in GATED_ENCODERS]),
        "note": "Frozen and LoRA DINOv3. Requires accepting the licence first.",
    }

    # Per-encoder LoRA jobs, so a single encoder can be re-run without repeating
    # the ones that already succeeded.
    for encoder in ENCODERS + GATED_ENCODERS:
        batches[f"lora-{encoder}"] = {
            "flavor": "l40sx1", "timeout": "150m",
            "groups": [lora_chain(encoder)],
            "note": f"LoRA on {encoder} alone, seed 42.",
        }

    # Attribution has to re-derive its own embeddings: jobs are ephemeral and
    # the results repo deliberately excludes the .npz caches, which are large
    # and regenerable. Extraction is only a minute or two, so a self-contained
    # chain is cheaper than shipping embeddings around.
    lora_run = run_name_for("siglip2")
    batches["attribution"] = {
        "flavor": "l40sx1", "timeout": "90m",
        "groups": [[
            extract("siglip2"),
            classify(frozen_npz("siglip2"), "siglip2-frozen"),
            attribution(frozen_npz("siglip2"), "siglip2-frozen"),
        ], [
            extract("siglip2", adapter=f"outputs/runs/{lora_run}/adapter"),
            classify(run_npz("siglip2", lora_run), lora_run),
            attribution(run_npz("siglip2", lora_run), lora_run),
        ]],
        "needs_previous": True,
        "note": "SHAP, permutation importance and forward selection for frozen "
                "vs LoRA SigLIP2. Produces the paper's attribution figures.",
    }

    # Collects everything the manuscript quotes into one JSON plus figures.
    # Re-extracts because jobs are ephemeral; extraction is a couple of minutes.
    batches["summarize"] = {
        "flavor": "l40sx1", "timeout": "60m",
        "groups": [[
            extract("siglip2"),
            extract("siglip2", adapter=f"outputs/runs/{run_name_for('siglip2')}/adapter"),
            "summarize_results.py",
        ]],
        "needs_previous": True,
        "note": "Per-age table, McNemar test, seed spread and all paper figures.",
    }

    # Experiment B: run-to-run and split-to-split variation, currently unknown.
    # One job per seed. Chain-level resilience protects against a run failing,
    # but not against the whole job being preempted, which is what took out the
    # combined four-seed job partway through its second run.
    for seed in SEEDS:
        if seed == 42:
            continue
        batches[f"seed-{seed}"] = {
            "flavor": "l40sx1", "timeout": "150m",
            "groups": [lora_chain("siglip2", seed=seed)],
            "note": f"SigLIP2 LoRA at seed {seed}.",
        }

    # Experiment F: how much of the gain is LoRA, and how much is the pooling
    # head that the original code silently trained at full rank.
    batches["pooling-head"] = {
        "flavor": "l40sx1", "timeout": "3h",
        "groups": [lora_chain("siglip2", mode="lora+pool"),
                   [finetune("siglip2", mode="probe")]],
        "note": "LoRA plus unfrozen pooling head, against pure LoRA and a linear probe.",
    }

    # Experiment C: test the overfitting claim instead of citing it.
    batches["full-finetune"] = {
        "flavor": "a100-large", "timeout": "4h",
        "groups": [lora_chain("siglip2", mode="full")],
        "note": "Full fine-tuning baseline. Needs the 80 GB card.",
    }

    # Experiment E: replace the admission that rank was never tuned with a curve.
    batches["rank-sweep"] = {
        "flavor": "l40sx1", "timeout": "6h",
        "groups": [lora_chain("siglip2", rank=r, alpha=2 * r) for r in (4, 8, 32, 64)],
        "note": "Rank 4 to 64 at alpha = 2r; rank 16 comes from lora-all.",
    }

    # Experiment G: how many labelled otoliths does this actually need.
    curves = [lora_chain(e, fraction=f)
              for e in ("siglip2", "clip") for f in (0.1, 0.25, 0.5)]
    batches["learning-curves"] = {
        "flavor": "l40sx1", "timeout": "8h", "groups": curves,
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
                  f"{len(_flatten(batch['groups'])):>5}  {batch['note']}")
        print("\nSubmit one with:  python scripts/hf/submit.py --batch <name> --dry-run")
        return 0

    batch = batches[args.batch]
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
    if batch.get("needs_previous"):
        argv.append("--seed-outputs")
    # Only the batch name crosses argv. The runner reads the definition from
    # this same file inside the bundle, so there is one source of truth and no
    # argument long enough to trip the client's local-file detection.
    argv += ["--batch", args.batch]

    print(f"Batch '{args.batch}': {len(batch['groups'])} chains, "
          f"{len(_flatten(batch['groups']))} commands on {flavor}, timeout {timeout}")
    print(f"Note: {batch['note']}\n")
    for gi, group in enumerate(batch["groups"], 1):
        print(f"  chain {gi}:")
        for command in group:
            print(f"      {command}")

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
    """Hours in a duration string.

    Kept tolerant of compound forms even though batch definitions must use a
    single unit: the Jobs CLI rejects anything like "2h30m" server-side.
    """
    import re
    factors = {"s": 1 / 3600, "m": 1 / 60, "h": 1.0, "d": 24.0}
    parts = re.findall(r"([0-9]*\.?[0-9]+)\s*([smhd])", timeout.lower())
    if not parts:
        raise ValueError(f"Could not parse timeout {timeout!r}")
    return sum(float(value) * factors[unit] for value, unit in parts)


if __name__ == "__main__":
    raise SystemExit(main())
