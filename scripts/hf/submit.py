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


def finetune(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0,
             image_size=None, extra=""):
    cmd = (f"finetune_encoder.py --encoder {encoder} --mode {mode} --seed {seed} "
           f"--split-file {SPLIT}")
    if mode in ("lora", "lora+pool"):
        cmd += f" --rank {rank} --lora-alpha {alpha}"
    if fraction < 1.0:
        cmd += f" --train-fraction {fraction}"
    if image_size:
        cmd += f" --image-size {image_size}"
    return cmd + (f" {extra}" if extra else "")


def run_name_for(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0,
                 image_size=None):
    """Mirrors build_run_name() in finetune_encoder.py."""
    parts = [encoder, mode.replace("+", "-")]
    if mode in ("lora", "lora+pool"):
        parts.append(f"r{rank}a{alpha}")
    if fraction < 1.0:
        parts.append(f"f{fraction:g}")
    if image_size:
        parts.append(f"px{image_size}")
    parts.append(f"s{seed}")
    parts.append("clahe")
    return "_".join(parts)


def lora_chain(encoder, mode="lora", seed=42, rank=16, alpha=32, fraction=1.0,
               image_size=None):
    """Fine-tune, extract with the trained weights, then classify.

    One logical unit: the later steps are meaningless without the earlier ones,
    which is what makes it the right granularity for the runner to abandon on
    failure. Full fine-tuning saves a state dict rather than an adapter
    directory, so its extraction is invoked differently.

    `image_size` has to reach the extraction step as well as the training one.
    The adapter is trained against a particular patch grid, and reading it out
    at the default 384 would hand those weights a different grid from the one
    they were fitted to, which fails quietly rather than loudly: the shapes
    still match, because rotary embeddings do not care how many tokens arrive.
    """
    name = run_name_for(encoder, mode, seed, rank, alpha, fraction, image_size)
    weights = (f"--state-dict outputs/runs/{name}/encoder.pt" if mode == "full"
               else f"--adapter outputs/runs/{name}/adapter")
    px = f" --image-size {image_size}" if image_size else ""
    return [
        finetune(encoder, mode, seed, rank, alpha, fraction, image_size),
        f"extract_embeddings.py --encoder {encoder} {weights}{px}",
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
        "flavor": "a100-large", "timeout": "110m",
        "groups": [lora_chain("siglip2", mode="full")],
        "needs_previous": True,
        "note": "Full fine-tuning baseline. Needs the 80 GB card. Trains in about "
                "22 minutes; the earlier attempt failed only at its extraction step.",
    }

    # Experiment E: replace the admission that rank was never tuned with a curve.
    # One job per rank. Nothing has completed beyond about 1h43m regardless of the
    # timeout requested, and the combined sweep died mid-run at 1h47m, so batches
    # are now sized to stay comfortably inside that.
    for rank in (4, 8, 32, 64):
        batches[f"rank-{rank}"] = {
            "flavor": "l40sx1", "timeout": "110m",
            "groups": [lora_chain("siglip2", rank=rank, alpha=2 * rank)],
            "note": f"SigLIP2 LoRA at rank {rank}, alpha {2 * rank}.",
        }

    # Experiment G: how many labelled otoliths does this actually need.
    # Experiment G, also one job per point for the same reason. Smaller training
    # fractions finish quickly, so the two cheapest are paired.
    for encoder in ("siglip2", "clip"):
        batches[f"curve-{encoder}-small"] = {
            "flavor": "l40sx1", "timeout": "110m",
            "groups": [lora_chain(encoder, fraction=f) for f in (0.1, 0.25)],
            "note": f"{encoder} at 10 and 25 percent of the training set.",
        }
        batches[f"curve-{encoder}-half"] = {
            "flavor": "l40sx1", "timeout": "110m",
            "groups": [lora_chain(encoder, fraction=0.5)],
            "note": f"{encoder} at 50 percent of the training set.",
        }

    # Experiment G, second pass. The first pass gave one run per point, which
    # supports a picture but not an error bar, and four x-positions is too few to
    # fit the three-parameter saturating curve that would say how many labels the
    # method actually wants. This pass adds two positions (5% and 75%) and takes
    # every position to three seeds.
    #
    # Only the runs that do not already exist are listed. SigLIP2 at the full
    # training set is already covered by the five-seed batches above, and seed 42
    # is already done at 10, 25 and 50 percent for both encoders.
    #
    # Job sizing is deliberately timid. Runtime is a fixed cost of roughly a
    # quarter hour for extraction and classification plus a training cost that
    # scales with the fraction, but early stopping makes the training term vary
    # by a factor of two or more between seeds, and nothing has ever survived
    # past about 1h43m regardless of the requested timeout. A wasted container
    # start costs a few cents; a job killed at 1h50m with three completed chains
    # inside it costs all of them.
    curve_seeds = (42, 7, 13)
    for encoder in ("siglip2", "clip", "dinov2", "dinov3"):
        # What already exists differs by encoder. SigLIP2 and CLIP were the
        # first-pass curve, so they have seed 42 at three interior fractions;
        # the DINO pair only ever ran at the full training set, from the
        # four-encoder comparison.
        if encoder in ("siglip2", "clip"):
            done = {(0.1, 42), (0.25, 42), (0.5, 42), (1.0, 42)}
        else:
            done = {(1.0, 42)}
        if encoder == "siglip2":
            done |= {(1.0, s) for s in curve_seeds}

        def todo(fraction):
            return [s for s in curve_seeds if (fraction, s) not in done]

        # Cheapest position: all three seeds fit in one job with room to spare.
        batches[f"curve-{encoder}-f005"] = {
            "flavor": "l40sx1", "timeout": "110m",
            "groups": [lora_chain(encoder, seed=s, fraction=0.05)
                       for s in todo(0.05)],
            "note": f"{encoder} at 5 percent, seeds {todo(0.05)}.",
        }
        # Middle positions, grouped by seed count. Runtime scales with the
        # fraction, so three seeds at half the training set is around 90 minutes
        # for the DINO pair, which is close enough to the ceiling that a single
        # overrun would discard all three chains. Two fit comfortably; three do
        # not, so a three-seed half-set position is split per seed.
        middle = [(0.1, "f010"), (0.25, "f025")]
        expensive = [(0.75, "f075"), (1.0, "full")]
        (middle if len(todo(0.5)) <= 2 else expensive).append((0.5, "f050"))

        for fraction, tag in middle:
            seeds = todo(fraction)
            if not seeds:
                continue
            batches[f"curve-{encoder}-{tag}"] = {
                "flavor": "l40sx1", "timeout": "110m",
                "groups": [lora_chain(encoder, seed=s, fraction=fraction)
                           for s in seeds],
                "note": f"{encoder} at {fraction:.0%}, seeds {seeds}.",
            }
        # Expensive positions: one seed per job, since two would not fit.
        for fraction, tag in expensive:
            for seed in todo(fraction):
                batches[f"curve-{encoder}-{tag}-s{seed}"] = {
                    "flavor": "l40sx1", "timeout": "110m",
                    "groups": [lora_chain(encoder, seed=seed, fraction=fraction)],
                    "note": f"{encoder} at {fraction:.0%}, seed {seed}.",
                }

    # Experiment G, third pass: the same curve under full fine-tuning.
    #
    # This is the experiment that turns a null result into a positive one. At
    # the full training set LoRA and full fine-tuning are indistinguishable,
    # which says nothing about why. Mai et al. attribute PEFT's advantage to an
    # implicit regulariser, which predicts that the gap should open up as labels
    # become scarce. That prediction is testable here and nowhere else in the
    # paper.
    #
    # SigLIP2 rather than DINOv3, despite DINOv3 having the higher point
    # estimate: its lead is 1.4 points against 1.6 points of seed noise, so it
    # is not resolvably the best, and SigLIP2 is where both the existing
    # full-fine-tune baseline and the five-seed LoRA reference already sit.
    #
    # Needs the 80 GB card at every fraction, since the memory requirement comes
    # from the model rather than the dataset. Full fine-tuning trains in 28
    # minutes against LoRA's 62, because it reaches its best epoch at 7 rather
    # than 13, so the whole curve is cheaper than the LoRA one it mirrors.
    sft_done = {(1.0, 42)}
    for fraction, tag in ((0.05, "f005"), (0.1, "f010"), (0.25, "f025"),
                          (0.5, "f050"), (0.75, "f075"), (1.0, "full")):
        seeds = [s for s in curve_seeds if (fraction, s) not in sft_done]
        if not seeds:
            continue
        batches[f"sftcurve-siglip2-{tag}"] = {
            "flavor": "a100-large", "timeout": "110m",
            "groups": [lora_chain("siglip2", mode="full", seed=s, fraction=fraction)
                       for s in seeds],
            "note": f"siglip2 FULL fine-tune at {fraction:.0%}, seeds {seeds}.",
        }

    # Experiment H: is the ceiling in the labels or in the pixels?
    #
    # The learning curves are flat over their last segment for every encoder and
    # for full fine-tuning, so more labelled otoliths of the same kind buy
    # nothing. That makes the constraint representational, and there is an
    # obvious candidate. The median crop is 780 by 332 pixels. Padding it square
    # and resizing to 384 downsamples by 2.03x along the axis the increments are
    # counted on, and leaves 58% of the canvas as border. The increments that
    # separate an eight from a nine are the ones packed tightest at the margin,
    # and they are the first casualty of that.
    #
    # DINOv3 is the only encoder that can test this without a confound. Its
    # rotary embeddings make the patch grid free, so the resolution changes and
    # nothing else does. The learned-table encoders would need their position
    # embedding interpolated, which is a second intervention.
    #
    # 512 first as a gate, because it is 1.78x the tokens rather than 4x. If it
    # does not clear the 1.6 pp seed spread, 768 is unlikely to be worth $14.
    for px, tag, flavor, note in ((512, "px512", "l40sx1", "gate"),
                                  (768, "px768", "a100-large", "native scale")):
        for seed in curve_seeds:
            batches[f"res-dinov3-{tag}-s{seed}"] = {
                "flavor": flavor, "timeout": "110m",
                "groups": [lora_chain("dinov3", seed=seed, image_size=px)],
                "note": f"DINOv3 LoRA at {px}px, seed {seed} ({note}). "
                        f"Compare against the {px // 16}x{px // 16} token grid "
                        "of the 384px runs already in the paper.",
            }

    # Experiment I: everything that can be squeezed out downstream of the
    # encoder, which is all of it once the embeddings exist.
    #
    # This job does no training. It re-extracts the embeddings for the adapted
    # runs the analysis needs and ships the .npz caches, which the other
    # batches deliberately discard. Threshold placement, stacking the ordinal
    # head's logits, concatenating two encoders and test-time augmentation are
    # then all local arithmetic on those arrays, so they can be iterated on
    # without a GPU and without paying for a container per attempt.
    squeeze_runs = [("siglip2", run_name_for("siglip2", seed=s)) for s in curve_seeds]
    squeeze_runs += [(e, run_name_for(e)) for e in ("dinov3", "dinov2", "clip")]
    batches["squeeze"] = {
        "flavor": "l40sx1", "timeout": "90m",
        "groups": ([[extract("siglip2"), extract("dinov3")]]
                   + [[extract(e, adapter=f"outputs/runs/{name}/adapter")]
                      for e, name in squeeze_runs]),
        "needs_previous": True,
        "keep_embeddings": True,
        "note": "Inference only. Frozen and adapted embeddings for the "
                "downstream analysis, with the .npz caches kept.",
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

    script_args = ["--repo", args.bundle_repo,
                   "--output-repo", args.output_repo,
                   "--run-group", args.batch]
    if batch.get("needs_previous"):
        script_args.append("--seed-outputs")
    if batch.get("keep_embeddings"):
        script_args.append("--keep-embeddings")
    # Only the batch name crosses argv. The runner reads the definition from
    # this same file inside the bundle, so there is one source of truth and no
    # argument long enough to trip the client's local-file detection.
    script_args += ["--batch", args.batch]

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

    runner = str(project_root / "scripts/hf/job_runner.py")
    print(f"\nrun_uv_job({runner!r}, script_args={script_args!r}, "
          f"flavor={flavor!r}, timeout={timeout!r})\n")

    if args.dry_run:
        print("Dry run: nothing submitted.")
        return 0

    return _submit(runner, script_args, flavor, timeout, args.batch)


def _submit(runner, script_args, flavor, timeout, batch_name):
    """Submit through the Python API rather than the `hf` command line.

    The CLI was the original route and is no longer a good one. It requires a
    terminal (it aborts with an ioctl error when its output is redirected), it
    inspects every argument after `--` to decide whether it looks like a local
    file to upload, and its flags are not stable: `--name` and `--label` both
    existed when these batches were first submitted and have since been removed,
    which caused sixteen submissions to fail with a usage error while the
    surrounding shell loop reported nothing. The API takes the same arguments,
    has none of those behaviours, and returns the job id instead of printing it,
    so submissions can be recorded rather than scraped back out of the console.
    """
    from huggingface_hub import get_token, run_uv_job

    job = run_uv_job(
        runner,
        script_args=script_args,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": get_token()},
    )
    print(f"Submitted {batch_name}: job {job.id}")
    print(f"  https://huggingface.co/jobs/{job.owner.name}/{job.id}")

    log = project_root / "outputs/hf_jobs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps({"batch": batch_name, "job_id": job.id,
                                 "flavor": flavor, "timeout": timeout}) + "\n")
    return 0


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
