#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers==5.3.0",
#   "peft==0.20.0",
#   "timm",
#   "coral-pytorch",
#   "opencv-python-headless",
#   "scikit-learn",
#   "pandas",
#   "numpy",
#   "scipy",
#   "seaborn",
#   "matplotlib",
#   "tqdm",
#   "pyyaml",
#   "huggingface_hub>=1.9",
# ]
# ///
"""
Entry point executed inside a HuggingFace Job.

Fetches the code and data bundle from a private dataset repo, reconstructs the
directory layout the pipeline expects, runs one or more experiment commands, and
copies the results out to a persistent location before the container disappears.

transformers and peft are pinned to the versions the code was validated
against. An unpinned `transformers>=4.56` resolved to a release whose SigLIP
vision model has a different attribute layout, and the job failed only after
downloading the weights. torch and torchvision are deliberately left to
resolve together, since pinning one without the other makes them unsatisfiable.

Two further dependency notes. `opencv-python-headless` rather than `opencv-python`,
because the uv base image has no libGL and the normal wheel fails to import.
And no `shap`: it drags in numba and llvmlite, whose resolution picks a version
that cannot build on Python 3.12, and the GPU stage does not need it. Feature
attribution runs afterwards on cached embeddings, on CPU.

Invoked by scripts/hf/submit.py, not usually by hand:

    hf jobs uv run --flavor l40sx1 --timeout 4h --secrets HF_TOKEN \\
        scripts/hf/job_runner.py -- --command "finetune_encoder.py --encoder siglip2"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path("/tmp/otolith")
DEFAULT_REPO = "hafsteinn/otolith-cod-bundle"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.environ.get("BUNDLE_REPO", DEFAULT_REPO))
    # A JSON list rather than a repeatable flag: the `hf jobs uv run` client
    # scans post-`--` arguments for things that look like local file paths and
    # tries to upload them, so a bare "make_split.py" argument makes submission
    # fail before the job is ever created.
    p.add_argument("--batch", default=None,
                   help="Name of a batch defined in scripts/hf/submit.py inside "
                        "the bundle. Preferred over --commands-json: the batch "
                        "definitions stay in one place and nothing long travels "
                        "through argv.")
    p.add_argument("--commands-json", default=None,
                   help="JSON list of commands, for ad-hoc runs. Keep it short: "
                        "the Jobs client stats every argument to see whether it "
                        "is a local file, and a long string raises ENAMETOOLONG.")
    p.add_argument("--output-repo", default=os.environ.get("OUTPUT_REPO"),
                   help="Dataset repo that results are pushed to. Falls back to "
                        "--output-dir alone if unset.")
    p.add_argument("--output-dir", default="/out",
                   help="Mounted bucket path. Written to directly, so results "
                        "survive even if the upload step fails.")
    p.add_argument("--run-group", default=None,
                   help="Subdirectory grouping this batch of results.")
    return p.parse_args()


def log(message):
    print(f"[runner {time.strftime('%H:%M:%S')}] {message}", flush=True)


def report_environment():
    log(f"job={os.environ.get('JOB_ID', 'local')} "
        f"accelerator={os.environ.get('ACCELERATOR', 'unknown')} "
        f"cpu={os.environ.get('CPU_CORES', '?')} mem={os.environ.get('MEMORY', '?')}")
    try:
        import torch
        log(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            log(f"gpu={props.name} memory={props.total_memory/1e9:.0f} GB "
                f"bf16={torch.cuda.is_bf16_supported()}")
    except Exception as exc:  # torch import must never take the job down here
        log(f"could not query torch: {exc}")


def fetch_bundle(repo: str) -> Path:
    """Download code, images and metadata, and lay them out as the code expects."""
    from huggingface_hub import snapshot_download

    log(f"downloading bundle from {repo}")
    started = time.time()
    local = Path(snapshot_download(repo_id=repo, repo_type="dataset"))
    log(f"bundle downloaded in {time.time() - started:.0f}s to {local}")

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)

    for entry in (local / "code").iterdir():
        destination = WORKDIR / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)

    for csv in local.glob("*.csv"):
        shutil.copy2(csv, WORKDIR / csv.name)

    # The pipeline reads images from otolith_images/segmented_images/<age>/.
    # Symlink rather than copy: the snapshot is already on local disk and the
    # images are only read.
    images_root = WORKDIR / "otolith_images"
    images_root.mkdir(exist_ok=True)
    (images_root / "segmented_images").symlink_to(local / "images")

    n_images = sum(1 for _ in (local / "images").rglob("*.jpg"))
    log(f"{n_images} images available at {images_root / 'segmented_images'}")
    return WORKDIR


def resolve_commands(args, workdir: Path) -> list:
    """Read the command list, preferring the batch definition in the bundle."""
    if args.commands_json:
        return [[c] for c in json.loads(args.commands_json)]
    if not args.batch:
        raise SystemExit("Pass either --batch or --commands-json.")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "submit", workdir / "scripts" / "hf" / "submit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    batches = module.build_batches()
    if args.batch not in batches:
        raise SystemExit(f"Unknown batch '{args.batch}'. "
                         f"Available: {', '.join(sorted(batches))}")
    # The shared split is a prerequisite for every batch and is a no-op once it
    # exists, so it becomes a chain of its own rather than being duplicated in
    # each definition.
    return [["make_split.py"]] + batches[args.batch]["groups"]


def preflight(commands, workdir: Path):
    """Import-check every script before running any of them.

    Running a script with --help executes all its module-level imports and then
    exits, which catches a missing dependency in about a second. Without this a
    missing package in the last step of a batch is only discovered after the
    hours of GPU time that preceded it, which is how a seaborn import cost a
    full round of fine-tuning.
    """
    scripts = sorted({c.split()[0] for c in commands})
    log(f"preflight: import-checking {len(scripts)} scripts")

    broken = []
    for name in scripts:
        script = workdir / "scripts" / name
        if not script.exists():
            broken.append((name, "not present in the bundle"))
            continue
        result = subprocess.run([sys.executable, str(script), "--help"],
                                cwd=str(workdir), capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()
            broken.append((name, tail[-1] if tail else f"exit {result.returncode}"))

    for name, reason in broken:
        log(f"  BROKEN {name}: {reason}")
    if broken:
        raise SystemExit(
            f"{len(broken)} script(s) cannot even be imported. Fix the bundle "
            "before spending GPU time."
        )
    log("preflight: all scripts import cleanly")


def run_command(command: str, workdir: Path) -> dict:
    """Run one experiment command, streaming its output into the job log."""
    parts = command.split()
    script = workdir / "scripts" / parts[0]
    if not script.exists():
        raise FileNotFoundError(f"No such script in the bundle: {script}")

    argv = [sys.executable, str(script), *parts[1:]]
    log(f"running: {' '.join(parts)}")
    started = time.time()

    result = subprocess.run(argv, cwd=str(workdir))
    elapsed = time.time() - started

    status = "ok" if result.returncode == 0 else f"failed ({result.returncode})"
    log(f"{status} in {elapsed / 60:.1f} min: {parts[0]}")
    return {"command": command, "returncode": result.returncode,
            "minutes": round(elapsed / 60, 2)}


def publish(workdir: Path, args, summary):
    """Copy artifacts out of the container before it is destroyed."""
    outputs = workdir / "outputs"
    if not outputs.exists():
        log("no outputs directory was produced")
        return

    group = args.run_group or os.environ.get("JOB_ID", "run")

    with open(outputs / "job_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # The results dataset repo is the real persistence mechanism. A mounted
    # bucket, if one is present, is a convenience on top of it.
    if args.output_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.output_repo, repo_type="dataset", private=True,
                        exist_ok=True)
        log(f"uploading results to {args.output_repo} under {group}/")
        try:
            api.upload_folder(
                folder_path=str(outputs), path_in_repo=f"{group}/outputs",
                repo_id=args.output_repo, repo_type="dataset",
                commit_message=f"Results for {group}",
                # Model weights and embedding caches are large and regenerable;
                # keeping them out makes the results repo something you can
                # actually clone. Adapters are small and worth keeping.
                ignore_patterns=["*.npz", "encoder.pt"],
            )
            log("upload complete")
        except Exception as exc:
            log(f"UPLOAD FAILED: {exc}")
            log("results still exist inside the container; retrieve them with hf jobs ssh")
    else:
        log("no --output-repo given; results will be lost when the job ends")

    mounted = Path(args.output_dir)
    if mounted.is_dir():
        target = mounted / group
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(outputs, target / "outputs", dirs_exist_ok=True)
        log(f"also copied to the mounted volume at {target}")


def main():
    args = parse_args()
    report_environment()

    workdir = fetch_bundle(args.repo)
    sys.path.insert(0, str(workdir))

    chains = resolve_commands(args, workdir)
    commands = [c for chain in chains for c in chain]
    log(f"{len(chains)} chains, {len(commands)} commands")
    preflight(commands, workdir)

    summary = {"commands": [], "job_id": os.environ.get("JOB_ID"),
               "accelerator": os.environ.get("ACCELERATOR")}
    failed = False

    for index, chain in enumerate(chains, 1):
        for position, command in enumerate(chain):
            record = run_command(command, workdir)
            record["chain"] = index
            summary["commands"].append(record)
            if record["returncode"] != 0:
                failed = True
                skipped = len(chain) - position - 1
                log(f"chain {index} failed; skipping its remaining {skipped} "
                    f"command(s) and continuing with the next chain")
                summary.setdefault("skipped", []).extend(chain[position + 1:])
                break

    # Publish whatever exists, including partial results from a failed batch.
    # Losing four completed runs because a fifth crashed would be the expensive
    # kind of mistake here.
    publish(workdir, args, summary)

    log("finished with failures" if failed else "finished cleanly")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
