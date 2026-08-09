#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "torchvision",
#   "transformers>=4.56",
#   "peft>=0.13",
#   "coral-pytorch",
#   "opencv-python-headless",
#   "scikit-learn",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "tqdm",
#   "pyyaml",
#   "shap",
#   "huggingface_hub>=1.9",
# ]
# ///
"""
Entry point executed inside a HuggingFace Job.

Fetches the code and data bundle from a private dataset repo, reconstructs the
directory layout the pipeline expects, runs one or more experiment commands, and
copies the results out to a persistent location before the container disappears.

`opencv-python-headless` rather than `opencv-python`: the uv base image has no
libGL, and the normal wheel fails to import there.

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
    p.add_argument("--command", action="append", required=True,
                   help="A script under scripts/ plus its arguments. Repeatable; "
                        "commands run in order and the job stops at the first failure.")
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

    destination = Path(args.output_dir)
    if destination.parent.exists() or destination.exists():
        target = destination / group
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(outputs, target / "outputs", dirs_exist_ok=True)
        with open(target / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        log(f"results written to {target}")
    else:
        log(f"{destination} is not mounted; skipping bucket copy")

    if args.output_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.output_repo, repo_type="dataset", private=True,
                        exist_ok=True)
        log(f"uploading results to {args.output_repo} under {group}/")
        api.upload_folder(folder_path=str(outputs), path_in_repo=f"{group}/outputs",
                          repo_id=args.output_repo, repo_type="dataset",
                          commit_message=f"Results for {group}")


def main():
    args = parse_args()
    report_environment()

    workdir = fetch_bundle(args.repo)
    sys.path.insert(0, str(workdir))

    summary = {"commands": [], "job_id": os.environ.get("JOB_ID"),
               "accelerator": os.environ.get("ACCELERATOR")}
    failed = False

    for command in args.command:
        record = run_command(command, workdir)
        summary["commands"].append(record)
        if record["returncode"] != 0:
            failed = True
            log("stopping: a command failed")
            break

    # Publish whatever exists, including partial results from a failed batch.
    # Losing four completed runs because a fifth crashed would be the expensive
    # kind of mistake here.
    publish(workdir, args, summary)

    log("finished with failures" if failed else "finished cleanly")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
