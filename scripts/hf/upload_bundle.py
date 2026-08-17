#!/usr/bin/env python3
"""
Publish the images, metadata and code to a private Hub dataset repo.

Jobs run in an ephemeral container with no access to this laptop, so everything
a run needs has to live somewhere the job can reach. One private dataset repo
holds all of it:

    images/<age>/<measurement_id>.jpg     the 8,619 images named in the CSV
    cod_otolith_age_final_with_scale.csv  the metadata
    code/                                 src, scripts and configs

Images and metadata change almost never; code changes every time we iterate. The
two are uploaded separately so a code fix does not re-check 8,619 image files.

    python scripts/hf/upload_bundle.py --all          # first time
    python scripts/hf/upload_bundle.py --code         # after editing code
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pandas as pd
from huggingface_hub import HfApi

DEFAULT_REPO = "hafsteinn/otolith-cod-bundle"

#: Copied into the job container. Anything not listed here is unavailable at
#: run time, which is deliberate: it keeps the uploaded bundle small and makes
#: the job's dependencies explicit.
CODE_PATHS = ["src", "scripts", "configs", "pyproject.toml"]

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "outputs", ".pytest_cache"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--images", action="store_true", help="Upload the image files.")
    p.add_argument("--metadata", action="store_true", help="Upload the metadata CSV.")
    p.add_argument("--code", action="store_true", help="Upload src, scripts and configs.")
    p.add_argument("--all", action="store_true", help="Upload everything.")
    p.add_argument("--image-root", default="otolith_images/segmented_images")
    p.add_argument("--metadata-csv", default="cod_otolith_age_final_with_scale.csv")
    p.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def collect_images(image_root: Path, metadata_csv: Path):
    """Only the images named in the metadata CSV, keyed by measurement ID.

    Uploading the whole source tree would carry roughly 6,300 extra images that
    the pipeline filters out anyway, and would make the age distribution in the
    repo disagree with the one the paper reports.
    """
    valid = set(
        pd.read_csv(metadata_csv, usecols=["measurement_id"])["measurement_id"]
        .astype(int)
    )
    selected = []
    for age_dir in sorted(image_root.iterdir()):
        if not age_dir.is_dir():
            continue
        for image in sorted(age_dir.glob("*.jpg")):
            if image.stem.isdigit() and int(image.stem) in valid:
                selected.append((image, f"images/{age_dir.name}/{image.name}"))
    return selected


def stage_code(staging: Path):
    """Copy the code into a staging directory, dropping caches and outputs."""
    ignore = shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc", "*.npz", "*.pt", "*.pdf")
    for entry in CODE_PATHS:
        source = project_root / entry
        if not source.exists():
            print(f"  skipping {entry} (not present)")
            continue
        target = staging / "code" / entry
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return staging / "code"


def main():
    args = parse_args()
    if args.all:
        args.images = args.metadata = args.code = True
    if not any([args.images, args.metadata, args.code]):
        print("Nothing selected. Pass --all, or one of --images / --metadata / --code.")
        return 1

    api = HfApi()
    image_root = project_root / args.image_root
    metadata_csv = project_root / args.metadata_csv

    if not args.dry_run:
        api.create_repo(args.repo, repo_type="dataset", private=args.private,
                        exist_ok=True)
        print(f"Repo ready: https://huggingface.co/datasets/{args.repo}")

    if args.metadata:
        print(f"\nUploading {metadata_csv.name} ({metadata_csv.stat().st_size/1e6:.1f} MB)")
        if not args.dry_run:
            api.upload_file(path_or_fileobj=str(metadata_csv),
                            path_in_repo=metadata_csv.name,
                            repo_id=args.repo, repo_type="dataset",
                            commit_message="Update metadata CSV")

    if args.images:
        images = collect_images(image_root, metadata_csv)
        total_mb = sum(p.stat().st_size for p, _ in images) / 1e6
        print(f"\n{len(images)} images selected ({total_mb:.0f} MB)")
        if not args.dry_run:
            with tempfile.TemporaryDirectory() as tmp:
                staging = Path(tmp)
                for source, rel in images:
                    destination = staging / rel
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                api.upload_folder(folder_path=str(staging / "images"),
                                  path_in_repo="images",
                                  repo_id=args.repo, repo_type="dataset",
                                  commit_message=f"Upload {len(images)} otolith images")

    if args.code:
        print("\nStaging code")
        with tempfile.TemporaryDirectory() as tmp:
            code_dir = stage_code(Path(tmp))
            files = list(code_dir.rglob("*"))
            print(f"  {sum(1 for f in files if f.is_file())} files")
            if not args.dry_run:
                api.upload_folder(folder_path=str(code_dir), path_in_repo="code",
                                  repo_id=args.repo, repo_type="dataset",
                                  commit_message="Update code bundle",
                                  delete_patterns="*")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
