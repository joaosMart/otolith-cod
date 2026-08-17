#!/usr/bin/env python3
"""Copy finished batch outputs from the results repo into `results/`.

Jobs publish to `hafsteinn/otolith-cod-results` under `<batch>/outputs/...`.
This mirrors them into the local layout the manuscript scripts read,
`results/<batch>/...`, dropping the `outputs/` level.

Weights stay on the Hub. Only JSON, CSV and PNG are copied, which is what keeps
`results/` clonable, and it is the rule `results/README.md` documents.

    uv run python scripts/hf/fetch_results.py                 # everything new
    uv run python scripts/hf/fetch_results.py --batch curve   # name prefix
    uv run python scripts/hf/fetch_results.py --list          # show, copy nothing
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "hafsteinn/otolith-cod-results"
DEST = Path("results")
KEEP = {".json", ".csv", ".png"}

#: Adapter directories carry a config.json and a README.md that would otherwise
#: pass the extension filter and pull in per-run clutter with no numbers in it.
SKIP_PARTS = ("/adapter/", "/embeddings/")


def wanted(path: str) -> bool:
    return (Path(path).suffix in KEEP
            and not any(part in path for part in SKIP_PARTS))


def local_path(remote: str) -> Path:
    parts = [p for p in Path(remote).parts if p != "outputs"]
    return DEST.joinpath(*parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="", help="Only batches whose name starts with this.")
    ap.add_argument("--list", action="store_true", help="Show what would be copied.")
    ap.add_argument("--force", action="store_true", help="Re-copy files that already exist.")
    args = ap.parse_args()

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    todo = []
    for remote in sorted(files):
        if args.batch and not remote.startswith(args.batch):
            continue
        if not wanted(remote):
            continue
        target = local_path(remote)
        if target.exists() and not args.force:
            continue
        todo.append((remote, target))

    if args.list or not todo:
        print(f"{len(todo)} file(s) to copy")
        for remote, _ in todo[:40]:
            print("  ", remote)
        if len(todo) > 40:
            print(f"   ... and {len(todo) - 40} more")
        return 0

    for remote, target in todo:
        cached = hf_hub_download(REPO, remote, repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
    print(f"copied {len(todo)} file(s) into {DEST}/")

    batches = sorted({Path(t).parts[1] for _, t in todo})
    print("batches touched:", ", ".join(batches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
