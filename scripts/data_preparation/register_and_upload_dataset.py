#!/usr/bin/env python3
"""Register a local cache directory as a ClearML Dataset AND physically
upload its contents to the ClearML fileserver.

Counterpart to register_dataset.py, which only records file://... URIs —
useful when the data is already on a shared filesystem (Lustre). For
node-local caches (e.g. /mnt/datadisk3/... on a workstation), external_files
won't be reachable from DGX nodes; use this script instead so workers can
`Dataset.get().get_local_copy()` the data from the fileserver.

Example:
    python scripts/data_preparation/register_and_upload_dataset.py \
      --path /mnt/datadisk3/tmpoc_kamikado/cache/kamikado_v3_tiled \
      --name kamikado_v3_tiled \
      --tags kamikado tile v3 fisheye fcm \
      --description "Kamikado woven_sequence FCM fisheye tile cache."
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from clearml import Dataset


DEFAULT_PROJECT = "e2e_calib/datasets"


def _git_rev(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _du_bytes(path: Path) -> int:
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(out.split()[0])
    except Exception:
        tot = 0
        for r, _, fs in os.walk(path):
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
        return tot


def _count_files(path: Path) -> int:
    try:
        out = subprocess.check_output(
            f"find {path} -type f | wc -l", shell=True, text=True
        )
        return int(out.strip())
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--description", default="")
    ap.add_argument("--parents", nargs="*", default=[])
    ap.add_argument("--output-uri", default=None,
                    help="override output_uri (defaults to ClearML files server)")
    ap.add_argument("--chunk-size", type=int, default=1024,
                    help="MB per upload chunk (default 1024)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = args.path.resolve()
    if not path.is_dir():
        sys.exit(f"not a dir: {path}")

    size_b = _du_bytes(path)
    n_files = _count_files(path)
    header = (
        f"path:     {path}\n"
        f"size:     {size_b / 1e9:.2f} GB ({size_b} B)\n"
        f"n_files:  {n_files}\n"
        f"host:     {os.uname().nodename}\n"
        f"git_rev:  {_git_rev(Path(__file__).resolve().parents[2])}\n"
        f"upload:   add_files (physical copy to ClearML fileserver)\n"
    )
    full_description = f"{args.description.strip()}\n\n---\n{header}"

    print(f"[register+upload] {args.project}/{args.name}")
    print(f"  path : {path}")
    print(f"  size : {size_b / 1e9:.2f} GB  files: {n_files}")
    print(f"  tags : {args.tags}")
    if args.dry_run:
        print("  (dry-run)")
        return

    ds = Dataset.create(
        dataset_name=args.name,
        dataset_project=args.project,
        dataset_tags=args.tags,
        description=full_description,
        parent_datasets=args.parents or None,
        output_uri=args.output_uri,
    )
    ds.add_files(path=str(path), dataset_path='.', recursive=True,
                  verbose=False)
    ds.upload(show_progress=True, chunk_size=args.chunk_size)
    ds.finalize()
    print(f"  dataset_id: {ds.id}")


if __name__ == "__main__":
    main()
