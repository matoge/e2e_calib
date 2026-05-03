#!/usr/bin/env python3
"""Register a directory under /mnt/fsx/tmp/hfunaya as a ClearML Dataset.

We use **`add_external_files` with `file:///...` URLs** instead of `add_files`:
the Lustre FS (`/mnt/fsx`) is already visible from every DGX node as the
same absolute path, so physically copying 10-100+ GB of cache into the
ClearML fileserver is wasteful. With external_files:

- ClearML Dataset UI lists the dataset, its parent history, description,
  tags, file list (as URIs), and size stats.
- `Dataset.get(name=..., project=...).get_local_copy()` on any node just
  returns the registered `/mnt/fsx/...` path — no download.
- We stop "losing" caches, because every cache that has been built
  through this script is discoverable from the ClearML UI / API with a
  stable (project, name) tuple.

Usage
-----
  # single directory
  python scripts/data_preparation/register_dataset.py \
      --path /mnt/fsx/tmp/hfunaya/e2e_calib_cache/waymo_v3_tiled \
      --name waymo_v3_tiled \
      --tags waymo tile v3 5cam stride10 \
      --description "Waymo OD v2 tile cache. 512x512 / stride=384 / y_start=200. 5 cam, 798 seg, stride 10."

  # batch mode (read JSON config)
  python scripts/data_preparation/register_dataset.py --config scripts/data_preparation/datasets.yaml

Run from HEATRUN (or any machine with clearml.conf pointed at the right
ClearML server). Auth comes from ~/clearml.conf.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # yaml optional
    yaml = None

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
    """Approximate directory size using du -sb; falls back to os.walk."""
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


def register_one(
    path: Path,
    name: str,
    project: str = DEFAULT_PROJECT,
    tags: list[str] | None = None,
    description: str = "",
    parents: list[str] | None = None,
    dry_run: bool = False,
) -> str | None:
    """Register `path` as a ClearML Dataset (external_files mode).

    Returns the ClearML dataset_id (or None for dry-run).
    """
    path = path.resolve()
    if not path.exists():
        print(f"[skip] {name}: {path} does not exist", file=sys.stderr)
        return None
    if not path.is_dir():
        print(f"[skip] {name}: {path} is not a directory", file=sys.stderr)
        return None

    size_b = _du_bytes(path)
    n_files = _count_files(path)
    size_h = f"{size_b / 1e9:.2f} GB"

    header = (
        f"path:     file://{path}\n"
        f"size:     {size_h} ({size_b} B)\n"
        f"n_files:  {n_files}\n"
        f"host:     {os.uname().nodename}\n"
        f"git_rev:  {_git_rev(Path(__file__).resolve().parents[2])}\n"
    )
    full_description = f"{description.strip()}\n\n---\n{header}"

    print(f"[register] {project}/{name}")
    print(f"  path  : {path}")
    print(f"  size  : {size_h}  files: {n_files}")
    print(f"  tags  : {tags or []}")
    if dry_run:
        print("  (dry-run; skipping ClearML API call)")
        return None

    ds = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        dataset_tags=tags or [],
        description=full_description,
        parent_datasets=parents or None,
    )
    # Use add_external_files: register by URI without copying to fileserver.
    # The URI must be file:///... (triple slash) for absolute POSIX paths.
    uri = f"file://{path}"
    n_added = ds.add_external_files(source_url=uri, recursive=True)
    print(f"  added {n_added} external file refs")
    # Even with only external_files, the generated manifest itself must be
    # uploaded to the ClearML fileserver before finalize.
    ds.upload(show_progress=False)
    ds.finalize()
    print(f"  dataset_id: {ds.id}")
    return ds.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, help="directory to register")
    ap.add_argument("--name", type=str, help="Dataset name")
    ap.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--description", type=str, default="")
    ap.add_argument("--parents", nargs="*", default=[],
                    help="parent dataset IDs (for lineage)")
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML/JSON with list of datasets to register in batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.config:
        text = args.config.read_text()
        if args.config.suffix in (".yaml", ".yml"):
            if yaml is None:
                print("PyYAML not installed", file=sys.stderr)
                sys.exit(2)
            cfg = yaml.safe_load(text)
        else:
            cfg = json.loads(text)
        results = []
        for entry in cfg:
            try:
                ds_id = register_one(
                    path=Path(entry["path"]),
                    name=entry["name"],
                    project=entry.get("project", DEFAULT_PROJECT),
                    tags=entry.get("tags", []),
                    description=entry.get("description", ""),
                    parents=entry.get("parents", []),
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"[error] {entry['name']}: {e}", file=sys.stderr)
                ds_id = f"ERROR: {e}"
            results.append((entry["name"], ds_id))
        print("\n=== summary ===")
        for n, i in results:
            print(f"  {n}: {i}")
        return

    if not args.path or not args.name:
        print("either --config OR (--path AND --name) required", file=sys.stderr)
        sys.exit(2)
    register_one(
        path=args.path,
        name=args.name,
        project=args.project,
        tags=args.tags,
        description=args.description,
        parents=args.parents,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
