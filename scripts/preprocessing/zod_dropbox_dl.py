"""Download ZOD Frames subset from Zenseact's shared Dropbox.

Pulls only the bits needed for calib training:
  infos / annotations / images_dnat (×2 chunks) / lidar_velodyne_core (×10 chunks) / oxts.

~462 GB total. Resumes by skipping files already at correct on-disk size.
Requires DBX_TOKEN env var (Dropbox app token with sharing.read +
files.metadata.read + files.content.read).

Usage:
  python scripts/preprocessing/zod_dropbox_dl.py            # default dst
  python scripts/preprocessing/zod_dropbox_dl.py --dst /mnt/x/zod_frames
"""
import argparse
import os
import sys
import time
from pathlib import Path

import dropbox

URL = (
    "https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/"
    "AFuqa-QrSkGzHmnkhhpvbBE?rlkey=ocr9n0gq3u083zj8sn1yo1ak6"
)
KEEP_PREFIX = ("infos", "annotations", "images_dnat", "lidar_velodyne_core", "oxts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="/mnt/nvme6t/zod/frames")
    ap.add_argument("--chunk-size-mb", type=int, default=8)
    args = ap.parse_args()

    token = os.environ.get("DBX_TOKEN")
    if not token:
        sys.exit("DBX_TOKEN env var missing — `source ~/.bashrc` first")

    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    dbx = dropbox.Dropbox(token, timeout=600)
    shared = dropbox.files.SharedLink(url=URL)

    # List + filter
    res = dbx.files_list_folder(path="/single_frames", shared_link=shared)
    targets = []
    for e in res.entries:
        if not isinstance(e, dropbox.files.FileMetadata):
            continue
        if any(e.name.startswith(p) for p in KEEP_PREFIX):
            targets.append((e.name, e.size))
    targets.sort()
    total = sum(s for _, s in targets)
    print(f"target: {len(targets)} files, {total/1e9:.1f} GB → {dst}", flush=True)

    chunk = args.chunk_size_mb * 1024 * 1024
    grand_t0 = time.time()
    grand_dl = 0
    for i, (name, size) in enumerate(targets, 1):
        out = dst / name
        if out.exists() and out.stat().st_size == size:
            print(f"[{i:2}/{len(targets)}] SKIP {name}  ({size/1e9:.2f} GB cached)",
                  flush=True)
            continue

        # Resume: drop partial, restart this file (Dropbox sharing API doesn't
        # support range requests reliably — simpler to redo full file)
        if out.exists():
            out.unlink()

        print(f"[{i:2}/{len(targets)}] DL   {name}  ({size/1e9:.2f} GB)", flush=True)
        t0 = time.time()
        md, resp = dbx.sharing_get_shared_link_file(url=URL, path=f"/single_frames/{name}")
        wrote = 0
        with open(out, "wb") as f:
            for c in resp.iter_content(chunk_size=chunk):
                if not c:
                    continue
                f.write(c)
                wrote += len(c)
                if wrote % (chunk * 32) < chunk:
                    pct = wrote / size * 100
                    mbps = wrote / 1e6 / max(1e-9, time.time() - t0)
                    eta = (size - wrote) / 1e6 / max(1, mbps) / 60
                    print(f"           {pct:5.1f}%  {wrote/1e9:5.2f}/{size/1e9:.2f} GB  "
                          f"{mbps:6.1f} MB/s  eta {eta:.1f}min",
                          flush=True)
        elapsed = time.time() - t0
        grand_dl += wrote
        print(f"           done {wrote/1e9:.2f} GB in {elapsed/60:.1f} min "
              f"({wrote/1e6/elapsed:.1f} MB/s)", flush=True)

    grand_elapsed = time.time() - grand_t0
    print(f"=== ALL DONE: {grand_dl/1e9:.1f} GB in {grand_elapsed/60:.1f} min "
          f"({grand_dl/1e6/grand_elapsed:.1f} MB/s avg) ===", flush=True)


if __name__ == "__main__":
    main()
