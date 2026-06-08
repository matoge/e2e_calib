"""Pair the first and last frame from a SplatAD-rendered scene with their
GT counterparts and stack into a 3-up image (rendered | GT | residual).

Output layout per scene:
    <compare-dir>/ps<scene>_first.png   (3 panels: render, gt, |render-gt|)
    <compare-dir>/ps<scene>_last.png    (idem)
    <compare-dir>/all_scenes_grid.png   (one row per scene: first + last)
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_render_dir(out_root: Path, scene: str, ts: str) -> Path | None:
    """Locate render_train dir under <out_root>/ps<scene>_front_so3xr3_<ts>/."""
    exp_dir = out_root / f"ps{scene}_front_so3xr3_{ts}"
    cand = list(exp_dir.glob("render_train/**/*.png")) + \
           list(exp_dir.glob("render_train/**/*.jpg"))
    if not cand:
        return None
    # ns-render typically outputs <name>/<frame_id>.png; just return the dir holding rgb
    return cand[0].parent


def first_last_of(dir_: Path) -> tuple[Path, Path] | None:
    files = sorted([p for p in dir_.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if len(files) < 2:
        return None
    return files[0], files[-1]


def find_gt_for(rendered_path: Path, scene: str) -> Path | None:
    """Look up the matching GT image in /mnt/fsx/tmp/hfunaya/pandaset/<scene>/camera/front_camera/<NN>.jpg.
    rendered_path filename is typically <NNNN>.png — strip and convert."""
    stem = rendered_path.stem
    # try interpreting stem as a frame index (with leading zeros)
    try:
        idx = int(stem)
    except ValueError:
        return None
    gt_dir = Path(f"/mnt/fsx/tmp/hfunaya/pandaset/{scene}/camera/front_camera")
    candidates = [gt_dir / f"{idx:02d}.jpg",
                  gt_dir / f"{idx:03d}.jpg",
                  gt_dir / f"{idx:04d}.jpg",
                  gt_dir / f"{stem}.jpg"]
    for p in candidates:
        if p.exists():
            return p
    # fall back to listing dir + idx-th
    if gt_dir.is_dir():
        files = sorted(gt_dir.glob("*.jpg"))
        if 0 <= idx < len(files):
            return files[idx]
    return None


def make_3up(render: Path, gt: Path, out: Path, title: str = ""):
    r = np.asarray(Image.open(render).convert("RGB"))
    g = np.asarray(Image.open(gt).convert("RGB"))
    # match sizes (render res may differ from GT)
    if r.shape != g.shape:
        gpil = Image.open(gt).convert("RGB").resize((r.shape[1], r.shape[0]))
        g = np.asarray(gpil)
    diff = np.abs(r.astype(np.int16) - g.astype(np.int16)).clip(0, 255).astype(np.uint8)
    # stack horizontally
    panel = np.concatenate([r, g, diff], axis=1)
    img = Image.fromarray(panel)
    if title:
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 20)
        except Exception:
            f = ImageFont.load_default()
        d.text((10, 10), title, fill="yellow", font=f)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--compare-dir", type=Path, required=True)
    args = ap.parse_args()

    rows_for_grid = []  # list of (scene, first_panel_path, last_panel_path)
    for scene in args.scenes:
        rdir = find_render_dir(args.out_root, scene, args.ts)
        if rdir is None:
            print(f"[skip] ps{scene}: no rendered frames")
            continue
        fl = first_last_of(rdir)
        if fl is None:
            print(f"[skip] ps{scene}: < 2 frames")
            continue
        first_r, last_r = fl
        first_g = find_gt_for(first_r, scene)
        last_g = find_gt_for(last_r, scene)
        if first_g is None or last_g is None:
            print(f"[skip] ps{scene}: GT not found ({first_r.stem} / {last_r.stem})")
            continue
        a = make_3up(first_r, first_g,
                     args.compare_dir / f"ps{scene}_first.png",
                     title=f"ps{scene} frame {first_r.stem}  L=render  C=GT  R=|diff|")
        b = make_3up(last_r, last_g,
                     args.compare_dir / f"ps{scene}_last.png",
                     title=f"ps{scene} frame {last_r.stem}  L=render  C=GT  R=|diff|")
        rows_for_grid.append((scene, a, b))
        print(f"[ok] ps{scene}: {a.name} {b.name}")

    if rows_for_grid:
        # Build single big grid PNG: one scene per row, [first | last] horizontally
        rows_imgs = []
        for scene, a, b in rows_for_grid:
            ra = np.asarray(Image.open(a).convert("RGB"))
            rb = np.asarray(Image.open(b).convert("RGB"))
            # align widths if needed
            if ra.shape[1] != rb.shape[1]:
                w = min(ra.shape[1], rb.shape[1])
                ra = ra[:, :w]; rb = rb[:, :w]
            rows_imgs.append(np.concatenate([ra, rb], axis=1))
        # max-width across rows
        max_w = max(r.shape[1] for r in rows_imgs)
        padded = []
        for r in rows_imgs:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.concatenate([r, pad], axis=1)
            padded.append(r)
        grid = np.concatenate(padded, axis=0)
        grid_p = args.compare_dir / "all_scenes_grid.png"
        Image.fromarray(grid).save(grid_p, quality=88)
        print(f"[grid] {grid_p}")


if __name__ == "__main__":
    main()
