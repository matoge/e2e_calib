"""Waymo V3 cache pretrain vis: full-image + GT lidar projection.

Per camera (cam_id 1..5), pick N random insts, render with cached uv_full
overlaid on the JPEG image. No perturbation, no model — just sanity check
that lidar projection lines up with the camera image.
"""
import sys, pathlib, argparse, random, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def render_one(inst_path: Path, out_path: Path):
    inst = torch.load(inst_path, weights_only=False)
    img = np.asarray(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    IH, IW = img.shape[:2]
    uv = inst['uv_full'].numpy() if torch.is_tensor(inst['uv_full']) else inst['uv_full']
    z  = inst['z_cam'].numpy()  if torch.is_tensor(inst['z_cam'])   else inst['z_cam']
    cam = inst.get('cam_id', '?')
    seg = inst.get('seg', '?')
    ts  = inst.get('ts', '?')

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(img)
    # color by depth (closer = warm)
    z_norm = np.clip(z / 60.0, 0, 1) if len(z) else z
    ax.scatter(uv[:, 0], uv[:, 1], c=1 - z_norm, cmap='turbo', s=10, marker='x',
               linewidths=0.7, alpha=0.85)
    ax.set_title(f'{seg} cam={cam} ts={ts}  IW={IW} IH={IH}  pts={len(uv)}',
                 fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight'); plt.close()
    return cam, seg, ts, len(uv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/waymo_v3_full')
    ap.add_argument('--out',   default=None, help='default = <cache>/vis_pretrain')
    ap.add_argument('--n-per-cam', type=int, default=5)
    ap.add_argument('--seed',  type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out) if args.out else (cache / 'vis_pretrain')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.rglob('*.png'): old.unlink()

    inst_files = sorted((cache / 'inst').glob('*.pt'))
    print(f'cache: {cache}  total insts: {len(inst_files)}')

    # Group by cam_id (cheap: just open headers via torch.load lightly)
    by_cam = {}
    for f in inst_files:
        try: meta = torch.load(f, weights_only=False)
        except Exception: continue
        cid = int(meta.get('cam_id', 0))
        by_cam.setdefault(cid, []).append(f)

    for cid, files in sorted(by_cam.items()):
        print(f'  cam_id={cid}: {len(files)} insts')

    rng = random.Random(args.seed)
    for cid, files in sorted(by_cam.items()):
        cam_dir = out / f'cam_{cid}'
        cam_dir.mkdir(exist_ok=True)
        picked = rng.sample(files, min(args.n_per_cam, len(files)))
        for i, f in enumerate(picked):
            cam, seg, ts, n = render_one(f, cam_dir / f'pre_{i:02d}_{f.stem}.png')
            print(f'    cam{cid} pre_{i:02d}: {seg}@{ts}  pts={n}')

    print(f'saved → {out}')


if __name__ == '__main__':
    main()
