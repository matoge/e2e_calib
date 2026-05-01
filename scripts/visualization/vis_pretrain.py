"""Pre-training sanity vis: 2-panel per sample —
  left  : full image with GT lidar projection + the actual __getitem__ crop box (red)
  right : the cropped 64x64 tile that __getitem__ returns, with true_uvd (lime ×)
          and dist_uvd (gold ×, perturbed). This is exactly what the model sees.
"""
import sys, pathlib, argparse, random as _r
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.pandaset_full import PandaSetCalibDatasetFull, _is_obj_per_point, decode_inst_img


def _full_proj(inst):
    img = decode_inst_img(inst).permute(1, 2, 0).numpy()
    IH, IW = img.shape[:2]
    K = inst['K_full'].numpy()
    T_gt = inst['T_gt'].numpy()
    pts = inst['pts'].numpy()
    cubs = inst.get('cuboids', [])
    homo = np.column_stack([pts, np.ones(len(pts))])
    pcam = (T_gt @ homo.T)[:3].T
    z = pcam[:, 2]
    uv = ((K @ pcam.T)[:2] / np.maximum(pcam[:, 2:].T, 1e-6)).T
    vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    is_obj = _is_obj_per_point(pts, cubs).astype(bool)
    return img, uv, vis, is_obj, IH, IW


def render_pair(ds, idx, out_path, S=64):
    img_crop, true_uvd, dist_uvd, vfp = ds[idx]
    box = getattr(ds, '_last_crop', None)
    inst = ds._load_inst(idx)
    full, uv, vis, is_obj, IH, IW = _full_proj(inst)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7), dpi=110,
                                    gridspec_kw={'width_ratios': [16, 9]})
    axL.imshow(full)
    m = vis & ~is_obj
    axL.scatter(uv[m, 0], uv[m, 1], c='yellow', s=2, marker='x', linewidths=0.4, alpha=0.5)
    m = vis & is_obj
    axL.scatter(uv[m, 0], uv[m, 1], c='lime', s=10, marker='x', linewidths=0.9)
    if box is not None:
        u0, v0, cs = box['u0'], box['v0'], box['cs']
        axL.add_patch(plt.Rectangle((u0, v0), cs, cs, fill=False, ec='red', lw=2))
    axL.set_title(f'FULL  scene={inst.get("scene")} frame={inst.get("frame")}  '
                  f'crop=({box["u0"] if box else "?"},{box["v0"] if box else "?"},'
                  f'{box["cs"] if box else "?"})  cubs={len(inst.get("cuboids", []))}',
                  fontsize=10)
    axL.axis('off')

    crop_t = img_crop.permute(1, 2, 0).cpu()
    if crop_t.dtype == torch.uint8:
        crop_np = crop_t.numpy()  # imshow handles uint8 natively
    else:
        crop_np = np.clip(crop_t.float().numpy(), 0, 1)
    axR.imshow(crop_np)
    t = true_uvd.numpy(); d = dist_uvd.numpy()
    is_obj_pt = d[:, 3] > 0.5
    axR.scatter(d[~is_obj_pt, 0], d[~is_obj_pt, 1], c='gold', s=18, marker='x', linewidths=0.7,
                alpha=0.7, label=f'dist bg ({(~is_obj_pt).sum()})')
    axR.scatter(d[is_obj_pt, 0], d[is_obj_pt, 1], c='gold', s=30, marker='x', linewidths=1.0,
                label=f'dist obj ({is_obj_pt.sum()})')
    axR.scatter(t[~is_obj_pt, 0], t[~is_obj_pt, 1], c='yellow', s=10, marker='x',
                linewidths=0.5, alpha=0.6, label='GT bg')
    axR.scatter(t[is_obj_pt, 0], t[is_obj_pt, 1], c='lime', s=22, marker='x', linewidths=1.1,
                label='GT obj')
    for j in range(len(t)):
        axR.plot([d[j, 0], t[j, 0]], [d[j, 1], t[j, 1]],
                 '-', color=('orange' if is_obj_pt[j] else 'deepskyblue'),
                 lw=0.5, alpha=0.5)
    axR.set_xlim(0, S); axR.set_ylim(S, 0); axR.axis('off')
    axR.set_title(f'GETITEM crop  N={len(t)} obj={int(is_obj_pt.sum())}  vfp={float(vfp):.0f}',
                  fontsize=10)
    axR.legend(loc='upper right', fontsize=7, framealpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight'); plt.close()
    return inst.get('scene'), int(inst.get('frame', -1)), int((d[:, 3] > 0.5).sum()), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full')
    ap.add_argument('--out',   default=None)
    ap.add_argument('--n',     type=int, default=10)
    ap.add_argument('--seed',  type=int, default=0)
    ap.add_argument('--img-size',     type=int, default=64)
    ap.add_argument('--min-crop-px',  type=int, default=128)
    ap.add_argument('--max-crop-px',  type=int, default=384)
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out) if args.out else (cache / 'vis_pretrain')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'): old.unlink()

    ds = PandaSetCalibDatasetFull(cache, split='val', img_size=args.img_size,
                                   min_crop_px=args.min_crop_px,
                                   max_crop_px=args.max_crop_px, oversample=1)
    idxs = list(range(len(ds))); _r.Random(args.seed).shuffle(idxs)
    print(f'cache: {cache}  insts: {len(ds)}  rendering {args.n} → {out}')
    saved = 0
    for idx in idxs:
        if saved >= args.n: break
        try:
            s, fr, n_obj, n_tot = render_pair(ds, idx, out / f'pre_{saved:02d}_idx{idx:06d}.png',
                                               S=args.img_size)
        except Exception as e:
            continue
        print(f'  pre_{saved:02d}: idx={idx}  scene={s} frame={fr}  N={n_tot} obj={n_obj}')
        saved += 1


if __name__ == '__main__':
    main()
