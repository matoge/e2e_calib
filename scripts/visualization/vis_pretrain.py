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
    # Tile inst: K_full stays in PARENT coords (distortion-safe), so projection
    # lands at parent-image uv. Subtract (tile_u0, tile_v0) to bring into the
    # tile-local canvas the renderer is showing.
    tile_u0 = int(inst.get('tile_u0', 0)); tile_v0 = int(inst.get('tile_v0', 0))
    if tile_u0 or tile_v0:
        uv = uv - np.array([tile_u0, tile_v0], dtype=uv.dtype)
    vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    is_obj = _is_obj_per_point(pts, cubs).astype(bool)
    return img, uv, vis, is_obj, IH, IW


CUBOID_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def project_cuboids(cubs, T_gt, K, tile_u0: int = 0, tile_v0: int = 0):
    """Project each cuboid's 8 corners. Returns list of dicts {uv: (8,2), z: (8,), label}.

    For tile insts, K stays in parent-image coords (distortion-safe), so the
    projected uv must be brought into tile-local coords by subtracting the
    tile origin. Pass tile_u0/v0 from inst.get('tile_u0', 0) when drawing on
    a tile canvas.

    Dedupe near-duplicate annotations: PandaSet ships occasional same-label
    cuboids within ~0.5m of each other (annotation tooling artifact).
    Drawing both produces visible "double wireframe" overlap.
    """
    # dedupe: same label + center within 0.5m → keep first only
    seen = []
    cubs_d = []
    for c in cubs:
        pos = np.asarray(c['pos'], dtype=np.float32)
        lbl = c.get('label', '')
        is_dup = False
        for sp, sl in seen:
            if sl == lbl and float(np.linalg.norm(pos - sp)) < 0.5:
                is_dup = True; break
        if not is_dup:
            seen.append((pos, lbl))
            cubs_d.append(c)
    out = []
    off = np.array([tile_u0, tile_v0], dtype=np.float32)
    for c in cubs_d:
        pos = np.asarray(c['pos'], dtype=np.float32)
        dims = np.asarray(c['dims'], dtype=np.float32)
        yaw = float(c.get('yaw', 0.0))
        l2, w2, h2 = dims[0]/2, dims[1]/2, dims[2]/2
        corners = np.array([
            [+l2, +w2, -h2], [+l2, -w2, -h2], [-l2, -w2, -h2], [-l2, +w2, -h2],
            [+l2, +w2, +h2], [+l2, -w2, +h2], [-l2, -w2, +h2], [-l2, +w2, +h2],
        ], dtype=np.float32)
        cy, sn = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sn, 0], [sn, cy, 0], [0, 0, 1]], dtype=np.float32)
        corners_local = (R @ corners.T).T + pos[None, :]
        homo = np.column_stack([corners_local, np.ones(8, dtype=np.float32)])
        corners_cam = (T_gt @ homo.T).T[:, :3]
        z = corners_cam[:, 2]
        uv = ((K @ corners_cam.T)[:2] / np.maximum(corners_cam[:, 2:].T, 1e-6)).T
        uv = uv - off
        out.append({'uv': uv.astype(np.float32), 'z': z.astype(np.float32),
                    'label': c.get('label', '')})
    return out


def draw_cuboids(ax, cubs_proj, color='lime', lw=1.2, alpha=0.85,
                  uv_offset=(0, 0), uv_scale=1.0):
    """Draw 12-edge wireframes. uv_offset/scale lets us re-use for crop coords:
    crop_uv = (full_uv - (u0, v0)) * (S / cs).
    Skips a cuboid entirely if ANY corner is behind the camera — projecting a
    near-zero-z corner produces +/- inf pixel coords that would auto-expand the
    axes and squeeze the image into a corner."""
    u_off, v_off = uv_offset
    for cp in cubs_proj:
        uv = cp['uv']
        z = cp['z']
        if (z <= 0.5).any():
            continue
        u = (uv[:, 0] - u_off) * uv_scale
        v = (uv[:, 1] - v_off) * uv_scale
        for i, j in CUBOID_EDGES:
            ax.plot([u[i], u[j]], [v[i], v[j]],
                    color=color, lw=lw, alpha=alpha,
                    solid_capstyle='round', clip_on=True)


def patch_entropy(crop_uint8: np.ndarray) -> tuple[float, float]:
    """Returns (Shannon entropy in bits 0..8, Laplacian-variance proxy for sharpness).
    crop_uint8: (H, W, C) or (H, W), uint8."""
    if crop_uint8.dtype != np.uint8:
        crop_uint8 = np.clip(crop_uint8, 0, 255).astype(np.uint8)
    gray = crop_uint8.mean(axis=-1).astype(np.uint8) if crop_uint8.ndim == 3 else crop_uint8
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    p = hist[hist > 0].astype(np.float64) / hist.sum()
    H = float(-(p * np.log2(p)).sum())
    # cheap sharpness proxy: variance of 5-tap discrete Laplacian (no scipy dep)
    g = gray.astype(np.float32)
    lap = (-4.0 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return H, float(lap.var())


def render_pair(ds, idx, out_path, S=64):
    # pin np.random per idx → identical crop+perturb across runs / mid-train epochs
    np.random.seed(int(idx))
    out = ds[idx]
    img_crop, true_uvd, dist_uvd, vfp = out[:4]   # tolerate 6-tuple (uvd_full, pad_full ignored here)
    box = getattr(ds, '_last_crop', None)
    inst = ds._load_inst(idx)
    full, uv, vis, is_obj, IH, IW = _full_proj(inst)
    cubs_proj = project_cuboids(inst.get('cuboids', []),
                                  inst['T_gt'].numpy(), inst['K_full'].numpy(),
                                  tile_u0=int(inst.get('tile_u0', 0)),
                                  tile_v0=int(inst.get('tile_v0', 0)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7), dpi=110,
                                    gridspec_kw={'width_ratios': [16, 9]})
    axL.imshow(full)
    axL.set_xlim(0, IW); axL.set_ylim(IH, 0)  # lock before cuboid plot to prevent off-image lines from expanding axes
    m = vis & ~is_obj
    axL.scatter(uv[m, 0], uv[m, 1], c='yellow', s=2, marker='x', linewidths=0.4, alpha=0.5)
    m = vis & is_obj
    axL.scatter(uv[m, 0], uv[m, 1], c='lime', s=10, marker='x', linewidths=0.9)
    draw_cuboids(axL, cubs_proj, color='lime', lw=1.2, alpha=0.85)
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
    H_bits, lap_var = patch_entropy(crop_np)
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
    if box is not None and cubs_proj:
        scale = S / float(box['cs'])
        draw_cuboids(axR, cubs_proj, color='lime', lw=1.0, alpha=0.85,
                     uv_offset=(box['u0'], box['v0']), uv_scale=scale)
    axR.set_xlim(0, S); axR.set_ylim(S, 0); axR.axis('off')
    axR.set_title(f'GETITEM crop  N={len(t)} obj={int(is_obj_pt.sum())}  '
                  f'H={H_bits:.2f}b  lapV={lap_var:.0f}  vfp={float(vfp):.0f}',
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
    chosen = []
    saved = 0
    for idx in idxs:
        if saved >= args.n: break
        try:
            s, fr, n_obj, n_tot = render_pair(ds, idx, out / f'pre_{saved:02d}_idx{idx:06d}.png',
                                               S=args.img_size)
        except Exception as e:
            continue
        print(f'  pre_{saved:02d}: idx={idx}  scene={s} frame={fr}  N={n_tot} obj={n_obj}')
        chosen.append(int(idx))
        saved += 1
    # persist chosen idxs so mid-training vis can render the SAME samples
    import json
    (out / 'sample_idxs.json').write_text(json.dumps(chosen))


if __name__ == '__main__':
    main()
