"""1 raw 4K frame → tile_cutter (40 tiles) → SAME (t, ypr) perturbation
applied to every tile via PandaSetCalibDatasetFull.apply_perturbation_explicit
→ batch forward → 8×5 grid PNG.

Uses the SAME projection / crop / bucketing path as __getitem__ (the
trainer), but the perturbation is fixed (rig-level extrinsic drift), so
each tile is now an observation of the SAME extrinsic — exactly what
multi-tile BA expects.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import io
import numpy as np
import torch
from PIL import Image as _PIL, ImageDraw

from scripts.data.adapters.kamikado import load_frame, list_frames
from scripts.data.tile_cutter import frame_to_tiles
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model

EXP    = 'km_wv_wm_n6_img128_cs256_512_200ep_dgx2_8gpu'
SCENE  = '/raw_kamikado/scenes/points_ip664_D_20260226_224648_d005_3000_3020'
TILE   = 512
STRIDE = 448
S_IN   = 128
SEED   = 42
OUT    = REPO / 'experiments' / EXP / '_eval_vis' / 'frame_tiles_predict.png'

THUMB_H = 96
MAIN_PX = 256
GAP = 4
GRID_GAP = 16


class _InMemoryDS(PandaSetCalibDatasetFull):
    """Minimal subclass: feed a Python list of inst dicts instead of a cache."""
    def __init__(self, insts, **ds_kw):
        self._insts = insts
        self.fnames = list(range(len(insts)))
        self._cache = None
        self._use_lmdb = False
        self._lmdb_env = None
        self._cubs_map = {}
        self.img_size      = ds_kw.get('img_size', S_IN)
        self.min_crop_px   = ds_kw.get('min_crop_px', TILE)
        self.max_crop_px   = ds_kw.get('max_crop_px', TILE)
        self.max_rot_deg   = ds_kw.get('max_rot_deg', 1.5)
        self.max_offset_m  = ds_kw.get('max_offset_m', 0.6)
        self.max_fx_pct    = 0.0
        self.max_fy_pct    = 0.0
        self.pose_frame    = 'orig'
        self.grid_n        = ds_kw.get('grid_n', 16)
        self.n_full        = ds_kw.get('n_full', 1024)
        self.k_per_cell    = ds_kw.get('k_per_cell', 8)
        self.oversample    = 1
        self.zoom_aug      = False
        self.rep_strategy  = 'cell_center'
        self.center_band   = 0.0
        self.fixed_center_crop = False
        self.min_pts       = 8
        self.max_tries     = 16
        self.frame_stride  = 1

    def __len__(self):
        return len(self._insts)

    def _load_inst(self, idx):
        raw = self._insts[idx]
        out = {}
        for k, v in raw.items():
            out[k] = torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
        return out


def render_panel(parent_img, tile_inst, img_t, dist_uv, pred_uv, true_uv,
                  err_pre, err_post, sx_a, sy_a, rho_a):
    pH, pW = parent_img.shape[:2]
    thumb_w = int(round(THUMB_H * pW / pH))
    parent_thumb = _PIL.fromarray(parent_img).resize((thumb_w, THUMB_H), _PIL.BILINEAR)
    draw = ImageDraw.Draw(parent_thumb)
    sx_ = thumb_w / pW
    sy_ = THUMB_H / pH
    u0 = tile_inst['tile_u0']; v0 = tile_inst['tile_v0']
    iw = tile_inst['IW']; ih = tile_inst['IH']
    rx0, ry0 = int(u0 * sx_), int(v0 * sy_)
    rx1, ry1 = int((u0 + iw) * sx_), int((v0 + ih) * sy_)
    draw.rectangle([rx0, ry0, max(rx0 + 1, rx1 - 1), max(ry0 + 1, ry1 - 1)],
                    outline=(255, 0, 0), width=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img_np = img_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    fig, ax = plt.subplots(figsize=(MAIN_PX/96, MAIN_PX/96), dpi=96)
    ax.imshow(img_np)
    valid = ~((dist_uv[:, 0] == 0) & (dist_uv[:, 1] == 0))
    if valid.any():
        for k in np.where(valid)[0]:
            ax.annotate('', xy=(dist_uv[k, 0], dist_uv[k, 1]),
                         xytext=(true_uv[k, 0], true_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=0.4, alpha=0.6), zorder=2)
            ax.annotate('', xy=(pred_uv[k, 0], pred_uv[k, 1]),
                         xytext=(dist_uv[k, 0], dist_uv[k, 1]),
                         arrowprops=dict(arrowstyle='->', color='cyan', lw=0.4, alpha=0.85), zorder=3)
        ax.scatter(true_uv[valid, 0], true_uv[valid, 1], s=5, c='yellow', marker='x', zorder=5)
        ax.scatter(dist_uv[valid, 0], dist_uv[valid, 1], s=4, facecolors='none', edgecolors='red', linewidths=0.5, zorder=6)
        ax.scatter(pred_uv[valid, 0], pred_uv[valid, 1], s=4, facecolors='none', edgecolors='lime', linewidths=0.5, zorder=7)
    ax.set_xlim(0, S_IN); ax.set_ylim(S_IN, 0); ax.axis('off')
    ax.set_title(f'N={int(valid.sum())} pre={err_pre:.1f}→{err_post:.1f}px',
                  fontsize=6, pad=2)
    fig.subplots_adjust(0, 0, 1, 0.93)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=96)
    plt.close(fig)
    buf.seek(0)
    main_pil = _PIL.open(buf).convert('RGB').resize((MAIN_PX, MAIN_PX), _PIL.BILINEAR)

    panel_h = max(THUMB_H, MAIN_PX)
    combo = _PIL.new('RGB', (thumb_w + GAP + MAIN_PX, panel_h), 'black')
    combo.paste(parent_thumb, (0, (panel_h - THUMB_H) // 2))
    combo.paste(main_pil, (thumb_w + GAP, (panel_h - MAIN_PX) // 2))
    return combo


def main():
    device = torch.device('cuda')
    model = load_calib_model(EXP).eval()

    frames = list_frames(Path(SCENE))
    fidx = frames[len(frames) // 2]
    cf = load_frame(SCENE, fidx)
    parent = np.asarray(cf.img)
    pH, pW = parent.shape[:2]
    print(f'parent {pW}×{pH}  frame {fidx}')

    tile_insts = frame_to_tiles(cf, tile_w=TILE, tile_h=TILE, stride=STRIDE)
    print(f'{len(tile_insts)} tiles')

    # ONE shared (t, ypr) for the whole frame
    rng = np.random.default_rng(SEED)
    t   = (rng.random(3) * 2 - 1) * 0.6
    ypr = (rng.random(3) * 2 - 1) * 1.5
    print(f'shared perturbation t={t.round(3)} m  ypr={ypr.round(3)} deg')

    ds = _InMemoryDS(tile_insts, img_size=S_IN,
                      min_crop_px=TILE, max_crop_px=TILE)
    samples = []
    for i in range(len(tile_insts)):
        s = ds.apply_perturbation_explicit(i, t, ypr)
        samples.append((i, s))
    samples = [(i, s) for i, s in samples if s is not None]
    print(f'usable: {len(samples)}/{len(tile_insts)}')
    if not samples:
        print('no usable tiles'); return

    batch = collate_full([s for _, s in samples])
    imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
    imgs_g = imgs.to(device).float().div(255.0)
    pad_g = pad_mask.to(device); vfp_g = vfp.to(device)
    b_uvd_g = b_uvd.to(device); b_v_g = b_v.to(device)
    use_int = bool(getattr(model, 'use_intensity', False))
    pin = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
           if use_int else dist_uvd[..., :3]).to(device)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        params = model(imgs_g, pin, key_padding_mask=pad_g, vfp=vfp_g,
                        bucket_uvd=b_uvd_g, bucket_valid=b_v_g)
    p = params.float().cpu().numpy()
    dist_uv_b = dist_uvd[..., :2].numpy()
    true_uv_b = true_uvd[..., :2].numpy()
    pred_uv_b = dist_uv_b + p[..., :2]
    sx_b = np.exp(p[..., 2]); sy_b = np.exp(p[..., 3]); rho_b = np.tanh(p[..., 4])

    from collections import defaultdict
    rows_dict = defaultdict(list)
    for k, (i, _) in enumerate(samples):
        ti = tile_insts[i]
        rows_dict[ti['tile_v0']].append((ti['tile_u0'], k, i))
    row_keys = sorted(rows_dict)
    cols = max(len(rows_dict[r]) for r in row_keys)
    rows = len(row_keys)
    print(f'grid {cols}×{rows}')

    panels = {}
    pw = ph = 0
    err_post_list = []
    for r, v0 in enumerate(row_keys):
        for c, (u0, k, ti_idx) in enumerate(sorted(rows_dict[v0])):
            ti = tile_insts[ti_idx]
            valid = ~((dist_uv_b[k, :, 0] == 0) & (dist_uv_b[k, :, 1] == 0))
            if not valid.any():
                continue
            err_pre  = float(np.linalg.norm(dist_uv_b[k, valid] - true_uv_b[k, valid], axis=1).mean())
            err_post = float(np.linalg.norm(pred_uv_b[k, valid] - true_uv_b[k, valid], axis=1).mean())
            err_post_list.append(err_post)
            panels[(r, c)] = render_panel(parent, ti, imgs[k],
                                          dist_uv_b[k], pred_uv_b[k], true_uv_b[k],
                                          err_pre, err_post,
                                          sx_b[k], sy_b[k], rho_b[k])
            pw = max(pw, panels[(r,c)].size[0]); ph = max(ph, panels[(r,c)].size[1])

    if err_post_list:
        ep = np.array(err_post_list)
        print(f'mean post err: {ep.mean():.2f}px median {np.median(ep):.2f} max {ep.max():.2f}')

    grid_w = pw * cols + GRID_GAP * (cols - 1)
    grid_h = ph * rows + GRID_GAP * (rows - 1)
    grid = _PIL.new('RGB', (grid_w, grid_h), 'black')
    for (r, c), p_ in panels.items():
        grid.paste(p_, (c * (pw + GRID_GAP), r * (ph + GRID_GAP)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT)
    print(f'wrote → {OUT}  size {grid.size}')


if __name__ == '__main__':
    main()
