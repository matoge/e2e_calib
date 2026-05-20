"""1 raw 4K frame → tile_cutter (40 tiles) → SAME (t, rpy) perturbation
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

from scipy.spatial.transform import Rotation
from scripts.data.adapters.kamikado import load_frame, list_frames
from scripts.data.tile_cutter import frame_to_tiles
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model

EXP    = 'km_wv_wm_n4_img128_cs256_512_200ep_dgx1_16gpu_resume'
SCENES_ROOT = '/raw_kamikado/scenes'
SCENES = [
    'points_ip664_D_20260226_224648_d005_3000_3020',
    'points_ip664_D_20260301_222527_d006_800_820',
    'points_ip664_D_20260304_231950_d007-mdc_IWATESAN_inside_2',
    'points_ip664_D_20260404_041811_d005_510_530',
    'points_ip664_D_20260405_232105_d002_350_370',
]
SCENE  = f'{SCENES_ROOT}/{SCENES[0]}'  # default for single-scene path
TILE   = 512
STRIDE = 448
S_IN   = 128
SEED   = 42
import os as _os
OUT    = REPO / 'experiments' / EXP / '_eval_vis' / _os.environ.get(
    'BA_OUT_NAME', 'frame_tiles_predict.png')

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


def main(scene_path: str = SCENE, model=None, device=None, scene_tag: str = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = load_calib_model(EXP).eval().to(device)

    frames = list_frames(Path(scene_path))
    fidx = frames[len(frames) // 2]
    cf = load_frame(scene_path, fidx)
    if scene_tag is None:
        scene_tag = Path(scene_path).name
    # Per-scene output paths so scene loop doesn't overwrite each other.
    out_root = REPO / 'experiments' / EXP / '_eval_vis' / 'multi_scene' / scene_tag
    out_root.mkdir(parents=True, exist_ok=True)
    global OUT
    OUT = out_root / 'frame_tiles_predict.png'
    print(f'\n=== scene: {scene_tag}  frame {fidx} ===')
    parent = np.asarray(cf.img)
    pH, pW = parent.shape[:2]
    print(f'parent {pW}×{pH}  frame {fidx}')

    tile_insts = frame_to_tiles(cf, tile_w=TILE, tile_h=TILE, stride=STRIDE)
    print(f'{len(tile_insts)} tiles')

    # ONE shared (t, rpy) for the whole frame — fixed inside the training
    # regime (±0.6 m, ±1.5°) so every tile shows a readable GT→dist arrow.
    # NAMING: build_sample / pandaset_full feed this array to
    # Rotation.from_euler('zyx', arr, degrees=True). 'zyx' (lowercase →
    # intrinsic) means element[0] = z-axis rot (cam optical axis = ROLL),
    # element[1] = y-axis rot (cam vertical = YAW), element[2] = x-axis rot
    # (cam horizontal = PITCH). The legacy variable name `ypr` in the
    # trainer is misleading — actual order is [roll, yaw, pitch] in degrees.
    t       = np.zeros(3, dtype=np.float64)
    rpy_deg = np.array([0.0, +1.0, +0.5], dtype=np.float64)      # [roll, yaw, pitch] → yaw=+1°, pitch=+0.5°

    print(f'shared perturbation t={t.round(3)} m  rpy={rpy_deg.round(3)} deg [roll,yaw,pitch]')

    ds = _InMemoryDS(tile_insts, img_size=S_IN,
                      min_crop_px=TILE, max_crop_px=TILE)
    samples = []
    for i in range(len(tile_insts)):
        s = ds.apply_perturbation_explicit(i, t, rpy_deg)
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

    # ─── BA: stack all per-pt observations across tiles in PARENT coords,
    # solve one 6×6 normal-equation (Huber-IRLS down-weighted), report δ̂.
    K_full = tile_insts[samples[0][0]]['K_full']
    K_full = K_full.numpy() if hasattr(K_full, 'numpy') else np.asarray(K_full)
    scale_p = TILE / S_IN  # tile-px → parent-px

    uv_obs_list, par_list, z_list, gt_uv_list = [], [], [], []
    for k, (i, _) in enumerate(samples):
        ti = tile_insts[i]
        valid = ~((dist_uv_b[k, :, 0] == 0) & (dist_uv_b[k, :, 1] == 0))
        if not valid.any(): continue
        # tile-local (S_IN px) → parent-px
        u_p = dist_uv_b[k, valid, 0] * scale_p + ti['tile_u0']
        v_p = dist_uv_b[k, valid, 1] * scale_p + ti['tile_v0']
        gu_p = true_uv_b[k, valid, 0] * scale_p + ti['tile_u0']
        gv_p = true_uv_b[k, valid, 1] * scale_p + ti['tile_v0']
        # Δuv (model output) is correction direction in tile-px → parent-px
        du_p = (pred_uv_b[k, valid, 0] - dist_uv_b[k, valid, 0]) * scale_p
        dv_p = (pred_uv_b[k, valid, 1] - dist_uv_b[k, valid, 1]) * scale_p
        # σ in parent-px (variance scales by scale²; std by scale)
        su_p = sx_b[k, valid] * scale_p
        sv_p = sy_b[k, valid] * scale_p
        rho_p = rho_b[k, valid]
        # depth: DOF_JAC needs cam-Z (light-axis), not euclidean. Recover Z
        # from the GT uv + euclidean dist using:
        #   d_eucl² = X² + Y² + Z²,  X = (u-cx)·Z/fx,  Y = (v-cy)·Z/fy
        # ⇒ Z = d_eucl / √(1 + ((u-cx)/fx)² + ((v-cy)/fy)²)
        d_eucl = (true_uvd[k, valid, 2].numpy() * 100.0).astype(np.float64)
        _fx, _fy, _cx, _cy = K_full[0,0], K_full[1,1], K_full[0,2], K_full[1,2]
        _xn = (gu_p - _cx) / _fx
        _yn = (gv_p - _cy) / _fy
        d_proxy = d_eucl / np.sqrt(1.0 + _xn*_xn + _yn*_yn)
        uv_obs_list.append(np.stack([u_p, v_p], axis=1))
        par_list.append(np.stack([du_p, dv_p, su_p, sv_p, rho_p], axis=1))
        z_list.append(d_proxy)
        gt_uv_list.append(np.stack([gu_p, gv_p], axis=1))

    uv_obs = np.concatenate(uv_obs_list)
    par_obs = np.concatenate(par_list)
    z_obs = np.concatenate(z_list)
    gt_uv = np.concatenate(gt_uv_list)
    print(f'BA pool: {len(uv_obs)} pts  '
           f'u range [{uv_obs[:,0].min():.0f},{uv_obs[:,0].max():.0f}] / parent {pW}  '
           f'v range [{uv_obs[:,1].min():.0f},{uv_obs[:,1].max():.0f}] / parent {pH}')
    # Sanity: how big is the model's Δuv vs the geometric residual (obs-GT)?
    # For yaw=1° on this rig (fx=1888), |obs-GT| ≈ 33 px at centre. If the
    # model is healthy, |model Δuv| should approach the same magnitude.
    geom = uv_obs - gt_uv
    pred_duv = par_obs[:, :2]
    print(f'  geom |obs-GT|     mean Δu={geom[:,0].mean():+.2f} Δv={geom[:,1].mean():+.2f}  '
           f'|·| mean {np.linalg.norm(geom,axis=1).mean():.2f} median {np.median(np.linalg.norm(geom,axis=1)):.2f}')
    print(f'  model Δuv         mean Δu={pred_duv[:,0].mean():+.2f} Δv={pred_duv[:,1].mean():+.2f}  '
           f'|·| mean {np.linalg.norm(pred_duv,axis=1).mean():.2f} median {np.median(np.linalg.norm(pred_duv,axis=1)):.2f}')
    print(f'  ratio             mean |model Δuv| / |geom| = '
           f'{np.linalg.norm(pred_duv,axis=1).mean()/np.linalg.norm(geom,axis=1).mean():.2f}')

    from scripts.ba.ba_multicam_corr import solve_dofs, DOF_JAC
    dof_names = ['omega_x', 'omega_y']  # 2-DoF: pitch, yaw only (roll out)
    # Centre-band filter: keep pts within ±25% of parent height around cy and
    # ±50% of parent width around cx, where KB distortion is mildest and the
    # pinhole Jacobian is closest to truth.
    band_u = 0.50 * pW / 2
    band_v = 0.25 * pH / 2
    in_band = (np.abs(uv_obs[:,0] - K_full[0,2]) < band_u) & \
              (np.abs(uv_obs[:,1] - K_full[1,2]) < band_v)
    print(f'  centre band: {int(in_band.sum())}/{len(uv_obs)} pts kept '
           f'(|u-cx|<{band_u:.0f}, |v-cy|<{band_v:.0f})')
    uv_b   = uv_obs[in_band]
    par_b  = par_obs[in_band]
    z_b    = z_obs[in_band]
    gt_b   = gt_uv[in_band]
    # σ-top-K with spatial coverage cap: walk smallest-σ first, but limit
    # how many pts may come from each (GU × GV) parent-image cell so the
    # solver doesn't see only the left/right wall-top edges.
    TOPK = 200
    GU, GV = 8, 4
    PER_CELL = 10  # 200 budget over 20 cells avg
    sigma_mean = 0.5 * (par_b[:, 2] + par_b[:, 3])
    order = np.argsort(sigma_mean)
    cell_count = np.zeros((GU, GV), dtype=np.int32)
    sel = []
    cu = pW / GU
    cv = pH / GV
    for i in order:
        gx = int(min(uv_b[i, 0] / cu, GU - 1))
        gy = int(min(uv_b[i, 1] / cv, GV - 1))
        if cell_count[gx, gy] >= PER_CELL:
            continue
        sel.append(i)
        cell_count[gx, gy] += 1
        if len(sel) >= TOPK:
            break
    keep = np.asarray(sel, dtype=np.int64)
    uv_b   = uv_b[keep]
    par_b  = par_b[keep]
    z_b    = z_b[keep]
    gt_b   = gt_b[keep]
    occupied = int((cell_count > 0).sum())
    print(f'  σ-stratified: kept {len(keep)} pts (σ ≤ {sigma_mean[keep].max():.2f})  '
           f'over {occupied}/{GU*GV} cells (≤{PER_CELL}/cell)')
    delta = solve_dofs(uv_b, par_b.astype(np.float64),
                        z_b.astype(np.float64), K_full,
                        dof_names, damping=1e-3,
                        huber_k=2.5, n_iter=10)
    cov = solve_dofs._last_cov
    print('\n  [closed-form pinhole BA, centre-band σ-top-K]')
    print('  DoF        δ̂          ±σ')
    for j, name in enumerate(dof_names):
        print(f'  {name:8s}  {delta[j]:+9.4f}  ±{np.sqrt(max(cov[j,j],0.0)):.4f}')

    # ─── KB-aware non-linear BA via scipy.least_squares.
    # Forward = KB projection of (R(δ)·X + t(δ)). Residual = whitened Δuv
    # (Mahalanobis), Σ = [[σx², ρσxσy], [ρσxσy, σy²]] from model output.
    # All N points (no centre-band, no σ filter) → KB Jacobian handles the
    # image-edge bias automatically via numerical diff.
    from scipy.optimize import least_squares
    from scripts.util.projection import project_kannala
    fx_, fy_, cx_, cy_ = K_full[0,0], K_full[1,1], K_full[0,2], K_full[1,2]
    # KB pool: full frame, σ-stratified TOP-K (no centre-band restriction).
    # KB Jacobian handles the image-edge bias so we can use the whole frame.
    KB_TOPK = 300
    KB_GU, KB_GV = 8, 4
    KB_PER_CELL = 16  # 300 / ~19 occupied cells
    sigma_full = 0.5 * (par_obs[:, 2] + par_obs[:, 3])
    order = np.argsort(sigma_full)
    cell_ct = np.zeros((KB_GU, KB_GV), dtype=np.int32)
    cuk = pW / KB_GU; cvk = pH / KB_GV
    sel_kb = []
    for i in order:
        gx = int(min(uv_obs[i, 0] / cuk, KB_GU - 1))
        gy = int(min(uv_obs[i, 1] / cvk, KB_GV - 1))
        if cell_ct[gx, gy] >= KB_PER_CELL: continue
        sel_kb.append(i); cell_ct[gx, gy] += 1
        if len(sel_kb) >= KB_TOPK: break
    keep_kb = np.asarray(sel_kb, dtype=np.int64)
    uv_kb  = uv_obs[keep_kb]
    par_kb = par_obs[keep_kb]
    z_kb   = z_obs[keep_kb]
    print(f'  KB pool: {len(keep_kb)} pts (full-frame σ-stratified TOP-{KB_TOPK})')
    Xa = (uv_kb[:,0] - cx_) * z_kb / fx_
    Ya = (uv_kb[:,1] - cy_) * z_kb / fy_
    Za = z_kb
    pts3 = np.stack([Xa, Ya, Za], axis=1).astype(np.float64)  # (N,3) cam-frame
    K_64 = K_full.astype(np.float64)
    dist_kb = np.asarray(cf.dist, dtype=np.float64)
    uv_target = uv_kb + par_kb[:, :2]
    sx_p = par_kb[:, 2]; sy_p = par_kb[:, 3]; rho_p = par_kb[:, 4]
    det = sx_p**2 * sy_p**2 * (1 - rho_p**2)
    Wuu = (sy_p**2)        / det
    Wvv = (sx_p**2)        / det
    Wuv = (-rho_p * sx_p * sy_p) / det
    # Cholesky of 2×2 SPD W: a=√Wuu, b=Wuv/a, c=√(Wvv - b²)
    a_w = np.sqrt(np.maximum(Wuu, 1e-12))
    b_w = Wuv / np.maximum(a_w, 1e-12)
    c_w = np.sqrt(np.maximum(Wvv - b_w**2, 1e-12))
    def residual_fn(delta):
        R = Rotation.from_rotvec(np.deg2rad(delta[:3])).as_matrix()
        p = pts3 @ R.T + delta[3:6]
        uv_pred = project_kannala(p, K_64, dist_kb)
        r = uv_pred - uv_target
        # whiten: [a 0; b c] @ [ru; rv]
        ru = a_w * r[:, 0]
        rv = b_w * r[:, 0] + c_w * r[:, 1]
        return np.concatenate([ru, rv])
    # 2-DoF (pitch, yaw) only — match closed-form for direct comparison.
    def residual_2dof(d_xy):
        d6 = np.zeros(6)
        d6[0] = d_xy[0]; d6[1] = d_xy[1]
        return residual_fn(d6)
    # Initialise from closed-form δ̂ (warm start) so LM doesn't fall into
    # the pitch/yaw swap basin from zero.
    x0 = np.array([delta[0], delta[1]], dtype=np.float64)
    res = least_squares(residual_2dof, x0,
                        jac='2-point', loss='huber', f_scale=2.5,
                        method='trf', max_nfev=200, verbose=0)
    delta_kb = np.zeros(6); delta_kb[:2] = res.x

    # ─── KB closed-form (analytic Jacobian, GN with re-linearisation).
    from scripts.ba.ba_kb_jac import solve_dofs_kb
    kb_dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    # Larger damping on translations + roll: tz is degenerate with depth
    # scale, t_xy is degenerate with rotation, ω_z is small for an
    # automotive rig.
    damping_diag = np.array([1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1.0])
    x0_6 = np.zeros(6)
    x0_6[0] = delta[0]; x0_6[1] = delta[1]
    delta_kb_cf6 = solve_dofs_kb(
        uv_kb, par_kb.astype(np.float64), z_kb.astype(np.float64),
        K_full, dist_kb, kb_dof,
        damping=damping_diag, huber_k=2.5, n_iter=10,
        x0=x0_6,
    )
    cov_kbcf = solve_dofs_kb._last_cov
    print(f'\n  [KB closed-form (analytic Jac), full-frame σ-top {len(uv_kb)} pts, 6-DoF]')
    print('  DoF        δ̂          ±σ')
    for j, n in enumerate(kb_dof):
        print(f'  {n:8s}  {delta_kb_cf6[j]:+9.4f}  ±{np.sqrt(max(cov_kbcf[j,j],0.0)):.4f}')
    # 2-DoF view of 6-DoF result for the rest of the script.
    delta_kb_cf = np.array([delta_kb_cf6[0], delta_kb_cf6[1]])
    print(f'\n  [KB-aware scipy BA, full-frame σ-top {len(pts3)} pts, 2-DoF, warm-start from cf]')
    print(f'  status={res.status}  nfev={res.nfev}  cost={res.cost:.3f}')
    names6 = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    print('  DoF        δ̂_KB')
    for j, n in enumerate(names6):
        print(f'  {n:8s}  {delta_kb[j]:+9.4f}')
    # Reconstruct uv_corrected via the Jacobian (NOT euler — bypasses the
    # 'zyx' axis-naming ambiguity).
    fx, fy, cx, cy = K_full[0,0], K_full[1,1], K_full[0,2], K_full[1,2]
    # Evaluate residuals on the SAME band that solved δ̂ (not the full pool).
    uv_obs = uv_b
    par_obs = par_b
    z_obs = z_b
    gt_uv = gt_b
    X = (uv_obs[:,0] - cx) * z_obs / fx
    Y = (uv_obs[:,1] - cy) * z_obs / fy
    Z = z_obs
    Ju = np.column_stack([DOF_JAC[n](X,Y,Z,uv_obs,K_full)[0] for n in dof_names])
    Jv = np.column_stack([DOF_JAC[n](X,Y,Z,uv_obs,K_full)[1] for n in dof_names])
    uv_corr = uv_obs + np.column_stack([Ju @ delta, Jv @ delta])

    # Per-pt residuals: BA前 = 観測 (dist) ↔ GT, BA後 = 補正 ↔ GT
    res_pre = uv_obs - gt_uv          # 観測そのまま (model 補正前 = 摂動された投影)
    res_pred = (uv_obs + par_obs[:, :2]) - gt_uv  # model pred 補正後
    res_post = uv_corr - gt_uv        # BA 補正後
    print(f'\n  per-pt err [px]:')
    print(f'    raw (obs-GT)     mean {np.linalg.norm(res_pre,axis=1).mean():6.2f}  median {np.median(np.linalg.norm(res_pre,axis=1)):6.2f}')
    print(f'    after model Δuv  mean {np.linalg.norm(res_pred,axis=1).mean():6.2f}  median {np.median(np.linalg.norm(res_pred,axis=1)):6.2f}')
    print(f'    after BA δ̂      mean {np.linalg.norm(res_post,axis=1).mean():6.2f}  median {np.median(np.linalg.norm(res_post,axis=1)):6.2f}')

    # ─── Direction-coloured cell overlay (parent image), 2 panels: pre / post.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    GRID_U, GRID_V = 16, 9
    cu = pW / GRID_U
    cv = pH / GRID_V
    def cell_dir_image(res):
        img = np.zeros((GRID_V, GRID_U, 3), dtype=np.float32)
        for gu in range(GRID_U):
            for gv in range(GRID_V):
                u_lo, u_hi = gu*cu, (gu+1)*cu
                v_lo, v_hi = gv*cv, (gv+1)*cv
                mask = ((uv_obs[:,0] >= u_lo) & (uv_obs[:,0] < u_hi) &
                        (uv_obs[:,1] >= v_lo) & (uv_obs[:,1] < v_hi))
                if not mask.any(): continue
                mdu, mdv = res[mask,0].mean(), res[mask,1].mean()
                mag = np.hypot(mdu, mdv)
                hue = (np.arctan2(mdv, mdu) / (2*np.pi)) % 1.0
                sat = float(min(mag/10.0, 1.0))     # 10 px ≥ → fully saturated
                val = 0.85
                img[gv, gu] = hsv_to_rgb([hue, sat, val])
        return img
    pre_img = cell_dir_image(res_pre)
    post_img = cell_dir_image(res_post)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=100)
    titles = [
        f'BA前 (raw obs vs GT) — mean |r|={np.linalg.norm(res_pre,axis=1).mean():.1f}px',
        f'BA後 (corrected vs GT) — mean |r|={np.linalg.norm(res_post,axis=1).mean():.1f}px',
    ]
    for ax, im, title in zip(axes, [pre_img, post_img], titles):
        ax.imshow(parent)
        ax.imshow(im, extent=[0, pW, pH, 0], alpha=0.55,
                   interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    OUT_DIR = OUT.parent / 'frame_residual_dirmap.png'
    fig.tight_layout()
    fig.savefig(OUT_DIR)
    plt.close(fig)
    print(f'wrote → {OUT_DIR}')

    # ─── 3-stage overlay (centre-band BA pool, grid-rep pts).
    OUT3 = OUT.parent / 'ba_reproj_overlay_v2.png'
    fig3, axes3 = plt.subplots(3, 1, figsize=(pW/300, 3*pH/300), dpi=200)
    panels = [
        ('GT uv',                gt_uv,    'yellow'),
        ('Perturbed uv (input)', uv_obs,   'red'),
        (f'BA-corrected uv  δ̂[ω_x={delta[0]:+.3f}° ω_y={delta[1]:+.3f}°]',
                                 uv_corr,  'lime'),
    ]
    for ax, (title, pts_uv, colr) in zip(axes3, panels):
        ax.imshow(parent)
        ax.scatter(pts_uv[:, 0], pts_uv[:, 1], s=2.0, c=colr,
                    marker='.', linewidths=0, alpha=0.9)
        ax.set_xlim(0, pW); ax.set_ylim(pH, 0)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig3.tight_layout()
    fig3.savefig(OUT3, dpi=200)
    plt.close(fig3)
    print(f'wrote → {OUT3}')

    # ─── Pose summary table (values, not figures) so user can sanity-check
    #     before trusting any overlay.
    delta_full_pose = np.zeros(3, dtype=np.float64)
    for j, n in enumerate(dof_names):
        if   n == 'omega_x': delta_full_pose[0] = delta[j]   # cam-x = pitch
        elif n == 'omega_y': delta_full_pose[1] = delta[j]   # cam-y = yaw
        elif n == 'omega_z': delta_full_pose[2] = delta[j]   # cam-z = roll
    # rpy_deg is in build_sample's [roll, yaw, pitch] order (zyx euler).
    # Convert to (omega_x, omega_y, omega_z) = (pitch, yaw, roll) so it can
    # be compared element-wise with delta_full_pose.
    pert_xyz = np.array([rpy_deg[2], rpy_deg[1], rpy_deg[0]], dtype=np.float64)
    print('\n  Pose summary (cam-frame axes, deg):')
    print(f'    {"":<10} {"ω_x(pitch)":>12} {"ω_y(yaw)":>12} {"ω_z(roll)":>12}')
    print(f'    {"GT":<10} {0.0:>12.4f} {0.0:>12.4f} {0.0:>12.4f}')
    print(f'    {"perturbed":<10} {pert_xyz[0]:>+12.4f} {pert_xyz[1]:>+12.4f} {pert_xyz[2]:>+12.4f}')
    print(f'    {"BA δ̂(cf)":<10} {delta_full_pose[0]:>+12.4f} {delta_full_pose[1]:>+12.4f} {delta_full_pose[2]:>+12.4f}')
    resid = pert_xyz - delta_full_pose
    print(f'    {"residual":<10} {resid[0]:>+12.4f} {resid[1]:>+12.4f} {resid[2]:>+12.4f}    (=perturbed-δ̂_cf)')
    print(f'    {"BA δ̂(kb)":<10} {delta_kb[0]:>+12.4f} {delta_kb[1]:>+12.4f} {delta_kb[2]:>+12.4f}     +t=({delta_kb[3]:+.3f},{delta_kb[4]:+.3f},{delta_kb[5]:+.3f}) m')
    resid_kb = pert_xyz - delta_kb[:3]
    print(f'    {"residual":<10} {resid_kb[0]:>+12.4f} {resid_kb[1]:>+12.4f} {resid_kb[2]:>+12.4f}    (=perturbed-δ̂_kb)')
    print(f'    {"BA δ̂(kb-cf)":<10} {delta_kb_cf[0]:>+12.4f} {delta_kb_cf[1]:>+12.4f}      +0.0000')
    resid_kbcf = pert_xyz[:2] - delta_kb_cf
    print(f'    {"residual":<10} {resid_kbcf[0]:>+12.4f} {resid_kbcf[1]:>+12.4f}      +0.0000    (=perturbed-δ̂_kbcf)')

    # ─── 3-stage DENSE overlay: ALL LiDAR pts re-projected with KB.
    #   yellow = GT pose
    #   red    = perturbed pose (R_pert applied to cam pose)
    #   green  = perturbed pose corrected by BA δ̂ (should approach yellow)
    # δ̂ is in degrees, axes (omega_x, omega_y[, omega_z]) per solve_dofs
    # — same convention as DOF_JAC. Convert via from_rotvec then apply in
    # the same hand we applied R_pert.
    from scripts.util.projection import project_kannala
    pts_cam = cf.pts_cam.astype(np.float64)
    K_full_64 = K_full.astype(np.float64)
    dist_kb   = np.asarray(cf.dist, dtype=np.float64)
    R_pert = Rotation.from_euler('zyx', rpy_deg, degrees=True).as_matrix()
    # build_sample / apply_perturbation_explicit: pts_cam_off = R_off.T @ pts.T
    # with R_off = R_gt @ R_pert (R_gt = I for kamikado tile_inst). So the
    # perturbed cam-frame point is R_pert.T @ pts_cam_orig (column vec).
    pts_cam_pert = pts_cam @ R_pert      # equiv to (R_pert.T @ pts.T).T
    # Use the pinhole closed-form 2-DoF δ̂ (= the most stable solver
    # across 5 scenes) for the green overlay. delta = [ω_x (pitch),
    # ω_y (yaw)] from solve_dofs.
    delta_2 = np.array([delta[0], delta[1], 0.0])  # ω_x, ω_y, ω_z=0
    R_BA = Rotation.from_rotvec(np.deg2rad(delta_2)).as_matrix()
    pts_cam_ba = pts_cam_pert @ R_BA.T
    uv_gt_all   = project_kannala(pts_cam,      K_full_64, dist_kb)
    uv_pert_all = project_kannala(pts_cam_pert, K_full_64, dist_kb)
    uv_ba_all   = project_kannala(pts_cam_ba,   K_full_64, dist_kb)
    def _in(uv, z):
        return ((z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < pW)
                & (uv[:,1] >= 0) & (uv[:,1] < pH))
    z_gt   = pts_cam[:, 2]
    z_pert = pts_cam_pert[:, 2]
    z_ba   = pts_cam_ba[:, 2]
    OUT_DENSE = OUT.parent / 'ba_reproj_overlay_v3.png'
    figD, axD = plt.subplots(3, 1, figsize=(pW/300, 3*pH/300), dpi=200)
    densep = [
        (f'GT  ({int(_in(uv_gt_all,z_gt).sum())} pts)',           uv_gt_all,   z_gt,   'yellow'),
        (f'Perturbed  ({int(_in(uv_pert_all,z_pert).sum())} pts)', uv_pert_all, z_pert, 'red'),
        (f'BA-corrected  pinhole-cf 2-DoF  δ̂[ω_x={delta[0]:+.3f}° ω_y={delta[1]:+.3f}°]',
                                                                   uv_ba_all,   z_ba,   'lime'),
    ]
    for ax, (title, pts_uv, zz, colr) in zip(axD, densep):
        m = _in(pts_uv, zz)
        ax.imshow(parent)
        ax.scatter(pts_uv[m, 0], pts_uv[m, 1], s=2.0, c=colr,
                    marker='.', linewidths=0, alpha=0.9)
        ax.set_xlim(0, pW); ax.set_ylim(pH, 0)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    figD.tight_layout()
    figD.savefig(OUT_DENSE, dpi=200)
    plt.close(figD)
    print(f'wrote → {OUT_DENSE}')

    # ─── Truck-area 1024×1024 crop (3 rows: GT / perturbed / BA-corrected),
    # for the hero figure assembly. Crop is centred slightly below image
    # centre (truck-on-road region).
    CROP = 1024
    cu_truck = pW // 2
    cv_truck = pH // 2 + 200            # nudge down toward the road / truck
    u0 = max(0, cu_truck - CROP // 2)
    v0 = max(0, cv_truck - CROP // 2)
    u1 = min(pW, u0 + CROP)
    v1 = min(pH, v0 + CROP)
    parent_crop = parent[v0:v1, u0:u1]
    OUT_HCROP = OUT.parent / 'ba_reproj_overlay_truckcrop.png'
    figH, axH = plt.subplots(3, 1, figsize=(CROP/200, 3*CROP/200), dpi=200)
    for ax, (title, pts_uv, zz, colr) in zip(axH, densep):
        ax.imshow(parent_crop, extent=[u0, u1, v1, v0])
        m = _in(pts_uv, zz)
        ax.scatter(pts_uv[m, 0], pts_uv[m, 1], s=4.0, c=colr,
                    marker='.', linewidths=0, alpha=0.9)
        ax.set_xlim(u0, u1); ax.set_ylim(v1, v0)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    figH.tight_layout()
    figH.savefig(OUT_HCROP, dpi=200)
    plt.close(figH)
    print(f'wrote → {OUT_HCROP}')

    # ─── TOP-100 selection visualisation: parent + cyan ○ on the σ-top-100
    #     pts that BA actually solved with. uv_b is in parent coords.
    OUT_TOP = OUT.parent / 'ba_top100_selection.png'
    figT, axT = plt.subplots(1, 1, figsize=(pW/200, pH/200), dpi=200)
    axT.imshow(parent)
    axT.scatter(uv_b[:, 0], uv_b[:, 1], s=18, facecolors='none',
                 edgecolors='cyan', linewidths=1.0, alpha=0.95,
                 label=f'σ-top {len(uv_b)} (used by BA)')
    axT.set_xlim(0, pW); axT.set_ylim(pH, 0)
    axT.set_title(f'BA pool: σ-top {len(uv_b)} pts (max σ={float(par_b[:,2:4].mean(axis=0).mean()):.2f}px)',
                   fontsize=10)
    axT.legend(loc='upper right', fontsize=9)
    axT.axis('off')
    figT.tight_layout()
    figT.savefig(OUT_TOP, dpi=200)
    plt.close(figT)
    print(f'wrote → {OUT_TOP}')


if __name__ == '__main__':
    import os as _os
    if _os.environ.get('BA_MULTI_SCENE') == '1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_calib_model(EXP).eval().to(device)
        for s in SCENES:
            try:
                main(f'{SCENES_ROOT}/{s}', model=model, device=device, scene_tag=s)
            except Exception as e:
                print(f'[{s}] FAILED: {type(e).__name__}: {e}')
    else:
        main()
