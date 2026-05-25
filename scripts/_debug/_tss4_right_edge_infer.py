"""Single-tile inference + 3-panel overlay for TSS4 right-edge tile.

Why: TSS4 frame_0814 (seq16) right-edge tile (u0=3328, v0=677) lines up
exactly with the pole the user pointed at. KB4 distortion is strongest at
the periphery, so the "GT" reprojection from recalibration.json already
shows visible drift on the pole. We want to see, on this exact tile,
how the current best model (km_wv_wm img128 InfoHead) treats it: does
its δ̂ correction move points towards the pole or away?

We feed the tile's 4 quadrants (cs=256, S=128) through the σ-head, run
shared-GN BA with δ_target=0 (= "do not perturb, just see what the model
says about the current calib"), then render the standard 3-panel
overlay (yellow GT / red perturbed / lime corrected) over the parent
tile.

Output: scripts/_debug/_outputs/tss4_right_edge_<gid>_<tname>.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
import scripts.eval.eval_shared_256x800 as _ev
from scripts.eval.eval_shared_256x800 import (
    _build_model, _build_subwin, render_3panel_overlay,
    DEVICE, DOFS, PRIOR_DIAG, BA_N_ITER, DAMPING,
)
from scripts.ba.ba_torch import solve_kb_xyz_shared, make_info_from_sigma_rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--split', default='train')
    ap.add_argument('--tile', default='00081400_t19.pt',
                    help='target tile filename (default: seq16 frame14 right-edge)')
    ap.add_argument('--cs', type=int, default=256, choices=(128, 256, 512))
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--ckpt-run', default=None,
                    help='experiment dir name under experiments/ (selects '
                         'best_model.pt + config.py); default = use eval_shared default')
    ap.add_argument('--apply-fit-json', type=Path, default=None,
                    help='13-DoF (or 11-DoF) GN fit json. If given, the inst is '
                         "rewritten with K_fit/D_fit and pts/T_gt are rotated by "
                         'R(omega) before sub-crops are built. Tangential p is '
                         "ignored (KB4 inst can't express it).")
    args = ap.parse_args()

    if args.ckpt_run:
        run_dir = REPO / 'experiments' / args.ckpt_run
        _ev.CKPT = run_dir / 'best_model.pt'
        _ev.EXP_CFG_PATH = run_dir / 'config.py'
    CKPT = _ev.CKPT
    cfg = _ev._load_cfg()
    print(f'[viz] ckpt={CKPT}  cfg img_size={cfg["img_size"]}  cs={args.cs}  '
          f'max_rot_deg={cfg.get("max_rot_deg","?")}  '
          f'use_pose_emb={cfg.get("use_pose_emb", False)}  '
          f'deform_mode={cfg.get("deform_mode","sl")}')

    ds = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split=args.split,
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    if args.tile not in ds.fnames:
        raise SystemExit(f'tile {args.tile!r} not in {args.split} split '
                         f'({len(ds.fnames)} entries)')
    target_idx = ds.fnames.index(args.tile)
    print(f'[viz] tile={args.tile}  idx={target_idx}')
    inst = ds._load_inst(target_idx)

    if args.apply_fit_json is not None:
        import json
        from scipy.spatial.transform import Rotation as _R
        j = json.loads(args.apply_fit_json.read_text())
        K_old = inst['K_full'].numpy().astype(np.float64)
        D_old = inst['distortion'].numpy().astype(np.float64)
        K_new = K_old.copy()
        K_new[0, 0] = float(j['fx_fit']); K_new[1, 1] = float(j['fy_fit'])
        K_new[0, 2] = float(j['cx_fit']); K_new[1, 2] = float(j['cy_fit'])
        D_new = np.asarray(j['dist_fit'], dtype=np.float64)
        if D_new.size != 4:
            raise SystemExit(f'apply-fit-json: dist_fit must be 4 KB coefs '
                             f'(got {D_new.size}); cannot fold high-order KB '
                             f'into KB4 inst.')
        omega = np.asarray(j['omega_deg'], dtype=np.float64)
        R_om = _R.from_rotvec(np.deg2rad(omega)).as_matrix()
        # inst['pts'] is in REAR-AXLE (world) frame; T_gt = world→cam.
        # Fit applies pts_cam_fit = R_om @ pts_cam_init = (R_om @ T_gt) @ pts_w.
        # So left-multiply T_gt by R_om and DO NOT touch pts.
        pts_w = inst['pts'].numpy().astype(np.float64)
        if 'T_gt' not in inst:
            raise SystemExit('apply-fit-json: inst missing T_gt')
        T_gt = inst['T_gt'].numpy().astype(np.float64)
        T_new = T_gt.copy()
        T_new[:3, :3] = R_om @ T_gt[:3, :3]
        T_new[:3, 3]  = R_om @ T_gt[:3, 3]
        inst['T_gt'] = torch.from_numpy(T_new.astype(np.float32))
        inst['K_full']     = torch.from_numpy(K_new.astype(np.float32))
        inst['distortion'] = torch.from_numpy(D_new.astype(np.float32))
        # recompute pts in (new) cam frame for cached uv_full / z_cam
        homo = np.column_stack([pts_w, np.ones(len(pts_w))])
        pts_cam_new = (T_new @ homo.T)[:3].T  # (N, 3) m, in new cam frame
        if 'uv_full' in inst:
            from scripts.util.projection import project_kannala as _proj_kb
            uv_new = _proj_kb(pts_cam_new, K_new, D_new)
            inst['uv_full'] = torch.from_numpy(uv_new.astype(np.float32))
        if 'z_cam' in inst:
            inst['z_cam'] = torch.from_numpy(pts_cam_new[:, 2].astype(np.float32))
        p_str = ('  p=' + str(np.asarray(j.get('tangential_p', [])).round(4).tolist())
                 if 'tangential_p' in j else '')
        print(f'[viz] APPLIED fit from {args.apply_fit_json.name}')
        print(f'[viz]   ω={omega.round(4).tolist()}°  '
              f'fx*={j["fx_fit"]/j["fx_init"]:.4f} fy*={j["fy_fit"]/j["fy_init"]:.4f}  '
              f'k_fit={D_new.round(4).tolist()}{p_str}')
        if 'tangential_p' in j and float(np.abs(np.asarray(j['tangential_p'])).max()) > 1e-6:
            print('[viz]   WARN: tangential p≠0 ignored (KB4 cache cannot encode it)')
    print(f'[viz] tile_u0={int(inst.get("tile_u0",0))} '
          f'tile_v0={int(inst.get("tile_v0",0))}  IW×IH={int(inst["IW"])}×{int(inst["IH"])}  '
          f'npts={int(inst["pts"].shape[0])}  is_fisheye={inst.get("is_fisheye")}')

    # δ_target = 0: just see "current calib drift" + what δ̂ the model emits.
    ypr_target = np.zeros(3, dtype=np.float64)
    t_target = np.zeros(3, dtype=np.float64)
    dist_one = inst['distortion'].clone().detach().to(torch.float32).reshape(1, 4)

    if args.cs == 256:
        u0v0_list = [(0, 0), (256, 0), (0, 256), (256, 256)]
    elif args.cs == 128:
        # Right-most column of the 512×512 tile: u0=384 (=512-128), four
        # 128×128 sub-crops stacked along Y. These hit parent u∈[3712,3840]
        # (= the rightmost 128 px of the parent image) — the periphery the
        # user wants to probe at finest scale.
        u0v0_list = [(384, 0), (384, 128), (384, 256), (384, 384)]
    else:
        u0v0_list = [(0, 0)]
    wins = []
    for (u0, v0) in u0v0_list:
        w = _build_subwin(ds, inst, t_target, ypr_target, u0=u0, v0=v0, cs=args.cs)
        if w is not None:
            wins.append(w)
            print(f'[viz]   sub-crop (u0={u0:3d},v0={v0:3d}) ok')
        else:
            print(f'[viz]   sub-crop (u0={u0:3d},v0={v0:3d}) skipped (too few pts)')
    assert len(wins) >= 1, 'no sub-crops survived'

    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b, _delta1) = moved
    valid = ~pad_mask
    pad_full = ~valid
    B, N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        P0_orig[pad_full] = torch.tensor([0.0, 0.0, 1.0],
                                          dtype=P0_orig.dtype, device=P0_orig.device)
    dist = dist_one.to(DEVICE).expand(B, 4).contiguous()

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    use_intensity = getattr(model, 'use_intensity', True)
    if use_intensity:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
    W_direct = None
    if isinstance(out, tuple):
        last = out[-1]
        if torch.is_tensor(last) and last.dim() == 4 and last.shape[-2:] == (2, 2):
            W_direct = last
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    if W_direct is not None:
        W_sigma_local = W_direct.detach()
    else:
        sx = per_pt[..., 2].exp()
        sy = per_pt[..., 3].exp()
        rho = per_pt[..., 4]
        W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    scale_l2o = (cs_b / float(cfg['img_size'])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)

    prior = PRIOR_DIAG.to(DEVICE)
    with torch.no_grad():
        delta_shared, _ = solve_kb_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, dist, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING, prior_diag=prior,
        )
    d = delta_shared.detach().cpu().numpy()
    print(f'[viz] B={B}  δ̂ ω=[{d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}]°  '
          f't=[{d[3]:+.4f},{d[4]:+.4f},{d[5]:+.4f}]m')

    ckpt_tag = (args.ckpt_run or 'HEAD').split('/')[-1]
    suffix = f'_cs{args.cs}' if args.cs != 256 else ''
    if args.apply_fit_json is not None:
        suffix = suffix + f'_fit{args.apply_fit_json.stem.split("_")[-2]}'
    out_path = args.out or (REPO / 'scripts' / '_debug' / '_outputs'
                              / f'tss4_right_edge_{args.tile.replace(".pt","")}_{ckpt_tag}{suffix}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suptitle = (f'TSS4 {args.tile}  (u0={int(inst.get("tile_u0",0))},'
                f'v0={int(inst.get("tile_v0",0))})  '
                f'cs={args.cs} B={B}  δ_target=0  '
                f'δ̂ω=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})° '
                f't=({d[3]:+.3f},{d[4]:+.3f},{d[5]:+.3f})m')
    render_3panel_overlay(inst, ypr_target, t_target, delta_shared,
                          out_path=out_path, suptitle=suptitle,
                          panel_label='Model δ̂ applied')
    print(f'[viz] wrote {out_path}')

    # ---- 4th panel: per-point duv_pred arrows on the parent tile.
    # The model emits duv in S=img_size px (cs sub-crop resampled to S×S).
    # Convert each sub-crop's local uv back to parent-tile px:
    #   uv_parent = uv_local * (cs / S) + (u0_sub, v0_sub)
    # Same for duv_pred. true_uvd carries the input uv in S-units.
    import io as _io
    import numpy as _np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt
    from PIL import Image as _Image

    if args.cs == 256:
        u0v0_used = u0v0_list[:B] if B == len(u0v0_list) else u0v0_list
    else:
        u0v0_used = [(0, 0)] * B

    parent_arr = _np.array(_Image.open(_io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    pH, pW = parent_arr.shape[:2]
    S = float(cfg['img_size'])
    scale = float(args.cs) / S  # local→parent px per sub-crop

    true_local = true_uvd[..., :2].detach().cpu().numpy()    # (B, N, 2)
    duv_local  = duv_pred_local.detach().cpu().numpy()       # (B, N, 2)
    valid_np   = valid.detach().cpu().numpy()                # (B, N)

    arrow_path = out_path.with_name(out_path.stem + '_arrows.png')
    fig, ax = _plt.subplots(1, 1, figsize=(pW/180, pH/180), dpi=140)
    ax.imshow(parent_arr)
    arr_norms = []
    for b in range(B):
        u0s, v0s = u0v0_used[b]
        m = valid_np[b]
        u_in = true_local[b, m, 0] * scale + u0s
        v_in = true_local[b, m, 1] * scale + v0s
        du   = duv_local[b, m, 0]  * scale
        dv   = duv_local[b, m, 1]  * scale
        ax.scatter(u_in, v_in, s=2, c='cyan', marker='.',
                   linewidths=0, alpha=0.7, label=('input uv' if b == 0 else None))
        ax.quiver(u_in, v_in, du, dv,
                  angles='xy', scale_units='xy', scale=1.0,
                  color='magenta', width=0.0015, headwidth=3, headlength=4,
                  alpha=0.85, label=('duv_pred (uv → uv+duv)' if b == 0 else None))
        # sub-crop box
        ax.add_patch(_plt.Rectangle((u0s, v0s), args.cs, args.cs,
                                     fill=False, edgecolor='yellow', lw=1.0,
                                     alpha=0.6))
        arr_norms.append(_np.linalg.norm(_np.stack([du, dv], axis=-1), axis=-1))
    norms_all = _np.concatenate(arr_norms) if arr_norms else _np.array([0.0])
    ax.set_xlim(0, pW); ax.set_ylim(pH, 0)
    ax.set_title(f'TSS4 {args.tile}  per-point duv_pred  '
                 f'mean={float(norms_all.mean()):.2f}px  '
                 f'med={float(_np.median(norms_all)):.2f}px  '
                 f'max={float(norms_all.max()):.2f}px  (parent-px)',
                 fontsize=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(arrow_path, dpi=140, bbox_inches='tight')
    _plt.close(fig)
    print(f'[viz] wrote {arrow_path}')
    print(f'[viz] duv_pred (parent-px): mean={float(norms_all.mean()):.3f}  '
          f'median={float(_np.median(norms_all)):.3f}  '
          f'p90={float(_np.quantile(norms_all,0.90)):.3f}  '
          f'max={float(norms_all.max()):.3f}')


if __name__ == '__main__':
    main()
