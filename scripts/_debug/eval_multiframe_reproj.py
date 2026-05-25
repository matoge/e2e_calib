"""Multi-frame eval: 1 frame = N tiles → 1 pose. Compares
do-nothing / W=I / W=σ / W=learned / oracle on N_FRAMES held-out frames.

Each frame: same (t, ypr) perturbation applied to every tile of that frame
(via apply_perturbation_explicit), per-tile inference, then concatenate the
per-tile (P0_orig, duv_pred_orig, W_orig, valid) along the points dim and
solve once for the rig-level 6-DoF δ. Reports rot°RMS, t·m·RMS, reproj
mean/med/p95 across frames — the eval that 1-tile aperture made impossible.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import (
    solve_kb_xyz, make_info_from_sigma_rho, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEFAULT_HEAD = (REPO / 'scripts' / '_debug' / '_outputs'
                / 'overfit_6dof_ba_stream_orig_idx17_6dof_kb' / 'info_head.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build_model(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'], self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def _frame_groups(fnames):
    """Group tile fname indices by parent frame ('00000000_t17.pt' → '00000000')."""
    groups = defaultdict(list)
    for i, fn in enumerate(fnames):
        groups[fn.rsplit('_t', 1)[0]].append(i)
    return groups


def _project_after_delta(P0, delta, K, dist, dofs):
    d = _split_delta(delta, dofs)
    omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
    t_v = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
    P = _apply_extrinsic(P0, omega, t_v)
    Kn = _K_with_delta(K, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
    return project_kb(P, Kn, dist)


def _per_point_mahalanobis(duv_pred, duv_oracle, W, valid):
    """m_k = r_kᵀ W_k r_k. r in orig-px, W in orig-px⁻².
    Under correct W, m ~ χ²(2): mean=2, median≈1.386."""
    r = (duv_pred - duv_oracle).unsqueeze(-1)               # (B,N,2,1)
    Wr = torch.matmul(W, r)                                 # (B,N,2,1)
    m = (r * Wr).sum(dim=(-2, -1))                          # (B,N)
    return m.masked_fill(~valid, float('nan'))


def _build_one_frame_batch(ds, tile_idxs, t_delta, ypr_deg):
    """Apply same (t, ypr) to every tile of the frame; return collated batch
    (B = N_tiles_with_a_valid_window). Drops tiles that return None."""
    wins = []
    for ti in tile_idxs:
        win = ds.apply_perturbation_explicit(int(ti), t_delta, ypr_deg)
        if win is not None:
            wins.append(win)
    if not wins:
        return None
    return collate_full(wins)


def _forward_and_concat(model, cfg, batch, dist_one):
    """Run model on a per-frame batch (B tiles), then concat per-tile points
    into a single (1, sum_N, ...) for the solver. Returns dict of orig-frame
    tensors plus the rig-level distortion."""
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs) = moved
    B, N = pts_cam_orig.shape[:2]
    valid = ~pad_mask
    pad_full = ~valid

    captured = {}
    def _hk(_m, _i, _o):
        captured['q'] = _i[0].detach()
    h = model.info_head.mlp[0].register_forward_hook(_hk)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    per_pt = out[0]                                          # (B,N,5)
    q = captured['q']                                        # (B,N,d)

    duv_pred_local = per_pt[..., :2].detach()
    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()
    eye2 = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)
    with torch.no_grad():
        W_learn_local = model.info_head(q).detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone(); duv_pred_local[pad_full] = 0.0

    # local-px → orig-px: duv·(cs/S), W·(S/cs)²
    S_local = float(cfg['img_size'])
    scale_l2o = (cs / S_local).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_eye_orig = eye2 * inv_l2o.pow(2)
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)
    W_learn_orig = W_learn_local * inv_l2o.pow(2)

    # Sanitize padded slots
    duv_oracle_orig = duv_orig.clone()
    P0_orig = pts_cam_orig.clone()
    if pad_full.any():
        duv_oracle_orig[pad_full] = 0.0
        P0_orig[pad_full] = torch.tensor([0., 0., 1.],
                                          dtype=P0_orig.dtype, device=DEVICE)

    # Concatenate (B tiles → 1 rig batch) along the points dim
    def _cat_pts(x):  # (B,N,*)
        return x.reshape(1, B * N, *x.shape[2:])

    P0_cat = _cat_pts(P0_orig)
    duv_pred_cat = _cat_pts(duv_pred_orig)
    duv_oracle_cat = _cat_pts(duv_oracle_orig)
    W_eye_cat = _cat_pts(W_eye_orig)
    W_sigma_cat = _cat_pts(W_sigma_orig)
    W_learn_cat = _cat_pts(W_learn_orig)
    valid_cat = valid.reshape(1, B * N)

    # K_orig and dist are rig-level (parent camera); all B tiles share them.
    # Verify K_orig is identical across tiles before collapsing.
    K_first = K_orig[0:1]                                   # (1,3,3)
    dist_first = dist_one.to(DEVICE).reshape(1, 4)          # (1,4)
    return dict(
        P0=P0_cat, duv_pred=duv_pred_cat, duv_oracle=duv_oracle_cat,
        W_eye=W_eye_cat, W_sigma=W_sigma_cat, W_learn=W_learn_cat,
        valid=valid_cat, K=K_first, dist=dist_first, n_tiles=B, n_pts=int(valid_cat.sum().item()),
    )


def _solve_and_eval(d, n_iter=8, damping=1e-3):
    """Run solver under each W; return per-frame metrics dict."""
    P0 = d['P0']; K = d['K']; dist = d['dist']; valid = d['valid']
    duv_pred = d['duv_pred']; duv_oracle = d['duv_oracle']

    uv0 = project_kb(P0, K, dist)
    uv_target = uv0 + duv_oracle
    res = {}

    # Oracle (Δuv=GT, W=I, all valid) — gold standard for this rig
    delta_oracle, _ = solve_kb_xyz(P0, duv_oracle, d['W_eye'], K, dist, DOFS,
                                    valid=valid, n_iter=n_iter, damping=damping)

    variants = [
        ('do-nothing', None, None),
        ('W=I',        duv_pred, d['W_eye']),
        ('W=σ',        duv_pred, d['W_sigma']),
        ('W=learned',  duv_pred, d['W_learn']),
        ('oracle',     duv_oracle, d['W_eye']),
    ]
    for name, duv, W in variants:
        if duv is None:
            delta = torch.zeros(1, len(DOFS), device=DEVICE)
        else:
            delta, _ = solve_kb_xyz(P0, duv, W, K, dist, DOFS,
                                     valid=valid, n_iter=n_iter, damping=damping)
        uv_proj = _project_after_delta(P0, delta, K, dist, DOFS)
        err = torch.linalg.vector_norm(uv_proj - uv_target, dim=-1)
        err = err.masked_fill(~valid, float('nan'))
        flat = err[~torch.isnan(err)].cpu().numpy()
        rot_rms = (delta[:, :3] - delta_oracle[:, :3]).pow(2).mean(dim=-1).sqrt().item()
        t_rms = (delta[:, 3:] - delta_oracle[:, 3:]).pow(2).mean(dim=-1).sqrt().item()
        res[name] = dict(
            reproj_mean=float(flat.mean()) if flat.size else float('nan'),
            reproj_med=float(np.median(flat)) if flat.size else float('nan'),
            reproj_p95=float(np.percentile(flat, 95)) if flat.size else float('nan'),
            rot_rms=rot_rms, t_rms=t_rms,
            delta=delta[0].cpu().numpy(),
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head-pt', type=Path, default=DEFAULT_HEAD)
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m', type=float, default=0.05)
    ap.add_argument('--n-frames', type=int, default=98)
    ap.add_argument('--seed', type=int, default=7 + 1000)
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs'
                            / 'multiframe_reproj.txt')
    args = ap.parse_args()

    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val', img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0, oversample=1,
        grid_n=cfg.get('grid_n', 16), center_band=0.0, preload=False,
    )
    groups = _frame_groups(ds.fnames)
    frame_keys = list(groups.keys())[:args.n_frames]
    print(f"[mf] val has {len(groups)} frames, evaluating {len(frame_keys)}")

    # All tiles share parent camera → take dist from any tile (idx 0)
    inst0 = ds._load_inst(0)
    assert inst0.get('is_fisheye', False), 'expected KB fisheye'
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    head_sd = torch.load(args.head_pt, map_location=DEVICE, weights_only=False)
    model.info_head.load_state_dict(head_sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[mf] loaded backbone + info_head from {args.head_pt}")

    rng = np.random.RandomState(args.seed)
    rows_per_variant = defaultdict(list)
    n_skipped = 0
    for fi, fkey in enumerate(frame_keys):
        tile_idxs = groups[fkey]
        # one rig perturbation, applied to every tile of the frame
        ox = float(rng.uniform(-args.rot_deg, args.rot_deg))
        oy = float(rng.uniform(-args.rot_deg, args.rot_deg))
        oz = float(rng.uniform(-args.rot_deg, args.rot_deg))
        ypr = np.array([oz, oy, ox], dtype=np.float64)
        t = (rng.uniform(-1.0, 1.0, size=3) * args.t_m).astype(np.float64) \
            if args.t_m > 0 else np.zeros(3)
        batch = _build_one_frame_batch(ds, tile_idxs, t, ypr)
        if batch is None:
            n_skipped += 1
            continue
        d = _forward_and_concat(model, cfg, batch, dist_one)
        if d['n_pts'] < 12:
            n_skipped += 1
            continue
        res = _solve_and_eval(d)
        for name, m in res.items():
            rows_per_variant[name].append(m)
        if (fi + 1) % 10 == 0 or fi == 0:
            r0 = res
            print(f"[mf] frame {fi+1}/{len(frame_keys)} {fkey}  "
                  f"tiles={d['n_tiles']} pts={d['n_pts']}  | "
                  f"do-nothing {r0['do-nothing']['reproj_mean']:5.2f}  "
                  f"σ {r0['W=σ']['reproj_mean']:5.2f}  "
                  f"learned {r0['W=learned']['reproj_mean']:5.2f}  "
                  f"oracle {r0['oracle']['reproj_mean']:5.2f}")

    print()
    print(f"[mf] processed {sum(len(v) for v in rows_per_variant.values()) // 5} frames "
          f"({n_skipped} skipped)")
    print()
    hdr = (f"{'variant':12s}  "
           f"{'reproj mean':>12s}  {'reproj med':>11s}  {'reproj p95':>11s}  "
           f"{'rot°RMS':>9s}  {'t·m·RMS':>9s}")
    print(hdr)
    print('-' * len(hdr))
    out_lines = [hdr, '-' * len(hdr)]
    for name in ['do-nothing', 'W=I', 'W=σ', 'W=learned', 'oracle']:
        rows = rows_per_variant[name]
        if not rows:
            continue
        rep_mean = np.mean([r['reproj_mean'] for r in rows])
        rep_med = np.mean([r['reproj_med'] for r in rows])
        rep_p95 = np.mean([r['reproj_p95'] for r in rows])
        rot_rms = float(np.sqrt(np.mean([r['rot_rms'] ** 2 for r in rows])))
        t_rms = float(np.sqrt(np.mean([r['t_rms'] ** 2 for r in rows])))
        line = (f"{name:12s}  "
                f"{rep_mean:12.3f}  {rep_med:11.3f}  {rep_p95:11.3f}  "
                f"{rot_rms:9.4f}  {t_rms:9.4f}")
        print(line)
        out_lines.append(line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(out_lines) + '\n')
    print(f"\n[mf] wrote → {args.out}")


if __name__ == '__main__':
    main()
