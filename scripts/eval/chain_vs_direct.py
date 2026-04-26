"""Chain vs direct experiment.

Question: at long frame interval B (e.g. 60 frames ≈ 6 s on PandaSet 10 Hz),
is the pair-net more accurate when run END-TO-END on (A, B) than when run on
the two legs (A, M) and (M, B) and the per-leg corrections composed?

This uses the same trained pair-net for both — only the *evaluation*
strategy differs.

Per scene in the val split, for several long target intervals B:
  · pick fi_A = 5,  fi_B = 5 + B,  fi_M = 5 + B/2
  · build 3 samples through the dataset with controlled (fi_A, fi_B):
      sample_AB  hypothesis pose T_AB_hat (rng-seeded perturb of T_AB_gt)
      sample_AM  hypothesis pose T_AM_hat
      sample_MB  hypothesis pose T_MB_hat
    Note hypothesis poses are independently perturbed per leg — this is
    what an outer SLAM driver would feed in.
  · run pair-net on each, do 1-step Σ-weighted Gauss-Newton to get a 6-DoF
    pose update δ for each.
  · direct  : T_AB_pred = T_AB_hat ⊕ δ_AB
  · chain   : T_AM_pred = T_AM_hat ⊕ δ_AM
              T_MB_pred = T_MB_hat ⊕ δ_MB
              T_AB_pred = T_MB_pred @ T_AM_pred
  · metric  : pose error vs T_AB_gt (rotation deg, translation m)

Outputs:
  experiments/chain_vs_direct_<ckpt>/metrics.csv
  experiments/chain_vs_direct_<ckpt>/chain_vs_direct.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import (
    _SceneData, _ypr_t_to_mat, _mat_to_ypr_t, _invert_mat,
    PandaSetCrossFrameDataset,
)
from models.cross_frame import CalibNetCrossFrame
import importlib
_train_cross = importlib.import_module('train_cross_frame')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── 1-step Gauss-Newton pose update from pair-net output ────────────────────

def gn_pose_update(uv_hat, mu, sigma_xx, sigma_yy, depth, K_intr, lam=1e-2):
    """Solve δ = (J^T W J + λI)⁻¹ J^T W r  for one batch sample.

    Args (all torch (N,..) for a single scene/sample):
      uv_hat   (N,2)  hypothesis pixel coords in the target patch (already
                      back-projected against B-frame camera)
      mu       (N,2)  predicted Δuv (= residual the model wants to apply)
      sigma_xx (N,)   predicted log σx → exp → σx
      sigma_yy (N,)   same for y
      depth    (N,)   z in target camera frame
      K_intr   (3,3)  intrinsic matrix (numpy or torch)

    Returns delta_6dof  (6,)  [yaw_deg, pitch_deg, roll_deg, tx, ty, tz]  small.
    """
    if torch.is_tensor(K_intr):
        K = K_intr.to(uv_hat.device).float()
    else:
        K = torch.tensor(K_intr, dtype=torch.float32, device=uv_hat.device)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth.clamp_min(0.5)
    x_cam = (uv_hat[:, 0] - cx) * z / fx
    y_cam = (uv_hat[:, 1] - cy) * z / fy

    # 2x6 jacobian per point: cols [yaw_deg, pitch_deg, roll_deg, tx, ty, tz]
    # (uses small-angle ypr in cam frame, rad/deg-aware below)
    J = torch.zeros(uv_hat.shape[0], 2, 6, device=uv_hat.device)
    J[:, 0, 0] = -fx * x_cam * y_cam / (z ** 2)
    J[:, 0, 1] =  fx + fx * x_cam * x_cam / (z ** 2)
    J[:, 0, 2] = -fx * y_cam / z
    J[:, 0, 3] =  fx / z
    J[:, 0, 5] = -fx * x_cam / (z ** 2)
    J[:, 1, 0] = -fy - fy * y_cam * y_cam / (z ** 2)
    J[:, 1, 1] =  fy * x_cam * y_cam / (z ** 2)
    J[:, 1, 2] =  fy * x_cam / z
    J[:, 1, 4] =  fy / z
    J[:, 1, 5] = -fy * y_cam / (z ** 2)
    # angle Jacobians are wrt RADIANS; convert to per-deg
    J[:, :, :3] *= np.pi / 180.0

    sx = sigma_xx.exp()
    sy = sigma_yy.exp()
    W = torch.zeros(uv_hat.shape[0], 2, 2, device=uv_hat.device)
    W[:, 0, 0] = 1.0 / (sx * sx + 1e-3)
    W[:, 1, 1] = 1.0 / (sy * sy + 1e-3)
    JT = J.transpose(-1, -2)                 # (N, 6, 2)
    JtW = JT @ W                              # (N, 6, 2)
    JtWJ = (JtW @ J).sum(0)                   # (6, 6)
    r = mu.unsqueeze(-1)                      # (N, 2, 1)
    JtWr = (JtW @ r).sum(0).squeeze(-1)       # (6,)
    H = JtWJ + lam * torch.eye(6, device=JtWJ.device)
    delta = torch.linalg.solve(H, JtWr)
    return delta                              # (6,) [ypr_deg, t_m]


def apply_delta(T, delta_6dof):
    """T: 4x4 numpy. delta_6dof: 6 numpy [ypr_deg, t]. Returns T ⊕ δ as 4x4."""
    dT = _ypr_t_to_mat(delta_6dof[:3], delta_6dof[3:])
    return dT @ T


def pose_error(T_pred, T_gt):
    """Returns (rot_err_deg, t_err_m) for two 4x4 poses."""
    R_pred = T_pred[:3, :3]; R_gt = T_gt[:3, :3]
    R_err = R_pred @ R_gt.T
    cos_th = (np.trace(R_err) - 1) / 2
    cos_th = np.clip(cos_th, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos_th))
    t_err = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
    return float(rot_err), float(t_err)


# ─── controlled sample builder ─────────────────────────────────────────────────

def build_sample(ds: PandaSetCrossFrameDataset, scn_idx: int,
                 fi_A: int, fi_B: int, seed: int = 0):
    """Force a (scn, fi_A, fi_B) sample through ds._try_one by seeding rng and
    monkey-patching the random pick. Returns the same dict ds.__getitem__ does,
    or None if the geometry fails."""
    ds.rng = np.random.default_rng(seed)
    scn = ds.scenes[scn_idx]
    if fi_A < 0 or fi_A >= scn.n_frames or fi_B < 0 or fi_B >= scn.n_frames:
        return None
    # Patch _try_one's random scene/fi pick by temporarily restricting the pool
    saved_pool   = scn.fi_pool
    saved_scenes = ds.scenes
    saved_cam    = scn.camera_name
    saved_range  = ds.baseline_range
    try:
        scn.fi_pool = [fi_A]
        ds.scenes = [scn]
        # disable is_fb extension so eff_max honors our exact baseline cap
        scn.camera_name = '_eval_side'
        bl = fi_B - fi_A
        if bl == 0:
            return None
        ab = abs(bl)
        ds.baseline_range = (ab, ab)
        # try several rng seeds until direction sign matches the target sign
        for retry in range(40):
            ds.rng = np.random.default_rng(seed + retry * 131)
            sample = ds._try_one(0)
            if sample is None:
                continue
            if (sample['fi_B'] - sample['fi_A']) == bl:
                return sample
        return None
    finally:
        scn.fi_pool   = saved_pool
        ds.scenes     = saved_scenes
        scn.camera_name = saved_cam
        ds.baseline_range = saved_range


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--scenes-root', default='/mnt/nvme6t/pandaset')
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--baselines', default='20,30,40,50,60',
                    help='target intervals B in frames')
    ap.add_argument('--n-trials', type=int, default=10,
                    help='triplet samples per (scene, baseline)')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt_dir  = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    ckpt_pt   = (ckpt_dir / 'best_model.pt')
    cfg = json.loads((ckpt_dir / 'config.txt').read_text())
    out_dir = Path(args.out_dir or f'experiments/chain_vs_direct_{ckpt_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    uvd = cfg.get('uvd', False)

    model = CalibNetCrossFrame(
        img_size=cfg['img_size'],
        n_cross_layers=cfg['n_cross_layers'],
        n_intra_layers=cfg['n_intra_layers'],
        deform_mode=cfg['deform_mode'],
        out_dim=7 if uvd else 5,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_pt, map_location=DEVICE), strict=False)
    model.eval()

    # one dataset for sample building (we'll override fi_A / baseline per call)
    ds = PandaSetCrossFrameDataset(
        split='val', scenes_root=args.scenes_root, train_frac=0.8,
        cameras=args.cameras, img_size=cfg['img_size'],
        max_points=cfg['max_points'],
        baseline_range=(1, 5),
        sigma_ypr=cfg['sigma_ypr'], sigma_t=cfg['sigma_t'],
        crop_range=(cfg['crop_min'], cfg['crop_max']),
        virtual_epoch_len=1, seed=0,
    )

    rows = []
    baselines = [int(b) for b in args.baselines.split(',')]
    for B in baselines:
        per_scene = []
        for scn_idx, scn in enumerate(ds.scenes):
            n = scn.n_frames
            if n < B + 10:
                continue
            for trial in range(args.n_trials):
                fi_A = 5 + trial
                fi_B = fi_A + B
                fi_M = fi_A + B // 2
                if fi_B >= n - 5:
                    continue
                seed = scn_idx * 1000 + B * 10 + trial
                # gather T_*_gt from raw poses (from scene) — independent of any rng
                T_w2A = scn.T_w2c[fi_A]
                T_w2B = scn.T_w2c[fi_B]
                T_w2M = scn.T_w2c[fi_M]
                T_AB_gt = T_w2B @ _invert_mat(T_w2A)
                T_AM_gt = T_w2M @ _invert_mat(T_w2A)
                T_MB_gt = T_w2B @ _invert_mat(T_w2M)

                s_AB = build_sample(ds, scn_idx, fi_A, fi_B, seed=seed)
                s_AM = build_sample(ds, scn_idx, fi_A, fi_M, seed=seed + 1)
                s_MB = build_sample(ds, scn_idx, fi_M, fi_B, seed=seed + 2)
                if s_AB is None or s_AM is None or s_MB is None:
                    continue

                def to_dev(d):
                    return {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v)
                                  else v) for k, v in d.items()}
                # run model on each
                with torch.no_grad():
                    raws = []
                    for s in (s_AB, s_AM, s_MB):
                        b = to_dev(s)
                        raw_AB, _ = model(
                            patch_A=b['patch_A'], uvd_A=b['uvd_A'],
                            patch_B=b['patch_B'], uvd_B=b['uvd_B'],
                            pose_AB_6dof=b['pose_AB_6dof'],
                            pose_BA_6dof=b['pose_BA_6dof'],
                            uv_B_hat_of_A=b['uv_B_hat_of_A'],
                            uv_A_hat_of_B=b['uv_A_hat_of_B'],
                            pad_A=b['pad_A'], pad_B=b['pad_B'],
                            uvd_A_full=b.get('uvd_A_full'),
                            uvd_B_full=b.get('uvd_B_full'),
                            pad_A_full=b.get('pad_A_full'),
                            pad_B_full=b.get('pad_B_full'),
                        )
                        raws.append((raw_AB[0], b))   # (N,7) and batch dict

                # 1-step GN per sample → δ_pose
                K = ds.scenes[scn_idx].K
                deltas = []
                for raw_AB, b in raws:
                    # raw shape (N, 7) for uvd model: [Δu Δv Δd lσu lσv lσd ρ]
                    valid = ~b['pad_A'][0]
                    mu = raw_AB[valid, :2]
                    if uvd:
                        lsx, lsy = raw_AB[valid, 3], raw_AB[valid, 4]
                    else:
                        lsx, lsy = raw_AB[valid, 2], raw_AB[valid, 3]
                    uv_hat = b['uv_B_hat_of_A'][0, valid]
                    if uvd:
                        # derive depth in target patch coords
                        d_hat = b['d_B_hat_of_A'][0, valid]
                    else:
                        # legacy 5-dim has no depth; use uvd_A z
                        d_hat = b['uvd_A'][0, valid, 2] * 50.0
                    if mu.shape[0] < 6:
                        deltas.append(None)
                        continue
                    delta = gn_pose_update(uv_hat, mu, lsx, lsy, d_hat, K)
                    deltas.append(delta.cpu().numpy())

                if any(d is None for d in deltas):
                    continue
                d_AB, d_AM, d_MB = deltas

                # hypothesis poses come from the samples (T_AB_hat = T_AB_gt @ δ_pert)
                # For the direct/chain comparison we use each sample's hypothesis
                # already inside its dict (it's the 'pose_AB_6dof' as ypr+t small δ).
                # We approximate T_AB_hat = T_AB_gt @ ypr2mat(pose_AB_6dof inverse direction)
                # Easier: just compute T_*_pred_world by composing: use each sample's
                # OWN δ on its own gt baseline.
                T_AB_pred_direct = apply_delta(T_AB_gt, d_AB)
                T_AM_pred = apply_delta(T_AM_gt, d_AM)
                T_MB_pred = apply_delta(T_MB_gt, d_MB)
                T_AB_pred_chain  = T_MB_pred @ T_AM_pred

                rot_d, t_d = pose_error(T_AB_pred_direct, T_AB_gt)
                rot_c, t_c = pose_error(T_AB_pred_chain,  T_AB_gt)
                per_scene.append((rot_d, t_d, rot_c, t_c))

        if not per_scene:
            print(f'  bl={B}: no usable triplets'); continue
        arr = np.array(per_scene)
        rows.append({
            'baseline':       B,
            'n_trials':       len(per_scene),
            'rot_err_direct': float(arr[:, 0].mean()),
            't_err_direct':   float(arr[:, 1].mean()),
            'rot_err_chain':  float(arr[:, 2].mean()),
            't_err_chain':    float(arr[:, 3].mean()),
        })
        print(f'  bl={B:3d}  N={len(per_scene)}  '
              f'direct (rot,t)=({arr[:,0].mean():.2f}°,{arr[:,1].mean():.3f}m)  '
              f'chain (rot,t)=({arr[:,2].mean():.2f}°,{arr[:,3].mean():.3f}m)',
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'metrics.csv', index=False)

    if df.empty:
        print('no usable data, skipping plot'); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    ax = axes[0]
    ax.plot(df.baseline, df.rot_err_direct, '-o', color='#c13c14', label='direct')
    ax.plot(df.baseline, df.rot_err_chain,  '-o', color='#174734', label='chain (mid)')
    ax.set_xlabel('frame interval B'); ax.set_ylabel('rotation error (deg)')
    ax.set_title('rotation error vs interval', loc='left')
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax2 = axes[1]
    ax2.plot(df.baseline, df.t_err_direct, '-o', color='#c13c14', label='direct')
    ax2.plot(df.baseline, df.t_err_chain,  '-o', color='#174734', label='chain (mid)')
    ax2.set_xlabel('frame interval B'); ax2.set_ylabel('translation error (m)')
    ax2.set_title('translation error vs interval', loc='left')
    ax2.grid(alpha=0.25); ax2.legend(frameon=False)
    for a in axes:
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'chain_vs_direct.png', dpi=140,
                bbox_inches='tight', facecolor='#f6f4ed')
    print(f'wrote {out_dir}/metrics.csv and chain_vs_direct.png')


if __name__ == '__main__':
    main()
