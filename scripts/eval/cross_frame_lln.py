"""LLN + BA pose-recovery test for CalibNetCrossFrame.

Goal: prove or refute the central hypothesis — a model trained on SHORT-BASELINE
(1–20 frame) pairs still gives useful per-point residual predictions at LONGER
baselines, and the averaging across many points via BA recovers pose better
than the raw dead-reckoning hypothesis.

Flow per (fi_A, fi_B) pair, at each baseline:
  1. Build the exact same training-style sample (one pivot, in-patch LiDAR,
     T_AB_hat = T_AB_gt · δT).
  2. Run the model → per-point (Δu, Δv, Σ) for A→B direction.
  3. Solve 6-DoF θ via pyceres minimising Σᵢ ||π_B(R(θ)·P_i + t) − (uv_hatᵢ + Δᵢ)||_Σᵢ².
  4. Compare (θ·T_AB_hat) vs T_AB_gt → rotation err (°) / translation err (cm).
     And compare T_AB_hat vs T_AB_gt for the baseline "no correction".

Output: experiments/{ckpt_dir}/lln/
  - summary.json
  - lln_curves.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import pyceres
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import (
    _SceneData, _project, _ypr_t_to_mat, _mat_to_ypr_t, _invert_mat,
)
from models.cross_frame import CalibNetCrossFrame
import ba_singleframe as bas

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bas.DEVICE = DEVICE


# ─── one pair → one training-style sample (but we keep full-res info for BA) ─

def make_pair(scn: _SceneData, rng, fi_A, fi_B, img_size, max_points,
              sigma_ypr, sigma_t, crop_range):
    """Same sampling logic as PandaSetCrossFrameDataset._try_one but retains
    P_A (3D points in A-cam frame) and box coords so we can reconstruct
    full-res projections for BA.
    """
    IW, IH, K = scn.IW, scn.IH, scn.K
    T_w2A = scn.T_w2c[fi_A]
    T_w2B = scn.T_w2c[fi_B]
    T_A2w = _invert_mat(T_w2A)
    T_A_to_B_gt = T_w2B @ T_A2w

    pts_w_A, uv_Af, z_Af, in_A = scn.frame_data(fi_A)
    pts_w_B, uv_Bf, z_Bf, in_B = scn.frame_data(fi_B)
    if in_A.sum() < 50:
        return None

    pts_w_A_in = pts_w_A[in_A]
    uv_A_all   = uv_Af[in_A]
    z_A_all    = z_Af[in_A]

    # pivot point
    ci = int(rng.integers(len(pts_w_A_in)))
    P_center_w = pts_w_A_in[ci]
    uc_A, vc_A = uv_A_all[ci]

    # pose perturbation → T_AB_hat
    ypr_pert = rng.standard_normal(3).astype(np.float32) * sigma_ypr
    t_pert   = rng.standard_normal(3).astype(np.float32) * sigma_t
    δT       = _ypr_t_to_mat(ypr_pert, t_pert)
    T_A_to_B_hat = T_A_to_B_gt @ δT

    # pivot in B under hypothesis
    P_center_A   = (T_w2A @ np.append(P_center_w, 1.0))[:3]
    P_center_Bh  = (T_A_to_B_hat @ np.append(P_center_A, 1.0))[:3]
    if P_center_Bh[2] < 1.0:
        return None
    uc_B_hat = (K @ P_center_Bh)[:2] / P_center_Bh[2]
    uc_B_hat = uc_B_hat.astype(np.float32)

    # crop both patches (random size shared)
    CROP = int(rng.integers(crop_range[0], crop_range[1] + 1))
    half = CROP / 2
    u0_A = max(0, min(IW - CROP, uc_A - half))
    v0_A = max(0, min(IH - CROP, vc_A - half))
    u0_B = max(0, min(IW - CROP, uc_B_hat[0] - half))
    v0_B = max(0, min(IH - CROP, uc_B_hat[1] - half))

    def _crop(img, u0, v0):
        u0i, v0i = int(u0), int(v0)
        u1i, v1i = u0i + CROP, v0i + CROP
        u0i = max(0, u0i); v0i = max(0, v0i)
        u1i = min(IW, u1i); v1i = min(IH, v1i)
        if u1i - u0i < 2 or v1i - v0i < 2:
            return None
        patch = img[v0i:v1i, u0i:u1i]
        t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(img_size, img_size),
                           mode='bilinear', align_corners=False).squeeze(0)
        return t, (u0i, v0i, u1i - u0i, v1i - v0i)

    img_A = scn.load_image(fi_A)
    img_B = scn.load_image(fi_B)
    pa = _crop(img_A, u0_A, v0_A)
    pb = _crop(img_B, u0_B, v0_B)
    if pa is None or pb is None:
        return None
    patch_A, box_A = pa
    patch_B, box_B = pb

    # A-queries = in-patch_A A-points
    u0, v0, cw, ch = box_A
    in_box_A = ((uv_A_all[:, 0] >= u0) & (uv_A_all[:, 0] < u0 + cw) &
                (uv_A_all[:, 1] >= v0) & (uv_A_all[:, 1] < v0 + ch))
    if in_box_A.sum() < 4:
        return None
    pts_w_QA = pts_w_A_in[in_box_A]
    uv_A_full = uv_A_all[in_box_A]
    z_A_all_  = z_A_all[in_box_A]

    # transform to A-cam + to B-cam (both gt and hat)
    homo = np.concatenate([pts_w_QA, np.ones((len(pts_w_QA), 1), dtype=np.float32)], axis=1)
    P_QA_in_A     = (T_w2A @ homo.T)[:3].T
    P_QA_in_B_gt  = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_gt.T)[:, :3]
    P_QA_in_B_hat = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_hat.T)[:, :3]
    good = (P_QA_in_B_gt[:, 2] > 0.5) & (P_QA_in_B_hat[:, 2] > 0.5)
    if good.sum() < 4:
        return None

    # full-res uv for gt and hat
    uv_B_gt_full  = (K @ P_QA_in_B_gt[good].T)[:2]  / P_QA_in_B_gt[good, 2]
    uv_B_hat_full = (K @ P_QA_in_B_hat[good].T)[:2] / P_QA_in_B_hat[good, 2]
    uv_B_gt_full  = uv_B_gt_full.T.astype(np.float32)
    uv_B_hat_full = uv_B_hat_full.T.astype(np.float32)

    # patch-local uv (for model input)
    def _to_local(uv_full, box):
        u0, v0, cw, ch = box
        return np.stack([(uv_full[:, 0] - u0) * (img_size / cw),
                          (uv_full[:, 1] - v0) * (img_size / ch)], axis=1).astype(np.float32)

    uv_A_patch       = _to_local(uv_A_full[good], box_A)
    uv_B_hat_local   = _to_local(uv_B_hat_full, box_B)
    uv_B_gt_local    = _to_local(uv_B_gt_full,  box_B)

    # drop points falling outside patch_B (for model input consistency)
    inb = ((uv_B_hat_local[:, 0] >= 0) & (uv_B_hat_local[:, 0] < img_size) &
           (uv_B_hat_local[:, 1] >= 0) & (uv_B_hat_local[:, 1] < img_size) &
           (uv_B_gt_local[:, 0]  >= 0) & (uv_B_gt_local[:, 0]  < img_size) &
           (uv_B_gt_local[:, 1]  >= 0) & (uv_B_gt_local[:, 1]  < img_size))
    if inb.sum() < 4:
        return None

    P_QA_in_A      = P_QA_in_A[good][inb]
    P_QA_in_B_gt   = P_QA_in_B_gt[good][inb]
    P_QA_in_B_hat  = P_QA_in_B_hat[good][inb]
    uv_A_patch     = uv_A_patch[inb]
    z_A_all_       = z_A_all_[good][inb]
    uv_B_hat_local = uv_B_hat_local[inb]
    uv_B_gt_local  = uv_B_gt_local[inb]
    uv_B_hat_full  = uv_B_hat_full[inb]
    uv_B_gt_full   = uv_B_gt_full[inb]

    # Pad/truncate to max_points
    N = len(uv_A_patch)
    N_use = min(N, max_points)
    if N > N_use:
        pick = rng.choice(N, size=N_use, replace=False)
    else:
        pick = np.arange(N)
    P_QA_in_A      = P_QA_in_A[pick]
    uv_A_patch     = uv_A_patch[pick]
    z_A_all_       = z_A_all_[pick]
    uv_B_hat_local = uv_B_hat_local[pick]
    uv_B_gt_local  = uv_B_gt_local[pick]
    uv_B_hat_full  = uv_B_hat_full[pick]
    uv_B_gt_full   = uv_B_gt_full[pick]

    # Pad to max_points
    pad = np.zeros(max_points, dtype=bool)
    if N_use < max_points:
        pad[N_use:] = True
        P_QA_in_A      = np.concatenate([P_QA_in_A,      np.zeros((max_points - N_use, 3), np.float32)])
        uv_A_patch     = np.concatenate([uv_A_patch,     np.zeros((max_points - N_use, 2), np.float32)])
        z_A_all_       = np.concatenate([z_A_all_,       np.zeros(max_points - N_use, np.float32)])
        uv_B_hat_local = np.concatenate([uv_B_hat_local, np.zeros((max_points - N_use, 2), np.float32)])
        uv_B_gt_local  = np.concatenate([uv_B_gt_local,  np.zeros((max_points - N_use, 2), np.float32)])
        uv_B_hat_full  = np.concatenate([uv_B_hat_full,  np.zeros((max_points - N_use, 2), np.float32)])
        uv_B_gt_full   = np.concatenate([uv_B_gt_full,   np.zeros((max_points - N_use, 2), np.float32)])

    # Symmetric B-side (minimal, just enough for model forward)
    in_box_B = ((uv_Bf[:, 0] >= box_B[0]) & (uv_Bf[:, 0] < box_B[0] + box_B[2]) &
                (uv_Bf[:, 1] >= box_B[1]) & (uv_Bf[:, 1] < box_B[1] + box_B[3]) &
                (z_Bf > 1.0))
    idx_B = np.where(in_box_B)[0]
    if len(idx_B) > max_points:
        idx_B = rng.choice(idx_B, size=max_points, replace=False)
    z_B_here = z_Bf[idx_B]
    uv_B_full_here = uv_Bf[idx_B]
    uv_B_local = np.stack([(uv_B_full_here[:, 0] - box_B[0]) * (img_size / box_B[2]),
                            (uv_B_full_here[:, 1] - box_B[1]) * (img_size / box_B[3])], axis=1).astype(np.float32)
    pad_B = np.zeros(max_points, dtype=bool)
    if len(idx_B) < max_points:
        pad_B[len(idx_B):] = True
        uv_B_local = np.concatenate([uv_B_local, np.zeros((max_points - len(idx_B), 2), np.float32)])
        z_B_here   = np.concatenate([z_B_here,   np.zeros(max_points - len(idx_B), np.float32)])

    # Pose 6DoF
    ypr_AB_hat, t_AB_hat = _mat_to_ypr_t(T_A_to_B_hat)
    ypr_BA_hat, t_BA_hat = _mat_to_ypr_t(_invert_mat(T_A_to_B_hat))

    z_A_norm = z_A_all_ / 50.0
    z_B_norm = z_B_here / 50.0

    # Dummy A_of_B values (model needs them but we don't BA B→A side here)
    dummy_B = np.zeros_like(uv_B_local)

    # depth of A-points projected into B under hyp / gt (meters, B-cam frame z)
    d_B_hat_of_A = P_QA_in_B_hat[:, 2].astype(np.float32)
    d_B_gt_of_A  = P_QA_in_B_gt[:, 2].astype(np.float32)
    if N_use < max_points:
        d_B_hat_of_A = np.concatenate([d_B_hat_of_A, np.zeros(max_points - N_use, np.float32)])
        d_B_gt_of_A  = np.concatenate([d_B_gt_of_A,  np.zeros(max_points - N_use, np.float32)])

    return dict(
        patch_A=patch_A.float(), patch_B=patch_B.float(),
        uvd_A=torch.from_numpy(np.concatenate([uv_A_patch, z_A_norm[:, None]], 1)).float(),
        uvd_B=torch.from_numpy(np.concatenate([uv_B_local, z_B_norm[:, None]], 1)).float(),
        uv_B_hat_of_A=torch.from_numpy(uv_B_hat_local).float(),
        uv_A_hat_of_B=torch.from_numpy(dummy_B).float(),
        pose_AB_6dof=torch.from_numpy(np.concatenate([ypr_AB_hat, t_AB_hat]).astype(np.float32)),
        pose_BA_6dof=torch.from_numpy(np.concatenate([ypr_BA_hat, t_BA_hat]).astype(np.float32)),
        pad_A=torch.from_numpy(pad),
        pad_B=torch.from_numpy(pad_B),
        # BA helpers (full-res)
        P_A_cam   = P_QA_in_A,          # (N, 3) — for BA
        uv_hat_full = uv_B_hat_full,    # (N, 2) — for BA target
        uv_gt_full  = uv_B_gt_full,
        d_hat_full  = d_B_hat_of_A,     # (N,) — hyp depth in B-cam (meters) for UVD BA
        d_gt_full   = d_B_gt_of_A,
        T_AB_gt = T_A_to_B_gt,
        T_AB_hat = T_A_to_B_hat,
        K = K,
        box_B = np.array(box_B, dtype=np.float32),    # (u0, v0, cw, ch) for patch→full scale
        N_valid = N_use,
        fi_A = fi_A, fi_B = fi_B, scene = scn.root.name,
    )


# ─── BA pose recovery ────────────────────────────────────────────────────────

class CalibBACost3D(pyceres.CostFunction):
    """3D residual variant of CalibBACost.

    Residuals per point = sqrt_info_3x3 @ (π3(R(ypr)·P + t) - target_3),
    where target_3 = (uv_target, d_target) and π3 returns (u, v, z).
    sqrt_info_3x3 is block-diagonal: [L_uv (2×2), 1/σ_d] (depth
    decoupled from uv by the model's clamp_params_uvd parameterisation).

    Block = (6,). N points → 3N residuals.
    """

    def __init__(self, P, uv_target, d_target, L_uv, inv_sigma_d, K):
        import pyceres
        super().__init__()
        self.P, self.uv_target, self.d_target = P, uv_target, d_target
        self.L_uv, self.inv_sd = L_uv, inv_sigma_d
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.N = P.shape[0]
        self.set_num_residuals(3 * self.N)
        self.set_parameter_block_sizes([6])

    def _residuals(self, theta):
        from ba_singleframe import ypr_to_R
        R = ypr_to_R(theta[:3]); t = theta[3:6]
        Q = self.P @ R.T + t
        proj_uv = np.stack([self.fx * Q[:, 0] / Q[:, 2] + self.cx,
                            self.fy * Q[:, 1] / Q[:, 2] + self.cy], axis=1)
        r_uv_raw = proj_uv - self.uv_target
        r_uv = np.einsum('nij,nj->ni', self.L_uv, r_uv_raw)       # (N,2)
        r_d  = (Q[:, 2] - self.d_target) * self.inv_sd             # (N,)
        return np.concatenate([r_uv, r_d[:, None]], axis=1).ravel()

    def Evaluate(self, params, residuals, jacobians):
        theta = np.asarray(params[0]).copy()
        residuals[:] = self._residuals(theta)
        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-5
            J = np.zeros((3 * self.N, 6))
            for k in range(6):
                tp = theta.copy(); tp[k] += eps
                tm = theta.copy(); tm[k] -= eps
                J[:, k] = (self._residuals(tp) - self._residuals(tm)) / (2 * eps)
            jacobians[0][:] = J.ravel()
        return True


def ba_recover(sample, delta_pred, Sigma_pred, delta_d_pred=None, sigma_d_pred=None):
    """Solve 6-DoF θ that aligns hypothesis+delta with A→B transformation.

    We use T_A_to_B_hat as the hypothesis. BA variable θ = (ypr, t) composes
    with the hypothesis multiplicatively:
        T_recovered = δT(θ) · T_AB_hat
    At θ=0 the output matches the hypothesis.

    If `delta_d_pred` and `sigma_d_pred` are provided, the BA uses 3D
    residuals (uv + depth, block-diagonal Mahalanobis). Otherwise falls
    back to the 2D uv-only cost.
    """
    N = sample['N_valid']
    if N < 6:
        return None
    P_A = sample['P_A_cam'][:N]             # (N, 3) A-cam frame
    uv_hat_full = sample['uv_hat_full'][:N]  # (N, 2) full-res px
    K = sample['K']
    T_AB_hat = sample['T_AB_hat']

    # Convert model's patch-local delta → full-res delta using box_B
    # patch-local coord: (u_full - u0) * (img_size / cw) = u_patch
    # ∴ 1 patch-px = cw / img_size full-px
    # Assume img_size is the model output scale (can reconstruct from uvd_A range)
    u0, v0, cw, ch = sample['box_B']
    # img_size is fixed across pair; infer from uv_B_hat_of_A (patch-local max)
    img_size = 64.0   # must match dataset.img_size
    scale_u = cw / img_size
    scale_v = ch / img_size
    delta_full = np.stack([delta_pred[:N, 0] * scale_u,
                            delta_pred[:N, 1] * scale_v], axis=1)
    # Sigma scales as (scale)^2 in each dim, with cross-term scale_u*scale_v
    Sigma_full = np.zeros_like(Sigma_pred[:N])
    Sigma_full[:, 0, 0] = Sigma_pred[:N, 0, 0] * scale_u * scale_u
    Sigma_full[:, 1, 1] = Sigma_pred[:N, 1, 1] * scale_v * scale_v
    Sigma_full[:, 0, 1] = Sigma_pred[:N, 0, 1] * scale_u * scale_v
    Sigma_full[:, 1, 0] = Sigma_full[:, 0, 1]

    uv_target_full = uv_hat_full + delta_full

    # transform P_A by T_AB_hat to get "P in B-cam under hypothesis"
    homo = np.concatenate([P_A, np.ones((N, 1))], axis=1)
    P_Bh = (homo @ T_AB_hat.T)[:, :3]

    # sqrt_info from Sigma
    L = np.zeros_like(Sigma_full)
    for i in range(N):
        S = 0.5 * (Sigma_full[i] + Sigma_full[i].T) + 1e-3 * np.eye(2)
        try:
            L[i] = np.linalg.cholesky(np.linalg.inv(S)).T
        except np.linalg.LinAlgError:
            L[i] = np.eye(2)

    θ = np.zeros(6)
    prob = pyceres.Problem()
    if delta_d_pred is not None and sigma_d_pred is not None:
        # UVD mode: compose 3D target (uv + depth) and use block-diagonal 3×3 sqrt_info.
        # Floor σ_d generously: the model's σ_d is trained against training depth GT noise
        # (≤0.05m seen) but actual operating-condition noise (pose perturbation σ_t=0.2m
        # + segment-level pose GT drift) is an order of magnitude larger. A too-confident
        # σ_d lets depth dominate the Mahalanobis sum and blows up the pose solve.
        d_hat_all = sample['d_hat_full'][:N]
        d_target  = d_hat_all + delta_d_pred[:N]
        sd_safe   = np.clip(sigma_d_pred[:N], 1.0, 50.0)   # ≥ 1 m floor
        inv_sd    = 1.0 / sd_safe
        prob.add_residual_block(
            CalibBACost3D(P_Bh, uv_target_full, d_target, L, inv_sd, K),
            None, [θ])
    else:
        prob.add_residual_block(
            bas.CalibBACost(P_Bh, uv_target_full, np.zeros_like(uv_target_full), L, K),
            None, [θ])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 80
    opts.minimizer_progress_to_stdout = False
    summ = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summ)
    return θ, summ


def pose_errors(T_hat, T_gt):
    R_rel = T_hat[:3, :3] @ T_gt[:3, :3].T
    ang = np.rad2deg(np.arccos(np.clip((np.trace(R_rel) - 1) * 0.5, -1, 1)))
    t_err = np.linalg.norm(T_hat[:3, 3] - T_gt[:3, 3])
    return float(ang), float(t_err)


# ──────────────────────────────────────────────────────────────────── main ──

@torch.no_grad()
def infer_batch(model, samples):
    keys = ['patch_A', 'uvd_A', 'patch_B', 'uvd_B',
            'pose_AB_6dof', 'pose_BA_6dof',
            'uv_B_hat_of_A', 'uv_A_hat_of_B', 'pad_A', 'pad_B']
    batch = {k: torch.stack([s[k] for s in samples]).to(DEVICE) for k in keys}
    raw_AB, _ = model(**batch)
    return raw_AB.cpu().numpy()


def run(args):
    out_dir = Path(args.ckpt_dir) / 'lln'
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt_dir) / 'best_model.pt'
    print(f'loading ckpt {ckpt_path}')
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd.keys()) else 'none'
    n_cross = sum(1 for k in sd.keys()
                  if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd.keys()
                  if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w_keys = [k for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = sd[proj_w_keys[0]].shape[0] if proj_w_keys else 5
    print(f'  detected: deform={deform}, n_cross={n_cross}, n_intra={n_intra}, out_dim={out_dim}')
    model = CalibNetCrossFrame(img_size=args.img_size,
                                n_cross_layers=n_cross, n_intra_layers=n_intra,
                                deform_mode=deform, out_dim=out_dim).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # Build val scenes (same scene-level split as training)
    root = Path(args.scenes_root)
    scene_names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in scene_names])
    random.Random(args.seed).shuffle(shuffled)
    cutoff = int(len(shuffled) * args.train_frac)
    val_roots = shuffled[cutoff:]
    print(f'val scenes ({len(val_roots)}): {[Path(r).name for r in val_roots]}')

    scenes = []
    for sr in val_roots:
        scn = _SceneData(Path(sr))
        scn.precompute_all()
        scenes.append(scn)

    results = {}
    for bl in args.baselines:
        rng = np.random.default_rng(200 + bl)
        per_pair = []
        print(f'\n=== baseline ±{bl} frames ===')
        tried = 0
        while len(per_pair) < args.n_pairs and tried < args.n_pairs * 5:
            tried += 1
            scn = scenes[int(rng.integers(len(scenes)))]
            fi_A = int(rng.integers(scn.n_frames))
            delta = bl * int(rng.choice([-1, 1]))
            fi_B = fi_A + delta
            if fi_B < 0 or fi_B >= scn.n_frames:
                continue
            s = make_pair(scn, rng, fi_A, fi_B, args.img_size, args.max_points,
                           args.sigma_ypr, args.sigma_t, (args.crop_min, args.crop_max))
            if s is None:
                continue
            per_pair.append(s)

        if not per_pair:
            print('  (no valid pairs)')
            continue

        # batched inference
        raw_list = []
        for i in range(0, len(per_pair), args.batch_size):
            chunk = per_pair[i : i + args.batch_size]
            raw_list.append(infer_batch(model, chunk))
        raw_all = np.concatenate(raw_list, axis=0)   # (N_pairs, N_max, 5)

        # BA per-pair
        rot_hat_list, t_hat_list, rot_ba_list, t_ba_list = [], [], [], []
        point_err_pred, point_err_base = [], []

        for s, raw in zip(per_pair, raw_all):
            # baseline (no correction)
            rot_hat, t_hat = pose_errors(s['T_AB_hat'], s['T_AB_gt'])
            rot_hat_list.append(rot_hat); t_hat_list.append(t_hat)

            # per-point stats
            N = s['N_valid']
            if N < 6:
                rot_ba_list.append(rot_hat); t_ba_list.append(t_hat)
                continue

            mu = raw[:N, :2]             # patch-local delta (uv only, ignore td if uvd)
            if out_dim == 7:
                log_sx = raw[:N, 3]; log_sy = raw[:N, 4]
                rho = raw[:N, 6]  # already clamped by clamp_params_uvd
            else:
                log_sx = raw[:N, 2]; log_sy = raw[:N, 3]
                rho = np.tanh(raw[:N, 4]) * 0.99
            sx = np.exp(log_sx); sy = np.exp(log_sy)
            Sigma = np.zeros((N, 2, 2), np.float32)
            Sigma[:, 0, 0] = sx * sx; Sigma[:, 1, 1] = sy * sy
            Sigma[:, 0, 1] = Sigma[:, 1, 0] = rho * sx * sy

            # per-point pixel err (patch-local)
            pred_local = s['uv_B_hat_of_A'][:N].numpy() + mu
            gt_local = ((s['uv_hat_full'][:N]  # reuse: compute local gt via transforming
                          - s['uv_hat_full'][:N])  # placeholder 0 — use stored gt_local? skip —
                         + 0)  # we didn't store gt_local on sample; use full-res diff instead
            # Use full-res pred vs gt:
            scale_full = None  # TBD — see ba_recover
            # Simpler: recompute per-point err in full-res using patch mu scaled.
            # box_B isn't stored — approximate by retrieved from sample? skip.
            # For now, use raw patch-local comparison:
            tgt_local = (s['uv_hat_full'][:N] * 0)  # placeholder
            # Actually we need full-res delta to compare with full-res base.
            # Easiest: just track raw L2 in patch-local for diagnostics.
            base_px = np.linalg.norm(s['uv_hat_full'][:N] - s['uv_gt_full'][:N], axis=-1)
            # approximate pred err: use BA-less patch-local + we'll not plot here
            point_err_base.append(base_px)
            # Placeholder pred err (to refine post-BA)
            point_err_pred.append(base_px * 0.5)

            # BA — pass depth residual + σ_d when the model is 7-dim (UVD)
            if out_dim == 7:
                # raw[:, 2] is td in meters (already scaled by D_SCALE=50 in clamp_params_uvd)
                delta_d_pred = raw[:N, 2]
                sigma_d_pred = np.exp(raw[:N, 5])
                res = ba_recover(s, mu, Sigma,
                                 delta_d_pred=delta_d_pred,
                                 sigma_d_pred=sigma_d_pred)
            else:
                res = ba_recover(s, mu, Sigma)
            if res is None:
                rot_ba_list.append(rot_hat); t_ba_list.append(t_hat)
                continue
            θ, _ = res
            δT = _ypr_t_to_mat(θ[:3], θ[3:6])
            T_BA = δT @ s['T_AB_hat']
            rot_ba, t_ba = pose_errors(T_BA, s['T_AB_gt'])
            rot_ba_list.append(rot_ba); t_ba_list.append(t_ba)

        rot_hat = np.array(rot_hat_list); t_hat = np.array(t_hat_list)
        rot_ba  = np.array(rot_ba_list);  t_ba  = np.array(t_ba_list)

        results[bl] = dict(
            n_pairs = len(per_pair),
            rot_hat_mean_deg  = float(np.mean(rot_hat)),
            rot_ba_mean_deg   = float(np.mean(rot_ba)),
            t_hat_mean_cm     = float(np.mean(t_hat) * 100),
            t_ba_mean_cm      = float(np.mean(t_ba) * 100),
            rot_hat_med_deg   = float(np.median(rot_hat)),
            rot_ba_med_deg    = float(np.median(rot_ba)),
            t_hat_med_cm      = float(np.median(t_hat) * 100),
            t_ba_med_cm       = float(np.median(t_ba) * 100),
            rot_improvement   = float(1.0 - np.mean(rot_ba) / max(np.mean(rot_hat), 1e-6)),
            t_improvement     = float(1.0 - np.mean(t_ba)  / max(np.mean(t_hat),  1e-6)),
        )
        r = results[bl]
        print(f'  {r["n_pairs"]} pairs  rot: hat {r["rot_hat_mean_deg"]:.3f}° → BA {r["rot_ba_mean_deg"]:.3f}° '
              f'(↓{r["rot_improvement"]*100:.0f}%)   '
              f't: hat {r["t_hat_mean_cm"]:.1f}cm → BA {r["t_ba_mean_cm"]:.1f}cm '
              f'(↓{r["t_improvement"]*100:.0f}%)')

    # save
    (out_dir / 'summary.json').write_text(json.dumps(dict(
        ckpt=str(args.ckpt_dir), baselines=args.baselines,
        n_pairs=args.n_pairs, sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        results=results,
    ), indent=2))

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    bls = args.baselines
    ax[0].plot(bls, [results[b]['rot_hat_mean_deg'] for b in bls if b in results],
                '--', color='#6b6a63', lw=1, label='dead-reckon')
    ax[0].plot(bls, [results[b]['rot_ba_mean_deg']  for b in bls if b in results],
                'o-', color='#c13c14', lw=2, label='after BA')
    ax[0].set_xlabel('baseline (frames)'); ax[0].set_ylabel('rot err (°)')
    ax[0].set_title('rotation recovery vs baseline', loc='left'); ax[0].legend(frameon=False); ax[0].grid(alpha=0.25)
    ax[1].plot(bls, [results[b]['t_hat_mean_cm'] for b in bls if b in results],
                '--', color='#6b6a63', lw=1, label='dead-reckon')
    ax[1].plot(bls, [results[b]['t_ba_mean_cm']  for b in bls if b in results],
                'o-', color='#c13c14', lw=2, label='after BA')
    ax[1].set_xlabel('baseline (frames)'); ax[1].set_ylabel('translation err (cm)')
    ax[1].set_title('translation recovery vs baseline', loc='left'); ax[1].legend(frameon=False); ax[1].grid(alpha=0.25)
    for a in ax:
        for sp in ('top','right'): a.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'lln_curves.png', dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'\nsaved → {out_dir}/summary.json  {out_dir}/lln_curves.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='experiments/cross_frame_v04_multi39')
    ap.add_argument('--scenes-root', default='/mnt/mininas/datasets/pandaset')
    ap.add_argument('--train-frac', type=float, default=0.80)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--baselines', type=int, nargs='+', default=[1, 5, 10, 20, 40, 60])
    ap.add_argument('--n-pairs', type=int, default=100)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--crop-min', type=int, default=64)
    ap.add_argument('--crop-max', type=int, default=192)
    args = ap.parse_args()
    run(args)
