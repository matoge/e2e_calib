"""Train CalibNet2 (CND2) on tiled-cache data — minimal Accelerate DDP trainer.

Forked from train_ps_v3_ddp.py but stripped:
  * model = CalibNet2 (own RoPEPoseEmb, plain cross-attn, single final head)
  * forward signature differs (distorted_uvd / dpose_R / vfp / bucket_uvd)
  * No BA eval hook, no per-dataset breakdown, no warm-start, no ClearML;
    those wire to CalibNetDepth-specific surfaces and aren't worth the
    bloat for the first CND2 run.

Loss: gaussian2d_nll over per_pt[..., :5] vs gt_duv (=true - dist), same as
legacy. CalibNet2 returns (per_pt, W) when use_info_head=True; we ignore W
for the first kick (matches "sigma-head only" baseline).

Launch (DGX2 GPU 3-10 + 15 = 9 processes, fp16):
    CUDA_VISIBLE_DEVICES=3,4,5,6,7,8,9,10,15 \
    /home/hfunaya/.pyenv/versions/3.10.4/bin/python -m accelerate.commands.launch \
        --num_processes=9 --mixed_precision=fp16 \
        scripts/training/train_cnd2_ddp.py \
        --name cnd2_km_os16_50ep \
        --cache /home/hfunaya/cache/kamikado_v3_tiled \
        --epochs 50 --oversample 16 --batch-size 64
"""
import sys, os, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, math, time, torch
import torch.multiprocessing as _tmp
try: _tmp.set_sharing_strategy('file_system')
except Exception: pass
from pathlib import Path
from datetime import datetime

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full, collate_pair
from models.calibnet2 import CalibNet2
from models.model_cov import gaussian2d_nll
from scripts.ba.ba_torch import (
    pinhole_jacobian, project_pinhole, gn_step, make_info_from_sigma_rho,
    _apply_extrinsic,
)


def _ba_pose_loss(per_pt, dist_uvd, pad_mask, batch, ba_iter=4, damping=1e-3):
    """BA-based pose loss for cnd2.

    per_pt: (B, N, 5)  [du, dv, log_sx, log_sy, rho]
    dist_uvd: (B, N, ≥3)  observation uv in tile-local coords
    pad_mask: (B, N) bool, True = invalid
    batch: full batch tuple (PS calib mode: 7=pert, 9=K, etc.)
    Returns scalar loss + diagnostics dict.
    """
    pert_vec = batch[7]                              # (B, 8) [tx,ty,tz, ypr_deg, dfx, dfy]
    K_orig = batch[10] if len(batch) > 10 else None
    cs_t   = batch[11] if len(batch) > 11 else None
    u0_t   = batch[13] if len(batch) > 13 else None
    v0_t   = batch[14] if len(batch) > 14 else None
    if K_orig is None or cs_t is None or u0_t is None:
        return None, {}
    B, N, _ = dist_uvd.shape
    valid = ~pad_mask
    # Build tile-local K from parent K + crop origin + crop size
    # img_size = tile size used by model (default 128). Inferred from dist_uvd range.
    img_size = 128.0
    scale = img_size / cs_t                          # (B,)
    fx_l = K_orig[..., 0, 0] * scale
    fy_l = K_orig[..., 1, 1] * scale
    cx_l = (K_orig[..., 0, 2] - u0_t) * scale
    cy_l = (K_orig[..., 1, 2] - v0_t) * scale
    K_local = torch.zeros_like(K_orig)
    K_local[..., 0, 0] = fx_l; K_local[..., 1, 1] = fy_l
    K_local[..., 0, 2] = cx_l; K_local[..., 1, 2] = cy_l
    K_local[..., 2, 2] = 1.0
    K = K_local
    mu = per_pt[..., :2]
    sx = torch.exp(per_pt[..., 2]).clamp(0.1, 50.0)
    sy = torch.exp(per_pt[..., 3]).clamp(0.1, 50.0)
    rho = torch.tanh(per_pt[..., 4]) * 0.95
    W = make_info_from_sigma_rho(sx, sy, rho)
    # 3D points from dist_uvd (tile-local u, v, d)
    fx = K[..., 0, 0:1]; fy = K[..., 1, 1:2]
    cx = K[..., 0, 2:3]; cy = K[..., 1, 2:3]
    d = dist_uvd[..., 2] * 100.0                     # normalised → m
    # Padded points have d=0 → divisions blow up. Replace invalid d with safe 10m
    # (won't matter for loss since `valid` masks them out, but keeps Jacobian finite).
    safe_d = torch.where(valid, d, torch.full_like(d, 10.0)).clamp(min=0.5)
    u, v = dist_uvd[..., 0], dist_uvd[..., 1]
    X = (u - cx) * safe_d / fx
    Y = (v - cy) * safe_d / fy
    pts_cam = torch.stack([X, Y, safe_d], dim=-1)
    uv_target = dist_uvd[..., :2] + mu               # network corrected
    dof_names = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')
    prior_diag = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09],
                              device=dist_uvd.device, dtype=dist_uvd.dtype)
    omega = torch.zeros(B, 3, device=dist_uvd.device, dtype=dist_uvd.dtype)
    t = torch.zeros(B, 3, device=dist_uvd.device, dtype=dist_uvd.dtype)
    for _ in range(ba_iter):
        pts_t = _apply_extrinsic(pts_cam, omega, t)
        uv_proj = project_pinhole(pts_t, K)
        r = uv_target - uv_proj
        Xp, Yp, Zp = pts_t.unbind(-1)
        J = pinhole_jacobian(Xp, Yp, Zp, K, uv_proj.detach(), dof_names)
        delta, _ = gn_step(J, W, r, valid=valid, damping=damping, prior_diag=prior_diag)
        omega = omega + delta[:, :3]
        t = t + delta[:, 3:]
    delta_solved = torch.cat([omega, t], dim=-1)     # (B, 6)
    # pert_vec layout: [tx, ty, tz, yaw, pitch, roll, dfx, dfy]
    t_gt = pert_vec[:, :3]
    ypr_gt = pert_vec[:, 3:6]
    rot_loss = (omega - ypr_gt).pow(2).mean()
    t_loss = (t - t_gt).pow(2).mean()
    loss = rot_loss + 100.0 * t_loss
    with torch.no_grad():
        rot_err = (omega - ypr_gt).abs().mean()
        t_err = (t - t_gt).abs().mean()
    return loss, {'rot_err': rot_err.item(), 't_err': t_err.item()}
from torch.utils.data import DataLoader, Subset, ConcatDataset, RandomSampler
import random as _r

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
# Local hack: repo has its own `datasets/` package (= datasets.pandaset_full).
# Accelerate's prepare_data_loader does `from datasets import IterableDataset`
# when is_datasets_available() is True; since the repo dir shadows HF datasets
# on sys.path, that import fails. We don't use HF datasets at all → force the
# probe to False before accel.prepare() runs.
import accelerate.utils.imports as _ai
_ai.is_datasets_available = lambda: False
import accelerate.data_loader as _adl
_adl.is_datasets_available = lambda: False

torch.set_float32_matmul_precision("high")


def _R_from_zyx_deg(ypr_deg: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) rotation matrices from intrinsic ZYX Euler in degrees.

    `ypr_deg` is (B, 3) [yaw, pitch, roll]; matches dataset's
    Rotation.from_euler('zyx', ..., degrees=True) convention.
    """
    yaw = ypr_deg[..., 0] * (math.pi / 180.0)
    pit = ypr_deg[..., 1] * (math.pi / 180.0)
    rol = ypr_deg[..., 2] * (math.pi / 180.0)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pit), torch.sin(pit)
    cr, sr = torch.cos(rol), torch.sin(rol)
    z = torch.zeros_like(yaw); o = torch.ones_like(yaw)
    Rz = torch.stack([cy, -sy, z, sy, cy, z, z, z, o], dim=-1).view(*yaw.shape, 3, 3)
    Ry = torch.stack([cp, z, sp, z, o, z, -sp, z, cp], dim=-1).view(*yaw.shape, 3, 3)
    Rx = torch.stack([o, z, z, z, cr, -sr, z, sr, cr], dim=-1).view(*yaw.shape, 3, 3)
    return Rz @ Ry @ Rx


def render_pair_debug_samples(cap: dict, ep: int, n_samples: int = 6,
                              out_dir: Path = None,
                              k_show: int = 12) -> list[Path]:
    """Numbered-correspondence pair vis (port of _cnd2_pair_hero.py).

    For each captured (A, B) sample, draws a 2-panel figure:
      A panel: numbered circles at uv_A_local (= dist_A), one per selected
               LiDAR query, colour = tab10[k%10].
      B panel: same coloured per-point glyphs at:
               * x  = HAT projection (dist_B, model input)
               * o (large) = GT (true_B, target)
               * o (small) = pred = dist_B + Δuv from per_pt
               * 2σ ellipse around pred from log_sx/log_sy/rho.
      Cross-panel ConnectionPatch (dotted) links A circle ↔ B GT circle so
      the same world point is visible across the two cameras.

    Title shows hyp (= dist_B vs true_B) and pred (= pred_B vs true_B) px err.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Ellipse, ConnectionPatch
    import numpy as _np
    out_dir = Path(out_dir) if out_dir is not None else Path('/tmp/cnd2_dbg')
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_A = cap['imgs_A']; imgs_B = cap['imgs_B']
    distA = cap['distA']
    trueB = cap['trueB']; distB = cap['distB']
    per_pt = cap['per_pt']
    padA = cap['padA']; padB = cap['padB']
    pert_B = cap['pert_B']; dpose = cap['dpose']
    split = cap.get('split', 'train')
    B = imgs_A.shape[0]
    N = min(n_samples, B)
    paths = []

    def _cov_ellipse_axes(sx, sy, rho):
        cxx, cyy, cxy = sx * sx, sy * sy, rho * sx * sy
        cov = _np.array([[cxx, cxy], [cxy, cyy]])
        w, v = _np.linalg.eigh(cov)
        w = _np.maximum(w, 1e-9)
        order = w.argsort()[::-1]
        w = w[order]; v = v[:, order]
        ang = float(_np.degrees(_np.arctan2(v[1, 0], v[0, 0])))
        return float(2 * _np.sqrt(w[0])), float(2 * _np.sqrt(w[1])), ang

    for i in range(N):
        img_A = imgs_A[i].permute(1, 2, 0).numpy()
        img_B = imgs_B[i].permute(1, 2, 0).numpy()
        img_A = _np.clip(img_A * 255.0, 0, 255).astype(_np.uint8)
        img_B = _np.clip(img_B * 255.0, 0, 255).astype(_np.uint8)
        IMG_SIZE = img_A.shape[0]
        valid = (~padA[i].bool().numpy()) & (~padB[i].bool().numpy())
        uv_A_local = distA[i].numpy()[:, :2]
        uv_B_hat   = distB[i].numpy()[:, :2]
        uv_B_gt    = trueB[i].numpy()[:, :2]
        pp = per_pt[i].numpy()
        mu = pp[:, :2]; sx = _np.exp(pp[:, 2]); sy = _np.exp(pp[:, 3]); rho = pp[:, 4]
        uv_B_pred = uv_B_hat + mu

        in_A = ((uv_A_local[:, 0] >= 0) & (uv_A_local[:, 0] < IMG_SIZE) &
                (uv_A_local[:, 1] >= 0) & (uv_A_local[:, 1] < IMG_SIZE))
        in_B = ((uv_B_hat[:, 0] >= 0) & (uv_B_hat[:, 0] < IMG_SIZE) &
                (uv_B_hat[:, 1] >= 0) & (uv_B_hat[:, 1] < IMG_SIZE) &
                (uv_B_gt[:, 0]  >= 0) & (uv_B_gt[:, 0]  < IMG_SIZE) &
                (uv_B_gt[:, 1]  >= 0) & (uv_B_gt[:, 1]  < IMG_SIZE))
        ok = valid & in_A & in_B
        n_ok = int(ok.sum())
        if n_ok < 4:
            continue
        err_hyp = float(_np.linalg.norm(uv_B_hat[ok] - uv_B_gt[ok], axis=-1).mean())
        err_pred = float(_np.linalg.norm(uv_B_pred[ok] - uv_B_gt[ok], axis=-1).mean())

        cand = _np.where(ok)[0]
        step = max(1, len(cand) // k_show)
        sel = cand[::step][:k_show]

        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(11, 5.5))
        ax_A.imshow(img_A); ax_A.set_xticks([]); ax_A.set_yticks([])
        ax_B.imshow(img_B); ax_B.set_xticks([]); ax_B.set_yticks([])
        cmap = cm.get_cmap('tab10')
        for k, idx in enumerate(sel):
            c = cmap(k % 10)
            uA = uv_A_local[idx]; uH = uv_B_hat[idx]
            uG = uv_B_gt[idx];    uP = uv_B_pred[idx]
            ax_A.plot(uA[0], uA[1], 'o', color=c, markersize=10,
                       markeredgecolor='white', mew=1.2, zorder=5)
            ax_A.annotate(str(k), (uA[0], uA[1]), ha='center', va='center',
                           fontsize=7, color='black', weight='bold', zorder=6)
            w, h, ang = _cov_ellipse_axes(sx[idx], sy[idx], rho[idx])
            ax_B.plot(uH[0], uH[1], 'x', color=c, markersize=7, mew=1.2,
                       alpha=0.7, zorder=4)
            ax_B.plot(uG[0], uG[1], 'o', color=c, markersize=5,
                       markeredgecolor='white', mew=0.8, zorder=5)
            ax_B.plot(uP[0], uP[1], 'o', color=c, markersize=3,
                       markeredgecolor='white', mew=0.5, zorder=6)
            ax_B.add_patch(Ellipse((uP[0], uP[1]), width=w, height=h, angle=ang,
                                    facecolor=c, edgecolor='none',
                                    alpha=0.30, zorder=7))
            ax_B.add_patch(Ellipse((uP[0], uP[1]), width=w, height=h, angle=ang,
                                    facecolor='none', edgecolor=c,
                                    alpha=1.0, lw=1.5, zorder=8))
            fig.add_artist(ConnectionPatch(
                xyA=(uA[0], uA[1]), xyB=(uG[0], uG[1]),
                coordsA='data', coordsB='data',
                axesA=ax_A, axesB=ax_B,
                color=c, lw=0.6, ls=':', alpha=0.6, zorder=3))
        ax_B.text(2, IMG_SIZE - 4,
                   'x  HAT (model input)\n'
                   'o (large)  GT\n'
                   'o (small) + ellipse  pred ± 2σ',
                   fontsize=6, color='white', va='bottom',
                   bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none'))
        ax_A.set_title(f'A frame  N_ok={n_ok}', fontsize=9, loc='left')
        ax_B.set_title(f'B frame  hyp {err_hyp:.1f} → pred {err_pred:.2f} px',
                        fontsize=9, loc='left')
        pB = pert_B[i].numpy(); dp = dpose[i].numpy()
        fig.suptitle(
            f'CND2 pair ep{ep} {split} #{i}  '
            f'ε_pose t=({pB[0]:+.2f},{pB[1]:+.2f},{pB[2]:+.2f})m '
            f'ypr=({pB[3]:+.2f},{pB[4]:+.2f},{pB[5]:+.2f})°  '
            f'dpose_AB GT t=({dp[0]:+.2f},{dp[1]:+.2f},{dp[2]:+.2f})m '
            f'ypr=({dp[3]:+.2f},{dp[4]:+.2f},{dp[5]:+.2f})°',
            fontsize=8)
        fig.tight_layout()
        p = out_dir / f'ep{ep:03d}_{split}_{i}.png'
        fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig)
        paths.append(p)
    return paths


def epoch_loop_pair(model, loader, optimizer, accel: Accelerator, train: bool,
                    img_size: int, vis_capture: dict | None = None):
    """Cross-frame pair epoch.

    For each batch (dict from collate_pair): forward A's LiDAR Q through the
    shared block stack on KV_A, apply RoPE(R_AB) once, then continue on
    KV_B. Loss = NLL of predicted Δ vs (true_uv_in_B - dist_uv_in_B), where:
      - dist_uv = HAT projection of A's points in B image (= dist_uvd_B)
      - true_uv = GT projection of A's points in B image (= true_uvd_B)
    If `vis_capture` is given (dict, rank-0 only), the LAST batch's tensors
    are stashed into it for downstream report_image; nothing else changes.
    Both come from the dataset's pair builder which already aligns A↔B
    indices when same_frame_self_sup=True. For genuine cross-frame pairs we
    rely on the same-coordinate alignment that build_crop emits.
    """
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    total_nll_calib, total_mse_calib, n_calib = 0.0, 0.0, 0
    _t_start = time.time()
    _last_log_step = 0
    for batch in loader:
        A = batch['A']; B = batch['B']
        dpose_AB = batch['dpose_AB']                          # (B, 6) GT [tx,ty,tz, ypr_deg]
        # A side: image + LiDAR Q.
        imgs_A, _trueA, distA, padA, vfpA, buA, bvA = A[:7]
        imgs_A = imgs_A.float().div_(255.0)
        # B side: image + bucket KV. trueB / distB describe the SAME A points
        # under POSE_GT (target) and POSE_HAT (sampling anchor) on B image.
        imgs_B, trueB, distB, padB, vfpB, buB, bvB = B[:7]
        imgs_B = imgs_B.float().div_(255.0)
        pertB = B[7]                                          # (B, 8) [tx,ty,tz, ypr, dfx, dfy]
        # PoseEmb input = POSE_HAT_AB = POSE_GT_AB ⊕ ε.  ε is what the dataset
        # sampled in pertB[:, :6].  Compose as: ypr_HAT = ypr_GT + ypr_pert
        # (small-angle approx; exact composition is built dataset-side already
        # via POSE_HAT projection — we just need the network to know HAT).
        ypr_GT  = dpose_AB[..., 3:6]
        ypr_eps = pertB[..., 3:6].to(ypr_GT)
        ypr_HAT = ypr_GT + ypr_eps
        R_AB = _R_from_zyx_deg(ypr_HAT).to(imgs_A.device, dtype=imgs_A.dtype)
        # Translation hat: GT t_AB ⊕ ε (small-additive composition, matches
        # rotation HAT recipe). Routed to RoPEPoseEmb's translation_mlp (type-0
        # scalar chunk) so the block stack sees the full SE(3) hat, not just R.
        t_GT  = dpose_AB[..., 0:3]
        t_eps = pertB[..., 0:3].to(t_GT)
        t_HAT = (t_GT + t_eps).to(imgs_A.device, dtype=imgs_A.dtype)

        # CalibNet2 input on Q: A's [u, v, d, intensity] (drop is_obj col 3).
        point_in_A = torch.cat([distA[..., :3], distA[..., 4:5]], dim=-1)

        # Dispatch through model.forward(...) so DDP's grad-sync hook fires.
        # CalibNet2.forward routes to forward_cross_frame when mode='cross'
        # and now returns (out_A, out_B) where each is per_pt or (per_pt, W).
        out = model(
            imgs_A, point_in_A,
            mode='cross', image_B=imgs_B, R_AB=R_AB, t_AB=t_HAT,
            vfp=vfpA, vfp_B=vfpB,
            bucket_uvd=buA, bucket_valid=bvA,
            bucket_uvd_B=buB, bucket_valid_B=bvB,
            key_padding_mask=padA,
        )
        # forward_cross_frame returns (out_A, out_B). out_X is per_pt OR
        # (per_pt, W) when use_info_head=True. Older legacy single-frame
        # path returned just per_pt or (per_pt, W); guard both.
        if isinstance(out, tuple) and len(out) == 2 and (
                isinstance(out[0], tuple) or
                (torch.is_tensor(out[0]) and out[0].dim() == 3)):
            out_A, out_B = out
            per_pt_A = out_A[0] if isinstance(out_A, tuple) else out_A
            per_pt_B = out_B[0] if isinstance(out_B, tuple) else out_B
        else:
            per_pt_A = None
            per_pt_B = out[0] if isinstance(out, tuple) else out

        # Target B = δ between POSE_GT and POSE_HAT B-projection of A's points.
        # build_crop emits A and B in lex-deterministic sub_idx order; in
        # same_frame_self_sup mode the leading min(N_A, N_B) tokens describe
        # the same world points.  Crop to that prefix; mask via padA up to it.
        Nmin = min(per_pt_B.shape[1], trueB.shape[1])
        per_pt_c = per_pt_B[:, :Nmin]
        gt = trueB[:, :Nmin, :2] - distB[:, :Nmin, :2]
        valid = ~(padA[:, :Nmin] | padB[:, :Nmin])
        loss_pose = gaussian2d_nll(per_pt_c[valid], gt[valid])
        # Calib head loss: A-side ε_calib supervision when calib_pert is on.
        # gt_A = trueA - distA (= δ_calib-induced A-image shift). The mid-stack
        # readout already produced per_pt_A from Δ_cum after A blocks only.
        w_calib = float(getattr(epoch_loop_pair, '_w_calib', 0.0))
        w_pose  = float(getattr(epoch_loop_pair, '_w_pose',  1.0))
        if per_pt_A is not None and w_calib > 0.0:
            per_pt_A_c = per_pt_A[:, :Nmin]
            gt_A = _trueA[:, :Nmin, :2] - distA[:, :Nmin, :2]
            valid_A = ~padA[:, :Nmin]
            loss_calib = gaussian2d_nll(per_pt_A_c[valid_A], gt_A[valid_A])
            loss = w_pose * loss_pose + w_calib * loss_calib
        else:
            loss_calib = None
            loss = w_pose * loss_pose
        if train:
            optimizer.zero_grad(set_to_none=True)
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # pose-side metrics (always)
        with torch.no_grad():
            err = (per_pt_c[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            total_mse += err.mean().item()
        total_nll += loss_pose.item(); n += 1
        # calib-side metrics (only when ε_calib_A is enabled and pred_A exists)
        if loss_calib is not None:
            with torch.no_grad():
                err_A = (per_pt_A_c[valid_A][..., :2].float() - gt_A[valid_A]).norm(dim=-1)
                total_mse_calib += err_A.mean().item()
            total_nll_calib += loss_calib.item(); n_calib += 1
        # Stash the most recent batch (rank-0 only) for end-of-epoch
        # debug-sample rendering. Detach + cpu so we don't pin GPU memory.
        if vis_capture is not None and accel.is_main_process:
            with torch.no_grad():
                vis_capture['imgs_A']  = imgs_A.detach().cpu()
                vis_capture['imgs_B']  = imgs_B.detach().cpu()
                vis_capture['trueB']   = trueB[:, :Nmin].detach().cpu()
                vis_capture['distB']   = distB[:, :Nmin].detach().cpu()
                vis_capture['_trueA']  = _trueA[:, :Nmin].detach().cpu()
                vis_capture['distA']   = distA[:, :Nmin].detach().cpu()
                vis_capture['padA']    = padA[:, :Nmin].detach().cpu()
                vis_capture['padB']    = padB[:, :Nmin].detach().cpu()
                vis_capture['per_pt']  = per_pt_c.detach().float().cpu()
                vis_capture['per_pt_B'] = per_pt_c.detach().float().cpu()
                if per_pt_A is not None:
                    vis_capture['per_pt_A'] = per_pt_A[:, :Nmin].detach().float().cpu()
                else:
                    vis_capture['per_pt_A'] = None
                vis_capture['pert_B']  = pertB.detach().cpu()
                vis_capture['dpose']   = dpose_AB.detach().cpu()
                vis_capture['split']   = 'train' if train else 'val'
        if train and accel.is_main_process and (n - _last_log_step >= 25):
            _dt = time.time() - _t_start
            sps_per  = n * imgs_A.shape[0] / _dt if _dt > 0 else 0
            sps_glob = sps_per * accel.num_processes
            calib_str = (f"  calib={loss_calib.item():+.3f}"
                         if loss_calib is not None else "")
            print(f"  step {n}  pose={loss_pose.item():+.3f}{calib_str}  "
                  f"sps/rank={sps_per:.0f}  sps(global)={sps_glob:.0f}", flush=True)
            _last_log_step = n
    pose_metrics = (total_nll / max(n, 1), total_mse / max(n, 1))
    if n_calib > 0:
        calib_metrics = (total_nll_calib / n_calib, total_mse_calib / n_calib)
    else:
        calib_metrics = (float('nan'), float('nan'))
    return pose_metrics + calib_metrics  # (nll, mse, nll_calib, mse_calib)


def epoch_loop(model, loader, optimizer, accel: Accelerator, train: bool,
               img_size: int, vis_capture: dict | None = None):
    """Single-frame calib epoch.

    If `vis_capture` is given (dict, rank-0 only), the LAST batch's tensors
    are stashed into it for downstream debug-sample rendering — parallels
    the pair-mode `vis_capture` hook so calib runs also produce per-epoch
    PNG samples under <exp>/debug_samples/.
    """
    model.train(train)
    total_nll, total_mse, n = 0.0, 0.0, 0
    _t_start = time.time()
    _last_log_step = 0
    for batch in loader:
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
        # last tuple slot from PandaSetCalibDatasetFull calib mode is
        # pert_vec (8,); collate_full stacks to (B, 8). Optional — older
        # caches may not emit it, so guard.
        pert_vec_b = batch[7] if len(batch) > 7 else None
        imgs = imgs.float().div_(255.0)
        gt   = true_uvd[..., :2] - dist_uvd[..., :2]
        # CalibNet2.use_intensity is True by default; pass [u,v,d,intensity].
        # dist_uvd cols: [u, v, d, is_obj, intensity] → drop is_obj (col 3).
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
        out = model(imgs, point_in,
                    dpose_R=None, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                    key_padding_mask=pad_mask)
        # use_info_head=False here → out is per_pt only.
        per_pt = out[0] if isinstance(out, tuple) else out
        valid  = ~pad_mask
        nll_loss = gaussian2d_nll(per_pt[valid], gt[valid])
        if getattr(accel, '_ba_loss_mode', False):
            ba_l, ba_diag = _ba_pose_loss(per_pt, dist_uvd, pad_mask, batch,
                                          ba_iter=getattr(accel, '_ba_iter', 4),
                                          damping=getattr(accel, '_ba_damping', 1e-3))
            ba_weight = getattr(accel, '_ba_weight', 0.5)
            if ba_l is None:
                loss = nll_loss
            else:
                loss = (1.0 - ba_weight) * nll_loss + ba_weight * ba_l
        else:
            loss = nll_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            accel.backward(loss)
            accel.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            err = (per_pt[valid][..., :2].float() - gt[valid]).norm(dim=-1)
            total_mse += err.mean().item()
        total_nll += loss.item(); n += 1
        # Stash last batch (rank-0 only) for end-of-epoch render. Same
        # contract as epoch_loop_pair.vis_capture but with calib fields.
        if vis_capture is not None and accel.is_main_process:
            with torch.no_grad():
                vis_capture['imgs']     = imgs.detach().cpu()
                vis_capture['true_uvd'] = true_uvd.detach().cpu()
                vis_capture['dist_uvd'] = dist_uvd.detach().cpu()
                vis_capture['pad_mask'] = pad_mask.detach().cpu()
                vis_capture['per_pt']   = per_pt.detach().float().cpu()
                if pert_vec_b is not None:
                    vis_capture['pert_vec'] = pert_vec_b.detach().cpu()
                vis_capture['split']    = 'train' if train else 'val'
        if train and accel.is_main_process and (n - _last_log_step >= 25):
            _dt = time.time() - _t_start
            sps_per  = n * imgs.shape[0] / _dt if _dt > 0 else 0
            sps_glob = sps_per * accel.num_processes
            print(f"  step {n}  loss={loss.item():+.3f}  "
                  f"sps/rank={sps_per:.0f}  sps(global)={sps_glob:.0f}", flush=True)
            _last_log_step = n
    return (total_nll / max(n, 1), total_mse / max(n, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cache', required=True,
                   help='comma-separated v3-tiled cache path(s)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=10,
                   help='Run validation + per-epoch debug-sample render only '
                        'every N epochs. Always evals on ep1 and the last ep. '
                        'Default 10 — matches CND1 cadence.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-6)
    p.add_argument('--oversample', type=int, default=16)
    p.add_argument('--rot-deg', type=float, default=1.5)
    p.add_argument('--t-m', type=float, default=0.20)
    p.add_argument('--min-crop-px', type=int, default=256)
    p.add_argument('--max-crop-px', type=int, default=512)
    p.add_argument('--grid-n', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--ba-loss', action='store_true',
                   help='Use BA-based pose loss instead of per-point gaussian2d_nll')
    p.add_argument('--ba-iter', type=int, default=4)
    p.add_argument('--ba-damping', type=float, default=1e-3)
    p.add_argument('--ba-weight', type=float, default=0.5,
                   help='weight of BA loss in mixture; (1-w) * nll + w * ba')
    p.add_argument('--prefetch', type=int, default=4)
    p.add_argument('--n-iter', type=int, default=3)
    p.add_argument('--n-heads', type=int, default=4)
    # kick #3: per-layer KV schedule. Named presets only for now; arbitrary
    # config could be loaded from a yaml later. None = legacy 3-level shared-
    # block stack (kick #1, #2 default).
    p.add_argument('--kv-schedule', type=str, default='',
                   choices=['', 'kick3'],
                   help='Per-layer KV pyramid schedule. "kick3" = '
                        'L0:coarse+lidar np=4, L1:coarse+lidar np=4, '
                        'L2:fine+lidar np=4, L3:super_fine+lidar np=8 '
                        '(layer-specific weights, A/B共有). '
                        'Empty = legacy shared-block 3-level stack.')
    # Fourier-feature head (NeRF / Tancik 2020). Lifts MLP NTK out of its
    # low-frequency regime — useful when sub-pixel residuals are not
    # being expressed by the plain Linear(d, 5) head. n_freq=0 disables.
    p.add_argument('--fourier-head-n-freq', type=int, default=0,
                   help='Fourier features prepended to final_head input '
                        '(0 = disabled, NeRF default ~16).')
    p.add_argument('--fourier-head-scale', type=float, default=10.0,
                   help='σ of the random B matrix in Fourier features '
                        '(NeRF default ~10).')
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--use-info-head', action='store_true',
                   help='enable InfoHead2x2 (still ignored by NLL loss; for ckpt only)')
    p.add_argument('--u-band', type=str, default='',
                   help='comma-separated <cache_path>:<frac> overrides for u_band '
                        '(central horizontal pivot band). 0=disabled, 0.8=keep '
                        'central 80%. Example: '
                        '"/home/hfunaya/cache_v5/tss4_v3_full_iter2kb4_yaw3:0.8"')
    p.add_argument('--frame-stride', type=str, default='',
                   help='comma-separated <cache_path>:<int> per-cache frame_stride '
                        'overrides (sub-sample fnames). 1=keep all, 10=1/10. '
                        'Example: "/home/hfunaya/cache_v5/waymo_v3_full:10"')
    p.add_argument('--per-cache-oversample', type=str, default='',
                   help='comma-separated <cache_path>:<int> per-cache oversample '
                        'overrides. Falls back to --oversample. Example: '
                        '"/home/hfunaya/cache_v5/kamikado_v3_full:8"')
    p.add_argument('--per-cache-mode', type=str, default='',
                   help='comma-separated <cache_path>:<pair|calib> per-cache '
                        'training mode. When set, OVERRIDES --pair-mode and '
                        'turns the trainer into mixed mode: pair-mode caches '
                        'feed forward_cross_frame, calib caches feed the '
                        'single-frame head. Each step alternates one batch '
                        'from each enabled mode. Example: '
                        '"<ps>:pair,<ns>:pair,<wm>:pair,<kami>:calib,<woven>:calib,<tss4>:calib"')
    p.add_argument('--pair-stride', type=int, default=10,
                   help='In pair-mode, max |Δframe| sampled per anchor. The '
                        'dataset emits all (i_A, i_A+δ) pairs with δ ∈ '
                        '[-S..-1, +1..+S] within the same (scene, cam). '
                        'Default 10 (= ±1 sec at 10 Hz).')
    p.add_argument('--pair-mode', action='store_true',
                   help='cross-frame pair training. Dataset emits (A, B, '
                        'dpose_AB) tuples; trainer calls '
                        'CalibNet2.forward_cross_frame(A, B, R_AB) and '
                        'targets the GT projection of A points in B image.')
    p.add_argument('--pair-bidir', action='store_true',
                   help='in pair-mode, also emit B->A swapped sample for the '
                        'same frame pair (doubles dataset throughput).')
    p.add_argument('--point-mlp-fourier-n-freq', type=int, default=0,
                   help='NeRF-style Fourier feature lift on PointMLP3 input '
                        '(uvd). 0 = off (default), 8/10 typical for sub-pixel '
                        'frequency capacity.')
    p.add_argument('--calib-pert', action='store_true',
                   help='enable ε_calib on frame_A inputs (build_pair side). '
                        'Activates the multi-task head: A blocks supervised '
                        'with calib NLL via mid-stack readout, B blocks '
                        'supervised with pose NLL as before. ε_calib_A is '
                        'sampled INDEPENDENTLY per frame (no leakage to B).')
    p.add_argument('--w-calib', type=float, default=1.0,
                   help='weight on the A-side calib NLL when --calib-pert is '
                        'on. 0 disables the head; ignored when --calib-pert '
                        'is off (loss_calib stays None).')
    p.add_argument('--w-pose', type=float, default=1.0,
                   help='weight on the B-side pose NLL.  Default 1.0; tune '
                        'with --w-calib for multi-task balance.')
    p.add_argument('--clearml', action='store_true',
                   help='register a ClearML Task with cfg + why + git context')
    p.add_argument('--clearml-project', type=str, default='e2e_calib/calib',
                   help='ClearML project namespace')
    p.add_argument('--why', type=str, default='',
                   help='WHY blob: rationale, hypothesis, expected outcome '
                        '(stored in ClearML task comment, free-form prose).')
    args = p.parse_args()

    # find_unused_parameters=True: in pair_mode cross-frame path the legacy
    # entry-time pose_emb / info_head / etc. don't see gradients in some
    # batches; without this DDP raises a "reduction not finished" error.
    accel = Accelerator(kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True),
    ])
    set_seed(args.split_seed + accel.process_index)
    # Stash BA flags on accel so epoch_loop can read without arg-thread
    accel._ba_loss_mode = bool(args.ba_loss)
    accel._ba_iter      = int(args.ba_iter)
    accel._ba_damping   = float(args.ba_damping)
    accel._ba_weight    = float(args.ba_weight)

    # ClearML init (rank-0 only, BEFORE main loop so cml_logger lookups work)
    if args.clearml and int(os.environ.get('RANK', '0')) == 0:
        try:
            from scripts.util.clearml_context import init_with_context
            init_with_context(
                project=args.clearml_project, name=args.name,
                cfg={k: v for k, v in vars(args).items()
                     if not callable(v) and not k.startswith('_')},
                why=args.why,
                baseline={'name': 'cnd2_km_os16_50ep_dgx2_9gpu',
                          'metric': 'val_nll', 'value': 2.4666})
        except Exception as _e:
            print(f'[clearml init failed] {_e}', flush=True)

    exp_dir = Path("experiments") / args.name
    if accel.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
        log_path.write_text("")
        # Save full config (every CLI arg) for reproducibility
        (exp_dir / "config.py").write_text(
            "# auto-generated by train_cnd2_ddp.py at run start\n"
            "CFG = dict(\n" +
            "".join(f"    {k:<20}= {v!r},\n"
                    for k, v in vars(args).items()) + ")\n")
    accel.wait_for_everyone()

    def log(msg):
        if not accel.is_main_process: return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        print(line, flush=True)
        with open(exp_dir / "train.log", "a") as f: f.write(line + "\n")

    log(f"accelerate: num_processes={accel.num_processes} "
        f"mixed_precision={accel.mixed_precision} device={accel.device}")

    # --- dataset(s) ---
    # Note: pair_mode is set per-cache below (mixed mode), so it's NOT in
    # the shared ds_kw. Each cache passes its own pair_mode via cp_kw.
    ds_kw = dict(img_size=args.img_size,
                 max_offset_m=args.t_m, max_rot_deg=args.rot_deg,
                 min_crop_px=args.min_crop_px, max_crop_px=args.max_crop_px,
                 grid_n=args.grid_n, oversample=args.oversample,
                 split_pert=False,
                 pair_stride=int(args.pair_stride),
                 pair_bidir=bool(args.pair_bidir),
                 calib_pert=bool(args.calib_pert))
    cache_paths = [s.strip() for s in args.cache.split(',') if s.strip()]
    # Per-cache mode override (mixed pair+calib training)
    mode_map: dict[str, str] = {}
    if getattr(args, 'per_cache_mode', ''):
        for tok in args.per_cache_mode.split(','):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.rsplit(':', 1)
            v = v.strip().lower()
            assert v in ('pair', 'calib'), f"per-cache-mode must be pair|calib, got {v!r}"
            mode_map[k.strip()] = v
    mixed_mode = bool(mode_map)
    # Per-cache u_band override
    ub_map = {}
    if getattr(args, 'u_band', ''):
        for tok in args.u_band.split(','):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.rsplit(':', 1)
            ub_map[k.strip()] = float(v)
    # Per-cache oversample override
    os_map = {}
    if getattr(args, 'per_cache_oversample', ''):
        for tok in args.per_cache_oversample.split(','):
            tok = tok.strip()
            if not tok: continue
            k, v = tok.rsplit(':', 1)
            os_map[k.strip()] = int(v)
    tr_parts, va_parts = [], []
    tr_modes:  list[str] = []
    va_modes:  list[str] = []
    for cp in cache_paths:
        ub = ub_map.get(cp, 0.0)
        os_i = os_map.get(cp, args.oversample)
        # In mixed mode, override pair_mode per cache. Outside mixed mode,
        # all caches share args.pair_mode (legacy behaviour).
        if mixed_mode:
            cp_mode = mode_map.get(cp, 'calib')
            cp_pair = (cp_mode == 'pair')
        else:
            cp_mode = 'pair' if args.pair_mode else 'calib'
            cp_pair = bool(args.pair_mode)
        cp_kw = {**ds_kw, 'oversample': os_i, 'pair_mode': cp_pair}
        tr = PandaSetCalibDatasetFull(cp, split='train', u_band=ub, **cp_kw)
        va = PandaSetCalibDatasetFull(cp, split='val',
                                       center_band=0.5, u_band=ub, **cp_kw)
        log(f"  [{cp}] train={len(tr)} val={len(va)} (os={os_i}) u_band={ub} mode={cp_mode}")
        tr_parts.append(tr); va_parts.append(va)
        tr_modes.append(cp_mode); va_modes.append(cp_mode)
    if mixed_mode:
        # Build separate ConcatDatasets / loaders per mode (pair vs calib)
        # since their collate fns + tuple shapes differ. Each mode's loader
        # serves one batch per step in the alternating epoch loop below.
        pair_tr   = [tr for tr, m in zip(tr_parts, tr_modes) if m == 'pair']
        calib_tr  = [tr for tr, m in zip(tr_parts, tr_modes) if m == 'calib']
        pair_va   = [va for va, m in zip(va_parts, va_modes) if m == 'pair']
        calib_va  = [va for va, m in zip(va_parts, va_modes) if m == 'calib']
        log(f"mixed mode: pair caches={len(pair_tr)} calib caches={len(calib_tr)}")
        tr_full   = (ConcatDataset(pair_tr)  if len(pair_tr)  > 1 else (pair_tr[0]  if pair_tr  else None))
        tr_full_c = (ConcatDataset(calib_tr) if len(calib_tr) > 1 else (calib_tr[0] if calib_tr else None))
        va_full   = (ConcatDataset(pair_va)  if len(pair_va)  > 1 else (pair_va[0]  if pair_va  else None))
        va_full_c = (ConcatDataset(calib_va) if len(calib_va) > 1 else (calib_va[0] if calib_va else None))
        # train_ds / val_ds keep the pair side (= primary loop driver).
        # The calib side rides alongside via train_loader_calib / val_loader_calib.
        train_ds = tr_full
        val_ds   = va_full
        train_ds_calib = tr_full_c
        val_ds_calib   = va_full_c
    else:
        tr_full = ConcatDataset(tr_parts) if len(tr_parts) > 1 else tr_parts[0]
        va_full = ConcatDataset(va_parts) if len(va_parts) > 1 else va_parts[0]
        full_ds = ConcatDataset([tr_full, va_full])
        # __len__ = frame count (oversample handled in __getitem__).
        idxs = list(range(len(full_ds)))
        _r.Random(args.split_seed).shuffle(idxs)
        n_val = int(len(idxs) * args.val_fraction)
        val_idxs, train_idxs = idxs[:n_val], idxs[n_val:]
        train_ds = Subset(full_ds, train_idxs)
        val_ds   = Subset(full_ds, val_idxs)
        train_ds_calib = None
        val_ds_calib   = None
        log(f"frame-level split: train={len(train_ds)} val={len(val_ds)} frames")

    nw = args.workers
    val_nw = min(4, nw)
    def _kw(coll):
        return dict(num_workers=nw, pin_memory=True,
                    persistent_workers=(nw > 0),
                    prefetch_factor=args.prefetch if nw > 0 else None,
                    collate_fn=coll,
                    multiprocessing_context='spawn' if nw > 0 else None)
    def _val_kw(coll):
        return dict(num_workers=val_nw, pin_memory=True,
                    persistent_workers=(val_nw > 0),
                    prefetch_factor=args.prefetch if val_nw > 0 else None,
                    collate_fn=coll,
                    multiprocessing_context='spawn' if val_nw > 0 else None)
    # primary (pair-mode if mixed_mode + has pair caches, else legacy single-mode)
    primary_collate = collate_pair if (mixed_mode and train_ds is not None) or args.pair_mode else collate_full
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **_kw(primary_collate)) if train_ds is not None else None
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **_val_kw(primary_collate)) if val_ds is not None else None
    # secondary (calib side in mixed mode only)
    train_loader_calib = None; val_loader_calib = None
    if mixed_mode and train_ds_calib is not None:
        train_loader_calib = DataLoader(train_ds_calib, batch_size=args.batch_size,
                                         shuffle=True, **_kw(collate_full))
        val_loader_calib   = DataLoader(val_ds_calib, batch_size=args.batch_size,
                                         shuffle=False, **_val_kw(collate_full))

    # --- model ---
    KV_SCHEDULES = {
        'kick3': [
            {'image': 'coarse',     'lidar': True, 'n_points': 4},
            {'image': 'coarse',     'lidar': True, 'n_points': 4},
            {'image': 'fine',       'lidar': True, 'n_points': 4},
            {'image': 'super_fine', 'lidar': True, 'n_points': 8},
        ],
    }
    kv_schedule = KV_SCHEDULES.get(args.kv_schedule) if args.kv_schedule else None
    model = CalibNet2(d=128, img_size=args.img_size, in_channels=3,
                      use_intensity=True, frustum_grid_n=args.grid_n,
                      n_iter=args.n_iter, n_heads=args.n_heads,
                      d_scalar=8, n_type1=40,
                      kv_schedule=kv_schedule,
                      fourier_head_n_freq=int(args.fourier_head_n_freq),
                      fourier_head_scale=float(args.fourier_head_scale),
                      point_mlp_fourier_n_freq=int(args.point_mlp_fourier_n_freq),
                      use_info_head=args.use_info_head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    if mixed_mode and train_loader_calib is not None:
        (model, optimizer, train_loader, val_loader,
         train_loader_calib, val_loader_calib) = accel.prepare(
            model, optimizer, train_loader, val_loader,
            train_loader_calib, val_loader_calib)
    else:
        model, optimizer, train_loader, val_loader = accel.prepare(
            model, optimizer, train_loader, val_loader)

    log(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
        f"(world_size={accel.num_processes}, bs/rank={args.batch_size}, "
        f"global_bs={args.batch_size*accel.num_processes})")

    epochs   = args.epochs
    lr_min_r = args.lr_min / args.lr
    def lr_lambda(e):
        if e < 5: return (e + 1) / 5
        t = (e - 5) / max(1, epochs - 5)
        return lr_min_r + (1 - lr_min_r) * 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val = float("inf")
    ckpt = exp_dir / "best_model.pt"
    t0 = time.time()

    # ClearML logger handle (rank-0 only)
    cml_logger = None
    cml_task = None
    if args.clearml and accel.is_main_process:
        try:
            from clearml import Task as _ClearMLTask
            cml_task = _ClearMLTask.current_task()
            if cml_task is not None:
                cml_logger = cml_task.get_logger()
        except Exception as _e:
            log(f"[clearml] logger init failed: {_e}")

    # Primary epoch fn: pair if mixed has pair caches, else legacy choice.
    primary_is_pair = (mixed_mode and train_loader is not None) or args.pair_mode
    _epoch_fn = epoch_loop_pair if primary_is_pair else epoch_loop
    # Mixed-mode: secondary fn drives the calib-only caches with the
    # single-frame `epoch_loop`, alternating with the pair loop. The model
    # parameters are shared so both updates are accumulated on the same net.
    _epoch_fn_calib = epoch_loop if mixed_mode and train_loader_calib is not None else None
    # Stash multi-task weights on the function object so epoch_loop_pair
    # can read them without threading another arg through every call site.
    # When --calib-pert is off, w_calib is forced to 0 so the head is dead.
    epoch_loop_pair._w_calib = float(args.w_calib) if args.calib_pert else 0.0
    epoch_loop_pair._w_pose  = float(args.w_pose)
    # Vis stash — populated by epoch_loop_{pair,(calib)} on the last batch
    # when not None. Allocated on rank-0 in either mode so debug samples
    # land under <exp>/debug_samples/ for both pair and calib runs.
    vis_cap_train: dict | None = {} if accel.is_main_process else None
    vis_cap_val:   dict | None = {} if accel.is_main_process else None
    vis_dir = exp_dir / 'debug_samples'
    vis_dir.mkdir(parents=True, exist_ok=True) if accel.is_main_process else None

    # ── PRE-FLIGHT debug samples (rank-0): render N samples directly from
    # the train DataLoader BEFORE training starts so we can sanity-check
    # dataset emissions without waiting for an epoch. Pred panel uses the
    # untrained model (zero-init head → pred ≈ dist + bias).
    # Preflight = dataset emission sanity only. NO model forward; the
    # untrained head's output is bias-only and conveys nothing useful, and
    # running model forward on rank 0 alone triggers a DDP allreduce that
    # ranks 1-N don't issue → 10-min NCCL watchdog timeout (cf. task
    # 4adad157, 2026-06-02). The same render is uploaded to title
    # vis_check_pair_getitem so the training task surfaces the A↔B
    # correspondence sanity directly.
    # CRITICAL: pull samples directly from the underlying train_ds (CPU
    # Subset/ConcatDataset), NOT from the prepared train_loader. Going via
    # train_loader (a) advances the DistributedSampler so other ranks see
    # shifted batches, and (b) accelerate's send_to_device hook moves
    # tensors onto cuda:0, breaking .numpy() inside render_one_pair.
    if args.pair_mode and accel.is_main_process:
        try:
            log("preflight: vis_check pair_getitem (dataset emission only)")
            from scripts.vis_check._pair_render import render_one_pair
            n_check = 6
            for ds_label, ds in (('train', train_ds), ('val', val_ds)):
                for i in range(n_check):
                    sample = ds[i % len(ds)]
                    if isinstance(sample, list):
                        if not sample:
                            continue
                        sample = sample[0]
                    built_A_t, built_B_t, dpose_AB_t = sample
                    # Preserve the dataset's pert_vec_A (slot 6); zero when
                    # --calib-pert is off, ε_calib_A when on.
                    built_A_i = built_A_t[:7]
                    built_B_i = built_B_t[:7]
                    out_p = vis_dir / f'vis_check_pair_{ds_label}_{i:02d}.png'
                    res = render_one_pair(built_A_i, built_B_i, dpose_AB_t,
                                           out_path=out_p,
                                           img_size=args.img_size,
                                           k_show=12,
                                           suptitle_prefix=f'vis_check {ds_label} #{i}  ')
                    if res and cml_logger is not None:
                        cml_logger.report_image(
                            title=f'vis_check_pair_getitem_{ds_label}',
                            series=f'sample_{i:02d}',
                            iteration=0, local_path=str(res[0]))
                    if res:
                        log(f"  vis_check[{ds_label}] {res[0]}  "
                            f"N_ok={res[1]['n_ok']}  "
                            f"HAT→GT={res[1]['err_hyp']:.1f}px")
        except Exception as _e:
            log(f"  ↳ preflight vis_check skipped: {_e}")
    accel.wait_for_everyone()
    for ep in range(epochs):
        ep_t = time.time()
        # epoch_loop_pair returns 4-tuple (nll, mse, nll_calib, mse_calib);
        # epoch_loop returns 2-tuple. Unpack with a length guard so this runs
        # in both modes without changing the legacy single-frame epoch_loop.
        # vis_capture is passed in BOTH modes (525f130) so calib path also
        # emits debug_samples/ on rank-0.
        # eval/vis stride: run validation + per-epoch debug-sample render
        # only every N epochs (default 10). On other epochs, only training
        # runs and val_* / *_calib metrics are reported as NaN. The legacy
        # CND1 loop did the same — full eval is expensive (val loader pulls
        # tiles + decodes JPEG every epoch) and 50ep × 10ep stride still
        # gives 5 eval points per run.
        eval_stride = int(getattr(args, 'eval_every', 10))
        do_eval = (ep + 1) % eval_stride == 0 or (ep + 1) == epochs or ep == 0
        tr_out = _epoch_fn(model, train_loader, optimizer, accel, True,
                            args.img_size, vis_capture=vis_cap_train if do_eval else None)
        # Mixed-mode: also run the calib-only secondary epoch (alternates
        # parameter updates with the primary pair epoch, both share model).
        tr_out_calib = None
        va_out_calib = None
        if _epoch_fn_calib is not None and train_loader_calib is not None:
            tr_out_calib = _epoch_fn_calib(model, train_loader_calib, optimizer,
                                            accel, True, args.img_size)
        if do_eval:
            with torch.no_grad():
                va_out = _epoch_fn(model, val_loader, optimizer, accel, False,
                                    args.img_size, vis_capture=vis_cap_val)
                if _epoch_fn_calib is not None and val_loader_calib is not None:
                    va_out_calib = _epoch_fn_calib(model, val_loader_calib, optimizer,
                                                    accel, False, args.img_size)
        else:
            va_out = (float('nan'),) * len(tr_out)
        if len(tr_out) == 4:
            tr_nll, tr_mse, tr_nll_cal, tr_mse_cal = tr_out
            va_nll, va_mse, va_nll_cal, va_mse_cal = va_out
        else:
            tr_nll, tr_mse = tr_out; va_nll, va_mse = va_out
            tr_nll_cal = tr_mse_cal = va_nll_cal = va_mse_cal = float('nan')
        scheduler.step()
        if accel.is_main_process:
            elapsed = time.time() - t0
            calib_str = ""
            if tr_nll_cal == tr_nll_cal:    # not NaN
                calib_str = (f"  tr_calib_nll={tr_nll_cal:.4f} tr_calib_mse={tr_mse_cal:.3f}"
                             f"  va_calib_nll={va_nll_cal:.4f} va_calib_mse={va_mse_cal:.3f}")
            mixed_str = ""
            if tr_out_calib is not None:
                mixed_str = (f"  calibOnly tr_nll={tr_out_calib[0]:.3f} tr_mse={tr_out_calib[1]:.2f}")
                if va_out_calib is not None:
                    mixed_str += f"  va_nll={va_out_calib[0]:.3f} va_mse={va_out_calib[1]:.2f}"
            log(f"ep{ep+1:03d}/{epochs}  "
                f"tr_nll={tr_nll:.4f} tr_mse={tr_mse:.3f}  "
                f"va_nll={va_nll:.4f} va_mse={va_mse:.3f}{calib_str}{mixed_str}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"ep_t={time.time()-ep_t:.1f}s  total={elapsed/60:.1f}min")
            if cml_logger is not None:
                rs = cml_logger.report_scalar
                rs(title='loss/nll', series='train', value=tr_nll, iteration=ep+1)
                rs(title='loss/mse', series='train', value=tr_mse, iteration=ep+1)
                rs(title='lr', series='lr',
                   value=scheduler.get_last_lr()[0], iteration=ep+1)
                if do_eval:
                    rs(title='loss/nll', series='val',   value=va_nll, iteration=ep+1)
                    rs(title='loss/mse', series='val',   value=va_mse, iteration=ep+1)
                # Mixed-mode: report calib-side scalars in their own series
                # so the pair (pose) and calib-only sides are visible side
                # by side in ClearML.
                if tr_out_calib is not None:
                    rs(title='loss/nll_calib_only', series='train',
                        value=tr_out_calib[0], iteration=ep+1)
                    rs(title='loss/mse_calib_only', series='train',
                        value=tr_out_calib[1], iteration=ep+1)
                if do_eval and va_out_calib is not None:
                    rs(title='loss/nll_calib_only', series='val',
                        value=va_out_calib[0], iteration=ep+1)
                    rs(title='loss/mse_calib_only', series='val',
                        value=va_out_calib[1], iteration=ep+1)
                if tr_nll_cal == tr_nll_cal:
                    rs(title='loss/nll_calib', series='train', value=tr_nll_cal, iteration=ep+1)
                    rs(title='loss/mse_calib', series='train', value=tr_mse_cal, iteration=ep+1)
                    if do_eval and va_nll_cal == va_nll_cal:
                        rs(title='loss/nll_calib', series='val', value=va_nll_cal, iteration=ep+1)
                        rs(title='loss/mse_calib', series='val', value=va_mse_cal, iteration=ep+1)
            # Pair-mode debug samples: render the SAME 16 fixed samples each
            # eval epoch (8 train + 8 val) so a person can watch convergence
            # on consistent panels. Only on do_eval epochs (every N).
            if args.pair_mode and accel.is_main_process and do_eval:
                try:
                    from scripts.vis_check._pair_render import render_one_pair
                    n_per_split = 8
                    _m = accel.unwrap_model(model)
                    was_train = _m.training
                    _m.eval()
                    for ds_label, ds in (('train', train_ds), ('val', val_ds)):
                        for i in range(n_per_split):
                            try:
                                sample = ds[i % len(ds)]
                            except Exception:
                                continue
                            if isinstance(sample, list):
                                if not sample: continue
                                sample = sample[0]
                            built_A_t, built_B_t, dpose_AB_t = sample
                            # Forward 1-batch through model on rank-0 only.
                            dev = accel.device
                            img_A_b = built_A_t[0].unsqueeze(0).float().div(255.0).to(dev)
                            img_B_b = built_B_t[0].unsqueeze(0).float().div(255.0).to(dev)
                            distA_b = built_A_t[2].unsqueeze(0).to(dev)
                            buA_b   = built_A_t[4].unsqueeze(0).to(dev)
                            bvA_b   = built_A_t[5].unsqueeze(0).to(dev)
                            buB_b   = built_B_t[4].unsqueeze(0).to(dev)
                            bvB_b   = built_B_t[5].unsqueeze(0).to(dev)
                            vfpA_b  = built_A_t[3].unsqueeze(0).to(dev)
                            vfpB_b  = built_B_t[3].unsqueeze(0).to(dev)
                            pertB_b = built_B_t[6].unsqueeze(0)
                            dpose_b = dpose_AB_t.unsqueeze(0)
                            ypr_HAT = (dpose_b[..., 3:6] + pertB_b[..., 3:6]).to(dev,
                                          dtype=img_A_b.dtype)
                            R_AB_b = _R_from_zyx_deg(ypr_HAT)
                            t_HAT_b = (dpose_b[..., 0:3]
                                        + pertB_b[..., 0:3]).to(dev, dtype=img_A_b.dtype)
                            point_in_A_b = torch.cat(
                                [distA_b[..., :3], distA_b[..., 4:5]], dim=-1)
                            with torch.no_grad():
                                out_b = _m(img_A_b, point_in_A_b,
                                           mode='cross', image_B=img_B_b,
                                           R_AB=R_AB_b, t_AB=t_HAT_b,
                                           vfp=vfpA_b, vfp_B=vfpB_b,
                                           bucket_uvd=buA_b, bucket_valid=bvA_b,
                                           bucket_uvd_B=buB_b, bucket_valid_B=bvB_b,
                                           key_padding_mask=None)
                            # forward_cross_frame returns (out_A, out_B); each
                            # is per_pt or (per_pt, W) when use_info_head=True.
                            if (isinstance(out_b, tuple) and len(out_b) == 2 and
                                    (isinstance(out_b[0], tuple) or
                                     (torch.is_tensor(out_b[0])
                                      and out_b[0].dim() == 3))):
                                out_b_A, out_b_B = out_b
                                pp_b_A = out_b_A[0] if isinstance(out_b_A, tuple) else out_b_A
                                pp_b_B = out_b_B[0] if isinstance(out_b_B, tuple) else out_b_B
                            else:
                                pp_b_A = None
                                pp_b_B = out_b[0] if isinstance(out_b, tuple) else out_b
                            per_pt_np   = pp_b_B[0].detach().float().cpu().numpy()
                            per_pt_np_A = (pp_b_A[0].detach().float().cpu().numpy()
                                            if pp_b_A is not None else None)
                            # built_A_i preserves the dataset's pert_vec_A
                            # (zeros when --calib-pert is off, ε_calib when on);
                            # the renderer uses it to label ε_calib in the
                            # suptitle when nonzero.
                            built_A_i = built_A_t[:7]
                            built_B_i = built_B_t[:7]
                            out_p = vis_dir / f'pair_debug_{ds_label}_ep{ep+1:03d}_{i:02d}.png'
                            res = render_one_pair(
                                built_A_i, built_B_i, dpose_AB_t,
                                out_path=out_p, img_size=args.img_size,
                                k_show=12,
                                suptitle_prefix=f'ep{ep+1} {ds_label} #{i}  ',
                                pred_per_pt=per_pt_np,
                                pred_per_pt_A=per_pt_np_A)
                            if res and cml_logger is not None:
                                cml_logger.report_image(
                                    title=f'pair_debug_{ds_label}',
                                    series=f'sample_{i:02d}',
                                    iteration=ep + 1, local_path=str(res[0]))
                    if was_train:
                        _m.train()
                except Exception as _e:
                    log(f"  ↳ pair debug-sample render skipped: {_e}")
            # Calib-mode debug samples: take the last batch's stash and
            # render N samples per split. Pure CPU; no model forward (the
            # epoch_loop already produced per_pt).
            if (not args.pair_mode) and accel.is_main_process and do_eval:
                # Per-cache calib debug samples: pull 4 fixed samples from each
                # tr_parts[k] / va_parts[k] (NOT from the shuffled ConcatDataset
                # batch that vis_cap stashes — that mixes datasets and you
                # can't tell which one is doing well). One forward per sample
                # on rank-0 (CPU dataset → cuda:0 model).
                try:
                    from scripts.vis_check._calib_render import render_one_calib
                    n_per_cache = 4
                    _m = accel.unwrap_model(model); was_train = _m.training; _m.eval()
                    for parts, ds_label in ((tr_parts, 'train'),
                                              (va_parts, 'val')):
                        for k, ds_k in enumerate(parts):
                            cache_name = Path(cache_paths[k]).name
                            for i in range(n_per_cache):
                                try:
                                    sample = ds_k[i % len(ds_k)]
                                except Exception:
                                    continue
                                if isinstance(sample, list):
                                    if not sample: continue
                                    sample = sample[0]
                                # legacy single-frame tuple layout (from collate_full):
                                #   (img, true_uvd, dist_uvd, vfp, bucket_uvd,
                                #    bucket_valid, pert_vec, ...)
                                img_t      = sample[0].unsqueeze(0).float().div(255.0).to(accel.device)
                                true_uvd_t = sample[1]
                                dist_uvd_t = sample[2]
                                vfp_t      = sample[3].unsqueeze(0).to(accel.device)
                                bu_t       = sample[4].unsqueeze(0).to(accel.device)
                                bv_t       = sample[5].unsqueeze(0).to(accel.device)
                                pert_v     = sample[6] if len(sample) > 6 else None
                                point_in   = torch.cat(
                                    [dist_uvd_t[..., :3], dist_uvd_t[..., 4:5]], dim=-1
                                ).unsqueeze(0).to(accel.device)
                                pad_t      = torch.zeros(1, dist_uvd_t.shape[0],
                                                          dtype=torch.bool,
                                                          device=accel.device)
                                with torch.no_grad():
                                    out_t = _m(img_t, point_in,
                                                dpose_R=None, vfp=vfp_t,
                                                bucket_uvd=bu_t, bucket_valid=bv_t,
                                                key_padding_mask=pad_t)
                                per_pt = out_t[0] if isinstance(out_t, tuple) else out_t
                                per_pt_np = per_pt[0].detach().float().cpu().numpy()
                                out_p = vis_dir / (
                                    f'calib_debug_{ds_label}_{cache_name}_'
                                    f'ep{ep+1:03d}_{i:02d}.png')
                                render_one_calib(
                                    img_t[0].cpu() * 255.0,
                                    true_uvd_t,
                                    dist_uvd_t,
                                    per_pt_np[:, :2] if per_pt_np.ndim > 1 else per_pt_np,
                                    out_path=out_p,
                                    img_size=args.img_size,
                                    pert_vec=pert_v,
                                    title_prefix=f'ep{ep+1} {ds_label} {cache_name} #{i}',
                                )
                                if cml_logger is not None:
                                    cml_logger.report_image(
                                        title=f'calib_debug_{cache_name}_{ds_label}',
                                        series=f'sample_{i:02d}',
                                        iteration=ep + 1, local_path=str(out_p))
                    if was_train: _m.train()
                except Exception as _e:
                    log(f"  ↳ calib debug-sample render skipped: {_e}")
            if va_nll < best_val:
                best_val = va_nll
                sd_cpu = {k: v.detach().cpu() for k, v in
                          accel.unwrap_model(model).state_dict().items()}
                accel.save(sd_cpu, ckpt)
                # belt-and-braces: under some ClearML + accelerate combos
                # accel.save was observed to silently no-op (file never
                # appears on disk despite log line firing). Verify + fallback
                # so we never lose a full training run to that interaction.
                if (not ckpt.exists()) or ckpt.stat().st_size < 1024:
                    torch.save(sd_cpu, ckpt)
                log(f"  ↳ best val_nll={best_val:.4f}  saved {ckpt} "
                    f"({ckpt.stat().st_size/1e6:.2f} MB)")
                # Upload best.pt as ClearML OutputModel + pair config.py
                if cml_task is not None:
                    try:
                        from clearml import OutputModel as _OM
                        _om = _OM(task=cml_task, name=f'{args.name}_best')
                        _om.update_weights(weights_filename=str(ckpt))
                        cfg_p = exp_dir / 'config.py'
                        if cfg_p.exists():
                            cml_task.upload_artifact('config.py',
                                                     artifact_object=str(cfg_p),
                                                     auto_pickle=False)
                    except Exception as _e:
                        log(f"  ↳ ClearML upload skipped: {_e}")
        accel.wait_for_everyone()


if __name__ == '__main__':
    main()
