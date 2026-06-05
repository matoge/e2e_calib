"""Shared render for pair-mode vis_check.

Used by both:
  * scripts/vis_check/pair_getitem.py  (standalone vis_check ClearML task)
  * datasets/train_cnd2_ddp.py preflight  (trainer-side hook)

Input: a single (built_A, built_B, dpose_AB) sample from the dataset (NOT
collated). Output: list of PNG paths, one per drawn sample (here only one,
the function is meant to be called per-sample).

The figure is the cross-frame "what does the network see" panel:
  Left  (A): img_A + numbered ○ at true_uvd_A  (POSE_GT_A projection)
                     × at dist_uvd_A          (POSE_GT_A + ε_calib projection,
                                                = same as true_A when calib pert
                                                isn't applied to A)
  Right (B): img_B + numbered ○ at true_uvd_B  (POSE_GT_AB projection of pts_A)
                     × at dist_uvd_B           (POSE_GT_AB + ε_pose projection)
  ConnectionPatch dotted lines tie A's i-th point to B's i-th GT point.

dist_uvd_B never enters the network — it's only used as the loss reference
(target_for_loss = true_uvd_B − dist_uvd_B) and for this visualization.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import ConnectionPatch, Ellipse


def render_one_pair(built_A, built_B, dpose_AB,
                     out_path: Path, *,
                     img_size: int = 128,
                     k_show: int = 12,
                     suptitle_prefix: str = '',
                     pred_per_pt: 'np.ndarray | None' = None,
                     pred_per_pt_A: 'np.ndarray | None' = None,
                     ) -> tuple[Path, dict] | None:
    """Render one (A, B, dpose_AB) sample. Returns (path, stats) or None
    when the sample has too few in-image points.

    Tensors are accepted as torch.Tensor or numpy.ndarray.

    pred_per_pt (optional):    (N, 5) — B-side pred (pose-error supervision).
    pred_per_pt_A (optional):  (N, 5) — A-side pred from the mid-stack
        readout in forward_cross_frame (calib-error supervision). Drawn as
        ○pred + 2σ ellipse on the A panel; reports err_pred_A in title.
    """
    def _np(x):
        return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)

    img_A      = _np(built_A[0]).transpose(1, 2, 0).astype(np.uint8)
    true_uvd_A = _np(built_A[1])
    dist_uvd_A = _np(built_A[2])
    img_B      = _np(built_B[0]).transpose(1, 2, 0).astype(np.uint8)
    true_uvd_B = _np(built_B[1])
    dist_uvd_B = _np(built_B[2])
    pert_B     = _np(built_B[6])  # collate-pre layout: [6] is pert_vec
    pert_A     = _np(built_A[6])  # ε_calib_A vec when calib_pert is on
    dpose      = _np(dpose_AB)
    pp_np   = _np(pred_per_pt)   if pred_per_pt   is not None else None
    pp_np_A = _np(pred_per_pt_A) if pred_per_pt_A is not None else None

    Nmin = min(true_uvd_A.shape[0], true_uvd_B.shape[0])
    in_A = ((true_uvd_A[:Nmin, 0] >= 0) & (true_uvd_A[:Nmin, 0] < img_size) &
            (true_uvd_A[:Nmin, 1] >= 0) & (true_uvd_A[:Nmin, 1] < img_size))
    in_B = ((true_uvd_B[:Nmin, 0] >= 0) & (true_uvd_B[:Nmin, 0] < img_size) &
            (true_uvd_B[:Nmin, 1] >= 0) & (true_uvd_B[:Nmin, 1] < img_size) &
            (dist_uvd_B[:Nmin, 0] >= 0) & (dist_uvd_B[:Nmin, 0] < img_size) &
            (dist_uvd_B[:Nmin, 1] >= 0) & (dist_uvd_B[:Nmin, 1] < img_size))
    ok = in_A & in_B
    n_ok = int(ok.sum())
    if n_ok < 4:
        return None
    err_hyp = float(np.linalg.norm(true_uvd_B[:Nmin][ok, :2]
                                   - dist_uvd_B[:Nmin][ok, :2],
                                   axis=-1).mean())

    # Stratified pick over the A image: split into a (gx, gy) grid (gx*gy
    # ≥ k_show), pick one valid point per cell. Avoids the "all selected
    # points clustered in one image region" failure mode of stride-pick.
    cand = np.where(ok)[0]
    if len(cand) <= k_show:
        sel = cand
    else:
        gx = int(np.ceil(np.sqrt(k_show * 4 / 3)))   # roughly 4:3 grid
        gy = int(np.ceil(k_show / gx))
        cell_w = img_size / gx
        cell_h = img_size / gy
        cand_uv_A = true_uvd_A[cand, :2]
        cu = np.clip((cand_uv_A[:, 0] / cell_w).astype(int), 0, gx - 1)
        cv = np.clip((cand_uv_A[:, 1] / cell_h).astype(int), 0, gy - 1)
        cell_id = cv * gx + cu
        # one point per cell, stable order: pick first cand in each cell
        order = np.argsort(cell_id, kind='stable')
        cand_sorted = cand[order]
        cell_sorted = cell_id[order]
        _, first_pos = np.unique(cell_sorted, return_index=True)
        per_cell = cand_sorted[first_pos]
        if len(per_cell) >= k_show:
            sel = per_cell[:k_show]
        else:
            # not enough occupied cells — fill remainder by stride from the rest
            remain = np.setdiff1d(cand, per_cell, assume_unique=False)
            extra_step = max(1, len(remain) // (k_show - len(per_cell)))
            sel = np.concatenate([per_cell, remain[::extra_step][:k_show - len(per_cell)]])

    fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(11, 5.5))
    ax_A.imshow(img_A); ax_A.set_xticks([]); ax_A.set_yticks([])
    ax_B.imshow(img_B); ax_B.set_xticks([]); ax_B.set_yticks([])
    cmap = cm.get_cmap('tab10')
    for k, i in enumerate(sel):
        c = cmap(k % 10)
        uA_gt = true_uvd_A[i, :2]
        uA_hat = dist_uvd_A[i, :2]
        uG = true_uvd_B[i, :2]
        uH = dist_uvd_B[i, :2]
        ax_A.plot(uA_gt[0], uA_gt[1], 'o', color=c, markersize=11,
                   markeredgecolor='white', mew=1.2, zorder=5)
        ax_A.annotate(str(k), (uA_gt[0], uA_gt[1]), ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=6)
        ax_A.plot(uA_hat[0], uA_hat[1], 'x', color=c, markersize=8, mew=1.5,
                   alpha=0.85, zorder=4)
        if pp_np_A is not None:
            muA = pp_np_A[i, :2]
            uPA = uA_hat + muA   # A-side pred = HAT + Δuv
            sxA = float(np.exp(pp_np_A[i, 2])); syA = float(np.exp(pp_np_A[i, 3]))
            rhoA = float(pp_np_A[i, 4])
            cxxA, cyyA, cxyA = sxA * sxA, syA * syA, rhoA * sxA * syA
            covA = np.array([[cxxA, cxyA], [cxyA, cyyA]])
            wA, vA = np.linalg.eigh(covA)
            wA = np.maximum(wA, 1e-9)
            ord_A = wA.argsort()[::-1]
            wA = wA[ord_A]; vA = vA[:, ord_A]
            angA = float(np.degrees(np.arctan2(vA[1, 0], vA[0, 0])))
            ax_A.plot(uPA[0], uPA[1], 'o', color=c, markersize=4,
                       markeredgecolor='white', mew=0.6, zorder=7)
            ax_A.add_patch(Ellipse((uPA[0], uPA[1]),
                                    width=float(2 * np.sqrt(wA[0])),
                                    height=float(2 * np.sqrt(wA[1])),
                                    angle=angA,
                                    facecolor='none', edgecolor=c,
                                    lw=1.2, alpha=0.9, zorder=8))
        ax_B.plot(uG[0], uG[1], 'o', color=c, markersize=8,
                   markeredgecolor='white', mew=1.0, zorder=5)
        ax_B.annotate(str(k), (uG[0], uG[1]), ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=6)
        ax_B.plot(uH[0], uH[1], 'x', color=c, markersize=8, mew=1.5,
                   alpha=0.85, zorder=4)
        if pp_np is not None:
            mu = pp_np[i, :2]
            uP = uH + mu  # pred = HAT + Δuv
            sx = float(np.exp(pp_np[i, 2])); sy = float(np.exp(pp_np[i, 3]))
            rho = float(pp_np[i, 4])
            cxx, cyy, cxy = sx * sx, sy * sy, rho * sx * sy
            cov = np.array([[cxx, cxy], [cxy, cyy]])
            w_, v_ = np.linalg.eigh(cov)
            w_ = np.maximum(w_, 1e-9)
            order = w_.argsort()[::-1]
            w_ = w_[order]; v_ = v_[:, order]
            ang = float(np.degrees(np.arctan2(v_[1, 0], v_[0, 0])))
            ax_B.plot(uP[0], uP[1], 'o', color=c, markersize=4,
                       markeredgecolor='white', mew=0.6, zorder=7)
            ax_B.add_patch(Ellipse((uP[0], uP[1]),
                                    width=float(2 * np.sqrt(w_[0])),
                                    height=float(2 * np.sqrt(w_[1])),
                                    angle=ang,
                                    facecolor='none', edgecolor=c,
                                    lw=1.2, alpha=0.9, zorder=8))
        fig.add_artist(ConnectionPatch(
            xyA=(uA_gt[0], uA_gt[1]), xyB=(uG[0], uG[1]),
            coordsA='data', coordsB='data',
            axesA=ax_A, axesB=ax_B,
            color=c, lw=0.7, ls=':', alpha=0.7, zorder=3))
    ax_A.text(2, img_size - 4,
               'o (large)  POSE_GT_A     (calib target = true_uvd_A)\n'
               'x  POSE_GT_A + ε_calib   (calib HAT = dist_uvd_A)',
               fontsize=6, color='white', va='bottom',
               bbox=dict(facecolor='black', alpha=0.55, pad=2,
                          edgecolor='none'))
    legend_B = ('o (large)  POSE_GT_AB           (pose target = true_uvd_B)\n'
                'x  POSE_GT_AB + ε_pose          (pose HAT = dist_uvd_B)')
    if pp_np is not None:
        legend_B += '\no (small) + ellipse  pred = HAT + Δuv ± 2σ'
    ax_B.text(2, img_size - 4, legend_B,
               fontsize=6, color='white', va='bottom',
               bbox=dict(facecolor='black', alpha=0.55, pad=2,
                          edgecolor='none'))
    if pp_np_A is not None:
        # err on A panel: GT − (HAT + pred)
        pred_uv_A = dist_uvd_A[:Nmin, :2] + pp_np_A[:Nmin, :2]
        hyp_A = float(np.linalg.norm(true_uvd_A[:Nmin][ok, :2]
                                       - dist_uvd_A[:Nmin][ok, :2],
                                       axis=-1).mean())
        err_pred_A = float(np.linalg.norm(true_uvd_A[:Nmin][ok, :2]
                                           - pred_uv_A[ok], axis=-1).mean())
        ax_A.set_title(
            f'A calib  HAT→GT {hyp_A:.1f} → pred {err_pred_A:.2f} px',
            fontsize=9, loc='left')
    else:
        err_pred_A = None
        ax_A.set_title(f'A frame  N_ok={n_ok}', fontsize=9, loc='left')
    if pp_np is not None:
        pred_uv = dist_uvd_B[:Nmin, :2] + pp_np[:Nmin, :2]
        err_pred = float(np.linalg.norm(true_uvd_B[:Nmin][ok, :2]
                                         - pred_uv[ok], axis=-1).mean())
        ax_B.set_title(
            f'B frame  HAT→GT {err_hyp:.1f} px → pred {err_pred:.2f} px',
            fontsize=9, loc='left')
    else:
        err_pred = None
        ax_B.set_title(f'B frame  HAT→GT shift {err_hyp:.1f} px',
                        fontsize=9, loc='left')
    calib_str = ''
    if np.any(np.abs(pert_A[:6]) > 1e-9):
        calib_str = (f'ε_calib t=({pert_A[0]:+.2f},{pert_A[1]:+.2f},{pert_A[2]:+.2f})m '
                     f'ypr=({pert_A[3]:+.2f},{pert_A[4]:+.2f},{pert_A[5]:+.2f})°  ')
    fig.suptitle(
        f'{suptitle_prefix}'
        f'{calib_str}'
        f'ε_pose t=({pert_B[0]:+.2f},{pert_B[1]:+.2f},{pert_B[2]:+.2f})m '
        f'ypr=({pert_B[3]:+.2f},{pert_B[4]:+.2f},{pert_B[5]:+.2f})°  '
        f'dpose_AB GT t=({dpose[0]:+.2f},{dpose[1]:+.2f},{dpose[2]:+.2f})m '
        f'ypr=({dpose[3]:+.2f},{dpose[4]:+.2f},{dpose[5]:+.2f})°',
        fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    stats = dict(n_ok=n_ok, err_hyp=err_hyp)
    if err_pred is not None:
        stats['err_pred'] = err_pred
    if err_pred_A is not None:
        stats['err_pred_A'] = err_pred_A
    return out_path, stats
