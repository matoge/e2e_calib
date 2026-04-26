"""train_cross_frame.py — PoC trainer for CalibNetCrossFrame.

Default mode is OVERFIT: cache ~N_OVERFIT pair samples in RAM, loop on them for
~EPOCHS epochs, watch mean Δ-reproj error drop to near zero. Symmetric dual
loss (A→B + B→A) on the Gaussian NLL of per-point (Δu, Δv) residuals.

Run:
    python train_cross_frame.py                 # default overfit run
    python train_cross_frame.py --full          # full train on scene 015

Outputs go to experiments/cross_frame_{name}/:
    best_model.pt, train.log, config.txt, curve.png
"""
import argparse, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.cross_frame import CalibNetCrossFrame
from models.cross_frame_multi import CalibNetMultiFrame
from models.cross_frame_unified import CalibNetUnifiedFrame
from models.model_cov import gaussian2d_nll, gaussian_uvd_nll

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── loss helpers ─────────────────────────────────────────────────────────────

def residual_nll_and_metrics(raw, uv_hat, uv_gt, pad_mask, img_size):
    """raw: (B,N,5)  — (Δu, Δv, log σu, log σv, ρ_raw) already clamped.
       uv_hat, uv_gt: (B,N,2) in patch-local pixel coords.
       pad_mask: (B,N) bool, True = padding.

    Target residual Δ_gt = uv_gt - uv_hat  (in pixel space).
    `gaussian2d_nll` is fed (Δ_gt) as the target and expects raw params where
    the first two channels ARE the predicted Δ (not an absolute position).
    """
    target = uv_gt - uv_hat                              # (B, N, 2) — target Δ
    B, N, _ = raw.shape
    # mask out padding: replace with zeros and reduce over valid only
    valid = ~pad_mask                                     # (B,N)

    # gaussian2d_nll expects params shaped (B,N,5) + target (B,N,2)
    # it returns mean NLL; we do per-sample and mask manually
    mu   = raw[..., :2]
    log_sx, log_sy = raw[..., 2], raw[..., 3]
    rho  = torch.tanh(raw[..., 4]) * 0.99
    sx, sy = torch.exp(log_sx), torch.exp(log_sy)
    dx = (target[..., 0] - mu[..., 0]) / sx
    dy = (target[..., 1] - mu[..., 1]) / sy
    r2 = 1.0 - rho * rho
    z  = (dx * dx + dy * dy - 2 * rho * dx * dy) / r2
    nll = 0.5 * z + log_sx + log_sy + 0.5 * torch.log(r2)

    # reduce over valid
    nll_masked = torch.where(valid, nll, torch.zeros_like(nll))
    n_valid    = valid.sum().clamp_min(1)
    loss = nll_masked.sum() / n_valid

    # diagnostic metrics: mean |pred|  and  pred err
    pred_uv = uv_hat + mu                                 # (B,N,2)
    with torch.no_grad():
        err_px = (pred_uv - uv_gt).pow(2).sum(-1).sqrt()  # (B,N)
        err_masked = torch.where(valid, err_px, torch.zeros_like(err_px))
        err_mean   = (err_masked.sum() / n_valid).item()
        # base error (no correction): |uv_hat - uv_gt|
        base = (uv_hat - uv_gt).pow(2).sum(-1).sqrt()
        base_masked = torch.where(valid, base, torch.zeros_like(base))
        base_mean   = (base_masked.sum() / n_valid).item()

    return loss, dict(err_px=err_mean, base_px=base_mean)


def residual_uvd_nll_and_metrics(raw, uv_hat, uv_gt, d_hat, d_gt, pad_mask, img_size):
    """raw: (B,N,7) [Δu, Δv, Δd, log σu, log σv, log σd, ρ_uv] already clamped.
       d_hat, d_gt: (B,N,) in meters (target-camera frame z).
       Returns (loss, dict(err_px, base_px, err_d, base_d))."""
    target_uv = uv_gt - uv_hat                              # (B,N,2) pixel Δ target
    target_d  = d_gt - d_hat                                # (B,N,)  meter Δ target
    valid = ~pad_mask                                       # (B,N)

    tx, ty, td = raw[..., 0], raw[..., 1], raw[..., 2]
    log_sx, log_sy, log_sd = raw[..., 3], raw[..., 4], raw[..., 5]
    rho = raw[..., 6]
    sx, sy, sd = log_sx.exp(), log_sy.exp(), log_sd.exp()
    dx = (target_uv[..., 0] - tx) / sx
    dy = (target_uv[..., 1] - ty) / sy
    dd = (target_d - td) / sd
    r2 = (1.0 - rho * rho).clamp(min=1e-6)
    maha_uv = (dx * dx - 2 * rho * dx * dy + dy * dy) / r2
    log_det_uv = 2 * log_sx + 2 * log_sy + torch.log(r2)
    maha_d = dd * dd
    log_det_d = 2 * log_sd
    nll = 0.5 * (log_det_uv + maha_uv + log_det_d + maha_d)

    nll_masked = torch.where(valid, nll, torch.zeros_like(nll))
    n_valid    = valid.sum().clamp_min(1)
    loss = nll_masked.sum() / n_valid

    with torch.no_grad():
        pred_uv = uv_hat + raw[..., :2]
        pred_d  = d_hat + td
        err_px  = (pred_uv - uv_gt).pow(2).sum(-1).sqrt()
        err_d   = (pred_d  - d_gt).abs()
        base_px = (uv_hat - uv_gt).pow(2).sum(-1).sqrt()
        base_d  = (d_hat  - d_gt).abs()
        m = lambda x: (torch.where(valid, x, torch.zeros_like(x)).sum() / n_valid).item()
        metrics = dict(err_px=m(err_px), base_px=m(base_px),
                       err_d=m(err_d),   base_d=m(base_d))
    return loss, metrics


# ─── one step ─────────────────────────────────────────────────────────────────

_LIDAR_SENTINEL_Z = -9999.0 / 50.0   # uvd[..., 2] is z/50 in normalised units


def _apply_lidar_dropout(batch, max_rate, p_zero, multi_frame):
    """Hierarchical LiDAR dropout. Per sample:
        - with probability `p_zero` → drop ALL depth (rate = 1.0, image-only mode)
        - else                       → drop rate ~ U(0, max_rate) (partial)

    Even one surviving depth-anchored pt is a strong cue, so partial drop alone
    isn't enough to break the LiDAR-dependence — `p_zero` forces a fraction of
    samples to image-only so the encoder learns to handle no-depth regions.
    `uvd[..., 2]` is set to a sentinel; (u, v) coords stay intact so model can
    still attend; loss/pad unchanged so per-point residual still supervised.
    """
    if max_rate <= 0 and p_zero <= 0:
        return
    sentinel = torch.tensor(_LIDAR_SENTINEL_Z, device=DEVICE)
    for tag in (['A', 'B', 'M'] if multi_frame else ['A', 'B']):
        for full in ('', '_full'):
            key = f'uvd_{tag}{full}'
            if key not in batch:
                continue
            t = batch[key]                                       # (B, N, 3)
            B, N, _ = t.shape
            zero_mode = torch.rand(B, 1, device=t.device) < p_zero
            partial   = torch.rand(B, 1, device=t.device) * max_rate
            rate      = torch.where(zero_mode, torch.ones_like(partial), partial)
            mask      = torch.rand(B, N, device=t.device) < rate
            t[..., 2] = torch.where(mask, sentinel, t[..., 2])


def step(model, batch, uvd_mode=False, frustum_full=True, multi_frame=False,
         lidar_dropout=0.0, lidar_dropout_zero=0.0, per_frame_emb=False):
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if model.training and (lidar_dropout > 0 or lidar_dropout_zero > 0):
        _apply_lidar_dropout(batch, lidar_dropout, lidar_dropout_zero, multi_frame)
    kw = {}
    if frustum_full:
        kw['uvd_A_full'] = batch.get('uvd_A_full')
        kw['uvd_B_full'] = batch.get('uvd_B_full')
        kw['pad_A_full'] = batch.get('pad_A_full')
        kw['pad_B_full'] = batch.get('pad_B_full')
    if multi_frame:
        kw['patch_M']        = batch['patch_M']
        kw['uvd_M']          = batch['uvd_M']
        kw['pad_M']          = batch['pad_M']
        kw['uvd_M_full']     = batch.get('uvd_M_full')
        kw['pad_M_full']     = batch.get('pad_M_full')
        kw['pose_AM_6dof']   = batch['pose_AM_6dof']
        kw['uv_M_hat_of_A']  = batch['uv_M_hat_of_A']
        kw['uv_M_hat_of_B']  = batch['uv_M_hat_of_B']
        if 'patch_M2' in batch:
            kw['patch_M2']       = batch['patch_M2']
            kw['uvd_M2']         = batch['uvd_M2']
            kw['pad_M2']         = batch['pad_M2']
            kw['uvd_M2_full']    = batch.get('uvd_M2_full')
            kw['pad_M2_full']    = batch.get('pad_M2_full')
            kw['pose_AM2_6dof']  = batch['pose_AM2_6dof']
            kw['uv_M2_hat_of_A'] = batch['uv_M2_hat_of_A']
            kw['uv_M2_hat_of_B'] = batch['uv_M2_hat_of_B']
    kw['per_frame_emb'] = per_frame_emb
    raw_AB, raw_BA = model(
        patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
        patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
        pose_AB_6dof=batch['pose_AB_6dof'], pose_BA_6dof=batch['pose_BA_6dof'],
        uv_B_hat_of_A=batch['uv_B_hat_of_A'], uv_A_hat_of_B=batch['uv_A_hat_of_B'],
        pad_A=batch['pad_A'], pad_B=batch['pad_B'],
        **kw,
    )
    if uvd_mode:
        loss_AB, m_AB = residual_uvd_nll_and_metrics(
            raw_AB, batch['uv_B_hat_of_A'], batch['uv_B_gt_of_A'],
            batch['d_B_hat_of_A'], batch['d_B_gt_of_A'], batch['pad_A'], 64)
        loss_BA, m_BA = residual_uvd_nll_and_metrics(
            raw_BA, batch['uv_A_hat_of_B'], batch['uv_A_gt_of_B'],
            batch['d_A_hat_of_B'], batch['d_A_gt_of_B'], batch['pad_B'], 64)
    else:
        loss_AB, m_AB = residual_nll_and_metrics(
            raw_AB, batch['uv_B_hat_of_A'], batch['uv_B_gt_of_A'], batch['pad_A'], 64)
        loss_BA, m_BA = residual_nll_and_metrics(
            raw_BA, batch['uv_A_hat_of_B'], batch['uv_A_gt_of_B'], batch['pad_B'], 64)

    loss = 0.5 * (loss_AB + loss_BA)
    metrics = dict(
        loss=loss.item(), loss_AB=loss_AB.item(), loss_BA=loss_BA.item(),
        err_AB=m_AB['err_px'], base_AB=m_AB['base_px'],
        err_BA=m_BA['err_px'], base_BA=m_BA['base_px'],
    )
    if uvd_mode:
        metrics.update(
            err_d_AB=m_AB['err_d'], base_d_AB=m_AB['base_d'],
            err_d_BA=m_BA['err_d'], base_d_BA=m_BA['base_d'],
        )
    return loss, metrics


# ─── n-frame stacked-tensor step ─────────────────────────────────────────────
# For datasets emitting (B, N, ...) per-frame and (B, N, N, ...) per-pair
# fields, run model.forward_n once and compute NLL on every (i, j) with i ≠ j,
# masked by pad_dir[i, j].

def step_n(model, batch, loss_ab_only=False, mix_mode='mix'):
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    raws = model.forward_n(
        patches       = batch['patches'],
        uvd           = batch['uvd'],
        pad           = batch['pad'],
        uvd_full      = batch['uvd_full'],
        pad_full      = batch['pad_full'],
        pose_hat_6dof = batch['pose_hat_6dof'],
        uv_hat        = batch['uv_hat'],
        mix_mode      = mix_mode,
    )
    B, N = batch['patches'].shape[0], batch['patches'].shape[1]
    losses, errs, bases = [], [], []
    pair_metrics = {}
    far = N - 1
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # `loss_ab_only` ablation: keep model's forward + diagnostic
            # err for every direction, but drop loss term for non-AB pairs
            # → isolates "M as KV (forward)" effect from "M as supervised
            # query (loss)" effect.
            include_in_loss = (not loss_ab_only) or (i == 0 and j == far) or (i == far and j == 0)
            raw_ij = raws[:, i, j]
            uv_hat_ij = batch['uv_hat'][:, i, j]
            uv_gt_ij  = batch['uv_gt' ][:, i, j]
            pad_ij    = batch['pad_dir'][:, i, j]
            loss_ij, m_ij = residual_nll_and_metrics(
                raw_ij, uv_hat_ij, uv_gt_ij, pad_ij, 64)
            if include_in_loss:
                losses.append(loss_ij)
            errs.append(m_ij['err_px'])
            bases.append(m_ij['base_px'])
            pair_metrics[f'err_{i}{j}']  = m_ij['err_px']
            pair_metrics[f'base_{i}{j}'] = m_ij['base_px']
    # Per-direction weight matches pair-mode (loss_AB + loss_BA)/2 → 0.5 each
    # so the A↔B gradient magnitude doesn't shrink as we add more directions.
    # Equivalent to sum(losses)/2; the extra directions add as bonus terms
    # rather than diluting AB.
    loss = torch.stack(losses).sum() * 0.5
    # Surface the legacy A↔B direction (frame 0 = anchor, N-1 = far frame)
    # via the err_AB / err_BA / base_AB keys so the trainer's existing logger
    # is directly comparable to pair runs. Other directions are kept as
    # err_ij / base_ij in pair_metrics for inspection.
    far = N - 1
    metrics = dict(
        loss=loss.item(),
        err_AB =pair_metrics[f'err_0{far}'],
        err_BA =pair_metrics[f'err_{far}0'],
        base_AB=pair_metrics[f'base_0{far}'],
        err_all=float(np.mean(errs)),       # diagnostic: mean across all dirs
        **pair_metrics,
    )
    return loss, metrics


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='v00_overfit')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--scene', default='/mnt/mininas/datasets/pandaset/015',
                    help='single-scene mode (legacy)')
    ap.add_argument('--scenes-root', default=None,
                    help='multi-scene mode: root directory containing all scene folders')
    ap.add_argument('--train-frac', type=float, default=0.80,
                    help='scene-level train fraction (only used with --scenes-root)')
    ap.add_argument('--cameras', default='front_camera',
                    help='comma-separated camera names per scene, or "all" for every camera present. '
                         'PandaSet has 6 (front/back/{front,,}_{left,right}_camera); '
                         'Waymo-converted scenes typically only have front_camera.')
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n-overfit', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--log-every', type=int, default=4)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--crop-min', type=int, default=64)
    ap.add_argument('--crop-max', type=int, default=192)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--virtual-epoch', type=int, default=4000)
    ap.add_argument('--deform-mode', default='none', choices=['none', 'sl'])
    ap.add_argument('--n-cross-layers', type=int, default=1)
    ap.add_argument('--n-intra-layers', type=int, default=2)
    ap.add_argument('--uvd', action=argparse.BooleanOptionalAction, default=True,
                    help='predict (Δu,Δv,Δd) with 3D gaussian NLL instead of 2D (default: on; '
                         'pass --no-uvd for legacy 5-dim)')
    ap.add_argument('--frustum-full', action=argparse.BooleanOptionalAction, default=True,
                    help='if set, FrustumLocalEncoder reads neighbors from the FULL '
                         'in-box LiDAR set (uvd_*_full padded to N=2048) instead of '
                         'just the stratified-256 query subset. --no-frustum-full = legacy.')
    ap.add_argument('--dataset', default='pandaset',
                    choices=['pandaset', 'waymo', 'pandaset+waymo'],
                    help='training dataset(s)')
    ap.add_argument('--waymo-root', default='/mnt/mininas/datasets/waymo/training',
                    help='Waymo training data root')
    ap.add_argument('--max-train-scenes', type=int, default=0,
                    help='cap number of train scenes (0 = all)')
    # online mining: overfit-triggered val→train migration
    ap.add_argument('--mine-val', action='store_true',
                    help='enable uniform-random val→train migration when val err plateaus')
    ap.add_argument('--val-pool-size', type=int, default=2000,
                    help='val virtual_epoch_len (enlarged pool for mining; default 2000)')
    ap.add_argument('--migrate-k', type=int, default=100,
                    help='samples to migrate from val→train per overfit trigger')
    ap.add_argument('--overfit-patience', type=int, default=2,
                    help='consecutive val checks w/o improvement before migration fires')
    ap.add_argument('--overfit-metric', default='nll', choices=['err', 'nll'],
                    help='which val metric triggers migration: err=val_err_px (easier to interpret), '
                         'nll=val_nll (catches σ-calibration breakdown earlier)')
    ap.add_argument('--rewind-back', type=int, default=3,
                    help='on MIGRATE, rewind model to N val-checks ago (0 = no rewind). '
                         'Default 3 ≈ log_every*3 epochs before trigger.')
    ap.add_argument('--rewind-lr-reset', action='store_true', default=True,
                    help='also reset LR to --lr on rewind and re-cosine for remaining epochs')
    ap.add_argument('--sentinel-size', type=int, default=0,
                    help='when >0, use sentinel-infinite mode: eval always runs on '
                         'idx [0..sentinel_size), mining pulls from idx [sentinel_size..∞). '
                         '0 = legacy finite val_pool_size mode.')
    ap.add_argument('--init-from', default=None,
                    help='path to a previous experiment dir (or .pt). Encoder weights '
                         '(cnn, point_mlp, frustum_enc, intra_blocks, pose_mlp) are loaded '
                         'and FROZEN. Only the new pieces (cross_blocks, pose_bias) train. '
                         'Use this to fine-tune multi-frame on top of a converged pair-net.')
    ap.add_argument('--no-freeze', action='store_true',
                    help='with --init-from, skip freezing the frame encoder so all '
                         'params train (full fine-tune). Useful for ablation: did freeze '
                         'matter, or does the multi-frame head need encoder co-training?')
    ap.add_argument('--motion-warp-gt', action='store_true',
                    help='dataset rewrites uv_gt for query points inside cuboids '
                         'flagged Moving by annotators, using the box A→B rigid '
                         'transform. Behind-camera warps drop; view-out warps '
                         'kept (model learns large Δuv). Trains motion-aware net.')
    ap.add_argument('--quad-frame', action='store_true',
                    help='extends --multi-frame to quad: A, M1, M2, B (M1 at 1/3 and '
                         'M2 at 2/3 between A and B). Requires --multi-frame and a '
                         'unified model with max_kv_frames>=3.')
    ap.add_argument('--multi-frame', action='store_true',
                    help='enable triplet dataset (A,M,B) + CalibNetMultiFrame model. '
                         'M is the middle frame; KV concatenates {M, B} with per-frame '
                         'pose-bias added to attention scores (relative-only, RPE-style).')
    ap.add_argument('--n-frames', type=int, default=0,
                    help='0 = legacy paths (default). 2/3 = new stacked-array path '
                         'with iid per-pair perturbation; dataset emits (N, ...) '
                         'per-frame and (N, N, ...) per-pair tensors, model.forward_n() '
                         'supervises every (i, j) direction.')
    ap.add_argument('--stop-no-improve-migrations', type=int, default=0,
                    help='stop training if this many consecutive migrations fail to '
                         'improve global-best val_nll. 0 = no early stop (run full epochs).')
    ap.add_argument('--lidar-dropout', type=float, default=0.0,
                    help='max per-sample LiDAR dropout rate (uvd[...,2] sentinel). '
                         'Per-sample rate ~ U(0, this). 0 = off.')
    ap.add_argument('--lidar-dropout-zero', type=float, default=0.0,
                    help='probability per sample of full-zero LiDAR dropout (image-only '
                         'mode). Combined with --lidar-dropout: zero-mode samples ignore '
                         'partial-rate. 0 = off.')
    ap.add_argument('--clearml', action='store_true',
                    help='upload metrics + curves to ClearML dashboard. Requires '
                         '`clearml-init` to be set up once with your creds.')
    ap.add_argument('--clearml-project', default='e2e_calib/cross-frame',
                    help='ClearML project name.')
    ap.add_argument('--loss-ab-only', action='store_true',
                    help='[stacked path only] supervise only A↔B (i.e. (0, N-1) '
                         'and (N-1, 0) directions); skip M-related directions. '
                         'Use to isolate "M as KV" benefit from "M as supervised '
                         'query" cost.')
    ap.add_argument('--mix-mode', default='mix', choices=['mix', 'pair'],
                    help='[stacked path only] "mix" = full N-frame KV (default); '
                         '"pair" = each (i, j) sees only frame j as KV — i.e. 6× '
                         'independent pair forwards, no internal multi-frame '
                         'fusion. Use as a clean baseline before adding any '
                         'mixing logic on top.')
    ap.add_argument('--model', default='multi', choices=['multi', 'unified'],
                    help='"multi" = legacy CalibNetMultiFrame (separate pt/img '
                         'KV, per-frame bias). "unified" = CalibNetUnifiedFrame '
                         '(img+pt scattered to same 16x16 frame_token grid; '
                         'cross-attn is single MSDeformAttn over multi-level '
                         'frame_token KV; per-point Q bilinear-sampled from '
                         'anchor frame_token).')
    ap.add_argument('--per-frame-emb', action='store_true',
                    help='Anchor-at-A absolute pose embedding: each frame\'s '
                         'tokens (Q, K, V both) carry their own pose-relative-'
                         'to-A embedding (broadcast across all tokens of that '
                         'frame). No Q-shift via target pose. Tested with pair '
                         'first; extends naturally to N frames.')
    args = ap.parse_args()

    out_dir = Path('experiments/cross_frame_' + args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    (out_dir / 'config.txt').write_text(json.dumps(vars(args), indent=2))

    # ─── ClearML hookup (opt-in) ─────────────────────────────────────────────
    cml_logger = None
    if args.clearml:
        from clearml import Task
        cml_task = Task.init(project_name=args.clearml_project, task_name=args.name,
                              auto_connect_frameworks={'pytorch': False})
        cml_task.connect(vars(args), name='args')
        cml_logger = cml_task.get_logger()

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'device = {DEVICE}')
    log(f'args = {vars(args)}')

    # dataset
    use_stacked = args.n_frames > 0
    n_frames_path = max(args.n_frames, 2)
    ds_kwargs = dict(
        img_size=args.img_size, max_points=args.max_points,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        crop_range=(args.crop_min, args.crop_max),
        cameras=args.cameras,
        triplet=args.multi_frame and not use_stacked,   # legacy aux-KV only
        quad=args.quad_frame,                            # adds M2 (forces triplet=True)
        motion_warp_gt=args.motion_warp_gt,
        n_frames=n_frames_path,
        use_stacked=use_stacked,
    )
    if args.scenes_root:
        ds_kwargs['scenes_root'] = args.scenes_root
        ds_kwargs['train_frac']  = args.train_frac
    else:
        ds_kwargs['scene_root']  = args.scene
    ds_train = PandaSetCrossFrameDataset(
        split='train',
        virtual_epoch_len=(args.n_overfit if not args.full else args.virtual_epoch),
        **ds_kwargs,
    )
    ds_val = PandaSetCrossFrameDataset(
        split='val',
        virtual_epoch_len=args.val_pool_size, seed=123,
        **ds_kwargs,
    )

    # pre-fetch overfit buffer: fix a small set of samples
    overfit = not args.full
    if overfit:
        log(f'caching {args.n_overfit} fixed overfit samples…')
        fixed = [ds_train[i] for i in range(args.n_overfit)]
        log(f'  done. sample[0] residual stats: '
            f'target_AB |Δ| mean = '
            f'{(fixed[0]["uv_B_gt_of_A"] - fixed[0]["uv_B_hat_of_A"])[~fixed[0]["pad_A"]].abs().mean().item():.2f} px')

        def collate(batch):
            keep_keys = [k for k in batch[0].keys() if not isinstance(batch[0][k], (int, str))]
            return {k: torch.stack([b[k] for b in batch]) for k in keep_keys}
        loader = [collate(fixed[i:i+args.batch_size])
                  for i in range(0, len(fixed), args.batch_size)]
    else:
        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, persistent_workers=True,
                             pin_memory=True, prefetch_factor=4)
        if args.sentinel_size > 0:
            # sentinel mode: eval on fixed idx [0..sentinel_size), mining pulls [sentinel_size..∞)
            from torch.utils.data import Subset
            ds_val.virtual_epoch_len = max(ds_val.virtual_epoch_len,
                                            args.sentinel_size)
            val_loader = DataLoader(
                Subset(ds_val, list(range(args.sentinel_size))),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=True, pin_memory=True, prefetch_factor=4)
        else:
            val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     persistent_workers=True, pin_memory=True, prefetch_factor=4)

    # ─── online mining state ──────────────────────────────────────────────────
    # val-pool mining: overfit triggered val→train migration.
    # Strategy: uniform-random sample K val examples when val_err plateaus and
    # append to a CPU-side buffer. Training batches additionally iterate through
    # this buffer every epoch so migrated samples keep getting gradient steps
    # — never restart from scratch.
    migrated_samples: list = []                         # list[dict of CPU tensors]
    # two mining source modes:
    #   - legacy  : remaining_val_idx starts as range(val_pool_size), shrinks on migration
    #   - sentinel: mining draws from idx [sentinel_size..∞) in ds_val, virtually infinite
    remaining_val_idx: list = list(range(args.val_pool_size))
    next_mining_idx = args.sentinel_size    # sentinel mode: ever-incrementing counter
    rng_mine = np.random.default_rng(0)
    best_val_for_mining = float('inf')
    global_best_val = float('inf')          # never resets — for early-stop criterion
    n_mig_since_global_best = 0             # migrations count since last global best update
    n_since_improve = 0
    n_migrations = 0
    # ring buffer of (ep, model_sd, opt_sd) for rewind on MIGRATE
    ckpt_ring: list = []
    ring_max = args.rewind_back + 1

    def _collate_dicts(batch_list):
        keep_keys = [k for k in batch_list[0].keys()
                     if not isinstance(batch_list[0][k], (int, str))]
        return {k: torch.stack([b[k] for b in batch_list]) for k in keep_keys}

    # model — multi (legacy) or unified (frame_token, single MSDeformAttn).
    if args.model == 'unified':
        model = CalibNetUnifiedFrame(
            img_size=args.img_size,
            n_intra_layers=args.n_intra_layers,
            n_cross_layers=args.n_cross_layers,
            out_dim=(7 if args.uvd else 5),
            max_kv_frames=max(2, n_frames_path - 1, 3 if args.quad_frame else 0),
        ).to(DEVICE)
    else:
        model = CalibNetMultiFrame(
            img_size=args.img_size,
            n_intra_layers=args.n_intra_layers,
            n_cross_layers=args.n_cross_layers,
            deform_mode=args.deform_mode,
            out_dim=(7 if args.uvd else 5),
            max_kv_frames=max(2, n_frames_path - 1, 3 if args.quad_frame else 0),
        ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'model params: {n_params/1e6:.3f} M')

    # ─── fine-tune mode: load matching keys + freeze frame encoder ─────────
    # Frame-encoder modules (data-dependent) get frozen; only the new
    # cross-frame pieces train. See docs/experiment_progression.html.
    if args.model == 'unified':
        FREEZE_PREFIXES = ('encoder.cnn.', 'encoder.point_mlp.',
                            'encoder.frustum_enc.', 'encoder.intra.',
                            'pose_mlp.')
    else:
        FREEZE_PREFIXES = ('cnn.', 'point_mlp.', 'frustum_enc.',
                            'intra_blocks.', 'pose_mlp.')
    if args.init_from:
        ckpt_path = Path(args.init_from)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / 'best_model.pt'
        sd = torch.load(ckpt_path, map_location=DEVICE)
        loaded = model.load_state_dict(sd, strict=False)
        log(f'init-from {ckpt_path}: '
            f'{len(sd) - len(loaded.unexpected_keys)} keys loaded, '
            f'{len(loaded.missing_keys)} missing (new), '
            f'{len(loaded.unexpected_keys)} unexpected (dropped)')
        n_frozen = 0
        if not args.no_freeze:
            for name, p in model.named_parameters():
                if any(name.startswith(pre) for pre in FREEZE_PREFIXES):
                    p.requires_grad = False
                    n_frozen += p.numel()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f'  frozen params: {n_frozen/1e6:.3f} M  trainable: {n_train/1e6:.3f} M'
            f'{" (no-freeze: full fine-tune)" if args.no_freeze else ""}')

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 1e-2)

    curves = dict(epoch=[], loss=[], err_AB=[], err_BA=[], base_AB=[],
                  val_loss=[], val_err_AB=[], val_err_BA=[], val_base_AB=[],
                  err_d=[], val_err_d=[], base_d=[], val_base_d=[])
    best_err = float('inf')
    t0 = time.time()

    @torch.no_grad()
    def _eval_val():
        if overfit:
            return None
        model.eval()
        vl, vA, vB, vbase, vdA, vdB, vd_base = [], [], [], [], [], [], []
        # Always use val_loader (parallel DataLoader). When mine_val=True, the loader
        # walks the full val_pool including samples that have been migrated — but
        # migrated ∈ train now, their residuals are near 0 → they inflate val slightly
        # (≤ migrated/pool frac). Accept the bias for 50-100× eval speedup.
        for batch in val_loader:
            if use_stacked:
                _, m = step_n(model, batch, loss_ab_only=args.loss_ab_only, mix_mode=args.mix_mode)
            else:
                _, m = step(model, batch, uvd_mode=args.uvd, frustum_full=args.frustum_full, multi_frame=args.multi_frame, lidar_dropout=args.lidar_dropout, lidar_dropout_zero=args.lidar_dropout_zero, per_frame_emb=args.per_frame_emb)
            vl.append(m['loss']); vA.append(m['err_AB'])
            vB.append(m['err_BA']); vbase.append(m['base_AB'])
            if args.uvd:
                vdA.append(m['err_d_AB']); vdB.append(m['err_d_BA'])
                vd_base.append(m['base_d_AB'])
        out = dict(loss=float(np.mean(vl)), err_AB=float(np.mean(vA)),
                   err_BA=float(np.mean(vB)), base_AB=float(np.mean(vbase)))
        if args.uvd:
            out.update(err_d_AB=float(np.mean(vdA)), err_d_BA=float(np.mean(vdB)),
                       base_d=float(np.mean(vd_base)))
        return out

    def _mig_batch_iter():
        """Yield collated batches from migrated_samples (reshuffled each epoch)."""
        if not migrated_samples:
            return
        idx = list(range(len(migrated_samples)))
        rng_mine.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            chunk = idx[i:i+args.batch_size]
            if len(chunk) < args.batch_size:
                break
            yield _collate_dicts([migrated_samples[j] for j in chunk])

    break_flag = False
    for ep in range(args.epochs):
        if break_flag:
            break
        model.train()
        ep_losses, ep_errs_AB, ep_errs_BA, ep_base_AB = [], [], [], []
        ep_errs_d_AB, ep_errs_d_BA, ep_base_d = [], [], []

        def _do_step(batch):
            opt.zero_grad(set_to_none=True)
            if use_stacked:
                loss, m = step_n(model, batch, loss_ab_only=args.loss_ab_only, mix_mode=args.mix_mode)
            else:
                loss, m = step(model, batch, uvd_mode=args.uvd, frustum_full=args.frustum_full, multi_frame=args.multi_frame, lidar_dropout=args.lidar_dropout, lidar_dropout_zero=args.lidar_dropout_zero, per_frame_emb=args.per_frame_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_losses.append(m['loss'])
            ep_errs_AB.append(m['err_AB']); ep_errs_BA.append(m['err_BA'])
            ep_base_AB.append(m['base_AB'])
            if args.uvd:
                ep_errs_d_AB.append(m['err_d_AB']); ep_errs_d_BA.append(m['err_d_BA'])
                ep_base_d.append(m['base_d_AB'])

        for batch in loader:
            _do_step(batch)
        # Additional migrated-sample passes each epoch (keeps them in the gradient diet).
        if args.mine_val:
            for batch in _mig_batch_iter():
                _do_step(batch)
        sched.step()

        curves['epoch'].append(ep)
        curves['loss'].append(float(np.mean(ep_losses)))
        curves['err_AB'].append(float(np.mean(ep_errs_AB)))
        curves['err_BA'].append(float(np.mean(ep_errs_BA)))
        curves['base_AB'].append(float(np.mean(ep_base_AB)))

        # periodic val pass (full mode only)
        do_val = (not overfit) and (ep % args.log_every == 0 or ep == args.epochs - 1)
        if do_val:
            v = _eval_val()
            curves['val_loss'].append(v['loss'])
            curves['val_err_AB'].append(v['err_AB'])
            curves['val_err_BA'].append(v['err_BA'])
            curves['val_base_AB'].append(v['base_AB'])

            # overfit-triggered val→train migration
            if args.mine_val:
                # Every val check: push current snapshot into ring buffer.
                # CPU-side copy so we don't bloat GPU memory.
                def _cpu_sd(sd):
                    return {k: v_.detach().cpu().clone() for k, v_ in sd.items()}
                ckpt_ring.append(dict(ep=ep,
                                       model=_cpu_sd(model.state_dict()),
                                       opt=opt.state_dict()))
                if len(ckpt_ring) > ring_max:
                    ckpt_ring.pop(0)

                if args.overfit_metric == 'nll':
                    cur = v['loss']          # val NLL — catches σ-calib collapse earliest
                else:
                    cur = 0.5 * (v['err_AB'] + v['err_BA'])
                if cur < best_val_for_mining - 1e-3:
                    best_val_for_mining = cur
                    n_since_improve = 0
                else:
                    n_since_improve += 1
                # global best (never resets) — for early-stop "data exhausted" detection
                if cur < global_best_val - 1e-3:
                    global_best_val = cur
                    n_mig_since_global_best = 0

                can_migrate = (n_since_improve >= args.overfit_patience)
                if args.sentinel_size > 0:
                    # sentinel mode: pull fresh idx [next_mining_idx..) — never hit sentinel or prior mines
                    pool_ok = True   # virtually infinite
                else:
                    pool_ok = len(remaining_val_idx) >= args.migrate_k

                if can_migrate and pool_ok:
                    if args.sentinel_size > 0:
                        pick_idx = list(range(next_mining_idx,
                                                next_mining_idx + args.migrate_k))
                        next_mining_idx += args.migrate_k
                        pool_log = f'next_mining_idx={next_mining_idx} (sentinel+∞ mode)'
                    else:
                        pick = rng_mine.choice(len(remaining_val_idx),
                                                size=args.migrate_k, replace=False)
                        pick_idx = [remaining_val_idx[i] for i in sorted(pick.tolist(), reverse=True)]
                        for i in sorted(pick.tolist(), reverse=True):
                            del remaining_val_idx[i]
                        pool_log = f'val pool remaining {len(remaining_val_idx)}'
                    # materialize samples (strip non-tensor scalars)
                    for ji in pick_idx:
                        s = ds_val[ji]
                        s = {k: v_ for k, v_ in s.items()
                             if not isinstance(v_, (int, str))}
                        migrated_samples.append(s)
                    n_migrations += 1
                    log(f'  [MIGRATE#{n_migrations}] '
                        f'moved {args.migrate_k} val→train (total migrated {len(migrated_samples)}, '
                        f'{pool_log})')

                    # rewind: pop oldest snapshot from ring and restore
                    if args.rewind_back > 0 and ckpt_ring:
                        rw = ckpt_ring[0]            # oldest in ring
                        model.load_state_dict({k: v_.to(DEVICE) for k, v_ in rw['model'].items()})
                        opt.load_state_dict(rw['opt'])
                        log(f'  [REWIND] model ← snapshot from ep {rw["ep"]} '
                            f'(current ep {ep}, rewound by {ep - rw["ep"]} epochs)')

                    # LR schedule reset: fresh cosine for remaining epochs
                    if args.rewind_lr_reset:
                        for pg in opt.param_groups:
                            pg['lr'] = args.lr
                        remaining = max(1, args.epochs - ep)
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=remaining, eta_min=args.lr * 1e-2)
                        log(f'  [LR-RESET] lr ← {args.lr:.1e}, cosine restart over {remaining} remaining epochs')

                    n_since_improve = 0
                    best_val_for_mining = float('inf')
                    ckpt_ring.clear()   # fresh ring — we just rewound, old entries irrelevant
                    n_mig_since_global_best += 1
                    if (args.stop_no_improve_migrations > 0 and
                            n_mig_since_global_best >= args.stop_no_improve_migrations):
                        log(f'  [EARLY-STOP] {n_mig_since_global_best} consecutive migrations '
                            f'without global val_nll improvement (best={global_best_val:.3f}) — '
                            f'data saturated, stopping training.')
                        break_flag = True

        if ep % args.log_every == 0 or ep == args.epochs - 1:
            val_str = (f'  val_err={0.5*(v["err_AB"]+v["err_BA"]):.2f}px '
                       f'(base {v["base_AB"]:.2f})  val_nll={v["loss"]:.2f}'
                       if do_val else '')
            depth_str = ''
            if args.uvd:
                err_d_mean = 0.5 * (float(np.mean(ep_errs_d_AB)) + float(np.mean(ep_errs_d_BA)))
                base_d_mean = float(np.mean(ep_base_d))
                depth_str = f'  d={err_d_mean:.2f}m(base {base_d_mean:.2f})'
                if do_val:
                    vd = 0.5 * (v['err_d_AB'] + v['err_d_BA'])
                    depth_str += f'  val_d={vd:.2f}m(base {v["base_d"]:.2f})'
            log(f'ep {ep:3d}  loss={curves["loss"][-1]:.3f}  '
                f'err_AB={curves["err_AB"][-1]:.2f}px  '
                f'err_BA={curves["err_BA"][-1]:.2f}px  '
                f'(base={curves["base_AB"][-1]:.2f}px){val_str}{depth_str}  '
                f'lr={opt.param_groups[0]["lr"]:.2e}  '
                f't={time.time()-t0:.0f}s')

            if cml_logger is not None:
                # Each metric gets its OWN plot title — ClearML splits same-title
                # different-series across subplots, which hid val_nll behind val_err.
                # Also dual-report under the legacy ('train','val') titles so old
                # dashboard layouts that pre-date the title split still populate.
                cml_logger.report_scalar('nll',     'train',  value=curves['loss'][-1], iteration=ep)
                cml_logger.report_scalar('err',     'train_AB', value=curves['err_AB'][-1], iteration=ep)
                cml_logger.report_scalar('err',     'train_BA', value=curves['err_BA'][-1], iteration=ep)
                cml_logger.report_scalar('err',     'base',   value=curves['base_AB'][-1], iteration=ep)
                cml_logger.report_scalar('lr',      'lr',     value=opt.param_groups[0]['lr'], iteration=ep)
                # legacy mirror
                cml_logger.report_scalar('train', 'loss',   value=curves['loss'][-1], iteration=ep)
                cml_logger.report_scalar('train', 'err_AB', value=curves['err_AB'][-1], iteration=ep)
                cml_logger.report_scalar('train', 'err_BA', value=curves['err_BA'][-1], iteration=ep)
                cml_logger.report_scalar('train', 'base',   value=curves['base_AB'][-1], iteration=ep)
                if do_val:
                    val_err = 0.5*(v['err_AB']+v['err_BA'])
                    cml_logger.report_scalar('err', 'val',  value=val_err, iteration=ep)
                    cml_logger.report_scalar('nll', 'val',  value=v['loss'], iteration=ep)
                    cml_logger.report_scalar('overfit', 'val_err / train_err',
                                              value=val_err/max(curves['err_AB'][-1], 1e-6), iteration=ep)
                    cml_logger.report_scalar('overfit', 'val_nll - train_nll',
                                              value=v['loss'] - curves['loss'][-1], iteration=ep)
                    # legacy mirror
                    cml_logger.report_scalar('val', 'err',  value=val_err, iteration=ep)
                    cml_logger.report_scalar('val', 'nll',  value=v['loss'], iteration=ep)

        # save best by train (overfit) or val (full) mean err
        if overfit:
            score = 0.5 * (curves['err_AB'][-1] + curves['err_BA'][-1])
        else:
            score = 0.5 * (v['err_AB'] + v['err_BA']) if do_val else best_err + 1
        if score < best_err:
            best_err = score
            torch.save(model.state_dict(), out_dir / 'best_model.pt')

    # save final (most-overfit) weights too — useful for inspecting whether
    # the train-time over-confidence is geometric memorisation or σ collapse.
    torch.save(model.state_dict(), out_dir / 'last_model.pt')

    # final plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    ax[0].plot(curves['epoch'], curves['loss'], color='#174734', lw=2, label='train NLL')
    if not overfit and curves['val_loss']:
        val_eps = [e for e in curves['epoch'] if (e % args.log_every == 0 or e == args.epochs - 1)]
        ax[0].plot(val_eps, curves['val_loss'], color='#c13c14', lw=2, marker='o', ms=4, label='val NLL')
    ax[0].set_title('NLL loss', loc='left')
    ax[0].set_title('NLL loss', loc='left'); ax[0].set_xlabel('epoch'); ax[0].grid(alpha=0.25)
    ax[0].legend(frameon=False)
    ax[1].plot(curves['epoch'], curves['err_AB'], color='#c13c14', lw=2, label='train err A→B')
    ax[1].plot(curves['epoch'], curves['err_BA'], color='#174734', lw=2, label='train err B→A')
    ax[1].plot(curves['epoch'], curves['base_AB'], color='#6b6a63', lw=1, ls='--', label='base (no correction)')
    if not overfit and curves['val_err_AB']:
        val_eps = [e for e in curves['epoch'] if (e % args.log_every == 0 or e == args.epochs - 1)]
        ax[1].plot(val_eps, curves['val_err_AB'], color='#c13c14', lw=1.5, ls=':', marker='o', ms=4, label='val A→B')
        ax[1].plot(val_eps, curves['val_err_BA'], color='#174734', lw=1.5, ls=':', marker='o', ms=4, label='val B→A')
    ax[1].set_title('mean reproj err (px)', loc='left'); ax[1].set_xlabel('epoch')
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=0.25)
    for a in ax:
        for sp in ('top','right'): a.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'curve.png', dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    log(f'saved curve → {out_dir}/curve.png')
    log(f'best {"val" if not overfit else "train"} err = {best_err:.2f} px')

    # auto-run train+val viz on last_model.pt (overfit weights so geometry is
    # easiest to read; train fit should be tight, val gap reveals σ collapse).
    try:
        import subprocess
        for split in ('train', 'val'):
            viz_out = out_dir / f'viz_{split.upper()}.png'
            cmd = ['python', 'scripts/visualization/vis_pred_check.py',
                   '--ckpt', str(out_dir),
                   '--ckpt-name', 'last_model.pt',
                   '--split', split,
                   '--out', str(viz_out),
                   '--n-pairs', '24',
                   '--scenes-root', args.scenes_root or str(Path(args.scene).parent),
                   '--cameras', args.cameras,
                   '--baseline-min', str(args.baseline_min),
                   '--baseline-max', str(args.baseline_max),
                   '--sigma-ypr', str(args.sigma_ypr),
                   '--sigma-t', str(args.sigma_t),
                   '--seed', '42']
            if args.multi_frame:
                cmd.append('--multi-frame')
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and viz_out.exists():
                log(f'saved viz → {viz_out}')
                if cml_logger is not None:
                    cml_logger.report_image('viz', split.upper(), iteration=args.epochs - 1,
                                             local_path=str(viz_out))
            else:
                log(f'[viz {split}] failed: {r.stderr.strip().splitlines()[-3:] if r.stderr else ""}')
    except Exception as e:
        log(f'[viz] error: {e}')

    # legacy BA-reproject viz (pair-only) — preserved for back-compat
    if args.multi_frame:
        return
    try:
        import subprocess
        viz_out = out_dir / 'ba_reproject.png'
        log(f'running vis_ba_reproject → {viz_out} ...')
        r = subprocess.run(
            ['python', 'scripts/visualization/vis_ba_reproject.py',
             '--ckpt', str(out_dir),
             '--n-pairs', '6',
             '--baseline-min', str(max(1, args.baseline_min)),
             '--baseline-max', str(min(args.baseline_max, 10)),
             '--scenes-root', args.scenes_root or str(Path(args.scene).parent),
             '--cameras', args.cameras,
             '--out', str(viz_out)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode == 0 and viz_out.exists():
            log(f'saved viz → {viz_out}')
        else:
            log(f'[viz] failed (rc={r.returncode}): {r.stderr.strip().splitlines()[-3:] if r.stderr else ""}')
    except Exception as e:
        log(f'[viz] skipped: {e!r}')


if __name__ == '__main__':
    main()
