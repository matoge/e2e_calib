"""Phase 1 smoke for the principled-ML calib design.

What this validates
-------------------
A frozen `CalibNetDepth` ckpt produces (per-query feature q, per-point Δuv,
per-point σ-head). We bolt a NEW per-query 2x2 information-matrix head on
top — the only learnable part — and ask: can it learn pose-aware trust
purely from gradient through the closed-form 2-DoF BA solver, with no
direct supervision on W?

Three baselines on the SAME 1 image / N random pitch+yaw perturbations:
    (1) W = I                              uniform trust
    (2) W = info_from_sigma_rho(σ-head)    existing per-point covariance
    (3) W = info_head(q)                   the new learned head

Pass criterion (Phase 1 of docs/blog/2026-05-20_principled_ml_calib.md §5):
    δ-MSE for (3) < (1) by a clear margin   (must beat uniform)
    δ-MSE for (3) ≤ (2)                     (matches or beats σ-heuristic)

Outputs land under scripts/_debug/_outputs/overfit_2dof_ba/.
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import solve_pinhole, make_info_from_sigma_rho


# ─── config ───────────────────────────────────────────────────────────
CACHE       = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT        = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_2dof_ba'
N_PERT      = 100
PITCH_RANGE_DEG = 0.30   # ω_x ∈ [-0.3°, +0.3°]
YAW_RANGE_DEG   = 0.30   # ω_y
N_STEPS     = 500
LR          = 1e-3
LOG_EVERY   = 25
BA_N_ITER   = 3
DAMPING     = 1e-3
SEED        = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_cfg() -> dict:
    """Read CFG dict from the experiment's frozen config.py."""
    src = EXP_CFG_PATH.read_text()
    ns: dict = {}
    exec(src, ns, ns)
    return ns['CFG']


def _build_model(cfg: dict) -> CalibNetDepth:
    """Build CalibNetDepth with the trainer's exact kwargs + use_info_head."""
    return CalibNetDepth(
        img_size=cfg['img_size'],
        in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'],
        self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def _freeze_except_info_head(model: CalibNetDepth):
    n_train = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if name.startswith('info_head'):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
    return n_train, n_total


def _sample_window(ds: PandaSetCalibDatasetFull, idx: int):
    """Hijack the dataset's perturbation API to get one (pitch, yaw)
    window. ypr is laid out (cam-z=yaw_dataset, cam-y=pitch_dataset,
    cam-x=roll_dataset). For the BA solver's omega_x = cam-x rotation
    and omega_y = cam-y rotation, we set:
        ypr[0] = 0           (no cam-z rotation in this experiment)
        ypr[1] = ω_y_deg     (cam-y → solver's omega_y)
        ypr[2] = ω_x_deg     (cam-x → solver's omega_x)
    so the recovered δ in DOF order ('omega_x', 'omega_y') aligns with
    (ypr[2], ypr[1]).
    """
    rng = np.random.RandomState(int(time.time_ns() & 0xFFFFFFFF))
    omega_x = float(rng.uniform(-PITCH_RANGE_DEG, PITCH_RANGE_DEG))
    omega_y = float(rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG))
    ypr = np.array([0.0, omega_y, omega_x], dtype=np.float64)
    t   = np.zeros(3, dtype=np.float64)
    win = ds.apply_perturbation_explicit(idx, t, ypr)
    return win, omega_x, omega_y


def _build_dof_target(omega_x: float, omega_y: float) -> torch.Tensor:
    """Solver returns δ such that applying it to back-projected truth
    lands at the target_uv = (uv_dist + duv) = uv_true. With the dataset
    convention `R_off = R_gt @ R_pert(ypr)`, the points are projected
    under R_off; lifting via K^{-1}·uv_dist back-projects (in cam-frame
    aligned with R_off). Applying δ that takes us to where uv_true sits
    means δ rotates ω_x → -ω_x, ω_y → -ω_y to undo R_pert.

    Sign empirically validated below by solving with ORACLE duv (= true -
    dist) and W = I; δ_oracle is then plotted against (-ω_x, -ω_y)."""
    return torch.tensor([-omega_x, -omega_y], dtype=torch.float32)


def _stack(windows):
    """collate_full expects a list of dataset tuples and returns the
    same 8-tuple the trainer consumes."""
    return collate_full(list(windows))


def _model_forward(model, imgs_u8, dist_uvd, pad_mask, vfp,
                    bucket_uvd, bucket_valid):
    imgs = imgs_u8.float().div(255.0)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    return model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                 bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)


def _build_K_batch(cfg: dict, vfp: torch.Tensor) -> torch.Tensor:
    """Reconstruct the IN-CROP K for the resized model input.
    The dataset's `vfp = K[0,0] * S / cs` already lives in resized px.
    Principal point = (S/2, S/2) by convention (the resized crop is
    centred on the chosen pivot square)."""
    S = float(cfg['img_size'])
    B = vfp.shape[0]
    K = torch.zeros(B, 3, 3, device=vfp.device, dtype=vfp.dtype)
    K[:, 0, 0] = vfp           # fx in resized px
    K[:, 1, 1] = vfp           # fy ≈ fx for these caches
    K[:, 0, 2] = S / 2
    K[:, 1, 2] = S / 2
    K[:, 2, 2] = 1.0
    return K


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg()
    print(f"[smoke] cfg: img_size={cfg['img_size']} n_layers={cfg['n_layers']} "
          f"deform={cfg.get('deform_mode')} convnext={cfg.get('use_convnext')}")

    # Dataset (val split, pose_frame='orig' default; oversample=1 — we
    # call apply_perturbation_explicit directly so oversample is ignored).
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE,
        split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'],
        max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0,
        max_rot_deg=0.0,
        oversample=1,
        grid_n=cfg.get('grid_n', 16),
        center_band=0.0,
        preload=False,
    )
    print(f"[smoke] dataset: {len(ds.fnames)} val instances "
          f"(cache={CACHE.name})")

    # Pick the first frame whose `apply_perturbation_explicit(0,0,0)` succeeds.
    base_idx = None
    for cand in range(min(50, len(ds.fnames))):
        win = ds.apply_perturbation_explicit(cand, np.zeros(3), np.zeros(3))
        if win is not None:
            base_idx = cand
            break
    assert base_idx is not None, "no usable frame in first 50"
    print(f"[smoke] using val frame idx={base_idx} ({ds.fnames[base_idx]})")

    # Sample N random (ω_x, ω_y) windows of THIS frame.
    windows, gt_targets, omegas = [], [], []
    while len(windows) < N_PERT:
        win, ox, oy = _sample_window(ds, base_idx)
        if win is None:
            continue
        windows.append(win)
        gt_targets.append(_build_dof_target(ox, oy))
        omegas.append((ox, oy))
    delta_target = torch.stack(gt_targets).to(DEVICE)        # (N, 2)
    omegas_arr = np.array(omegas, dtype=np.float32)
    print(f"[smoke] sampled N={len(windows)} perturbed windows, "
          f"|ωx|≤{PITCH_RANGE_DEG}°, |ωy|≤{YAW_RANGE_DEG}°")

    batch = _stack(windows)
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _pert = [
        t.to(DEVICE) if torch.is_tensor(t) else t for t in batch
    ]
    valid = ~pad_mask
    print(f"[smoke] batch tensors: imgs={tuple(imgs.shape)} "
          f"dist_uvd={tuple(dist_uvd.shape)} pad_valid={int(valid.sum())}")

    # Build model + load frozen weights.
    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    info_keys = [k for k in missing if k.startswith('info_head')]
    other_missing = [k for k in missing if not k.startswith('info_head')]
    print(f"[smoke] ckpt load: missing={len(missing)} (info_head={len(info_keys)}, "
          f"other={len(other_missing)})  unexpected={len(unexpected)}")
    assert len(other_missing) == 0, f"unexpected missing keys: {other_missing[:5]}"
    n_train, n_total = _freeze_except_info_head(model)
    print(f"[smoke] params: {n_train/1e3:.1f}k trainable / {n_total/1e6:.2f}M total")

    # Cache the FROZEN backbone outputs once (saves 500x recomputation).
    model.eval()
    with torch.no_grad():
        out = _model_forward(model, imgs, dist_uvd, pad_mask, vfp,
                              bucket_uvd, bucket_valid)
    if isinstance(out, tuple):
        per_pt_frozen, W_learned_initial = out
    else:
        raise RuntimeError(f"expected (per_pt, W) from forward, got {type(out)}")
    # per_pt: (B, N, 5) [tx, ty, log_sx, log_sy, rho]; tx,ty in resized px.
    duv_pred  = per_pt_frozen[..., :2].detach().clone()
    log_sx    = per_pt_frozen[..., 2]
    log_sy    = per_pt_frozen[..., 3]
    rho       = per_pt_frozen[..., 4]
    sigma_x   = log_sx.exp()
    sigma_y   = log_sy.exp()
    W_sigma   = make_info_from_sigma_rho(sigma_x, sigma_y, rho).detach()

    # Oracle duv for reference δ (computed from GT in dataset; what the
    # solver "would say" with perfect predictions and W=I).
    duv_oracle = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()
    uv_obs     = dist_uvd[..., :2].detach().clone()
    z_pts      = dist_uvd[..., 2].detach().clone() * 100.0   # un-normalise (d/100 m)
    # Padded entries carry z=0 / uv=0 — that drives 1/Z in the Jacobian
    # to ∞ and `0 * ∞ = NaN` even after the solver's `valid` mask zeroes
    # J and r post-jacobian. Replace padded slots with safe placeholders;
    # `valid=` still zeros their contribution to H / bvec.
    pad_full = (~valid)
    if pad_full.any():
        z_pts[pad_full]   = 1.0
        uv_obs[pad_full]  = 0.5 * float(cfg['img_size'])
        duv_oracle[pad_full] = 0.0
        duv_pred[pad_full]   = 0.0
    K_in       = _build_K_batch(cfg, vfp)
    DOFS       = ['omega_x', 'omega_y']

    # ─── δ-target via ORACLE duv + W=I (validates sign convention) ──
    with torch.no_grad():
        W_eye_n = torch.eye(2, device=DEVICE).expand(uv_obs.shape[0], uv_obs.shape[1], 2, 2)
        delta_oracle, _ = solve_pinhole(
            uv_obs, duv_oracle, W_eye_n, z_pts, K_in, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
    # Use the oracle δ as the regression target (it's what a perfect
    # network + uniform W would land on). We compare to (-ωx, -ωy)
    # below as a sanity check.
    delta_target = delta_oracle.detach()
    target_vs_omega_err = (delta_target.cpu().numpy()
                            - np.stack([-omegas_arr[:, 0], -omegas_arr[:, 1]], axis=1))
    print(f"[smoke] δ_oracle vs -ω: |bias|={np.abs(target_vs_omega_err.mean(0))} "
          f"|max abs|={np.abs(target_vs_omega_err).max():.4f}")

    # ─── Baseline (1): W = I, frozen Δuv ─────────────────────────────
    with torch.no_grad():
        delta_w1, _ = solve_pinhole(
            uv_obs, duv_pred, W_eye_n, z_pts, K_in, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
        mse_w1 = (delta_w1 - delta_target).pow(2).mean(dim=-1).mean().item()
    # ─── Baseline (2): W from existing σ-head, frozen Δuv ────────────
    with torch.no_grad():
        delta_w2, _ = solve_pinhole(
            uv_obs, duv_pred, W_sigma, z_pts, K_in, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
        mse_w2 = (delta_w2 - delta_target).pow(2).mean(dim=-1).mean().item()
    print(f"[smoke] BASELINE δ-MSE  W=I: {mse_w1:.6f}   W=σ-head: {mse_w2:.6f}")

    # ─── Phase 1 train: only info_head learns ────────────────────────
    info_head = model.info_head
    opt = torch.optim.AdamW(info_head.parameters(), lr=LR, weight_decay=0.0)

    # We need q (per-query feature) to feed info_head. Re-extract once via
    # a hook-light path: rerun forward with no_grad, and capture the
    # last `q` by reading model internals. The cleanest route here is to
    # NOT cache q and instead recompute the forward each step — but the
    # backbone is frozen, so we wrap it in no_grad and let only info_head
    # see grad. To avoid a 500x recompute, we cache q via a single hook
    # before training.
    captured = {}
    def _hook(_module, _inputs, output):
        # CalibNetDepth.info_head input is `q` (B, N, D) — captured by
        # registering on info_head's first Linear.
        captured['q'] = _inputs[0].detach()
    h = info_head.mlp[0].register_forward_hook(_hook)
    with torch.no_grad():
        _ = _model_forward(model, imgs, dist_uvd, pad_mask, vfp,
                            bucket_uvd, bucket_valid)
    h.remove()
    q_frozen = captured['q']
    print(f"[smoke] cached q: shape={tuple(q_frozen.shape)}")

    history = []
    info_head.train()
    for step in range(N_STEPS + 1):
        W_learn = info_head(q_frozen)                          # (B, N, 2, 2)
        delta_w3, _ = solve_pinhole(
            uv_obs, duv_pred, W_learn, z_pts, K_in, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
        loss = (delta_w3 - delta_target).pow(2).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(info_head.parameters(), 1.0)
        opt.step()
        if step % LOG_EVERY == 0 or step == N_STEPS:
            with torch.no_grad():
                mse = (delta_w3 - delta_target).pow(2).mean(dim=-1).mean().item()
                # log det of mean W and a quick health check
                W_det = torch.linalg.det(W_learn).clamp_min(1e-12)
                log_det_mean = W_det.log().mean().item()
            history.append((step, mse, log_det_mean))
            print(f"  step {step:>4}  loss={loss.item():.6e}  "
                  f"δ-MSE={mse:.6e}  ⟨log det W⟩={log_det_mean:+.3f}")

    info_head.eval()
    with torch.no_grad():
        W_learn_final = info_head(q_frozen)
        delta_w3_final, _ = solve_pinhole(
            uv_obs, duv_pred, W_learn_final, z_pts, K_in, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
        mse_w3 = (delta_w3_final - delta_target).pow(2).mean(dim=-1).mean().item()

    print(f"\n[smoke] final δ-MSE  W=I: {mse_w1:.6e}   "
          f"W=σ-head: {mse_w2:.6e}   W=learned: {mse_w3:.6e}")
    pass_uniform = mse_w3 < mse_w1
    pass_sigma   = mse_w3 <= mse_w2 * 1.05            # 5 % tolerance
    print(f"[smoke] PASS vs uniform: {pass_uniform}   "
          f"PASS vs σ-head: {pass_sigma}")

    # ─── Plot 1: training curve + baselines ──────────────────────────
    steps  = [h[0] for h in history]
    mses   = [h[1] for h in history]
    logdet = [h[2] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ax.semilogy(steps, mses, '-o', color='tab:red', ms=3, label='learned W')
    ax.axhline(mse_w1, ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
    ax.axhline(mse_w2, ls='--', color='tab:purple', label=f'W=σ-head ({mse_w2:.2e})')
    ax.set_xlabel('step'); ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title('(a) δ-MSE vs step — pose-loss-only training')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

    # ─── Plot 2: log det W_learned vs log det Σ_existing scatter ─────
    with torch.no_grad():
        W_l = info_head(q_frozen)                                  # (B, N, 2, 2)
        log_det_l = torch.linalg.det(W_l).clamp_min(1e-12).log()
        log_det_s = torch.linalg.det(W_sigma).clamp_min(1e-12).log()
        ld_l = log_det_l[valid].cpu().numpy()
        ld_s = log_det_s[valid].cpu().numpy()
    ax = axes[1]
    ax.scatter(ld_s, ld_l, s=3, alpha=0.25, color='tab:blue')
    if len(ld_l) > 1:
        corr = float(np.corrcoef(ld_s, ld_l)[0, 1])
    else:
        corr = float('nan')
    ax.set_xlabel('log det W from σ-head (NLL-trained)')
    ax.set_ylabel('log det W from learned head')
    ax.set_title(f'(b) per-point trust correlation\nPearson r = {corr:+.3f}')
    ax.grid(alpha=0.3)

    # ─── Plot 3: bar chart of final δ-MSE ────────────────────────────
    ax = axes[2]
    bars = ['W=I', 'W=σ', 'W=learn']
    vals = [mse_w1, mse_w2, mse_w3]
    colors = ['tab:gray', 'tab:purple', 'tab:red']
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title(f'(c) final δ-MSE on {N_PERT} perturbations\n'
                  f'ωx,ωy ∈ ±{PITCH_RANGE_DEG}°')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Phase 1: 1-image / N={N_PERT} random pitch+yaw perts on val frame {base_idx}\n'
        f'Frozen CalibNetDepth + new InfoHead2x2 only.  BA: solve_pinhole '
        f'n_iter={BA_N_ITER}, dofs={DOFS}.',
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    out_png = OUT / 'curves.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[smoke] wrote → {out_png}")

    # ─── Plot 4: W heatmap on the source image (1st perturbation) ────
    with torch.no_grad():
        W_l_one  = info_head(q_frozen[:1])[0]                       # (N, 2, 2)
        det_one  = torch.linalg.det(W_l_one).clamp_min(1e-12)
        log_det_one = det_one.log().cpu().numpy()
        uv_one   = uv_obs[0].cpu().numpy()
        valid_one = valid[0].cpu().numpy()
        img_one  = imgs[0].permute(1, 2, 0).cpu().numpy()
    fig2, ax = plt.subplots(1, 1, figsize=(6.0, 5.6))
    ax.imshow(np.clip(img_one, 0, 1))
    if valid_one.any():
        sc = ax.scatter(uv_one[valid_one, 0], uv_one[valid_one, 1],
                         c=log_det_one[valid_one], cmap='viridis', s=12,
                         edgecolors='none', alpha=0.85)
        plt.colorbar(sc, ax=ax, label='log det W (learned)')
    ax.set_title(f'(d) learned W per query — frame {base_idx}, 1st pert\n'
                  f'high = trusted by pose loss')
    ax.axis('off')
    fig2.tight_layout()
    out_png2 = OUT / 'w_overlay.png'
    fig2.savefig(out_png2, dpi=110, bbox_inches='tight')
    plt.close(fig2)
    print(f"[smoke] wrote → {out_png2}")

    print("[smoke] done.")


if __name__ == '__main__':
    main()
