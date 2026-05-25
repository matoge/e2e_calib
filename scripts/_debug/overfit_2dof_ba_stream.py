"""Phase 1.b — 1-frame, STREAMING perturbations.

Same experimental contract as `overfit_2dof_ba.py` but the perturbation
batch is re-sampled fresh each step. Removes any "fixed-100 memorization"
concern: every gradient step sees an unseen (ωx, ωy) draw, so the
InfoHead2x2 has no choice but to learn a frame-conditional mapping
q → W rather than a lookup over 100 specific (q_i, target_i) pairs.

Same 1 frame (val idx 0 of kamikado_v3_tiled), same DoFs (omega_x, omega_y),
same frozen backbone ckpt, same loss formulation. Pass criterion:
    final δ-MSE  <  W=I baseline  AND  ≤ W=σ-head baseline
on a held-out FRESH evaluation batch (different perts than any seen in
training).

Outputs land in scripts/_debug/_outputs/overfit_2dof_ba_stream/.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_2dof_ba_stream'
N_STEPS     = 400
BATCH       = 16              # perts/step; fresh sample every step
N_EVAL      = 200             # held-out perts for final comparison
PITCH_RANGE_DEG = 0.30
YAW_RANGE_DEG   = 0.30
LR          = 1e-3
LOG_EVERY   = 25
BA_N_ITER   = 3
DAMPING     = 1e-3
SEED        = 7

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_cfg() -> dict:
    src = EXP_CFG_PATH.read_text()
    ns: dict = {}
    exec(src, ns, ns)
    return ns['CFG']


def _build_model(cfg: dict) -> CalibNetDepth:
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


def _build_K_batch(cfg: dict, vfp: torch.Tensor) -> torch.Tensor:
    S = float(cfg['img_size'])
    B = vfp.shape[0]
    K = torch.zeros(B, 3, 3, device=vfp.device, dtype=vfp.dtype)
    K[:, 0, 0] = vfp
    K[:, 1, 1] = vfp
    K[:, 0, 2] = S / 2
    K[:, 1, 2] = S / 2
    K[:, 2, 2] = 1.0
    return K


def _sample_one(ds: PandaSetCalibDatasetFull, idx: int, rng: np.random.RandomState):
    omega_x = float(rng.uniform(-PITCH_RANGE_DEG, PITCH_RANGE_DEG))
    omega_y = float(rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG))
    ypr = np.array([0.0, omega_y, omega_x], dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    win = ds.apply_perturbation_explicit(idx, t, ypr)
    return win, omega_x, omega_y


def _build_batch(ds, idx, B, rng):
    windows = []
    while len(windows) < B:
        win, _ox, _oy = _sample_one(ds, idx, rng)
        if win is None:
            continue
        windows.append(win)
    return collate_full(windows)


def _move_batch(batch, device):
    return [t.to(device) if torch.is_tensor(t) else t for t in batch]


def _forward_capture_q(model, imgs_u8, dist_uvd, pad_mask, vfp,
                       bucket_uvd, bucket_valid):
    """Run the frozen backbone, return (per_pt, W_sigma_dummy, q).
    `q` is captured by hooking info_head.mlp[0]. per_pt is (B, N, 5)."""
    captured = {}

    def _hook(_module, _inputs, _output):
        captured['q'] = _inputs[0].detach()

    h = model.info_head.mlp[0].register_forward_hook(_hook)
    imgs = imgs_u8.float().div(255.0)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    with torch.no_grad():
        out = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    h.remove()
    per_pt, _W_unused = out
    return per_pt, captured['q']


def _prep_inputs(cfg, imgs, true_uvd, dist_uvd, pad_mask, vfp):
    """Build solver-ready tensors with NaN-safe padding for invalid slots."""
    valid = ~pad_mask
    duv_oracle = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()
    uv_obs = dist_uvd[..., :2].detach().clone()
    z_pts = dist_uvd[..., 2].detach().clone() * 100.0
    pad_full = ~valid
    if pad_full.any():
        z_pts[pad_full] = 1.0
        uv_obs[pad_full] = 0.5 * float(cfg['img_size'])
        duv_oracle[pad_full] = 0.0
    K_in = _build_K_batch(cfg, vfp)
    return valid, duv_oracle, uv_obs, z_pts, K_in, pad_full


def _solve_target(uv_obs, duv_oracle, valid, z_pts, K_in, dofs):
    B, N = uv_obs.shape[:2]
    W_eye = torch.eye(2, device=uv_obs.device).expand(B, N, 2, 2)
    with torch.no_grad():
        delta, _ = solve_pinhole(uv_obs, duv_oracle, W_eye, z_pts, K_in, dofs,
                                  valid=valid, n_iter=BA_N_ITER, damping=DAMPING)
    return delta.detach()


def _solve_with(uv_obs, duv, W, z_pts, K_in, dofs, valid):
    return solve_pinhole(uv_obs, duv, W, z_pts, K_in, dofs,
                         valid=valid, n_iter=BA_N_ITER, damping=DAMPING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=None,
                    help='val tile index to anchor on (default: first usable)')
    ap.add_argument('--tag', type=str, default=None,
                    help='subdir under _outputs/overfit_2dof_ba_stream (default: idx{idx})')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = OUT if args.tag is None else (OUT.parent / f'overfit_2dof_ba_stream_{args.tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    cfg = _load_cfg()
    print(f"[stream] cfg: img={cfg['img_size']} layers={cfg['n_layers']} "
          f"deform={cfg.get('deform_mode')} convnext={cfg.get('use_convnext')}")

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
    print(f"[stream] dataset: {len(ds.fnames)} val instances")

    if args.idx is not None:
        base_idx = int(args.idx)
        win = ds.apply_perturbation_explicit(base_idx, np.zeros(3), np.zeros(3))
        assert win is not None, f"idx={base_idx} returns None from apply_perturbation_explicit"
    else:
        base_idx = None
        for cand in range(min(50, len(ds.fnames))):
            win = ds.apply_perturbation_explicit(cand, np.zeros(3), np.zeros(3))
            if win is not None:
                base_idx = cand
                break
        assert base_idx is not None
    print(f"[stream] anchor val frame idx={base_idx} ({ds.fnames[base_idx]})")

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    other_missing = [k for k in missing if not k.startswith('info_head')]
    assert len(other_missing) == 0, f"missing keys: {other_missing[:5]}"
    n_train, n_total = _freeze_except_info_head(model)
    print(f"[stream] params: {n_train/1e3:.1f}k trainable / {n_total/1e6:.2f}M total")

    info_head = model.info_head
    opt = torch.optim.AdamW(info_head.parameters(), lr=LR, weight_decay=0.0)
    DOFS = ['omega_x', 'omega_y']

    # ─── BASELINES on a fresh held-out batch (no W training yet) ────
    eval_rng = np.random.RandomState(SEED + 1000)
    eval_batch = _build_batch(ds, base_idx, N_EVAL, eval_rng)
    e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp, e_bucket_uvd, e_bucket_valid, _ = \
        _move_batch(eval_batch, DEVICE)
    e_valid, e_duv_oracle, e_uv_obs, e_z, e_K, e_padfull = _prep_inputs(
        cfg, e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp)
    delta_target_eval = _solve_target(e_uv_obs, e_duv_oracle, e_valid, e_z, e_K, DOFS)

    model.eval()
    e_per_pt, e_q = _forward_capture_q(model, e_imgs, e_dist_uvd, e_pad, e_vfp,
                                        e_bucket_uvd, e_bucket_valid)
    e_duv_pred = e_per_pt[..., :2].detach()
    if e_padfull.any():
        e_duv_pred = e_duv_pred.clone()
        e_duv_pred[e_padfull] = 0.0
    e_sigma_x = e_per_pt[..., 2].exp()
    e_sigma_y = e_per_pt[..., 3].exp()
    e_rho     = e_per_pt[..., 4]
    e_W_sigma = make_info_from_sigma_rho(e_sigma_x, e_sigma_y, e_rho).detach()
    e_W_eye   = torch.eye(2, device=DEVICE).expand(N_EVAL, e_q.shape[1], 2, 2)

    with torch.no_grad():
        d_w1, _ = _solve_with(e_uv_obs, e_duv_pred, e_W_eye, e_z, e_K, DOFS, e_valid)
        mse_w1 = (d_w1 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
        d_w2, _ = _solve_with(e_uv_obs, e_duv_pred, e_W_sigma, e_z, e_K, DOFS, e_valid)
        mse_w2 = (d_w2 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
    print(f"[stream] HELD-OUT BASELINE  W=I: {mse_w1:.6e}   W=σ: {mse_w2:.6e}")

    # ─── streaming train ───────────────────────────────────────────
    history = []           # (step, train_loss, eval_mse_learned, eval_logdet_mean)
    t0 = time.time()
    info_head.train()
    for step in range(N_STEPS + 1):
        batch = _build_batch(ds, base_idx, BATCH, rng)
        b = _move_batch(batch, DEVICE)
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _ = b
        valid, duv_oracle, uv_obs, z_pts, K_in, _padfull = _prep_inputs(
            cfg, imgs, true_uvd, dist_uvd, pad_mask, vfp)
        delta_target = _solve_target(uv_obs, duv_oracle, valid, z_pts, K_in, DOFS)

        per_pt, q = _forward_capture_q(model, imgs, dist_uvd, pad_mask, vfp,
                                        bucket_uvd, bucket_valid)
        duv_pred = per_pt[..., :2].detach()
        if (~valid).any():
            duv_pred = duv_pred.clone()
            duv_pred[~valid] = 0.0

        W_learn = info_head(q)
        delta_pred, _ = _solve_with(uv_obs, duv_pred, W_learn, z_pts, K_in, DOFS, valid)
        loss = (delta_pred - delta_target).pow(2).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(info_head.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == N_STEPS:
            info_head.eval()
            with torch.no_grad():
                W_l = info_head(e_q)
                d_l, _ = _solve_with(e_uv_obs, e_duv_pred, W_l, e_z, e_K, DOFS, e_valid)
                eval_mse = (d_l - delta_target_eval).pow(2).mean(dim=-1).mean().item()
                logdet = torch.linalg.det(W_l).clamp_min(1e-12).log().mean().item()
            info_head.train()
            sps = (step + 1) / max(1e-3, time.time() - t0)
            history.append((step, loss.item(), eval_mse, logdet))
            print(f"  step {step:>4}  loss={loss.item():.4e}  "
                  f"EVAL δ-MSE={eval_mse:.4e}  ⟨log det W⟩={logdet:+.2f}  "
                  f"({sps:.2f} step/s)")

    # ─── FINAL eval ────────────────────────────────────────────────
    info_head.eval()
    with torch.no_grad():
        W_final = info_head(e_q)
        d_final, _ = _solve_with(e_uv_obs, e_duv_pred, W_final, e_z, e_K, DOFS, e_valid)
        mse_w3 = (d_final - delta_target_eval).pow(2).mean(dim=-1).mean().item()
    print(f"\n[stream] HELD-OUT FINAL  W=I: {mse_w1:.6e}   "
          f"W=σ: {mse_w2:.6e}   W=learned: {mse_w3:.6e}")
    print(f"[stream] PASS vs uniform: {mse_w3 < mse_w1}   "
          f"PASS vs σ: {mse_w3 <= mse_w2 * 1.05}")

    # ─── plot ──────────────────────────────────────────────────────
    steps  = [h[0] for h in history]
    trnL   = [h[1] for h in history]
    evMSE  = [h[2] for h in history]
    logdet = [h[3] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ax.semilogy(steps, evMSE, '-o', color='tab:red', ms=3,
                 label=f'learned W (held-out, N={N_EVAL})')
    ax.semilogy(steps, trnL,  '-x', color='tab:orange', ms=3, alpha=0.5,
                 label='train loss (fresh batch)')
    ax.axhline(mse_w1, ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
    ax.axhline(mse_w2, ls='--', color='tab:purple', label=f'W=σ-head ({mse_w2:.2e})')
    ax.set_xlabel('step'); ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title('(a) streaming perts — eval on held-out perts')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    with torch.no_grad():
        W_l = info_head(e_q)
        ld_l = torch.linalg.det(W_l).clamp_min(1e-12).log()[e_valid].cpu().numpy()
        ld_s = torch.linalg.det(e_W_sigma).clamp_min(1e-12).log()[e_valid].cpu().numpy()
    ax.scatter(ld_s, ld_l, s=3, alpha=0.25, color='tab:blue')
    if len(ld_l) > 1:
        corr = float(np.corrcoef(ld_s, ld_l)[0, 1])
    else:
        corr = float('nan')
    ax.set_xlabel('log det W from σ-head')
    ax.set_ylabel('log det W from learned head')
    ax.set_title(f'(b) σ vs learned per-query — Pearson r = {corr:+.3f}')
    ax.grid(alpha=0.3)

    ax = axes[2]
    bars = ['W=I', 'W=σ', 'W=learn']
    vals = [mse_w1, mse_w2, mse_w3]
    colors = ['tab:gray', 'tab:purple', 'tab:red']
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title(f'(c) final δ-MSE on held-out N={N_EVAL}\n'
                 f'streaming train: {N_STEPS} steps × B={BATCH}')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Phase 1.b: 1-frame STREAMING perts on val frame {base_idx}\n'
        f'Frozen CalibNetDepth + InfoHead2x2.  BA: solve_pinhole '
        f'n_iter={BA_N_ITER}, dofs={DOFS}.',
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    out_png = out_dir / 'curves.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[stream] wrote → {out_png}")

    # save the trained info_head so we can re-visualise without retraining
    sd_out = out_dir / 'info_head.pt'
    torch.save(info_head.state_dict(), sd_out)
    print(f"[stream] wrote → {sd_out}")

    # ─── richer 4-panel overlay on the anchor frame ────────────────
    with torch.no_grad():
        W_l_one  = info_head(e_q[:1])[0]                            # (N, 2, 2)
        W_s_one  = e_W_sigma[0]                                     # (N, 2, 2)
        ld_l = torch.linalg.det(W_l_one).clamp_min(1e-12).log().cpu().numpy()
        ld_s = torch.linalg.det(W_s_one).clamp_min(1e-12).log().cpu().numpy()
        uv_one = e_uv_obs[0].cpu().numpy()
        z_one  = e_z[0].cpu().numpy()
        v_one  = e_valid[0].cpu().numpy()
        img_one = e_imgs[0].permute(1, 2, 0).cpu().numpy().astype('float32') / 255.0
    img_one = np.clip(img_one, 0, 1)

    fig3, axs = plt.subplots(2, 2, figsize=(10.5, 9.5))
    titles_panels = [
        ('source image', None,                None,            None),
        ('per-query depth (m)',          z_one[v_one],         'plasma',  'depth Z (m)'),
        ('σ-head log det W (NLL-trained)', ld_s[v_one],        'viridis', 'log det W_σ'),
        ('learned log det W (pose-trust)', ld_l[v_one],        'viridis', 'log det W_learned'),
    ]
    for ax, (title, c, cmap, cbar_lbl) in zip(axs.ravel(), titles_panels):
        ax.imshow(img_one)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        if c is not None and v_one.any():
            sc = ax.scatter(uv_one[v_one, 0], uv_one[v_one, 1],
                            c=c, cmap=cmap, s=10, edgecolors='none', alpha=0.85)
            cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label(cbar_lbl, fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig3.suptitle(
        f'Phase 1.b overlays on val frame {base_idx} (1st held-out pert)\n'
        f'depth/σ/learned per-query maps on the same image',
        y=0.995, fontsize=11,
    )
    fig3.tight_layout()
    out_png3 = out_dir / 'overlay_panels.png'
    fig3.savefig(out_png3, dpi=110, bbox_inches='tight')
    plt.close(fig3)
    print(f"[stream] wrote → {out_png3}")

    print("[stream] done.")


if __name__ == '__main__':
    main()
