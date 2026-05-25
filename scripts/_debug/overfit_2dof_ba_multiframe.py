"""Phase 1.c — MULTI-FRAME streaming perturbations.

Same contract as overfit_2dof_ba_stream.py but each train step samples a
fresh (frame, ωx, ωy) from the KAMIKADO TRAIN split, and held-out eval
is fixed (frame, pert) pairs from the VAL split. Tests whether the
InfoHead2x2 design generalises across frames, not just across
perturbations of one frame.

  - frozen CalibNetDepth backbone (km_wv_wm_dgx2_n4_img128_8gpu_HEAD)
  - InfoHead2x2 trained from scratch (~99.6 k params)
  - DoFs: omega_x, omega_y; pinhole solver with n_iter=3
  - loss: ||δ - δ_gt||²  (no direct supervision on W)

Reports δ-MSE on a fixed held-out (val frame, pert) batch:
  W=I uniform   /   W=σ-head (NLL)   /   W=learned (ours)

Outputs in scripts/_debug/_outputs/overfit_2dof_ba_multiframe/.
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


CACHE       = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT        = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_2dof_ba_multiframe'

DEFAULT_N_STEPS = 2000
DEFAULT_BATCH   = 16
DEFAULT_N_EVAL  = 512
PITCH_RANGE_DEG = 0.30
YAW_RANGE_DEG   = 0.30
LR              = 1e-3
LOG_EVERY       = 50
BA_N_ITER       = 3
DAMPING         = 1e-3
SEED            = 7

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


def _sample_one(ds, idx, rng):
    omega_x = float(rng.uniform(-PITCH_RANGE_DEG, PITCH_RANGE_DEG))
    omega_y = float(rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG))
    ypr = np.array([0.0, omega_y, omega_x], dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    win = ds.apply_perturbation_explicit(idx, t, ypr)
    return win


def _build_batch(ds, idx_iter, B, rng, max_tries=64):
    """Sample B (frame, pert) windows. idx_iter: callable() -> int."""
    windows = []
    tries = 0
    while len(windows) < B and tries < max_tries * B:
        i = idx_iter()
        win = _sample_one(ds, i, rng)
        tries += 1
        if win is None:
            continue
        windows.append(win)
    if len(windows) < B:
        return None
    return collate_full(windows)


def _move_batch(batch, device):
    return [t.to(device) if torch.is_tensor(t) else t for t in batch]


def _forward_capture_q(model, imgs_u8, dist_uvd, pad_mask, vfp,
                       bucket_uvd, bucket_valid):
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
    per_pt, _ = out
    return per_pt, captured['q']


def _prep_inputs(cfg, true_uvd, dist_uvd, pad_mask, vfp):
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
    ap.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    ap.add_argument('--batch',   type=int, default=DEFAULT_BATCH)
    ap.add_argument('--n-eval',  type=int, default=DEFAULT_N_EVAL,
                    help='# of (val frame, pert) pairs in held-out batch')
    ap.add_argument('--tag',     type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = OUT if args.tag is None else (OUT.parent / f'overfit_2dof_ba_multiframe_{args.tag}')
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    cfg = _load_cfg()
    print(f"[mf] cfg: img={cfg['img_size']} layers={cfg['n_layers']} "
          f"deform={cfg.get('deform_mode')} convnext={cfg.get('use_convnext')}")

    ds_train = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='train',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    ds_val = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    print(f"[mf] train tiles: {len(ds_train.fnames)}   val tiles: {len(ds_val.fnames)}")

    # ─── model ────────────────────────────────────────────────────────
    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    other_missing = [k for k in missing if not k.startswith('info_head')]
    assert len(other_missing) == 0, f"missing: {other_missing[:5]}"
    n_train, n_total = _freeze_except_info_head(model)
    print(f"[mf] params: {n_train/1e3:.1f}k trainable / {n_total/1e6:.2f}M total")

    info_head = model.info_head
    opt = torch.optim.AdamW(info_head.parameters(), lr=LR, weight_decay=0.0)
    DOFS = ['omega_x', 'omega_y']

    # ─── BUILD HELD-OUT VAL EVAL BATCH ────────────────────────────────
    eval_rng = np.random.RandomState(SEED + 1000)
    val_idx_pool = list(range(len(ds_val.fnames)))
    eval_rng.shuffle(val_idx_pool)
    eval_pool_iter = iter(val_idx_pool)
    n_eval = args.n_eval

    def _eval_idx_iter():
        nonlocal eval_pool_iter
        try:
            return next(eval_pool_iter)
        except StopIteration:
            eval_rng.shuffle(val_idx_pool)
            eval_pool_iter = iter(val_idx_pool)
            return next(eval_pool_iter)

    print(f"[mf] building held-out eval batch (N={n_eval} val tiles × 1 pert) ...")
    t_eval_build = time.time()
    eval_batch = _build_batch(ds_val, _eval_idx_iter, n_eval, eval_rng)
    assert eval_batch is not None, "could not build eval batch"
    print(f"[mf]   eval batch built in {time.time()-t_eval_build:.1f}s")
    e = _move_batch(eval_batch, DEVICE)
    e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp, e_bucket_uvd, e_bucket_valid, _ = e
    e_valid, e_duv_oracle, e_uv_obs, e_z, e_K, e_padfull = _prep_inputs(
        cfg, e_true_uvd, e_dist_uvd, e_pad, e_vfp)
    delta_target_eval = _solve_target(e_uv_obs, e_duv_oracle, e_valid, e_z, e_K, DOFS)

    model.eval()
    # chunked backbone forward to avoid OOM at large N_eval
    EVAL_CHUNK = 32
    e_per_pt_list, e_q_list = [], []
    for s in range(0, n_eval, EVAL_CHUNK):
        sl = slice(s, min(s + EVAL_CHUNK, n_eval))
        pp, qq = _forward_capture_q(model, e_imgs[sl], e_dist_uvd[sl], e_pad[sl],
                                     e_vfp[sl], e_bucket_uvd[sl], e_bucket_valid[sl])
        e_per_pt_list.append(pp); e_q_list.append(qq)
    e_per_pt = torch.cat(e_per_pt_list, dim=0)
    e_q      = torch.cat(e_q_list,      dim=0)
    e_duv_pred = e_per_pt[..., :2].detach()
    if e_padfull.any():
        e_duv_pred = e_duv_pred.clone()
        e_duv_pred[e_padfull] = 0.0
    e_sigma_x = e_per_pt[..., 2].exp()
    e_sigma_y = e_per_pt[..., 3].exp()
    e_rho     = e_per_pt[..., 4]
    e_W_sigma = make_info_from_sigma_rho(e_sigma_x, e_sigma_y, e_rho).detach()
    e_W_eye   = torch.eye(2, device=DEVICE).expand(n_eval, e_q.shape[1], 2, 2)

    SOLVE_CHUNK = 32

    def _eval_mse_with_W(W_full):
        d_parts = []
        for s in range(0, n_eval, SOLVE_CHUNK):
            sl = slice(s, min(s + SOLVE_CHUNK, n_eval))
            d_part, _ = _solve_with(e_uv_obs[sl], e_duv_pred[sl], W_full[sl],
                                     e_z[sl], e_K[sl], DOFS, e_valid[sl])
            d_parts.append(d_part)
        d_full = torch.cat(d_parts, dim=0)
        return (d_full - delta_target_eval).pow(2).mean(dim=-1).mean().item(), d_full

    with torch.no_grad():
        mse_w1, _    = _eval_mse_with_W(e_W_eye)
        mse_w2, _    = _eval_mse_with_W(e_W_sigma)
        d_zero       = torch.zeros_like(delta_target_eval)
        mse_zero     = (d_zero - delta_target_eval).pow(2).mean(dim=-1).mean().item()
    print(f"[mf] HELD-OUT BASELINE  do-nothing: {mse_zero:.6e}  "
          f"W=I: {mse_w1:.6e}  W=σ: {mse_w2:.6e}")

    # ─── streaming TRAIN ──────────────────────────────────────────────
    train_idx_pool = list(range(len(ds_train.fnames)))
    rng.shuffle(train_idx_pool)
    train_iter = iter(train_idx_pool)

    def _train_idx_iter():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            rng.shuffle(train_idx_pool)
            train_iter = iter(train_idx_pool)
            return next(train_iter)

    history = []  # (step, train_loss, eval_mse, logdet_mean, eval_time_s)
    t0 = time.time()
    info_head.train()
    for step in range(args.n_steps + 1):
        batch = _build_batch(ds_train, _train_idx_iter, args.batch, rng)
        if batch is None:
            print(f"  step {step}: skipped (could not build train batch)")
            continue
        b = _move_batch(batch, DEVICE)
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _ = b
        valid, duv_oracle, uv_obs, z_pts, K_in, _padfull = _prep_inputs(
            cfg, true_uvd, dist_uvd, pad_mask, vfp)
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

        if step % LOG_EVERY == 0 or step == args.n_steps:
            info_head.eval()
            with torch.no_grad():
                W_l = info_head(e_q)
                eval_mse, _ = _eval_mse_with_W(W_l)
                logdet = torch.linalg.det(W_l).clamp_min(1e-12).log().mean().item()
            info_head.train()
            sps = (step + 1) / max(1e-3, time.time() - t0)
            history.append((step, loss.item(), eval_mse, logdet))
            print(f"  step {step:>4}  loss={loss.item():.4e}  "
                  f"VAL δ-MSE={eval_mse:.4e}  ⟨log det W⟩={logdet:+.2f}  "
                  f"({sps:.2f} step/s)")

    info_head.eval()
    with torch.no_grad():
        W_final = info_head(e_q)
        mse_w3, _ = _eval_mse_with_W(W_final)
    print(f"\n[mf] HELD-OUT FINAL  do-nothing: {mse_zero:.6e}  "
          f"W=I: {mse_w1:.6e}  W=σ: {mse_w2:.6e}  W=learned: {mse_w3:.6e}")
    print(f"[mf] PASS vs uniform: {mse_w3 < mse_w1}   "
          f"PASS vs σ: {mse_w3 < mse_w2}")

    # ─── plots ────────────────────────────────────────────────────────
    steps  = [h[0] for h in history]
    trnL   = [h[1] for h in history]
    evMSE  = [h[2] for h in history]
    logdet = [h[3] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ax.semilogy(steps, evMSE, '-o', color='tab:red', ms=3,
                 label=f'learned W (val held-out N={n_eval})')
    ax.semilogy(steps, trnL,  '-x', color='tab:orange', ms=3, alpha=0.5,
                 label='train loss (fresh tile+pert)')
    ax.axhline(mse_zero, ls=':',  color='tab:green',  label=f'do-nothing ({mse_zero:.2e})')
    ax.axhline(mse_w1,   ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
    ax.axhline(mse_w2,   ls='--', color='tab:purple', label=f'W=σ-head ({mse_w2:.2e})')
    ax.set_xlabel('step'); ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title('(a) train: KAMIKADO train tiles · eval: val held-out')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    with torch.no_grad():
        W_l = info_head(e_q)
        ld_l = torch.linalg.det(W_l).clamp_min(1e-12).log()[e_valid].cpu().numpy()
        ld_s = torch.linalg.det(e_W_sigma).clamp_min(1e-12).log()[e_valid].cpu().numpy()
    ax.scatter(ld_s, ld_l, s=2, alpha=0.18, color='tab:blue')
    if len(ld_l) > 1:
        corr = float(np.corrcoef(ld_s, ld_l)[0, 1])
    else:
        corr = float('nan')
    ax.set_xlabel('log det W from σ-head')
    ax.set_ylabel('log det W from learned head')
    ax.set_title(f'(b) σ vs learned per-query (val held-out) — Pearson r = {corr:+.3f}')
    ax.grid(alpha=0.3)

    ax = axes[2]
    bars = ['do-nothing', 'W=I', 'W=σ', 'W=learn']
    vals = [mse_zero, mse_w1, mse_w2, mse_w3]
    colors = ['tab:green', 'tab:gray', 'tab:purple', 'tab:red']
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('δ-MSE (deg²)')
    ax.set_title(f'(c) final δ-MSE on val held-out N={n_eval}\n'
                 f'multiframe train: {args.n_steps} steps × B={args.batch}')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Phase 1.c: MULTIFRAME streaming (KAMIKADO train, val held-out)\n'
        f'Frozen CalibNetDepth + InfoHead2x2.  BA: solve_pinhole '
        f'n_iter={BA_N_ITER}, dofs={DOFS}.',
        y=1.04, fontsize=10,
    )
    fig.tight_layout()
    out_png = out_dir / 'curves.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[mf] wrote → {out_png}")

    sd_out = out_dir / 'info_head.pt'
    torch.save(info_head.state_dict(), sd_out)
    print(f"[mf] wrote → {sd_out}")

    summary = {
        'n_steps': args.n_steps, 'batch': args.batch, 'n_eval': n_eval,
        'mse_zero': mse_zero, 'mse_W_I': mse_w1, 'mse_W_sigma': mse_w2,
        'mse_W_learned': mse_w3,
        'rmse_axis_zero':    float(np.sqrt(mse_zero)),
        'rmse_axis_W_I':     float(np.sqrt(mse_w1)),
        'rmse_axis_W_sigma': float(np.sqrt(mse_w2)),
        'rmse_axis_W_learn': float(np.sqrt(mse_w3)),
        'pearson_r': corr,
        'history': history,
    }
    torch.save(summary, out_dir / 'summary.pt')
    print(f"[mf] wrote → {out_dir / 'summary.pt'}")
    print("[mf] done.")


if __name__ == '__main__':
    main()
