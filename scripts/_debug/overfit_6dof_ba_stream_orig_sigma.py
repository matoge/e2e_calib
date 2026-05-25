"""Phase 1.b-orig-σ — frozen 6-DoF BA overfit, same as
overfit_6dof_ba_stream_orig.py BUT with a fresh σ-head (q → log σx, log σy, ρ)
that produces W via make_info_from_sigma_rho.

Goal: variant (c) of "HEAD だけ動かす" — train a small MLP that re-creates the
σ_x, σ_y, ρ slice (the existing per_pt 5-vec carries [Δu, Δv, log σx, log σy,
ρ], but the σ part of that vector was supervised by per-pt Gaussian NLL during
pretraining). Here we add a NEW σ-head from q, train ONLY this head with the
pose-MSE objective on a 1-frame streaming-pert anchor. Δuv stays frozen
(taken from per_pt[..., :2] of the frozen baseline backbone+heads). The
existing info_head is NOT used.

  q (B,N,D)  →  σ-head (small MLP)  →  [log σx, log σy, ρ]
                                       │
                                       └→ make_info_from_sigma_rho → W (B,N,2,2)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import solve_kb_xyz, make_info_from_sigma_rho

try:
    from clearml import Task as _ClearMLTask
except Exception:  # pragma: no cover
    _ClearMLTask = None


CACHE       = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT        = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
OUT         = REPO / 'scripts' / '_debug' / '_outputs' / 'overfit_6dof_ba_stream_orig_sigma'

DEFAULT_N_STEPS = 600
DEFAULT_BATCH   = 16
DEFAULT_N_EVAL  = 200
LR              = 1e-3
LOG_EVERY       = 25
BA_N_ITER       = 6
DAMPING         = 1e-3
SEED            = 7

DOF_PRESETS = {
    '2dof':   ['omega_x', 'omega_y'],
    '2dof_t': ['tx', 'ty'],
    '3dof':   ['omega_x', 'omega_y', 'omega_z'],
    '4dof':   ['omega_x', 'omega_y', 'tx', 'ty'],
    '6dof':   ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz'],
    '10dof':  ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz',
               'dfx', 'dfy', 'dcx', 'dcy'],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SigmaHeadFromQ(nn.Module):
    """q → [log σx, log σy, ρ] per query. W = make_info_from_sigma_rho(σx, σy, ρ).

    Init so that σx=σy=1, ρ=0 → W ≈ I (PSD identity, neutral starting trust)."""

    def __init__(self, d: int, hidden_mul: int = 2):
        super().__init__()
        h = d * hidden_mul
        self.mlp = nn.Sequential(
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # raw output = 0 → log σx = 0, log σy = 0, ρ = 0 → σx=σy=1, ρ=0 → W=I.

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(q)
        log_sx = raw[..., 0]
        log_sy = raw[..., 1]
        # bound ρ in (-0.95, 0.95) for numerical safety in W = Σ⁻¹.
        rho = torch.tanh(raw[..., 2]) * 0.95
        sx = log_sx.exp()
        sy = log_sy.exp()
        W = make_info_from_sigma_rho(sx, sy, rho)
        return W


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


def _freeze_all(model: CalibNetDepth):
    for p in model.parameters():
        p.requires_grad_(False)


def _local_to_orig_scale(cs: torch.Tensor, S: float) -> torch.Tensor:
    return (cs / float(S)).reshape(-1, 1, 1)


def _sample_one(ds, idx, rng, *, rot_deg, t_m, dfx_pct, dfy_pct):
    if rot_deg > 0.0:
        omega_x = float(rng.uniform(-rot_deg, rot_deg))
        omega_y = float(rng.uniform(-rot_deg, rot_deg))
        omega_z = float(rng.uniform(-rot_deg, rot_deg))
    else:
        omega_x = omega_y = omega_z = 0.0
    ypr = np.array([omega_z, omega_y, omega_x], dtype=np.float64)
    if t_m > 0.0:
        t = (rng.uniform(-1.0, 1.0, size=3) * t_m).astype(np.float64)
    else:
        t = np.zeros(3, dtype=np.float64)
    dfx = float(rng.uniform(-dfx_pct, dfx_pct)) if dfx_pct > 0.0 else 0.0
    dfy = float(rng.uniform(-dfy_pct, dfy_pct)) if dfy_pct > 0.0 else 0.0
    return ds.apply_perturbation_explicit(idx, t, ypr,
                                           dfx_pct=dfx, dfy_pct=dfy)


def _build_batch(ds, idx, B, rng, *, rot_deg, t_m, dfx_pct, dfy_pct,
                 max_tries=64):
    windows = []
    tries = 0
    while len(windows) < B and tries < max_tries * B:
        win = _sample_one(ds, idx, rng, rot_deg=rot_deg, t_m=t_m,
                          dfx_pct=dfx_pct, dfy_pct=dfy_pct)
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
    per_pt, _W_unused = out
    return per_pt, captured['q']


def _prep_inputs(true_uvd, dist_uvd, pad_mask,
                 pts_cam_orig, duv_orig, K_orig, cs):
    valid = ~pad_mask
    pad_full = ~valid
    duv_oracle_local = (true_uvd[..., :2] - dist_uvd[..., :2]).detach()
    duv_oracle_orig = duv_orig.detach().clone()
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        duv_oracle_local = duv_oracle_local.clone()
        duv_oracle_local[pad_full] = 0.0
        duv_oracle_orig[pad_full] = 0.0
        P0_orig[pad_full] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=P0_orig.dtype, device=P0_orig.device)
    return valid, duv_oracle_local, duv_oracle_orig, P0_orig, K_orig, cs


def _solve_target(P0_orig, duv_oracle_orig, valid, K_orig, dist, dofs):
    B, N = P0_orig.shape[:2]
    W_eye = torch.eye(2, device=P0_orig.device,
                       dtype=P0_orig.dtype).expand(B, N, 2, 2)
    with torch.no_grad():
        delta, _ = solve_kb_xyz(
            P0_orig, duv_oracle_orig, W_eye, K_orig, dist, dofs,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
        )
    return delta.detach()


def _solve_with(P0_orig, duv_orig, W_orig, K_orig, dist, dofs, valid):
    return solve_kb_xyz(
        P0_orig, duv_orig, W_orig, K_orig, dist, dofs,
        valid=valid, n_iter=BA_N_ITER, damping=DAMPING,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17)
    ap.add_argument('--n-steps', type=int, default=DEFAULT_N_STEPS)
    ap.add_argument('--batch',   type=int, default=DEFAULT_BATCH)
    ap.add_argument('--n-eval',  type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--lr',      type=float, default=LR)
    ap.add_argument('--tag',     type=str, default=None)
    ap.add_argument('--dofs',    type=str, default='6dof',
                    choices=list(DOF_PRESETS.keys()))
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m',     type=float, default=0.05)
    ap.add_argument('--dfx-pct', type=float, default=0.0)
    ap.add_argument('--dfy-pct', type=float, default=0.0)
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', type=str,
                    default='e2e_calib/principled_ml')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tag = args.tag or f'idx{args.idx}_{args.dofs}_sigma'
    out_dir = OUT.parent / f'overfit_6dof_ba_stream_orig_sigma_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    cml_logger = None
    if args.clearml and _ClearMLTask is not None:
        cml_task = _ClearMLTask.init(
            project_name=args.clearml_project,
            task_name=f'overfit_orig_sigma_{tag}',
            task_type=_ClearMLTask.TaskTypes.optimizer,
            reuse_last_task_id=False,
            auto_connect_frameworks={'matplotlib': False, 'pytorch': False},
        )
        cml_task.connect({
            'idx': args.idx, 'dofs': args.dofs,
            'n_steps': args.n_steps, 'batch': args.batch,
            'n_eval': args.n_eval, 'lr': args.lr,
            'rot_deg': args.rot_deg, 't_m': args.t_m,
            'ba_n_iter': BA_N_ITER, 'damping': DAMPING, 'seed': SEED,
            'head': 'sigma_from_q (variant c)',
        })
        cml_logger = cml_task.get_logger()
        print(f'[sigma-head] ClearML: {cml_task.get_output_log_web_page()}')

    cfg = _load_cfg()
    DOFS = DOF_PRESETS[args.dofs]
    pert_kw = dict(rot_deg=args.rot_deg, t_m=args.t_m,
                   dfx_pct=args.dfx_pct, dfy_pct=args.dfy_pct)
    print(f"[sigma-head] DoFs={args.dofs}  rot=±{args.rot_deg}deg  t=±{args.t_m}m")

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )

    base_idx = int(args.idx)
    inst0 = ds._load_inst(base_idx)
    assert inst0.get('is_fisheye', False) and 'distortion' in inst0
    dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
    print(f"[sigma-head] anchor val frame idx={base_idx} ({ds.fnames[base_idx]})")
    print(f"[sigma-head] KB dist (k1..k4) = {dist_one[0].numpy()}")

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    other_missing = [k for k in missing if not k.startswith('info_head')]
    assert len(other_missing) == 0, f"missing keys: {other_missing[:5]}"
    _freeze_all(model)
    model.eval()

    # σ-head's input dim must match q's last dim. Pull it from info_head.mlp[0].
    d_q = model.info_head.mlp[0].in_features
    sigma_head = SigmaHeadFromQ(d_q).to(DEVICE)
    n_train = sum(p.numel() for p in sigma_head.parameters())
    print(f"[sigma-head] σ-head params: {n_train/1e3:.1f}k  (q dim = {d_q})")

    opt = torch.optim.AdamW(sigma_head.parameters(), lr=args.lr, weight_decay=0.0)

    # ─── HELD-OUT BASELINES ──────────────────────────────────────
    eval_rng = np.random.RandomState(SEED + 1000)
    eval_batch = _build_batch(ds, base_idx, args.n_eval, eval_rng, **pert_kw)
    assert eval_batch is not None
    e = _move_batch(eval_batch, DEVICE)
    (e_imgs, e_true_uvd, e_dist_uvd, e_pad, e_vfp,
     e_bucket_uvd, e_bucket_valid, _,
     e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs) = e
    (e_valid, e_duv_oracle_local, e_duv_oracle_orig,
     e_P0_orig, e_K_orig, e_cs) = _prep_inputs(
        e_true_uvd, e_dist_uvd, e_pad,
        e_pts_cam_orig, e_duv_orig, e_K_orig, e_cs)
    e_dist = dist_one.to(DEVICE).expand(args.n_eval, 4).contiguous()
    delta_target_eval = _solve_target(
        e_P0_orig, e_duv_oracle_orig, e_valid, e_K_orig, e_dist, DOFS)

    e_per_pt, e_q = _forward_capture_q(
        model, e_imgs, e_dist_uvd, e_pad, e_vfp, e_bucket_uvd, e_bucket_valid)
    e_duv_pred_local = e_per_pt[..., :2].detach()
    e_padfull = ~e_valid
    if e_padfull.any():
        e_duv_pred_local = e_duv_pred_local.clone()
        e_duv_pred_local[e_padfull] = 0.0
    e_sx = e_per_pt[..., 2].exp()
    e_sy = e_per_pt[..., 3].exp()
    e_rho_legacy = e_per_pt[..., 4]
    e_W_sigma_local = make_info_from_sigma_rho(e_sx, e_sy, e_rho_legacy).detach()
    e_W_eye_local = torch.eye(2, device=DEVICE).expand(
        args.n_eval, e_q.shape[1], 2, 2)

    e_scale_l2o = _local_to_orig_scale(e_cs, cfg['img_size'])
    e_inv_l2o   = (1.0 / e_scale_l2o).reshape(-1, 1, 1, 1)
    e_duv_pred_orig = e_duv_pred_local * e_scale_l2o
    e_W_eye_orig    = e_W_eye_local    * e_inv_l2o.pow(2)
    e_W_sigma_orig  = e_W_sigma_local  * e_inv_l2o.pow(2)

    with torch.no_grad():
        d_zero = torch.zeros_like(delta_target_eval)
        mse_zero = (d_zero - delta_target_eval).pow(2).mean(dim=-1).mean().item()
        d_w1, _ = _solve_with(e_P0_orig, e_duv_pred_orig, e_W_eye_orig,
                              e_K_orig, e_dist, DOFS, e_valid)
        mse_w1 = (d_w1 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
        d_w2, _ = _solve_with(e_P0_orig, e_duv_pred_orig, e_W_sigma_orig,
                              e_K_orig, e_dist, DOFS, e_valid)
        mse_w2 = (d_w2 - delta_target_eval).pow(2).mean(dim=-1).mean().item()
    print(f"[sigma-head] HELD-OUT BASELINE  do-nothing: {mse_zero:.6e}  "
          f"W=I: {mse_w1:.6e}   W=σ_legacy: {mse_w2:.6e}")
    if cml_logger is not None:
        cml_logger.report_single_value('mse_baseline_do_nothing', mse_zero)
        cml_logger.report_single_value('mse_baseline_W_I', mse_w1)
        cml_logger.report_single_value('mse_baseline_W_sigma_legacy', mse_w2)

    # ─── streaming train ──────────────────────────────────────────
    history = []
    t0 = time.time()
    for step in range(args.n_steps + 1):
        batch = _build_batch(ds, base_idx, args.batch, rng, **pert_kw)
        if batch is None:
            print(f"  step {step}: skipped", flush=True)
            continue
        b = _move_batch(batch, DEVICE)
        (imgs, true_uvd, dist_uvd, pad_mask, vfp,
         bucket_uvd, bucket_valid, _,
         pts_cam_orig, duv_orig, K_orig, cs) = b
        (valid, duv_oracle_local, duv_oracle_orig,
         P0_orig, K_orig, cs) = _prep_inputs(
            true_uvd, dist_uvd, pad_mask,
            pts_cam_orig, duv_orig, K_orig, cs)
        b_dist = dist_one.to(DEVICE).expand(P0_orig.shape[0], 4).contiguous()
        delta_target = _solve_target(P0_orig, duv_oracle_orig, valid, K_orig,
                                      b_dist, DOFS)

        per_pt, q = _forward_capture_q(
            model, imgs, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid)
        duv_pred_local = per_pt[..., :2].detach()
        if (~valid).any():
            duv_pred_local = duv_pred_local.clone()
            duv_pred_local[~valid] = 0.0

        # σ-head is the only learnable module — single live forward.
        W_learn_local = sigma_head(q)

        scale_l2o = _local_to_orig_scale(cs, cfg['img_size'])
        inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
        duv_pred_orig = duv_pred_local * scale_l2o
        W_learn_orig  = W_learn_local  * inv_l2o.pow(2)

        delta_pred, _ = _solve_with(
            P0_orig, duv_pred_orig, W_learn_orig, K_orig, b_dist, DOFS, valid)
        diff = delta_pred - delta_target
        loss = diff.pow(2).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sigma_head.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == args.n_steps:
            sigma_head.eval()
            with torch.no_grad():
                W_l_local = sigma_head(e_q)
                W_l_orig  = W_l_local * e_inv_l2o.pow(2)
                d_l, _ = _solve_with(
                    e_P0_orig, e_duv_pred_orig, W_l_orig,
                    e_K_orig, e_dist, DOFS, e_valid)
                eval_mse = (d_l - delta_target_eval).pow(2).mean(dim=-1).mean().item()
                logdet = torch.linalg.det(W_l_orig).clamp_min(1e-12).log().mean().item()
            sigma_head.train()
            sps = (step + 1) / max(1e-3, time.time() - t0)
            history.append((step, loss.item(), eval_mse, logdet))
            print(f"  step {step:>4}  loss={loss.item():.4e}  "
                  f"EVAL δ-MSE={eval_mse:.4e}  "
                  f"do-nothing={mse_zero:.4e}  W=I={mse_w1:.4e}  "
                  f"W=σ_legacy={mse_w2:.4e}  "
                  f"⟨log det W_orig⟩={logdet:+.2f}  "
                  f"({sps:.2f} step/s)", flush=True)
            if cml_logger is not None:
                cml_logger.report_scalar('loss', 'train', loss.item(), step)
                cml_logger.report_scalar('mse', 'W_learned', eval_mse, step)
                cml_logger.report_scalar('mse', 'do_nothing', mse_zero, step)
                cml_logger.report_scalar('mse', 'W_I', mse_w1, step)
                cml_logger.report_scalar('mse', 'W_sigma_legacy', mse_w2, step)
                cml_logger.report_scalar('logdet_W_orig', 'mean', logdet, step)

    # ─── FINAL eval ────────────────────────────────────────────────
    sigma_head.eval()
    with torch.no_grad():
        W_final_local = sigma_head(e_q)
        W_final_orig  = W_final_local * e_inv_l2o.pow(2)
        d_final, _ = _solve_with(
            e_P0_orig, e_duv_pred_orig, W_final_orig,
            e_K_orig, e_dist, DOFS, e_valid)
        mse_w3 = (d_final - delta_target_eval).pow(2).mean(dim=-1).mean().item()
    print(f"\n[sigma-head] HELD-OUT FINAL  do-nothing: {mse_zero:.6e}  "
          f"W=I: {mse_w1:.6e}   W=σ_legacy: {mse_w2:.6e}   "
          f"W=σ_learned: {mse_w3:.6e}")
    print(f"[sigma-head] PASS vs do-nothing: {mse_w3 < mse_zero}   "
          f"PASS vs uniform: {mse_w3 < mse_w1}   "
          f"PASS vs σ_legacy: {mse_w3 <= mse_w2 * 1.05}")

    steps  = [h[0] for h in history]
    trnL   = [h[1] for h in history]
    evMSE  = [h[2] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.semilogy(steps, evMSE, '-o', color='tab:red', ms=3,
                 label=f'σ-learned (held-out N={args.n_eval})')
    ax.semilogy(steps, trnL,  '-x', color='tab:orange', ms=3, alpha=0.5,
                 label='train loss')
    ax.axhline(mse_zero, ls=':',  color='tab:green',  label=f'do-nothing ({mse_zero:.2e})')
    ax.axhline(mse_w1,   ls='--', color='tab:gray',   label=f'W=I        ({mse_w1:.2e})')
    ax.axhline(mse_w2,   ls='--', color='tab:purple', label=f'W=σ_legacy ({mse_w2:.2e})')
    ax.set_xlabel('step'); ax.set_ylabel('δ-MSE')
    ax.set_title(f'(a) σ-head only (variant c) — idx={base_idx}, {args.dofs}')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    bars = ['do-nothing', 'W=I', 'W=σ_legacy', 'W=σ_learn']
    vals = [mse_zero, mse_w1, mse_w2, mse_w3]
    colors = ['tab:green', 'tab:gray', 'tab:purple', 'tab:red']
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('δ-MSE')
    ax.set_title(f'(b) final δ-MSE on held-out N={args.n_eval}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / 'curves.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[sigma-head] wrote → {out_png}")

    sd_out = out_dir / 'sigma_head.pt'
    torch.save(sigma_head.state_dict(), sd_out)
    summary = {
        'idx': base_idx, 'dofs': args.dofs, 'DOFS': DOFS,
        'n_steps': args.n_steps, 'batch': args.batch, 'n_eval': args.n_eval,
        'rot_deg': args.rot_deg, 't_m': args.t_m,
        'mse_zero': mse_zero, 'mse_W_I': mse_w1,
        'mse_W_sigma_legacy': mse_w2, 'mse_W_sigma_learned': mse_w3,
        'history': history,
    }
    torch.save(summary, out_dir / 'summary.pt')
    print(f"[sigma-head] wrote → {sd_out}, summary.pt")
    print("[sigma-head] done.")


if __name__ == '__main__':
    main()
