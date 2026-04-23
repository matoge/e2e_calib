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
from models.model_cov import gaussian2d_nll

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


# ─── one step ─────────────────────────────────────────────────────────────────

def step(model, batch):
    batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
    raw_AB, raw_BA = model(
        patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
        patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
        pose_AB_6dof=batch['pose_AB_6dof'], pose_BA_6dof=batch['pose_BA_6dof'],
        uv_B_hat_of_A=batch['uv_B_hat_of_A'], uv_A_hat_of_B=batch['uv_A_hat_of_B'],
        pad_A=batch['pad_A'], pad_B=batch['pad_B'],
    )
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
    return loss, metrics


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='v00_overfit')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--scene', default='/mnt/mininas/datasets/pandaset/015')
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n-overfit', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--log-every', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path('experiments/cross_frame_' + args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    (out_dir / 'config.txt').write_text(json.dumps(vars(args), indent=2))

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log(f'device = {DEVICE}')
    log(f'args = {vars(args)}')

    # dataset
    ds_train = PandaSetCrossFrameDataset(
        scene_root=args.scene, split='train',
        img_size=args.img_size, max_points=args.max_points,
        virtual_epoch_len=(args.n_overfit if not args.full else 4000),
    )
    ds_val = PandaSetCrossFrameDataset(
        scene_root=args.scene, split='val',
        img_size=args.img_size, max_points=args.max_points,
        virtual_epoch_len=200, seed=123,
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
            keep_keys = [k for k in batch[0].keys() if not isinstance(batch[0][k], int)]
            return {k: torch.stack([b[k] for b in batch]) for k in keep_keys}
        loader = [collate(fixed[i:i+args.batch_size])
                  for i in range(0, len(fixed), args.batch_size)]
    else:
        loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, persistent_workers=True, pin_memory=True)

    # model
    model = CalibNetCrossFrame(img_size=args.img_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'model params: {n_params/1e6:.3f} M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 1e-2)

    curves = dict(epoch=[], loss=[], err_AB=[], err_BA=[], base_AB=[])
    best_err = float('inf')
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        ep_losses, ep_errs_AB, ep_errs_BA, ep_base_AB = [], [], [], []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss, m = step(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_losses.append(m['loss'])
            ep_errs_AB.append(m['err_AB']); ep_errs_BA.append(m['err_BA'])
            ep_base_AB.append(m['base_AB'])
        sched.step()

        curves['epoch'].append(ep)
        curves['loss'].append(float(np.mean(ep_losses)))
        curves['err_AB'].append(float(np.mean(ep_errs_AB)))
        curves['err_BA'].append(float(np.mean(ep_errs_BA)))
        curves['base_AB'].append(float(np.mean(ep_base_AB)))

        if ep % args.log_every == 0 or ep == args.epochs - 1:
            log(f'ep {ep:3d}  loss={curves["loss"][-1]:.3f}  '
                f'err_AB={curves["err_AB"][-1]:.2f}px  '
                f'err_BA={curves["err_BA"][-1]:.2f}px  '
                f'(base_AB={curves["base_AB"][-1]:.2f}px)  '
                f'lr={opt.param_groups[0]["lr"]:.2e}  '
                f't={time.time()-t0:.0f}s')

        # save best by err_AB
        mean_err = 0.5 * (curves['err_AB'][-1] + curves['err_BA'][-1])
        if mean_err < best_err:
            best_err = mean_err
            torch.save(model.state_dict(), out_dir / 'best_model.pt')

    # final plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    ax[0].plot(curves['epoch'], curves['loss'], color='#174734', lw=2)
    ax[0].set_title('NLL loss', loc='left'); ax[0].set_xlabel('epoch'); ax[0].grid(alpha=0.25)
    ax[1].plot(curves['epoch'], curves['err_AB'], color='#c13c14', lw=2, label='err A→B')
    ax[1].plot(curves['epoch'], curves['err_BA'], color='#174734', lw=2, label='err B→A')
    ax[1].plot(curves['epoch'], curves['base_AB'], color='#6b6a63', lw=1, ls='--', label='base (no correction)')
    ax[1].set_title('mean reproj err (px)', loc='left'); ax[1].set_xlabel('epoch')
    ax[1].legend(frameon=False); ax[1].grid(alpha=0.25)
    for a in ax:
        for sp in ('top','right'): a.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'curve.png', dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    log(f'saved curve → {out_dir}/curve.png')
    log(f'best mean err = {best_err:.2f} px')


if __name__ == '__main__':
    main()
