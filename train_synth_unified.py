"""Train CalibNetUnifiedFrame on the synthetic GridDepth dataset to test
whether the unified architecture can solve the calib task at all.

Pure architecture probe: synthetic data isolates the model from real-data
noise, so if this fails we know the unified arch itself is the bottleneck."""
import argparse, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.synth_calib_unified import SyntheticCalibUnified
from models.cross_frame_unified import CalibNetUnifiedFrame
from train_cross_frame import residual_uvd_nll_and_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--max-offset', type=float, default=16.0)
    ap.add_argument('--n-cross-layers', type=int, default=4)
    ap.add_argument('--n-intra-layers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-6)
    ap.add_argument('--train-size', type=int, default=8000)
    ap.add_argument('--val-size', type=int, default=800)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--random-depths', action='store_true')
    args = ap.parse_args()

    out_dir = Path('experiments') / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    log_path.write_text('')

    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f'{ts}  {msg}'
        with open(log_path, 'a') as f:
            f.write(line + '\n')
        print(line, flush=True)

    log(f'name={args.name}  img_size={args.img_size}  max_offset={args.max_offset}  '
        f'n_cross={args.n_cross_layers}  n_intra={args.n_intra_layers}  '
        f'epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}→{args.lr_min}')

    train_ds = SyntheticCalibUnified(
        length=args.train_size, img_size=args.img_size, max_offset=args.max_offset,
        max_points=args.max_points, random_depths=args.random_depths,
        random_each_epoch=True)
    val_ds = SyntheticCalibUnified(
        length=args.val_size, img_size=args.img_size, max_offset=args.max_offset,
        max_points=args.max_points, random_depths=args.random_depths,
        base_seed=10**7)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=max(1, args.num_workers // 2), pin_memory=True)

    model = CalibNetUnifiedFrame(
        in_channels=3, img_size=args.img_size,
        n_intra_layers=args.n_intra_layers, n_cross_layers=args.n_cross_layers,
        out_dim=7, uv_only_query=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'params: {n_params/1e6:.2f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)

    def run(loader, train: bool):
        model.train(train)
        losses, metrics_acc = [], dict()
        for batch in loader:
            batch = {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.set_grad_enabled(train):
                raw_AB, raw_BA = model(
                    patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
                    patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
                    pose_AB_6dof=batch['pose_AB_6dof'],
                    pose_BA_6dof=batch['pose_BA_6dof'],
                    uv_B_hat_of_A=batch['uv_B_hat_of_A'],
                    uv_A_hat_of_B=batch['uv_A_hat_of_B'],
                    pad_A=batch['pad_A'], pad_B=batch['pad_B'],
                    uvd_A_full=batch['uvd_A_full'], uvd_B_full=batch['uvd_B_full'],
                    pad_A_full=batch['pad_A_full'], pad_B_full=batch['pad_B_full'],
                    feats_A=batch['feats_A'], feats_B=batch['feats_B'],
                    modality_A='cam', modality_B='lidar',
                )
                loss_AB, m_AB = residual_uvd_nll_and_metrics(
                    raw_AB, batch['uv_B_hat_of_A'], batch['uv_B_gt_of_A'],
                    batch['d_B_hat_of_A'], batch['d_B_gt_of_A'],
                    batch['pad_A'], args.img_size, is_obj=batch['is_obj_A'])
                loss_BA, m_BA = residual_uvd_nll_and_metrics(
                    raw_BA, batch['uv_A_hat_of_B'], batch['uv_A_gt_of_B'],
                    batch['d_A_hat_of_B'], batch['d_A_gt_of_B'],
                    batch['pad_B'], args.img_size, is_obj=batch['is_obj_B'])
                loss = 0.5 * (loss_AB + loss_BA)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
            for tag in ('err_px', 'base_px', 'err_px_obj', 'base_px_obj',
                        'nll_obj', 'err_px_bg', 'base_px_bg', 'nll_bg'):
                if tag in m_AB and tag in m_BA:
                    metrics_acc.setdefault(tag, []).append(0.5 * (m_AB[tag] + m_BA[tag]))
        return float(np.mean(losses)), {k: float(np.mean(v)) for k, v in metrics_acc.items()}

    best_val = float('inf')
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_m = run(train_loader, train=True)
        sched.step()
        do_val = (ep % 4 == 0) or (ep == args.epochs)
        val_str = ''
        if do_val:
            with torch.no_grad():
                vl_loss, vl_m = run(val_loader, train=False)
            val_str = (f'  val nll={vl_loss:.3f} '
                       f'err={vl_m["err_px"]:.2f}px(obj={vl_m["err_px_obj"]:.2f} '
                       f'bg={vl_m["err_px_bg"]:.2f}) '
                       f'(base={vl_m["base_px"]:.2f})')
            if vl_loss < best_val:
                best_val = vl_loss
                torch.save(model.state_dict(), out_dir / 'best_model.pt')
        log(f'[{ep:3d}/{args.epochs}]  train nll={tr_loss:.3f}  '
            f'err={tr_m["err_px"]:.2f}px(obj={tr_m["err_px_obj"]:.2f} '
            f'bg={tr_m["err_px_bg"]:.2f})  '
            f'(base={tr_m["base_px"]:.2f}px)'
            f'{val_str}  lr={opt.param_groups[0]["lr"]:.2e}  '
            f'tot={(time.time()-t0)/60:.1f}min')

    log(f'best_val={best_val:.4f}')


if __name__ == '__main__':
    main()
