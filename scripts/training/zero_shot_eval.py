"""Zero-shot cross-dataset eval: load a trained best_model.pt, run val_loop on
a DIFFERENT dataset's val split (no retraining, no fine-tuning).

Usage:
  python scripts/training/zero_shot_eval.py \
      --ckpt experiments/ps_tile_s15_stage1_dualDA_50ep/best_model.pt \
      --eval-cache /mnt/nvme6t/e2e_calib_cache/nuscenes_v3_tiled \
      --convnext --deform-mode ml --n-layers 4 \
      --use-pose-emb --frustum-dense \
      --img-size 128 --batch-size 128 --workers 12 \
      --rot-deg 1.5 --t-m 0.6 \
      --val-size 4000

Output: val_nll, val_obj, val_bg, mse stats — printed + (optional) ClearML.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, time
import torch
from torch.utils.data import DataLoader, Subset

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='best_model.pt path')
    ap.add_argument('--eval-cache', required=True, help='V3 cache to eval on')
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--in-channels', type=int, default=3)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--convnext', action='store_true')
    ap.add_argument('--deform-mode', default='sl')
    ap.add_argument('--frustum-dense', action='store_true')
    ap.add_argument('--use-lidar-kv', action='store_true')
    ap.add_argument('--use-pose-emb', action='store_true')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--min-crop-px', type=int, default=128)
    ap.add_argument('--max-crop-px', type=int, default=384)
    ap.add_argument('--rot-deg', type=float, default=1.5)
    ap.add_argument('--t-m', type=float, default=0.6)
    ap.add_argument('--grid-n', type=int, default=16)
    ap.add_argument('--val-size', type=int, default=None,
                    help='deterministic first-N subsample of val')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = CalibNetDepth(
        img_size=args.img_size, in_channels=args.in_channels,
        n_layers=args.n_layers, use_convnext=args.convnext,
        use_frustum=True, frustum_dense=args.frustum_dense,
        use_lidar_kv=args.use_lidar_kv, use_pose_emb=args.use_pose_emb,
        deform_mode=args.deform_mode,
    ).to(device)

    # Load weights
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[warn] missing keys: {len(missing)} unexpected keys: {len(unexpected)}')
    model.eval()

    # Build val loader on the OTHER dataset
    ds = PandaSetCalibDatasetFull(
        args.eval_cache, split='val',
        img_size=args.img_size,
        max_offset_m=args.t_m, max_rot_deg=args.rot_deg,
        min_crop_px=args.min_crop_px, max_crop_px=args.max_crop_px,
        grid_n=args.grid_n, oversample=1,
    )
    if args.val_size is not None:
        ds = Subset(ds, list(range(min(args.val_size, len(ds)))))
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=collate_full,
    )

    # Eval loop
    print(f'\n=== Zero-shot eval ===')
    print(f'  ckpt:       {args.ckpt}')
    print(f'  eval-cache: {args.eval_cache}')
    print(f'  val instances: {len(ds)}')
    print(f'  perturbation: ±{args.rot_deg} deg / ±{args.t_m} m')
    print()

    nll_sum = mse_sum = obj_nll = bg_nll = obj_n = bg_n = n_batch = 0
    obj_errs, bg_errs = [], []
    AMP = torch.bfloat16
    t0 = time.time()
    with torch.no_grad():
        for imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid in loader:
            imgs = imgs.to(device, non_blocking=True).float().div_(255.0)
            true_uvd = true_uvd.to(device, non_blocking=True)
            dist_uvd = dist_uvd.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            vfp = vfp.to(device, non_blocking=True)
            bucket_uvd = bucket_uvd.to(device, non_blocking=True)
            bucket_valid = bucket_valid.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=AMP):
                params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad_mask,
                                vfp=vfp, bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
            valid = ~pad_mask
            gt = true_uvd[..., :2] - dist_uvd[..., :2]
            loss = gaussian2d_nll(params[valid], gt[valid]).item()
            mse = ((params[valid][..., :2].float() - gt[valid]).norm(dim=-1) ** 2).mean().item()
            is_obj = (true_uvd[..., 3] > 0.5) & valid
            is_bg = (~(true_uvd[..., 3] > 0.5)) & valid
            if is_obj.any():
                obj_nll += gaussian2d_nll(params[is_obj], gt[is_obj]).item(); obj_n += 1
                obj_errs.append((params[is_obj][..., :2].float() - gt[is_obj]).norm(dim=-1).cpu())
            if is_bg.any():
                bg_nll += gaussian2d_nll(params[is_bg], gt[is_bg]).item(); bg_n += 1
                bg_errs.append((params[is_bg][..., :2].float() - gt[is_bg]).norm(dim=-1).cpu())
            nll_sum += loss; mse_sum += mse; n_batch += 1
    dt = time.time() - t0
    sps = len(ds) / dt
    print(f'val_nll = {nll_sum/n_batch:.4f}')
    print(f'  obj_nll = {obj_nll/max(obj_n,1):.4f}')
    print(f'  bg_nll  = {bg_nll/max(bg_n,1):.4f}')
    print(f'val_mse = {mse_sum/n_batch:.4f}')
    if obj_errs:
        all_obj = torch.cat(obj_errs)
        print(f'  obj median = {all_obj.median():.3f} px, 95p = {torch.quantile(all_obj, 0.95):.3f} px')
    if bg_errs:
        all_bg = torch.cat(bg_errs)
        print(f'  bg  median = {all_bg.median():.3f} px, 95p = {torch.quantile(all_bg, 0.95):.3f} px')
    print(f'eval time: {dt:.1f}s ({sps:.0f} sps)')


if __name__ == '__main__':
    main()
