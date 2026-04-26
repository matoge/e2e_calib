"""3-axis eval grid: lidar (full/none) × σ (small/large) × baseline (4 buckets).

For each cell, draws N samples from val split, runs model, reports mean
err_AB (model pred vs gt). Same dataset path as training (frustum_full ON).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
import numpy as np
import torch

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.cross_frame_multi import CalibNetMultiFrame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SENTINEL = -9999.0 / 50.0


def load_model(ckpt_dir, ckpt_name):
    sd = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=True)
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    m = CalibNetMultiFrame(img_size=64, deform_mode='sl', n_cross_layers=n_cross,
                            n_intra_layers=n_intra, out_dim=out_dim).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m


@torch.no_grad()
def measure_cell(model, ds, n_samples, bl_lo, bl_hi, lidar_mode, max_pool=400):
    """Sample up to n_samples from ds with baseline ∈ [bl_lo, bl_hi].
    lidar_mode ∈ {'full', 'none'}: 'none' → uvd[..., 2] = sentinel for all pts.
    Returns (mean_err, mean_base, n_used).
    """
    errs, bases, used = [], [], 0
    for i in range(max_pool):
        if used >= n_samples:
            break
        s = ds[i]
        bl = abs(s['fi_B'] - s['fi_A'])
        if not (bl_lo <= bl <= bl_hi):
            continue
        # build batch dict (clone tensors before mutating)
        b = {k: (v.unsqueeze(0).clone().to(DEVICE) if torch.is_tensor(v) else v)
             for k, v in s.items()}
        if lidar_mode == 'none':
            for tag in ['A', 'B', 'M']:
                for full in ('', '_full'):
                    key = f'uvd_{tag}{full}'
                    if key in b:
                        b[key][..., 2] = SENTINEL
        raw, _ = model(patch_A=b['patch_A'], uvd_A=b['uvd_A'],
                       patch_B=b['patch_B'], uvd_B=b['uvd_B'],
                       pose_AB_6dof=b['pose_AB_6dof'], pose_BA_6dof=b['pose_BA_6dof'],
                       uv_B_hat_of_A=b['uv_B_hat_of_A'], uv_A_hat_of_B=b['uv_A_hat_of_B'],
                       pad_A=b['pad_A'], pad_B=b['pad_B'],
                       uvd_A_full=b.get('uvd_A_full'), uvd_B_full=b.get('uvd_B_full'),
                       pad_A_full=b.get('pad_A_full'), pad_B_full=b.get('pad_B_full'),
                       patch_M=b.get('patch_M'), uvd_M=b.get('uvd_M'), pad_M=b.get('pad_M'),
                       uvd_M_full=b.get('uvd_M_full'), pad_M_full=b.get('pad_M_full'),
                       pose_AM_6dof=b.get('pose_AM_6dof'),
                       uv_M_hat_of_A=b.get('uv_M_hat_of_A'),
                       uv_M_hat_of_B=b.get('uv_M_hat_of_B'))
        delta = raw[0, :, :2].cpu().numpy()
        valid = ~s['pad_A'].numpy()
        e_pred = np.linalg.norm((s['uv_B_hat_of_A'].numpy() + delta)[valid] - s['uv_B_gt_of_A'].numpy()[valid], axis=-1).mean()
        e_base = np.linalg.norm( s['uv_B_hat_of_A'].numpy()[valid]         - s['uv_B_gt_of_A'].numpy()[valid], axis=-1).mean()
        errs.append(e_pred); bases.append(e_base); used += 1
    if not errs:
        return None, None, 0
    return float(np.mean(errs)), float(np.mean(bases)), used


def main(args):
    ckpt = Path(args.ckpt)
    model = load_model(ckpt, args.ckpt_name)
    print(f'loaded {ckpt.name}/{args.ckpt_name}')

    bl_buckets = [(1, 4), (5, 10), (11, 30), (31, 79)]
    sigmas     = [(1.0, 0.2), (2.0, 0.5)]
    lidar_modes = ['full', 'none']

    print()
    header = f'  {"baseline":>10s} '
    for σy, σt in sigmas:
        for lid in lidar_modes:
            header += f'  σ={σy}/{σt} {lid:4s} '
    print(header)

    for lo, hi in bl_buckets:
        row = f'  [{lo:>2d}-{hi:>2d}]   '
        bases_seen = []
        for σy, σt in sigmas:
            ds = PandaSetCrossFrameDataset(
                scenes_root=args.scenes_root, train_frac=args.train_frac,
                split=args.split, img_size=64, max_points=256,
                baseline_range=(1, 20), sigma_ypr=σy, sigma_t=σt,
                crop_range=(128, 256), cameras=args.cameras,
                triplet=args.multi_frame, virtual_epoch_len=2000, seed=42,
            )
            for lid in lidar_modes:
                e, base, n = measure_cell(model, ds, args.n_samples, lo, hi, lid)
                cell = f'  {e:5.2f}/{base:5.1f}({n:>2d})' if e is not None else f'   --   ({0:>2d})'
                row += cell
        print(row)
    print()
    print('  cell = pred_err / base_err (sample_count)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt-name', default='best_model.pt')
    ap.add_argument('--scenes-root', default='/mnt/nvme6t/pandaset_39')
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--multi-frame', action='store_true')
    ap.add_argument('--n-samples', type=int, default=20)
    args = ap.parse_args()
    main(args)
