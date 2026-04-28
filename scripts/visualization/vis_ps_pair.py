"""Visualize ps_v12+ cross-frame predictions.

Loads experiments/{exp}/best_model.pt + config, draws 24 sample tiles where
each tile shows:
  - patch_B (the target frame's image crop)
  - GT positions (uv_B_gt) as green X
  - naive forward-projected positions (uv_B_naive) as yellow X
  - predicted positions (= uv_B_naive + Δuv) as cyan X
  - per-point covariance σ as faint ellipses around predictions
  - quiver: dist→pred (orange) and dist→gt (green dashed)

Usage:
    python scripts/visualization/vis_ps_pair.py ps_v13_cross_frame_full
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, importlib.util, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from pathlib import Path

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.model_pair import CalibNetDepthPair
from models.model_cov import MIN_SIGMA, MAX_SIGMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("_cfg", exp_dir / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.CFG


def main(exp: str, n_vis: int = 24, split: str = 'val', max_scan: int = 200):
    exp_dir = Path("experiments") / exp
    c = load_cfg(exp_dir)

    ds_kw = dict(
        cameras           = c['cameras'],
        img_size          = c['img_size'],
        max_points        = c['max_points'],
        baseline_range    = (c['baseline_min'], c['baseline_max']),
        sigma_ypr         = c['sigma_ypr'],
        sigma_t           = c['sigma_t'],
        crop_range        = (c['crop_min'], c['crop_max']),
        virtual_epoch_len = max_scan,
        n_frames          = 2,
        use_stacked       = True,
    )
    if c.get('scenes_root'):
        ds = PandaSetCrossFrameDataset(scenes_root=c['scenes_root'],
                                        train_frac=c.get('train_frac', 0.8),
                                        split=split, **ds_kw)
    else:
        ds = PandaSetCrossFrameDataset(scene_root=c['scene'], **ds_kw)

    model = CalibNetDepthPair(
        img_size=c['img_size'], in_channels=c['in_channels'],
        n_layers=c['n_layers'], use_convnext=c.get('use_convnext', False),
        use_frustum=c.get('use_frustum', False),
        use_lidar_kv=c.get('use_lidar_kv', False),
        use_pose_emb=c.get('use_pose_emb', False),
    ).to(DEVICE)
    model.load_state_dict(torch.load(exp_dir / "best_model.pt",
                                      map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"[{exp}] split={split} scanning up to {max_scan} for {n_vis} samples")
    picked = []
    for i in range(max_scan):
        s = ds[i]
        # require at least some valid points in the (0,1) direction
        valid = ~s['pad_dir'][0, 1]
        if valid.sum() < 8:
            continue
        picked.append(s)
        if len(picked) >= n_vis:
            break
    print(f"  picked {len(picked)} samples")

    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    for old in vis_dir.glob("pair_*.png"):
        old.unlink()

    cols = 6
    rows = (len(picked) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=110)
    axes = np.atleast_2d(axes).flatten()

    for vi, s in enumerate(picked):
        with torch.no_grad():
            patches    = s['patches'].to(DEVICE).unsqueeze(0)         # (1, 2, 3, H, W)
            uvd        = s['uvd'].to(DEVICE).unsqueeze(0)             # (1, 2, N, 4)
            pad        = s['pad'].to(DEVICE).unsqueeze(0)
            pose_hat   = s['pose_hat_6dof'].to(DEVICE).unsqueeze(0)
            uv_hat     = s['uv_hat'].to(DEVICE).unsqueeze(0)
            uv_gt      = s['uv_gt'].to(DEVICE).unsqueeze(0)
            pad_dir    = s['pad_dir'].to(DEVICE).unsqueeze(0)

            image_A = patches[:, 0]; image_B = patches[:, 1]
            uvd_A   = uvd[:, 0, :, :3]; uvd_B = uvd[:, 1, :, :3]
            pose_AB = pose_hat[:, 0, 1]
            uv_B_naive = uv_hat[:, 0, 1]
            uv_B_gt    = uv_gt[:, 0, 1]
            valid      = ~pad_dir[:, 0, 1]
            vfp = torch.full((1,), float(c['img_size']), device=DEVICE)
            params = model.forward_pair(
                image_A, image_B, uvd_A, uvd_B, uv_B_naive,
                pose_AB, vfp, pad_A=pad[:, 0], pad_B=pad[:, 1], query_pad=pad[:, 0],
            )[0]
            valid = valid[0]

        img_B  = image_B[0].permute(1, 2, 0).cpu().numpy()
        uvN    = uv_B_naive[0].cpu().numpy()
        uvG    = uv_B_gt[0].cpu().numpy()
        delta  = params[:, :2].float().cpu().numpy()
        sx     = params[:, 2].float().exp().cpu().numpy()
        sy     = params[:, 3].float().exp().cpu().numpy()
        uvP    = uvN + delta
        v      = valid.cpu().numpy()

        ax = axes[vi]
        ax.imshow(img_B); ax.set_axis_off()
        ax.scatter(uvN[v, 0], uvN[v, 1], c='yellow', s=14, marker='x', linewidths=1.0, label='naive')
        ax.scatter(uvG[v, 0], uvG[v, 1], c='lime',   s=18, marker='x', linewidths=1.2, label='gt')
        ax.scatter(uvP[v, 0], uvP[v, 1], c='cyan',   s=16, marker='+', linewidths=1.2, label='pred')
        for i in np.where(v)[0]:
            sxi = float(sx[i].clip(0.5, 8))
            syi = float(sy[i].clip(0.5, 8))
            ax.add_patch(Ellipse((uvP[i,0], uvP[i,1]), 2*sxi, 2*syi, fill=False,
                                 ec='cyan', alpha=0.25, lw=0.6))
        for i in np.where(v)[0]:
            ax.plot([uvN[i,0], uvP[i,0]], [uvN[i,1], uvP[i,1]],
                    color='orange', lw=0.8, alpha=0.7)
            ax.plot([uvN[i,0], uvG[i,0]], [uvN[i,1], uvG[i,1]],
                    color='lime', lw=0.6, alpha=0.4, linestyle=':')
        err_naive = np.linalg.norm(uvN[v] - uvG[v], axis=-1).mean()
        err_pred  = np.linalg.norm(uvP[v] - uvG[v], axis=-1).mean()
        ax.set_title(f'naive→gt {err_naive:.2f}px → pred→gt {err_pred:.2f}px',
                     fontsize=7)
    for ax in axes[len(picked):]:
        ax.set_axis_off()
    plt.tight_layout()
    out = vis_dir / "pair_grid.png"
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "ps_v12_cross_frame_overfit"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    split = sys.argv[3] if len(sys.argv) > 3 else 'val'
    main(exp, n, split=split)
