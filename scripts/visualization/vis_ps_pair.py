"""Cross-frame Δuv visualization (vis_ps_pair).

For each sampled pair, shows A and B side-by-side. A few points are colour-
coded; on A we mark the source position, on B we mark three positions in the
same colour:
  ○ naive   (= forward-project from A through the perturbed pose, current guess)
  ×  gt     (= true projection in B)
  +  pred   (= naive + Δuv predicted by the model)
Plus arrows naive→pred (orange) and naive→gt (lime, dashed).

So per-point reading: starting at ○, did the model arrow (orange to +)
land at × (lime arrow target)? If yes the model is correct on that point.
σ is shown as a faint ellipse around +.

Usage:
    python scripts/visualization/vis_ps_pair.py <exp> [n_samples=12] [split=val] [points_per_sample=8]
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import importlib.util, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
import numpy as np
from pathlib import Path

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.model_pair import CalibNetDepthPair

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(exp_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("_cfg", exp_dir / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.CFG


def main(exp: str, n_samples: int = 12, split: str = 'val',
         pts_per_sample: int = 8, max_scan: int = 200,
         upload_to_clearml: bool = True):
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

    print(f"[{exp}] split={split}, scanning up to {max_scan} for {n_samples} samples")
    picked = []
    for i in range(max_scan):
        s = ds[i]
        valid = ~s['pad_dir'][0, 1]
        if valid.sum() < pts_per_sample:
            continue
        picked.append(s)
        if len(picked) >= n_samples:
            break
    print(f"  picked {len(picked)} samples")

    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    for old in vis_dir.glob("pair_*.png"): old.unlink()

    rng = np.random.default_rng(0)
    cmap = plt.get_cmap('tab10')

    cols = 2  # A | B per sample
    rows = len(picked)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.6), dpi=110)
    axes = np.atleast_2d(axes)

    for ri, s in enumerate(picked):
        with torch.no_grad():
            patches    = s['patches'].to(DEVICE).unsqueeze(0)
            uvd        = s['uvd'].to(DEVICE).unsqueeze(0)
            pad        = s['pad'].to(DEVICE).unsqueeze(0)
            pose_hat   = s['pose_hat_6dof'].to(DEVICE).unsqueeze(0)
            uv_hat     = s['uv_hat'].to(DEVICE).unsqueeze(0)
            uv_gt      = s['uv_gt'].to(DEVICE).unsqueeze(0)
            pad_dir    = s['pad_dir'].to(DEVICE).unsqueeze(0)

            uvd_A   = uvd[:, 0, :, :3]; uvd_B = uvd[:, 1, :, :3]
            pose_AB = pose_hat[:, 0, 1]
            uv_B_naive = uv_hat[:, 0, 1]
            uv_B_gt    = uv_gt[:, 0, 1]
            valid      = ~pad_dir[:, 0, 1]
            vfp = torch.full((1,), float(c['img_size']), device=DEVICE)
            params = model.forward_pair(
                patches[:, 0], patches[:, 1], uvd_A, uvd_B, uv_B_naive,
                pose_AB, vfp, pad_A=pad[:, 0], pad_B=pad[:, 1], query_pad=pad[:, 0],
            )[0]
            valid = valid[0]

        img_A  = patches[0, 0].permute(1,2,0).cpu().numpy()
        img_B  = patches[0, 1].permute(1,2,0).cpu().numpy()
        uvA    = uvd_A[0, :, :2].cpu().numpy()
        uvN    = uv_B_naive[0].cpu().numpy()
        uvG    = uv_B_gt[0].cpu().numpy()
        delta  = params[:, :2].float().cpu().numpy()
        sx     = params[:, 2].float().exp().cpu().numpy()
        sy     = params[:, 3].float().exp().cpu().numpy()
        uvP    = uvN + delta
        v_idx  = np.where(valid.cpu().numpy())[0]
        n_pick = min(pts_per_sample, len(v_idx))
        sel    = rng.choice(v_idx, size=n_pick, replace=False)

        ax_A = axes[ri, 0]
        ax_B = axes[ri, 1]
        ax_A.imshow(img_A); ax_A.set_axis_off()
        ax_B.imshow(img_B); ax_B.set_axis_off()

        # all valid (faint, for context)
        ax_A.scatter(uvA[v_idx, 0], uvA[v_idx, 1], c='white', s=4, alpha=0.25, marker='.')
        ax_B.scatter(uvN[v_idx, 0], uvN[v_idx, 1], c='white', s=4, alpha=0.25, marker='.')

        for ci, i in enumerate(sel):
            color = cmap(ci % 10)
            # A: source dot
            ax_A.scatter(uvA[i, 0], uvA[i, 1], c=[color], s=44, marker='o',
                         edgecolors='black', linewidths=0.7, zorder=4)
            ax_A.text(uvA[i, 0]+1.5, uvA[i, 1]-1.5, str(ci), color=color,
                       fontsize=7, fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

            # B: naive (○) → pred (+) arrow in orange
            ax_B.scatter(uvN[i, 0], uvN[i, 1], facecolors='none', edgecolors=[color],
                          s=70, marker='o', linewidths=1.2, zorder=4)
            # B: gt (×) in same color
            ax_B.scatter(uvG[i, 0], uvG[i, 1], c=[color], s=60, marker='x',
                          linewidths=1.6, zorder=5)
            # B: pred (+) in same color
            ax_B.scatter(uvP[i, 0], uvP[i, 1], c=[color], s=70, marker='+',
                          linewidths=1.6, zorder=5)
            # σ ellipse around pred
            ax_B.add_patch(Ellipse((uvP[i,0], uvP[i,1]),
                                    2*float(sx[i].clip(0.5, 8)),
                                    2*float(sy[i].clip(0.5, 8)),
                                    fill=False, ec=color, alpha=0.35, lw=0.7, zorder=3))
            # arrows: naive→pred (orange-ish solid), naive→gt (lime dashed)
            ax_B.annotate('', xy=(uvP[i,0], uvP[i,1]), xytext=(uvN[i,0], uvN[i,1]),
                          arrowprops=dict(arrowstyle='->', color='orange', lw=1.0, alpha=0.85),
                          zorder=4)
            ax_B.annotate('', xy=(uvG[i,0], uvG[i,1]), xytext=(uvN[i,0], uvN[i,1]),
                          arrowprops=dict(arrowstyle='->', color='lime', lw=0.8,
                                           alpha=0.65, linestyle=':'),
                          zorder=4)
            ax_B.text(uvN[i, 0]+1.5, uvN[i, 1]-1.5, str(ci), color=color,
                       fontsize=7, fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

        err_naive = float(np.linalg.norm(uvN[sel] - uvG[sel], axis=-1).mean())
        err_pred  = float(np.linalg.norm(uvP[sel] - uvG[sel], axis=-1).mean())
        ax_A.set_title(f'patch_A  (sample {ri})', fontsize=9)
        ax_B.set_title(f'patch_B   naive→gt {err_naive:.2f}px → pred→gt {err_pred:.2f}px',
                        fontsize=9)

    fig.suptitle(f'{exp}  •  ○ naive  ×  gt  + pred   (orange arrow = model, lime dotted = truth)',
                  fontsize=10, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = vis_dir / "pair_grid.png"
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out}")

    if upload_to_clearml:
        try:
            from clearml import Task
            tasks = Task.get_tasks(project_name='e2e_calib/cross-frame',
                                   task_filter={'order_by':['-last_update']})
            t = next((t for t in tasks if t.name == exp), None)
            if t is not None:
                t.get_logger().report_image(
                    title='vis', series='pair_grid',
                    iteration=0, local_path=str(out))
                print(f'uploaded → ClearML task {t.id[:8]}')
            else:
                print(f'(no matching ClearML task for {exp}, vis kept local only)')
        except Exception as e:
            print(f'(clearml upload skipped: {e})')


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "ps_v12_cross_frame_overfit"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    split = sys.argv[3] if len(sys.argv) > 3 else 'val'
    ppl   = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    main(exp, n, split=split, pts_per_sample=ppl)
