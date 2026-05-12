"""Standalone wrapper around the in-training `_midtrain_vis` pipeline so that
post-training demos, BA evaluations and ad-hoc spot checks render with the
EXACT same data flow + 2-panel layout the training loop uses each 10 epochs.

The single source of truth is `render_eval_samples(...)`. The train script's
`_midtrain_vis` is a thin wrapper that builds the dataset/exp_dir/logger and
delegates here. Ad-hoc demos do the same — never roll a private vis."""
from __future__ import annotations
import json
import random as _r
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.visualization.vis_pretrain import (
    _full_proj as _vp_full_proj, project_cuboids as _vp_proj_cubs,
    draw_cuboids as _vp_draw_cubs, patch_entropy as _vp_entropy,
)


def render_eval_samples(*, model, ds, out_dir, img_size: int,
                         device, amp_dtype=torch.bfloat16,
                         n: int = 10, epoch: int = 0,
                         sample_idxs: list | None = None,
                         obj_filter_when_no_idxs: bool = True,
                         cml_logger=None, log=print):
    """Render N val tiles using the model — same logic as train's _midtrain_vis.

    Args:
      model:     trained CalibNetDepth (eval mode; pose head OK).
      ds:        PandaSetCalibDatasetFull (or compatible) for the cache to vis.
      out_dir:   Path to write PNGs (created if absent; existing *.png removed).
      img_size:  model's input crop size (e.g. 128).
      device:    'cuda' / 'cuda:0' / 'cpu'.
      n:         number of samples to render.
      epoch:     used in panel title + cml_logger iteration tag.
      sample_idxs: explicit list of dataset idxs. If None, prefers
                   {exp_dir}/vis_pretrain/sample_idxs.json for cross-ep continuity;
                   else random + obj-presence filter.
      obj_filter_when_no_idxs: when generating idxs from scratch, drop samples
                   with no obj or with tiny obj bbox (matches train default).
      cml_logger:  optional ClearML logger.

    Returns: number of PNGs saved.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()

    # Pick idxs (caller wins; else sibling vis_pretrain/sample_idxs.json; else shuffled)
    if sample_idxs is None:
        sibling = out.parent / 'vis_pretrain' / 'sample_idxs.json'
        if sibling.exists():
            idxs = json.loads(sibling.read_text())
            obj_filter = False
        else:
            idxs = list(range(len(ds))); _r.Random(0).shuffle(idxs)
            obj_filter = bool(obj_filter_when_no_idxs)
    else:
        idxs = list(sample_idxs); obj_filter = False

    S = int(img_size)
    saved = 0
    for idx in idxs[:max(200, n * 10)]:
        if saved >= n: break
        try:
            np.random.seed(int(idx))   # deterministic crop + perturbation across epochs
            _sample = ds[idx]
            img, true_uvd, dist_uvd, vfp, bucket_uvd_v, bucket_valid_v = _sample[:6]
        except Exception:
            continue
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if obj_filter:
            if not is_obj.any(): continue
            d_obj = dist_uvd[is_obj, :2].numpy()
            if max(float(d_obj[:,0].max()-d_obj[:,0].min()),
                   float(d_obj[:,1].max()-d_obj[:,1].min())) < 16: continue

        Nmax = true_uvd.shape[0]
        pad = torch.zeros(1, Nmax, dtype=torch.bool, device=device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype), torch.no_grad():
            img_gpu = img.unsqueeze(0).to(device).float().div_(255.0)
            _out = model(img_gpu,
                          dist_uvd.unsqueeze(0).to(device)[..., :3],
                          key_padding_mask=pad,
                          vfp=vfp.view(1).to(device),
                          bucket_uvd=bucket_uvd_v.unsqueeze(0).to(device),
                          bucket_valid=bucket_valid_v.unsqueeze(0).to(device))
            per_pt = _out[0] if isinstance(_out, tuple) else _out
            p = per_pt[0].float().cpu().numpy()

        true_uv = true_uvd[:, :2].numpy(); dist_uv = dist_uvd[:, :2].numpy()
        pred_uv = dist_uv + p[:, :2]
        err_b_obj = float(np.linalg.norm(dist_uv[is_obj] - true_uv[is_obj], axis=1).mean()) if is_obj.any() else float('nan')
        err_a_obj = float(np.linalg.norm(pred_uv[is_obj] - true_uv[is_obj], axis=1).mean()) if is_obj.any() else float('nan')
        err_b_bg  = float(np.linalg.norm(dist_uv[~is_obj] - true_uv[~is_obj], axis=1).mean()) if (~is_obj).any() else float('nan')
        err_a_bg  = float(np.linalg.norm(pred_uv[~is_obj] - true_uv[~is_obj], axis=1).mean()) if (~is_obj).any() else float('nan')

        inst = ds._load_inst(idx)
        box  = getattr(ds, '_last_crop', None)
        full, uv_full, vis_full, is_obj_full, IH_f, IW_f = _vp_full_proj(inst)
        cubs_proj = _vp_proj_cubs(inst.get('cuboids', []),
                                   inst['T_gt'].numpy(), inst['K_full'].numpy(),
                                   tile_u0=int(inst.get('tile_u0', 0)),
                                   tile_v0=int(inst.get('tile_v0', 0)))
        crop_np = img.permute(1, 2, 0).cpu().numpy()
        H_bits, lap_var = _vp_entropy(crop_np)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), dpi=96,
                                        gridspec_kw={'width_ratios': [16, 9]})
        axL.imshow(full)
        axL.set_xlim(0, IW_f); axL.set_ylim(IH_f, 0)
        mfb = vis_full & ~is_obj_full
        axL.scatter(uv_full[mfb, 0], uv_full[mfb, 1], c='yellow', s=2, marker='x',
                    linewidths=0.4, alpha=0.5)
        mfo = vis_full & is_obj_full
        axL.scatter(uv_full[mfo, 0], uv_full[mfo, 1], c='lime', s=10, marker='x',
                    linewidths=0.9)
        _vp_draw_cubs(axL, cubs_proj, color='lime', lw=1.0, alpha=0.8)
        if box is not None:
            axL.add_patch(plt.Rectangle((box['u0'], box['v0']), box['cs'], box['cs'],
                                         fill=False, ec='red', lw=2))
        axL.set_xlim(0, IW_f); axL.set_ylim(IH_f, 0); axL.axis('off')
        axL.set_title(f'FULL  scene={inst.get("scene")} f={inst.get("frame")}  '
                      f'cubs={len(inst.get("cuboids", []))}', fontsize=8)

        axR.imshow(crop_np)
        uc = pred_uv[:, 0] - dist_uv[:, 0]; vc = pred_uv[:, 1] - dist_uv[:, 1]
        if (~is_obj).any():
            axR.quiver(dist_uv[~is_obj, 0], dist_uv[~is_obj, 1], uc[~is_obj], vc[~is_obj],
                       angles='xy', scale_units='xy', scale=1, color='deepskyblue',
                       width=0.004, headwidth=3.5, headlength=4, alpha=0.55, zorder=2)
        if is_obj.any():
            axR.quiver(dist_uv[is_obj, 0], dist_uv[is_obj, 1], uc[is_obj], vc[is_obj],
                       angles='xy', scale_units='xy', scale=1, color='orange',
                       width=0.006, headwidth=3.5, headlength=4, alpha=0.95, zorder=4)
        axR.scatter(true_uv[is_obj, 0], true_uv[is_obj, 1], c='lime', s=14,
                    marker='x', linewidths=0.9, zorder=5)
        axR.scatter(true_uv[~is_obj, 0], true_uv[~is_obj, 1], c='yellow', s=6,
                    marker='x', linewidths=0.5, alpha=0.5, zorder=3)
        if box is not None and cubs_proj:
            _vp_draw_cubs(axR, cubs_proj, color='lime', lw=0.9, alpha=0.85,
                          uv_offset=(box['u0'], box['v0']),
                          uv_scale=S / float(box['cs']))
        axR.set_xlim(0, S); axR.set_ylim(S, 0); axR.axis('off')
        axR.set_title(f'ep{epoch:03d}  obj:{err_b_obj:.2f}→{err_a_obj:.2f} '
                      f'bg:{err_b_bg:.2f}→{err_a_bg:.2f}px  H={H_bits:.2f}b  lapV={lap_var:.0f}',
                      fontsize=8)
        plt.tight_layout(pad=0.2)
        fp = out / f'val_{saved:02d}_idx{idx:06d}.png'
        plt.savefig(fp, dpi=96)
        if cml_logger is not None:
            try: cml_logger.report_matplotlib_figure(
                    title='vis_ep', series=f'sample_{saved:02d}',
                    iteration=epoch, figure=fig)
            except Exception: pass
        plt.close(fig)
        if cml_logger is not None:
            try: cml_logger.report_image('vis_ep', f'sample_{saved:02d}',
                                          iteration=epoch, local_path=str(fp))
            except Exception: pass
        saved += 1
    log(f'vis_ep{epoch:03d}: saved {saved} → {out}')
    return saved


def main():
    """CLI: render eval samples from a checkpoint against a cache.
    Example:
      python -m scripts.visualization.vis_eval \\
        --ckpt experiments/ps_multicam_corr_clspose_fxfy_100ep/best_model.pt \\
        --cache /mnt/nvme6t/e2e_calib_cache/pandaset_v3_tiled_multicam_corr \\
        --out experiments/ps_multicam_corr_clspose_fxfy_100ep/vis_eval --n 16
    """
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--max-rot-deg', type=float, default=1.5)
    ap.add_argument('--max-offset-m', type=float, default=0.6)
    ap.add_argument('--max-fx-pct', type=float, default=0.02)
    ap.add_argument('--max-fy-pct', type=float, default=0.02)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--use-frame-pose', action='store_true', default=True)
    ap.add_argument('--frame-pose-dof', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from datasets.pandaset_full import PandaSetCalibDatasetFull
    from models.model_depth import CalibNetDepth

    model = CalibNetDepth(img_size=args.img_size, in_channels=3,
                          n_layers=args.n_layers, use_convnext=True,
                          use_frustum=True, deform_mode='sl',
                          use_frame_pose=args.use_frame_pose,
                          frame_pose_dof=args.frame_pose_dof).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f'load: miss={len(miss)} unexp={len(unexp)}')
    model.eval()

    ds = PandaSetCalibDatasetFull(args.cache, split='val',
                                   img_size=args.img_size,
                                   max_offset_m=args.max_offset_m,
                                   max_rot_deg=args.max_rot_deg,
                                   max_fx_pct=args.max_fx_pct,
                                   max_fy_pct=args.max_fy_pct,
                                   min_crop_px=128, max_crop_px=384,
                                   grid_n=16, oversample=1)
    n = render_eval_samples(model=model, ds=ds, out_dir=args.out,
                             img_size=args.img_size, device=args.device,
                             n=args.n, epoch=999)
    print(f'wrote {n} samples → {args.out}')


if __name__ == '__main__':
    main()
