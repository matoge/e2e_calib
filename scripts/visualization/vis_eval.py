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


def render_pose_eval_samples(*, model, ds, out_dir, img_size: int,
                              device, amp_dtype=torch.bfloat16,
                              n: int = 6, ps_root='/mnt/nvme6t/pandaset',
                              sample_idxs: list | None = None, log=print):
    """Pose-head evaluation: apply predicted μ as extrinsic+intrinsic correction
    and reproject ALL raw lidar (d=0 + d=1) so foreground cars get covered.

    Renders 2-row layout per sample: top=BEFORE (perturbed projection,
    red=lidar, lime=true target), bottom=AFTER (μ-recovered projection,
    cyan=lidar, lime=true target). Title shows GT vs μ numeric comparison.
    """
    import pickle, gzip, io
    from PIL import Image as _PIL
    from scipy.spatial.transform import Rotation as _Rot
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'): old.unlink()

    if sample_idxs is None:
        sibling = out.parent / 'vis_pretrain' / 'sample_idxs.json'
        if sibling.exists():
            idxs = json.loads(sibling.read_text())
        else:
            idxs = list(range(min(200, len(ds))))
            _r.Random(0).shuffle(idxs)
    else:
        idxs = list(sample_idxs)

    saved = 0
    for idx in idxs[:max(200, n * 10)]:
        if saved >= n: break
        try:
            np.random.seed(int(idx))
            _sample = ds[idx]
            img, true_uvd, dist_uvd, vfp, bucket_uvd_v, bucket_valid_v = _sample[:6]
            pert = _sample[6] if len(_sample) >= 7 else None
            crop = getattr(ds, '_last_crop', None)
            inst = ds._load_inst(idx)
        except Exception:
            continue
        if pert is None:
            log(f'  skip idx={idx}: no pert vec'); continue
        if crop is None:
            log(f'  skip idx={idx}: no _last_crop'); continue

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
            if not isinstance(_out, tuple) or len(_out) < 2:
                log(f'  skip idx={idx}: model has no pose head output'); continue
            per_pt, head_out = _out
            mu_t_ = head_out[0]; ls_t_ = head_out[1]  # also have L at [2] post-Cholesky upgrade
            mu_p = mu_t_.float().cpu().numpy()[0]
            sig_p = ls_t_.exp().float().cpu().numpy()[0]
        gt_p = pert.numpy()
        dof = mu_p.shape[0]
        gt_pose = gt_p[:6]
        mu_pose = mu_p[:6]
        gt_fxfy = gt_p[6:8] if dof >= 8 and gt_p.shape[0] >= 8 else np.zeros(2)
        mu_fxfy = mu_p[6:8] if dof >= 8 else np.zeros(2)

        # Read raw lidar from disk (d=0 + d=1)
        scene = inst['scene']; frame = int(inst['frame'])
        ld = Path(ps_root) / scene / 'lidar' / f'{frame:02d}.pkl'
        if not ld.exists(): ld = ld.with_suffix('.pkl.gz')
        try:
            df = pickle.load(open(ld,'rb')) if ld.suffix == '.pkl' else pickle.load(gzip.open(ld,'rb'))
            pts_w_raw = df[['x','y','z']].values.astype(np.float32)
        except Exception:
            pts_w_raw = inst['pts'].numpy()

        R_gt = inst['R_gt'].numpy(); cp = inst['cam_pos'].numpy(); K = inst['K_full'].numpy()
        # Replay dataset's perturbation forward (GT)
        R_off = R_gt @ _Rot.from_euler('zyx', gt_pose[3:6], degrees=True).as_matrix()
        cp_off = cp + gt_pose[:3]
        K_off = K.copy()
        K_off[0,0] = K[0,0] * (1 + gt_fxfy[0]); K_off[1,1] = K[1,1] * (1 + gt_fxfy[1])
        # μ-recovered (undo μ from off)
        R_rec = R_off @ _Rot.from_euler('zyx', -mu_pose[3:6], degrees=True).as_matrix()
        cp_rec = cp_off - mu_pose[:3]
        K_rec = K_off.copy()
        K_rec[0,0] = K_off[0,0] / (1 + mu_fxfy[0])
        K_rec[1,1] = K_off[1,1] / (1 + mu_fxfy[1])

        def proj(Rx, cpx, Kx):
            pc = (Rx.T @ (pts_w_raw - cpx).T).T
            z = pc[:,2]
            uv = (pc[:,:2] * np.array([Kx[0,0], Kx[1,1]]) / np.maximum(z[:,None],1e-6)
                  + np.array([Kx[0,2], Kx[1,2]]))
            return uv, z
        uv_true, z_true = proj(R_gt, cp, K)
        uv_off, z_off = proj(R_off, cp_off, K_off)
        uv_rec, z_rec = proj(R_rec, cp_rec, K_rec)

        # Load full image (jpg_bytes in cache)
        jpg = inst.get('jpg_bytes', None)
        if jpg is not None:
            full_img = np.asarray(_PIL.open(io.BytesIO(jpg)).convert('RGB'))
        else:
            full_img = np.zeros((int(inst['IH']), int(inst['IW']), 3), dtype=np.uint8)
        IH, IW = full_img.shape[:2]
        vis_t = (z_true>0.5)&(uv_true[:,0]>=0)&(uv_true[:,0]<IW)&(uv_true[:,1]>=0)&(uv_true[:,1]<IH)
        vis_o = (z_off>0.5)&(uv_off[:,0]>=0)&(uv_off[:,0]<IW)&(uv_off[:,1]>=0)&(uv_off[:,1]<IH)
        vis_r = (z_rec>0.5)&(uv_rec[:,0]>=0)&(uv_rec[:,0]<IW)&(uv_rec[:,1]>=0)&(uv_rec[:,1]<IH)
        both = vis_t & vis_o & vis_r
        if both.sum() < 10: continue
        err_before = float(np.linalg.norm(uv_off[both]-uv_true[both], axis=1).mean())
        err_after  = float(np.linalg.norm(uv_rec[both]-uv_true[both], axis=1).mean())

        u0, v0, cs = crop['u0'], crop['v0'], crop['cs']
        fig, axes = plt.subplots(2, 1, figsize=(IW/100, 2*IH/100 + 0.8), dpi=100)
        # BEFORE
        axes[0].imshow(full_img)
        axes[0].scatter(uv_off[both,0], uv_off[both,1], s=2, c='red', alpha=0.6)
        axes[0].scatter(uv_true[both,0], uv_true[both,1], s=2, c='lime', alpha=0.6)
        axes[0].add_patch(plt.Rectangle((u0,v0), cs, cs, fill=False, ec='yellow', lw=2))
        axes[0].set_xlim(0, IW); axes[0].set_ylim(IH, 0); axes[0].axis('off')
        axes[0].set_title(
            f'BEFORE  mean err={err_before:.2f} px   scene={scene} f{frame}   tile=({u0},{v0},{cs})\n'
            f'GT pert:  t={gt_pose[:3].round(3)} m   ypr={gt_pose[3:6].round(3)}°   '
            f'Δfx%={gt_fxfy[0]*100:+.2f}  Δfy%={gt_fxfy[1]*100:+.2f}',
            fontsize=9)
        # AFTER
        axes[1].imshow(full_img)
        axes[1].scatter(uv_rec[both,0], uv_rec[both,1], s=2, c='cyan', alpha=0.7)
        axes[1].scatter(uv_true[both,0], uv_true[both,1], s=2, c='lime', alpha=0.6)
        axes[1].add_patch(plt.Rectangle((u0,v0), cs, cs, fill=False, ec='yellow', lw=2))
        axes[1].set_xlim(0, IW); axes[1].set_ylim(IH, 0); axes[1].axis('off')
        red_pct = (err_before-err_after)/max(err_before,1e-9)*100
        axes[1].set_title(
            f'AFTER (pose-head μ)  mean err={err_after:.2f} px   ({red_pct:.0f}% reduction)\n'
            f'μ:  t={mu_pose[:3].round(3)} m   ypr={mu_pose[3:6].round(3)}°   '
            f'Δfx%={mu_fxfy[0]*100:+.2f}  Δfy%={mu_fxfy[1]*100:+.2f}\n'
            f'residual:  t={(gt_pose[:3]-mu_pose[:3]).round(3)} m   '
            f'ypr={(gt_pose[3:6]-mu_pose[3:6]).round(3)}°   '
            f'Δfx%_err={(gt_fxfy[0]-mu_fxfy[0])*100:+.2f}  Δfy%_err={(gt_fxfy[1]-mu_fxfy[1])*100:+.2f}',
            fontsize=9)
        plt.tight_layout()
        fp = out / f'pose_{saved:02d}_idx{idx:06d}.png'
        plt.savefig(fp, dpi=100, bbox_inches='tight')
        plt.close(fig)
        log(f'  pose_{saved:02d} idx={idx} scene={scene} f{frame}  err {err_before:.1f}→{err_after:.1f}px  N={int(both.sum())}')
        saved += 1
    log(f'pose-eval: saved {saved} → {out}')
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
    ap.add_argument('--frame-pose-full-cov', action='store_true',
                    help='Construct pose head with full Cholesky cov (matches '
                         'training ran with --frame-pose-full-cov).')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--mode', choices=('perpt','pose'), default='perpt',
                    help='perpt: training-style 2-panel (full + quiver). '
                         'pose: pose-head μ eval — full-frame BEFORE/AFTER with '
                         'all raw lidar (d=0+d=1) reprojected.')
    args = ap.parse_args()

    from datasets.pandaset_full import PandaSetCalibDatasetFull
    from models.model_depth import CalibNetDepth

    model = CalibNetDepth(img_size=args.img_size, in_channels=3,
                          n_layers=args.n_layers, use_convnext=True,
                          use_frustum=True, deform_mode='sl',
                          use_frame_pose=args.use_frame_pose,
                          frame_pose_dof=args.frame_pose_dof,
                          frame_pose_full_cov=args.frame_pose_full_cov).to(args.device)
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
    if args.mode == 'pose':
        n = render_pose_eval_samples(model=model, ds=ds, out_dir=args.out,
                                      img_size=args.img_size, device=args.device,
                                      n=args.n)
    else:
        n = render_eval_samples(model=model, ds=ds, out_dir=args.out,
                                 img_size=args.img_size, device=args.device,
                                 n=args.n, epoch=999)
    print(f'wrote {n} samples → {args.out}')


if __name__ == '__main__':
    main()
