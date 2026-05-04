"""Reusable mid-training visualization helpers.

Provides two functions that were previously inlined in
`scripts/training/train_ps_v3.py`:

* `vis_pretrain_run(exp_dir, cache, cml_logger=None, n=10, zod_src=None)`
  — sanity-check the cache before training, save 10 sample PNGs under
  `exp_dir/vis_pretrain/` and (optionally) upload to ClearML.

* `midtrain_vis(model, exp_dir, cache, epoch, cfg, cml_logger=None,
                 device='cuda', amp_dtype=torch.float16, n=10,
                 zod_src=None)`
  — render N obj-centered val tiles with current model output to
  `exp_dir/vis_ep{epoch:03d}/` and upload to ClearML.

Both functions are process-safe: they only do file IO + model inference,
no DDP collectives. DDP callers MUST gate with
`accel.is_main_process` so only rank-0 runs this.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def vis_pretrain_run(exp_dir: Path, cache: str, cml_logger=None,
                     n: int = 10, zod_src: str | None = None,
                     log=print) -> None:
    """Pre-training cache sanity vis: n random insts with GT projection on
    full image. Idempotent — reuses existing `sample_idxs.json` if present.
    """
    exp_dir = Path(exp_dir)
    if zod_src:
        log("vis_pretrain skipped (ZOD direct-read; full-image vis not yet wired)")
        return
    try:
        from scripts.visualization.vis_pretrain import main as _vis_pretrain
        import sys as _sys
        _argv = _sys.argv[:]
        _sys.argv = ['vis_pretrain', '--cache', cache,
                     '--out', str(exp_dir / 'vis_pretrain'), '--n', str(n)]
        _vis_pretrain()
        _sys.argv = _argv
        log(f"vis_pretrain → {exp_dir / 'vis_pretrain'}")
        if cml_logger is not None:
            for p in sorted((exp_dir / 'vis_pretrain').glob('*.png')):
                try:
                    cml_logger.report_image('vis_pretrain', p.stem,
                                             iteration=0, local_path=str(p))
                except Exception:
                    pass
    except Exception as e:
        log(f"vis_pretrain skipped: {e}")


def midtrain_vis(model, exp_dir: Path, cache: str, epoch: int,
                 img_size: int, min_crop_px: int, max_crop_px: int,
                 cml_logger=None, device: str = 'cuda',
                 amp_dtype=torch.float16, n: int = 10,
                 zod_src: str | None = None, log=print,
                 max_rot_deg: float = 1.5, max_offset_m: float = 0.6) -> None:
    """Render N obj-centered val tiles with current model output to
    `exp_dir/vis_ep{epoch:03d}/`.

    Notes:
    - model MUST be the unwrapped module (caller should pass
      `accel.unwrap_model(model)` for DDP).
    - DDP callers MUST gate with `accel.is_main_process`.
    """
    import random as _r, numpy as _np, json as _json
    exp_dir = Path(exp_dir)
    out = exp_dir / f'vis_ep{epoch:03d}'
    out.mkdir(exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()

    if zod_src:
        from datasets.zod_full import ZODCalibDataset
        ds = ZODCalibDataset(zod_src, split='val',
                             img_size=img_size,
                             min_crop_px=min_crop_px,
                             max_crop_px=max_crop_px,
                             max_rot_deg=max_rot_deg,
                             max_offset_m=max_offset_m,
                             oversample=1)
    else:
        from datasets.pandaset_full import PandaSetCalibDatasetFull
        ds = PandaSetCalibDatasetFull(cache, split='val',
                                       img_size=img_size,
                                       min_crop_px=min_crop_px,
                                       max_crop_px=max_crop_px,
                                       max_rot_deg=max_rot_deg,
                                       max_offset_m=max_offset_m,
                                       oversample=1)

    sample_idx_fp = exp_dir / 'vis_pretrain' / 'sample_idxs.json'
    if sample_idx_fp.exists():
        idxs = _json.loads(sample_idx_fp.read_text())
        obj_filter = False  # curated already
    else:
        idxs = list(range(len(ds)))
        _r.Random(0).shuffle(idxs)
        obj_filter = True

    S = int(img_size)
    saved = 0
    was_training = model.training
    model.eval()
    try:
        for idx in idxs[:200]:
            if saved >= n:
                break
            try:
                # Pin np.random per idx so the crop box (u0, v0, cs) and the
                # extrinsic perturbation are identical across epochs/eval runs —
                # vis stays comparable. train_ps_v3._midtrain_vis does the same.
                _np.random.seed(int(idx))
                img, true_uvd, dist_uvd, vfp, uvd_full_v, pad_full_v = ds[idx]
            except Exception:
                continue
            is_obj = dist_uvd[:, 3].numpy() > 0.5
            if obj_filter:
                if not is_obj.any():
                    continue
                d_obj = dist_uvd[is_obj, :2].numpy()
                if max(float(d_obj[:, 0].max() - d_obj[:, 0].min()),
                       float(d_obj[:, 1].max() - d_obj[:, 1].min())) < 16:
                    continue
            Nmax = true_uvd.shape[0]
            pad = torch.zeros(1, Nmax, dtype=torch.bool, device=device)
            with torch.autocast(device_type='cuda', dtype=amp_dtype), torch.no_grad():
                img_gpu = img.unsqueeze(0).to(device).float().div_(255.0)
                p = model(img_gpu,
                          dist_uvd.unsqueeze(0).to(device)[..., :3],
                          key_padding_mask=pad,
                          vfp=vfp.view(1).to(device),
                          distorted_uvd_full=uvd_full_v.unsqueeze(0).to(device),
                          pad_full=pad_full_v.unsqueeze(0).to(device))[0].float().cpu().numpy()
            true_uv = true_uvd[:, :2].numpy()
            dist_uv = dist_uvd[:, :2].numpy()
            pred_uv = dist_uv + p[:, :2]
            err_b_obj = float(_np.linalg.norm(dist_uv[is_obj] - true_uv[is_obj], axis=1).mean())
            err_a_obj = float(_np.linalg.norm(pred_uv[is_obj] - true_uv[is_obj], axis=1).mean())
            err_b_bg = (float(_np.linalg.norm(dist_uv[~is_obj] - true_uv[~is_obj], axis=1).mean())
                        if (~is_obj).any() else float('nan'))
            err_a_bg = (float(_np.linalg.norm(pred_uv[~is_obj] - true_uv[~is_obj], axis=1).mean())
                        if (~is_obj).any() else float('nan'))

            from scripts.visualization.vis_pretrain import (
                _full_proj as _vp_full_proj,
                project_cuboids as _vp_proj_cubs,
                draw_cuboids as _vp_draw_cubs,
                patch_entropy as _vp_entropy,
            )
            inst = ds._load_inst(idx)
            box = getattr(ds, '_last_crop', None)
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
            uc = pred_uv[:, 0] - dist_uv[:, 0]
            vc = pred_uv[:, 1] - dist_uv[:, 1]
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
                          f'bg:{err_b_bg:.2f}→{err_a_bg:.2f}px  '
                          f'H={H_bits:.2f}b  lapV={lap_var:.0f}',
                          fontsize=8)
            plt.tight_layout(pad=0.2)
            fp = out / f'val_{saved:02d}_idx{idx:06d}.png'
            plt.savefig(fp, dpi=96, bbox_inches='tight')
            plt.close(fig)
            if cml_logger is not None:
                try:
                    cml_logger.report_image('vis_ep', f'sample_{saved:02d}',
                                             iteration=epoch, local_path=str(fp))
                except Exception:
                    pass
            saved += 1
    finally:
        model.train(was_training)
    log(f"vis_ep{epoch:03d}: saved {saved} → {out}")
