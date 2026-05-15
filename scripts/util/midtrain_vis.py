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

    # Render two pools: n samples passing obj_filter, and n samples without
    # filter (random scenes). 2*n images are saved per epoch per dataset.
    idxs = list(range(len(ds)))
    _r.Random(0).shuffle(idxs)

    S = int(img_size)
    saved_obj = 0
    saved_any = 0
    n_obj_target = n
    n_any_target = n
    was_training = model.training
    model.eval()
    _first_error_logged = False
    try:
        for idx in idxs[:600]:
            if saved_obj >= n_obj_target and saved_any >= n_any_target:
                break
            # Let real exceptions bubble — only catch the rare "no points
            # in crop / dataset retry" so the first stacktrace surfaces in
            # the log instead of being silently eaten.
            _np.random.seed(int(idx))
            try:
                out = ds[idx]
            except (RuntimeError, ValueError) as _e:
                if not _first_error_logged:
                    log(f"vis_ep{epoch:03d}: ds[{idx}] retryable: {_e}")
                    _first_error_logged = True
                continue
            img, true_uvd, dist_uvd, vfp = out[0], out[1], out[2], out[3]
            bucket_uvd_v   = out[4]
            bucket_valid_v = out[5]
            is_obj = dist_uvd[:, 3].numpy() > 0.5
            d_obj = dist_uvd[is_obj, :2].numpy() if is_obj.any() else None
            obj_qualifies = (
                d_obj is not None and d_obj.shape[0] > 0 and
                max(float(d_obj[:, 0].max() - d_obj[:, 0].min()),
                    float(d_obj[:, 1].max() - d_obj[:, 1].min())) >= 16
            )
            if obj_qualifies and saved_obj < n_obj_target:
                pool = 'obj'
            elif saved_any < n_any_target:
                pool = 'any'
            else:
                continue
            Nmax = true_uvd.shape[0]
            pad = torch.zeros(1, Nmax, dtype=torch.bool, device=device)
            use_intensity = bool(getattr(model, 'use_intensity', False))
            if use_intensity and dist_uvd.shape[-1] >= 5:
                point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
            else:
                point_in = dist_uvd[..., :3]
            with torch.autocast(device_type='cuda', dtype=amp_dtype), torch.no_grad():
                img_gpu = img.unsqueeze(0).to(device).float().div_(255.0)
                p = model(img_gpu,
                          point_in.unsqueeze(0).to(device),
                          key_padding_mask=pad,
                          vfp=vfp.view(1).to(device),
                          bucket_uvd=bucket_uvd_v.unsqueeze(0).to(device),
                          bucket_valid=bucket_valid_v.unsqueeze(0).to(device))[0].float().cpu().numpy()
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
            sub_idx = saved_obj if pool == 'obj' else saved_any
            fp = out / f'val_{pool}_{sub_idx:02d}_idx{idx:06d}.png'
            plt.savefig(fp, dpi=96, bbox_inches='tight')
            plt.close(fig)
            if cml_logger is not None:
                try:
                    cml_logger.report_image(f'vis_ep_{pool}',
                                             f'sample_{sub_idx:02d}',
                                             iteration=epoch, local_path=str(fp))
                except Exception:
                    pass
            if pool == 'obj':
                saved_obj += 1
            else:
                saved_any += 1
    finally:
        model.train(was_training)
        # Release lmdb env in this (parent) process so DataLoader workers
        # forked AFTER this call can open it without hitting lmdb's
        # 'already open in this process' guard.
        try:
            ds.close_lmdb()
        except Exception:
            pass
    log(f"vis_ep{epoch:03d}: saved obj={saved_obj} any={saved_any} → {out}")
