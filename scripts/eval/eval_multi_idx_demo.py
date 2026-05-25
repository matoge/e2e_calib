"""Multi-idx evaluation + full-image demo for the sub-pixel report.

Per idx, we run three configurations using the SAME frozen ckpt:
  (a) B=1, cs=512        — "full image" lightweight demo (1 forward + 1 GN)
  (b) B=200, cs=512      — original 200×512 baseline
  (c) B=800, cs=256 ×4   — split sub-crop, the headline 0.46-px@fx result

For each idx we additionally render a 3-panel overlay (GT / perturbed /
corrected) for the full-image (B=1) inference — that's the most readable
demo: one network call, one Gauss-Newton solve, sub-px reprojection.

Output:
  scripts/_debug/eval_multi_idx_demo/
    summary.csv
    summary.md
    overlay_idx{ID}_full.png
    overlay_idx{ID}_b800.png
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.eval_shared_256x800 import (
    _load_cfg, _build_model, _draw_pert, _solve_one,
    render_3panel_overlay,
    CACHE, CKPT, DEVICE,
)
from datasets.pandaset_full import PandaSetCalibDatasetFull


OUT = REPO / 'scripts' / '_debug' / 'eval_multi_idx_demo'


def _residual_norms(delta, ypr_target, t_target):
    target_xyz = np.array([ypr_target[2], ypr_target[1], ypr_target[0]],
                           dtype=np.float64)
    d = delta.detach().cpu().numpy()
    return (
        float(np.linalg.norm(d[:3] - target_xyz)),  # ω deg
        float(np.linalg.norm(d[3:] - t_target)),    # t  m
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idxs', type=int, nargs='+',
                    default=[17, 100, 500, 1000, 2000, 3000, 3800])
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m',     type=float, default=0.05)
    ap.add_argument('--seed',    type=int,   default=1007)
    ap.add_argument('--n-512',   type=int,   default=200)
    ap.add_argument('--n-256',   type=int,   default=200)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    print(f'[multi-idx] ckpt={CKPT.name}  img_size={cfg["img_size"]}  '
          f'min_crop_px={cfg["min_crop_px"]}  max_crop_px={cfg["max_crop_px"]}')
    print(f'[multi-idx] perturbation: ω±{args.rot_deg}°  t±{args.t_m}m  '
          f'(within training range ±0.5°/±0.05m)')

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )

    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    for idx in args.idxs:
        inst0 = ds._load_inst(int(idx))
        if not inst0.get('is_fisheye', False) or 'distortion' not in inst0:
            print(f'[idx {idx}] skipped: not fisheye / no KB distortion')
            continue
        dist_one = inst0['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
        fx = float(inst0['K_full'].numpy()[0, 0])
        scene = inst0.get('scene', '?')
        cam = inst0.get('cam', '?')

        rng = np.random.RandomState(args.seed)
        ypr_t, t_t = _draw_pert(rng, rot_deg=args.rot_deg, t_m=args.t_m)
        print()
        print(f'═══ idx={idx}  scene={scene}  cam={cam}  fx={fx:.1f}px  '
              f'═══════════════════════')
        print(f'   δ_target  ω=[{ypr_t[2]:+.4f},{ypr_t[1]:+.4f},{ypr_t[0]:+.4f}]°  '
              f't=[{t_t[0]:+.4f},{t_t[1]:+.4f},{t_t[2]:+.4f}]m')

        configs = [
            ('full',  1,            512, 1),
            ('b200', args.n_512,    512, 1),
            ('b800', args.n_256,    256, 4),
        ]
        per_idx = {}
        for tag, n_inst, cs, npi in configs:
            t0 = time.time()
            rng_b = np.random.RandomState(args.seed + 1)
            try:
                delta, B, _H = _solve_one(
                    model, ds, target_idx=idx, n_inst=n_inst, cs=cs,
                    n_per_inst=npi, rng=rng_b,
                    ypr_target=ypr_t, t_target=t_t, dist_one=dist_one,
                    cfg=cfg, label=f'idx{idx}-{tag}')
            except Exception as e:
                print(f'   {tag:5s}  FAILED  {e!r}')
                continue
            dw, dt = _residual_norms(delta, ypr_t, t_t)
            dwpx = float(fx * np.tan(np.deg2rad(dw)))
            tt = time.time() - t0
            print(f'   {tag:5s}  B={B:4d}  '
                  f'ω={dw:.4f}° ({dwpx:.3f}px@fx)  t={dt:.4f}m  '
                  f'({tt:.2f}s)')
            per_idx[tag] = dict(B=B, omega_deg=dw, omega_px=dwpx, t_m=dt,
                                wall_s=tt, delta=delta.cpu().numpy().tolist())

            # Render overlay for the 'full' (B=1) and 'b800' rows
            if tag in ('full', 'b800'):
                title_tag = 'full-image (B=1)' if tag == 'full' else 'B=800 4×256'
                ovr = OUT / f'overlay_idx{idx}_{tag}.png'
                try:
                    render_3panel_overlay(
                        inst0, ypr_t, t_t, delta,
                        out_path=ovr,
                        suptitle=(f'idx={idx}  {title_tag}  '
                                  f'ω={dw:.4f}° ({dwpx:.3f}px@fx)  t={dt:.4f}m'),
                        panel_label=f'BA-corrected ({tag})',
                    )
                    print(f'      → {ovr.relative_to(REPO)}')
                except Exception as e:
                    print(f'      overlay failed: {e!r}')

        rows.append(dict(idx=idx, scene=scene, cam=cam, fx=fx,
                         ypr_target=ypr_t.tolist(), t_target=t_t.tolist(),
                         **{f'{k}__{kk}': vv
                            for k, v in per_idx.items()
                            for kk, vv in v.items()
                            if kk != 'delta'}))

    # Save summary CSV + markdown
    csv_path = OUT / 'summary.csv'
    md_path = OUT / 'summary.md'
    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with csv_path.open('w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), list) else r.get(k))
                           for k in cols})
        # Markdown table: idx, scene, fx | full ω/t | b200 ω/t | b800 ω/t
        lines = []
        lines.append('| idx | scene | fx | full(B=1) ω(px@fx) | b200×512 ω(px@fx) | '
                     'b800×256 ω(px@fx) | full t(m) | b200 t(m) | b800 t(m) |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for r in rows:
            def fmt_px(tag):
                v = r.get(f'{tag}__omega_px')
                return f'{v:.3f}' if v is not None else '—'
            def fmt_t(tag):
                v = r.get(f'{tag}__t_m')
                return f'{v:.4f}' if v is not None else '—'
            lines.append(
                f'| {r["idx"]} | {r["scene"]} | {r["fx"]:.0f} | '
                f'{fmt_px("full")} | {fmt_px("b200")} | {fmt_px("b800")} | '
                f'{fmt_t("full")} | {fmt_t("b200")} | {fmt_t("b800")} |'
            )
        # Mean / median rows
        for tag in ('full', 'b200', 'b800'):
            vs = [r.get(f'{tag}__omega_px') for r in rows
                  if r.get(f'{tag}__omega_px') is not None]
            ts = [r.get(f'{tag}__t_m') for r in rows
                  if r.get(f'{tag}__t_m') is not None]
            if vs:
                print(f'   {tag:5s}  mean ω={np.mean(vs):.3f}px  '
                      f'median ω={np.median(vs):.3f}px  '
                      f'mean t={np.mean(ts):.4f}m')
        md_path.write_text('\n'.join(lines) + '\n')
        print()
        print(f'[multi-idx] wrote {csv_path.relative_to(REPO)}')
        print(f'[multi-idx] wrote {md_path.relative_to(REPO)}')


if __name__ == '__main__':
    main()
