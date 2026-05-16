"""End-to-end inference from RAW frame (PNG + points_V.txt) → BA δ.

No cache lookup. Goes:

    image_<f>.png + points_V_<f>.txt + calib.calib
        → kamikado adapter → CalibFrame
        → tile_cutter → list of tile inst dicts
        → infer_tiles (slides over the parent image, runs the model
                        on every tile, returns the BA pool)
        → solve_dofs → 6-DoF δ + Cov

Output is plain text on stdout so VSCode (or any terminal) can read it.

Usage:
    docker exec ... python3 scripts/_debug/infer_raw_frame.py \\
        --scene /raw/kamikado/scenes/<scene_dir> \\
        --frame 0 \\
        --exp   km_wv_wm_dgx2_n2_img128_v2
"""
import argparse
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from scripts.data.adapters.kamikado import load_frame, TILE_LAYOUT
from scripts.data.tile_cutter import frame_to_tiles
from scripts.inference.infer_calib import load_calib_model
from scripts.ba.ba_multicam_corr import (
    infer_tiles, solve_dofs, _DOF_PRESETS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True,
                    help='kamikado raw scene dir')
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--exp', default='km_wv_wm_dgx2_n2_img128_v2')
    ap.add_argument('--tile-size',   type=int, default=384,
                    help='must match the trained max_crop_px')
    ap.add_argument('--tile-stride', type=int, default=320,
                    help='= tile_size - 64 → ~64px overlap between tiles')
    ap.add_argument('--huber-k',   type=float, default=0.0)
    ap.add_argument('--n-iter',    type=int,   default=1)
    ap.add_argument('--sigma-max', type=float, default=0.0)
    args = ap.parse_args()

    print(f'== adapter: {args.scene}  frame={args.frame} ==')
    cf = load_frame(Path(args.scene), args.frame)
    print(f'  {cf}')
    print(f'  parent image:  {cf.hw[1]}x{cf.hw[0]} px')
    print(f'  intensity ∈ [{cf.intensity.min():.3f}, {cf.intensity.max():.3f}]')

    print(f'== tile_cutter ==')
    tiles = frame_to_tiles(cf, **TILE_LAYOUT, min_pts=8)
    print(f'  {len(tiles)} tiles')
    for t in tiles[:3]:
        print(f'    tile_id={t["tile_id"]}  origin=({t["tile_u0"]},{t["tile_v0"]})  N_pts={len(t["pts"])}')
    if len(tiles) > 3:
        print(f'    ... +{len(tiles)-3} more')

    # infer_tiles wants a parent-image array + per-pt arrays in PARENT
    # coords. The CalibFrame already has those.
    print(f'== infer_tiles + solve_dofs ==')
    img_arr = cf.img
    uv = cf.uv_full.astype(np.float32)
    z  = cf.z_cam.astype(np.float32)
    K  = cf.K.astype(np.float32)
    intensity = cf.intensity.astype(np.float32)

    model = load_calib_model(args.exp).eval()
    ba_cfg = dict(tile_size=args.tile_size,
                  model_input_size=128,  # model's S
                  max_pts_per_tile=256,
                  min_pts_per_tile=8,
                  tile_stride=args.tile_stride)
    res = infer_tiles(model, img_arr, uv, z, K, ba_cfg,
                       torch.device('cuda'), intensity=intensity)
    if res is None:
        print('  infer_tiles returned None'); sys.exit(1)
    uv_pool, par_pool, z_pool = res
    print(f'  BA pool N={len(uv_pool)}')

    if args.sigma_max > 0:
        s = np.sqrt(par_pool[:, 2] * par_pool[:, 3])
        m = s <= args.sigma_max
        uv_pool, par_pool, z_pool = uv_pool[m], par_pool[m], z_pool[m]
        print(f'  σ-pre-filter (≤{args.sigma_max}px): {m.sum()}/{len(m)} kept')

    delta = solve_dofs(uv_pool, par_pool, z_pool, K,
                       _DOF_PRESETS['6dof_ext'], damping=1e-3,
                       huber_k=(args.huber_k if args.huber_k > 0 else None),
                       n_iter=args.n_iter)
    sigma = np.sqrt(np.diag(solve_dofs._last_cov))

    print(f'\n== 6-DoF δ_pred ==')
    print(f'{"DoF":10s}  {"δ_pred":>10s}  {"σ":>9s}')
    print('-' * 36)
    for nm, dp, sp in zip(_DOF_PRESETS['6dof_ext'], delta, sigma):
        print(f'{nm:10s}  {float(dp):+.4f}  {float(sp):>9.4f}')


if __name__ == '__main__':
    main()
