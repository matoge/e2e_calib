"""Smoke: BA on a full frame (sliding-tile sweep) using infer_tiles."""
import sys, pathlib, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image

from scripts.inference.infer_pipeline import make_ds
from scripts.inference.infer_calib import load_calib_model
from scripts.ba.ba_multicam_corr import infer_tiles, solve_dofs, _DOF_PRESETS


def main():
    exp = 'km_wv_8gpu_200ep_os4'
    cache = '/cache/kamikado_v3_tiled'
    ds, c = make_ds(exp, cache, split='val', oversample=1)
    model = load_calib_model(exp).eval()

    # Pick the first val frame; all tiles of the same scene+frame share
    # uv_full / pts / z / intensity, so just take inst 0.
    inst = ds._load_inst(0)
    full_jpg = bytes(inst['jpg_bytes'])
    img = np.asarray(Image.open(io.BytesIO(full_jpg)).convert('RGB'))
    uv = inst['uv_full'].numpy().astype(np.float32)
    z  = inst['z_cam'].numpy().astype(np.float32)
    intensity = inst['intensity'].numpy().astype(np.float32) if 'intensity' in inst else None
    K = inst['K_full'].numpy().astype(np.float32)
    # uv_full is in PARENT-image coords; the cached image jpg is the tile
    # (size IH×IW) starting at (tile_u0, tile_v0). Shift uv to tile coords.
    tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
    if tu0 or tv0:
        uv = uv - np.array([tu0, tv0], dtype=np.float32)
        K = K.copy()
        K[0, 2] -= tu0
        K[1, 2] -= tv0
    # filter to in-image
    H, W = img.shape[:2]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
    uv = uv[keep]; z = z[keep]
    if intensity is not None:
        intensity = intensity[keep]

    # Normalize intensity (kamikado /128 quant — same as dataset side).
    if intensity is not None:
        intensity = np.clip(intensity / 128.0, 0.0, 1.0).astype(np.float32)

    print(f'frame img shape={img.shape}  N pts={len(uv)}', flush=True)

    ba_cfg = dict(tile_size=512, model_input_size=c['img_size'],
                  max_pts_per_tile=256, min_pts_per_tile=8,
                  tile_stride=384)
    res = infer_tiles(model, img, uv, z, K, ba_cfg, torch.device('cuda'),
                      intensity=intensity)
    if res is None:
        print('infer_tiles returned None'); return
    uv_full, par, z_full = res
    print(f'after sweep: pool N={len(uv_full)} (sum across tiles)', flush=True)

    delta_6 = solve_dofs(uv_full, par, z_full, K,
                          _DOF_PRESETS['6dof_ext'], damping=1e-3)
    cov = solve_dofs._last_cov
    print('=== 6-DoF δ (full frame) ===', flush=True)
    for nm, v, s in zip(_DOF_PRESETS['6dof_ext'], delta_6, np.sqrt(np.diag(cov))):
        print(f'  {nm:10s}  δ={v:+.4f}  σ={s:.4f}', flush=True)
    print('OK', flush=True)


if __name__ == '__main__':
    main()
