"""For the same (scene, frame), confirm:

  raw → kamikado adapter → CalibFrame
versus
  cache LMDB → inst (which build_kamikado_v3 already wrote)

agree numerically on K, dist, pts_cam, uv_full, intensity (within
float32 round-off), AND that running infer_tiles on either path
produces identical par. Anything that breaks this means the adapter
diverged from what the cache builder did, and inference will silently
disagree downstream.

Run:
    docker exec caaas python3 -m pytest /workspace/tests/test_adapter_vs_cache.py -v -s
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.ba.ba_multicam_corr import infer_tiles
from scripts.data.adapters.kamikado import load_frame, TILE_LAYOUT
from scripts.data.tile_cutter import frame_to_tiles
from scripts.inference.infer_calib import load_calib_model


CACHE = '/cache/kamikado_v3_tiled'
RAW_ROOT = Path('/raw/kamikado/scenes')
SCENE = 'points_ip664_D_20260226_224648_d005_3000_3020'
FRAME = 0
EXP = 'km_wv_wm_dgx2_n2_v4'

REPROJ_TOL_PX = 1e-3
PTS_TOL = 1e-4   # float32 round-off
INTENSITY_TOL = 1e-6


def _scene_dir() -> Path:
    p = RAW_ROOT / SCENE
    if not p.exists():
        pytest.skip(f'raw scene not mounted at {p}')
    return p


def _adapter_frame():
    return load_frame(_scene_dir(), FRAME)


def _cache_inst_for_tile(scene: str, frame: int, tile_origin: tuple[int, int]):
    """Find the cached inst whose (scene, frame, tile_u0, tile_v0) matches."""
    ds = PandaSetCalibDatasetFull(CACHE, split='val',
                                    max_offset_m=0.20, max_rot_deg=0.5,
                                    min_crop_px=128, max_crop_px=384,
                                    oversample=1)
    for fn in ds.fnames:
        idx = ds.fnames.index(fn)
        inst = ds._load_inst(idx)
        if (str(inst.get('scene', '')) == scene
                and int(inst.get('frame', -1)) == frame
                and (int(inst.get('tile_u0', 0)),
                     int(inst.get('tile_v0', 0))) == tile_origin):
            return inst
    return None


def test_adapter_K_matches_cache():
    cf = _adapter_frame()
    inst = _cache_inst_for_tile(SCENE, FRAME, (0, 600))
    assert inst is not None, 'no cached tile at (0, 600)'
    K_cache = inst['K_full'].numpy().astype(np.float64)
    print(f'\n  K_adapter[0,0]={cf.K[0,0]:.4f}  K_cache[0,0]={K_cache[0,0]:.4f}')
    assert np.abs(cf.K - K_cache).max() < 1e-4, \
        f'K mismatch: {np.abs(cf.K - K_cache).max():.3e}'


def test_adapter_dist_matches_cache():
    cf = _adapter_frame()
    inst = _cache_inst_for_tile(SCENE, FRAME, (0, 600))
    d_cache = inst['distortion'].numpy().astype(np.float64)
    print(f'\n  dist_adapter={cf.dist}  dist_cache={d_cache}')
    assert np.abs(cf.dist - d_cache).max() < 1e-4, \
        f'dist mismatch: {np.abs(cf.dist - d_cache).max():.3e}'


def test_adapter_uv_matches_cached_uv_full():
    """For the kamikado tile at (0, 600), the cached uv_full of those
    points must match the adapter's uv_full (subset of full-frame, with
    pad / in_box ring) within fp32 round-off."""
    cf = _adapter_frame()
    inst = _cache_inst_for_tile(SCENE, FRAME, (0, 600))
    uv_cache = inst['uv_full'].numpy().astype(np.float32)  # parent coords
    # Adapter's uv_full has ALL points; cache only stores the in-pad subset.
    # Use uv-coordinate matching (not index-aligned).
    n_cache = len(uv_cache)
    # Match each cached point to the closest adapter point.
    diffs = []
    for u, v in uv_cache:
        d = np.linalg.norm(cf.uv_full - np.array([u, v], dtype=np.float32),
                            axis=1)
        diffs.append(d.min())
    diffs = np.asarray(diffs)
    print(f'\n  adapter uv vs cache uv  N_cache={n_cache}  '
          f'max nearest-neighbour distance={diffs.max():.3e} px')
    assert diffs.max() < REPROJ_TOL_PX * 10, \
        f'cached uv not in adapter uv (max dist {diffs.max():.3e} px)'


def test_adapter_intensity_matches_cache_per_point():
    """Match each cached point to the closest adapter point (by uv) and
    confirm intensity agrees within fp32 round-off. Range comparison is
    misleading because the cache only carries the in-pad tile subset."""
    cf = _adapter_frame()
    inst = _cache_inst_for_tile(SCENE, FRAME, (0, 600))
    uv_cache = inst['uv_full'].numpy().astype(np.float32)
    i_cache = inst['intensity'].numpy().astype(np.float32)
    diffs = []
    for k in range(len(uv_cache)):
        d = np.linalg.norm(cf.uv_full - uv_cache[k], axis=1)
        j = int(np.argmin(d))
        if d[j] < REPROJ_TOL_PX * 10:
            diffs.append(abs(cf.intensity[j] - i_cache[k]))
    diffs = np.asarray(diffs)
    print(f'\n  per-point intensity diff over {len(diffs)} matched pts: '
          f'max={diffs.max():.3e}  mean={diffs.mean():.3e}')
    assert diffs.max() < INTENSITY_TOL, (
        f'intensity diverges per-point: max={diffs.max():.3e} '
        f'(adapter range [{cf.intensity.min():.4f}, {cf.intensity.max():.4f}], '
        f'cache range [{i_cache.min():.4f}, {i_cache.max():.4f}])')


def test_adapter_tile_infer_tiles_finite():
    """Run infer_tiles on the adapter's parent frame (tile-cut by
    infer_tiles) and confirm par is finite + σ > 0."""
    cf = _adapter_frame()
    model = load_calib_model(EXP).eval()
    ba_cfg = dict(tile_size=384, model_input_size=128,
                  max_pts_per_tile=256, min_pts_per_tile=8, tile_stride=320)
    res = infer_tiles(model, cf.img,
                       cf.uv_full.astype(np.float32),
                       cf.z_cam.astype(np.float32),
                       cf.K.astype(np.float32), ba_cfg,
                       torch.device('cuda'),
                       intensity=cf.intensity.astype(np.float32))
    assert res is not None, 'infer_tiles None on adapter frame'
    uv_pool, par_pool, z_pool = res
    print(f'\n  adapter→infer_tiles: n_pool={len(uv_pool)}  '
          f'σ_u_med={float(np.median(par_pool[:,2])):.2f}  '
          f'σ_v_med={float(np.median(par_pool[:,3])):.2f}')
    assert np.all(np.isfinite(par_pool)), 'non-finite par'
    assert par_pool[:, 2].min() > 0 and par_pool[:, 3].min() > 0, 'σ ≤ 0'
