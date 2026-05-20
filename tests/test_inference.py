"""Reproducible inference regression tests.

Two checks per (ckpt, cache):
  1. test_val_nll_matches_log
       Replay the trainer's val pass on the FIRST 800 cache instances
       through DataLoader + collate_full + model forward + gaussian2d_nll.
       Asserts |offline_nll - train.log's saved best val_nll| ≤ tol so
       any silent regression to inference-time pre-processing trips the
       test before it reaches users.

  2. test_infer_tiles_smoke
       Run scripts.ba.ba_multicam_corr.infer_tiles + solve_dofs on the
       first cache tile. Asserts no NaN, σ > 0, δ has the right shape.

Run:
    docker exec caaas python3 -m pytest /workspace/tests/test_inference.py -v -s

Markers / parametrisation: only ckpts whose `experiments/<name>/best_model.pt`
is present on this host are exercised (so the same file works on DGX1/DGX2/
laptops without forcing extra downloads).
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_cov import gaussian2d_nll
from scripts.ba.ba_multicam_corr import (
    _DOF_PRESETS, infer_tiles, solve_dofs,
)
from scripts.inference.infer_calib import load_calib_model


CACHE = '/cache/kamikado_v3_tiled'
VAL_SIZE = 800
NLL_TOL = 0.20            # acceptable drift between offline replay and log

CANDIDATE_CKPTS = [
    'km_wv_wm_dgx2_n2_v4',
    'km_wv_wm_dgx1_n4_v4_resume',
    'km_wv_wm_dgx2_n2_img128_v2',
]


def _available_ckpts() -> list[str]:
    out = []
    for name in CANDIDATE_CKPTS:
        if (REPO_ROOT / 'experiments' / name / 'best_model.pt').is_file():
            out.append(name)
    return out


def _trainlog_best_val(exp: str) -> float | None:
    log = REPO_ROOT / 'experiments' / exp / 'train.log'
    if not log.is_file():
        return None
    txt = log.read_text(errors='ignore')
    m = re.findall(r'saved \(val_nll=(\d+\.\d+)\)', txt)
    return float(m[-1]) if m else None


def _replay_val(exp: str, cache: str, val_size: int = VAL_SIZE
                 ) -> tuple[float, float]:
    """Drive the trainer's exact val pass; returns (mean nll, mean px err)."""
    ds = PandaSetCalibDatasetFull(
        cache, split='val',
        max_offset_m=0.20, max_rot_deg=0.5,
        min_crop_px=128, max_crop_px=384, oversample=1)
    val_subset = Subset(ds, list(range(val_size)))
    loader = DataLoader(val_subset, batch_size=64, num_workers=4,
                         collate_fn=collate_full, shuffle=False)
    model = load_calib_model(exp).eval()
    device = torch.device('cuda')
    nll_s = mse_s = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
            imgs = imgs.to(device).float().div_(255.0)
            true_uvd = true_uvd.to(device); dist_uvd = dist_uvd.to(device)
            pad_mask = pad_mask.to(device); vfp = vfp.to(device)
            b_uvd = b_uvd.to(device); b_v = b_v.to(device)
            gt = true_uvd[..., :2] - dist_uvd[..., :2]
            use_intensity = bool(getattr(model, 'use_intensity', False))
            point_in = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
                          if use_intensity else dist_uvd[..., :3])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                params = model(imgs, point_in, key_padding_mask=pad_mask,
                                vfp=vfp, bucket_uvd=b_uvd, bucket_valid=b_v)
            valid = ~pad_mask
            if valid.any():
                nll_s += gaussian2d_nll(params[valid], gt[valid]).item()
                err = (params[valid][..., :2].float()
                        - gt[valid]).norm(dim=-1).mean().item()
                mse_s += err
                n += 1
    return nll_s / max(n, 1), mse_s / max(n, 1)


def _smoke_infer_tiles(exp: str, cache: str) -> dict:
    ds = PandaSetCalibDatasetFull(
        cache, split='val',
        max_offset_m=0.20, max_rot_deg=0.5,
        min_crop_px=128, max_crop_px=384, oversample=1)
    inst = ds._load_inst(0)
    img = np.asarray(Image.open(io.BytesIO(bytes(inst['jpg_bytes'])))
                       .convert('RGB'))
    uv = inst['uv_full'].numpy().astype(np.float32)
    z  = inst['z_cam'].numpy().astype(np.float32)
    intensity = inst['intensity'].numpy().astype(np.float32)
    K = inst['K_full'].numpy().astype(np.float32)
    tu0, tv0 = int(inst.get('tile_u0', 0)), int(inst.get('tile_v0', 0))
    uv = uv - np.array([tu0, tv0], dtype=np.float32)
    K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
    H, W = img.shape[:2]
    keep = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0))
    uv = uv[keep]; z = z[keep]
    intensity = np.clip(intensity[keep], 0, 1).astype(np.float32)

    model = load_calib_model(exp).eval()
    ba_cfg = dict(tile_size=384, model_input_size=128,
                  max_pts_per_tile=256, min_pts_per_tile=8, tile_stride=320)
    res = infer_tiles(model, img, uv, z, K, ba_cfg, torch.device('cuda'),
                       intensity=intensity)
    assert res is not None, 'infer_tiles returned None'
    uv_pool, par_pool, z_pool = res
    delta = solve_dofs(uv_pool, par_pool, z_pool, K,
                        _DOF_PRESETS['6dof_ext'], damping=1e-3)
    return dict(uv_pool=uv_pool, par_pool=par_pool, delta=delta)


# ───────────────────────── tests ─────────────────────────

@pytest.mark.parametrize('exp', _available_ckpts())
def test_val_nll_matches_log(exp):
    expected = _trainlog_best_val(exp)
    nll, mse = _replay_val(exp, CACHE)
    print(f'\n  [{exp}] offline_nll={nll:+.4f}  mse={mse:.3f}  '
          f'log_best={expected}')
    if expected is None:
        pytest.skip(f'no saved best in train.log for {exp}')
    assert abs(nll - expected) <= NLL_TOL, (
        f'offline_nll={nll:.4f} vs log_best={expected:.4f} '
        f'(|diff|={abs(nll-expected):.4f} > tol {NLL_TOL})')


@pytest.mark.parametrize('exp', _available_ckpts())
def test_infer_tiles_smoke(exp):
    out = _smoke_infer_tiles(exp, CACHE)
    par = out['par_pool']; delta = out['delta']
    assert np.all(np.isfinite(par)), 'non-finite par from infer_tiles'
    sx, sy = par[:, 2], par[:, 3]
    assert sx.min() > 0 and sy.min() > 0, f'σ ≤ 0 (sx_min={sx.min()})'
    assert np.all(np.isfinite(delta)), 'non-finite δ from solve_dofs'
    assert delta.shape == (6,), f'δ shape {delta.shape} ≠ (6,)'
    print(f'\n  [{exp}] n_pool={len(out["uv_pool"])}  '
          f'σ_med=({np.median(sx):.1f}, {np.median(sy):.1f})  '
          f'δ={[round(x, 3) for x in delta]}')
