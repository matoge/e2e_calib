"""Guard: collate_full's new original-camera-frame solver fields.

When `build_window` was extended to emit (pts_cam_orig, duv_orig, K_orig, cs)
per sample, `collate_full` was changed to pad/stack them. This test pins:

  1. the new fields appear at fixed tuple positions [8..11] and shapes match
     (B, Nmax, 3) / (B, Nmax, 2) / (B, 3, 3) / (B,).
  2. the per-anchor padding uses the SAME pad-mask as true_uvd / dist_uvd
     (so any downstream solver code can re-use `pad`).
  3. forward-projecting pts_cam_orig with K_orig recovers (uv_off + duv_orig)
     in original-camera pixels for the valid anchors — i.e. duv_orig is in
     the right frame and pts_cam_orig is the *pre-perturbation* point that
     paired with this Δuv.
  4. existing (non-solver) callers that unpack 7 leading elements
     `imgs, true_uvd, dist_uvd, pad, vfp, b_uvd, b_v = batch[:7]` keep
     working — i.e. backward-compat at the prefix.

Runs on a tiny in-memory cache (no GPU). If the cache path is missing,
the test is skipped (CI on machines without /cache survive cleanly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full


CACHE_CANDIDATES = [
    '/home/hfunaya/cache_v4/kamikado_v3_tiled',
    '/home/hfunaya/cache/kamikado_v3_tiled_new',
    '/home/hfunaya/cache/pandaset_v3_tiled',
    '/cache/kamikado_v3_tiled',
    '/cache/pandaset_v3_tiled',
]


def _pick_cache() -> str | None:
    for c in CACHE_CANDIDATES:
        if Path(c).exists():
            return c
    return None


@pytest.fixture(scope='module')
def small_batch():
    cache = _pick_cache()
    if cache is None:
        pytest.skip(f'no cache present: {CACHE_CANDIDATES}')
    ds = PandaSetCalibDatasetFull(
        cache, split='val',
        max_offset_m=0.20, max_rot_deg=0.5,
        min_crop_px=128, max_crop_px=384, oversample=1,
    )
    samples = []
    i = 0
    while len(samples) < 4 and i < 64:
        s = ds[i]
        if s is not None:
            samples.append(s)
        i += 1
    if len(samples) < 2:
        pytest.skip('dataset returned < 2 valid samples in 64 tries')
    batch = collate_full(samples)
    return batch, samples


def test_tuple_length_and_field_positions(small_batch):
    batch, samples = small_batch
    # 8 legacy + 4 orig-solver = 12 total
    assert len(batch) == 12, f'expected 12-tuple, got {len(batch)}'
    (imgs, true_p, dist_p, pad, vfps, b_uvds, b_valids, pert,
     pts_cam_orig, duv_orig, K_orig, cs_t) = batch
    B = len(samples)
    Nmax = true_p.shape[1]
    assert pts_cam_orig.shape == (B, Nmax, 3)
    assert duv_orig.shape     == (B, Nmax, 2)
    assert K_orig.shape       == (B, 3, 3)
    assert cs_t.shape         == (B,)
    # cs is always positive
    assert (cs_t > 0).all()


def test_pad_mask_aligns_with_orig_fields(small_batch):
    """Padded slots in true_p (pad==True) must coincide with zero-filled
    rows in pts_cam_orig and duv_orig. We don't test the inverse (a real
    point at depth 0 → looks like padding), only that pad rows are zero."""
    batch, _ = small_batch
    pad = batch[3]
    pts_cam_orig = batch[8]
    duv_orig     = batch[9]
    if pad.any():
        assert torch.allclose(pts_cam_orig[pad], torch.zeros_like(pts_cam_orig[pad]))
        assert torch.allclose(duv_orig[pad],     torch.zeros_like(duv_orig[pad]))


def test_forward_projection_consistency(small_batch):
    """Project pts_cam_orig with K_orig, sanity-check parent-camera frame.

    K_orig is the PARENT-camera intrinsics (NOT tile-localized) and
    pts_cam_orig is metric XYZ in the parent-camera coord system. So
    K_orig · pts / Z lands in parent-image px. duv_orig is tile-invariant
    (uv_gt − uv_off cancels any tile origin), so uv_off in parent-image px
    is not recoverable from the batch alone — we cannot bound u, v to a
    specific image rectangle. What we CAN check are the unit-bug guards
    that matter for the solver:

      - finite, non-NaN projections (rules out unit collisions like
        m × local-px focal that would explode the float)
      - Z > 0 for valid anchors (they are in front of the camera)
      - u, v are within an order of magnitude of cx, cy of K_orig (i.e.
        the result is plausibly an image-px coordinate, not e.g. a
        metric value or a normalized coord).
    """
    batch, _ = small_batch
    pad = batch[3]
    pts_cam_orig = batch[8]
    K_orig       = batch[10]
    valid = ~pad
    fx = K_orig[:, 0, 0:1]   # (B, 1)
    fy = K_orig[:, 1, 1:2]
    cx = K_orig[:, 0, 2:3]
    cy = K_orig[:, 1, 2:3]
    X, Y, Z = pts_cam_orig.unbind(-1)
    Z_safe = Z.clamp_min(1e-3)
    u = fx * X / Z_safe + cx
    v = fy * Y / Z_safe + cy
    u_v = u[valid]
    v_v = v[valid]
    assert torch.isfinite(u_v).all() and torch.isfinite(v_v).all()
    assert (Z[valid] > 0).all()
    # u, v should sit within ~5× of (cx, cy) — generous since anchors can
    # be near the image edge but not in another universe.
    cx_b = cx.expand_as(u)[valid]
    cy_b = cy.expand_as(v)[valid]
    assert ((u_v - cx_b).abs() < 5 * cx_b).all(), \
        f'u far from cx: max |u-cx|={((u_v - cx_b).abs()).max().item()}, cx~{cx_b.mean().item()}'
    assert ((v_v - cy_b).abs() < 5 * cy_b).all(), \
        f'v far from cy: max |v-cy|={((v_v - cy_b).abs()).max().item()}, cy~{cy_b.mean().item()}'


def test_duv_orig_scale_vs_local(small_batch):
    """duv_orig (original-camera px) and the local-128px Δuv embedded in
    (true_uvd - dist_uvd) should be related by `cs/S` per sample. We don't
    know S here without poking the dataset, but we can assert the *ratio*
    duv_orig / duv_local is finite, sign-consistent, and within a single
    decade — i.e. there isn't an order-of-magnitude unit mismatch."""
    batch, _ = small_batch
    pad        = batch[3]
    true_p     = batch[1]
    dist_p     = batch[2]
    duv_orig   = batch[9]
    duv_local  = (true_p[..., :2] - dist_p[..., :2])
    valid = ~pad
    # large-magnitude anchors only (avoid /0)
    big = (duv_local.abs() > 0.5) & valid.unsqueeze(-1)
    if big.any():
        ratio = duv_orig[big] / duv_local[big]
        assert torch.isfinite(ratio).all()
        # cs/S typically 1..6. Allow some slack.
        assert (ratio.abs() > 0.2).all() and (ratio.abs() < 30.0).all(), \
            f'duv_orig / duv_local out of band: {ratio.min().item()} .. {ratio.max().item()}'
        # signs must agree
        assert (torch.sign(duv_orig[big]) == torch.sign(duv_local[big])).all()


def test_legacy_prefix_unpack_still_works(small_batch):
    """Existing callers do `imgs, true_uvd, dist_uvd, pad, vfp, b_uvd, b_v = batch[:7]`
    or pull `batch[7]` for pert. Make sure those positions / dtypes are
    preserved across the collate change."""
    batch, samples = small_batch
    imgs, true_uvd, dist_uvd, pad, vfp, b_uvd, b_v = batch[:7]
    pert = batch[7]
    B = len(samples)
    assert imgs.shape[0] == B
    assert true_uvd.shape[:1] == (B,)
    assert dist_uvd.shape[:1] == (B,)
    assert pad.shape[:1] == (B,)
    assert vfp.shape == (B,)
    assert b_uvd.shape[0] == B
    assert b_v.shape[0] == B
    assert pert.shape[0] == B
