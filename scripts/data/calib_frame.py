"""CalibFrame — the dataset-agnostic per-frame payload.

Every dataset (kamikado / woven / waymo / zod / nuscenes / pandaset / …)
ships pixel data with its own quirks: intensity scale, fisheye coeffs,
camera-shutter time correction, vehicle-frame conventions, etc.
Adapters in scripts/data/adapters/<name>.py convert raw inputs into a
single normalised in-memory shape — `CalibFrame` — and run
`validate()` before returning. Anything downstream (tile cutter, training
crop, BA, CaaaS demo) only ever sees CalibFrame, so dataset-specific
drift cannot leak past the adapter boundary.

Validation guarantees (enforced by `CalibFrame.validate()`):

  1. intensity ∈ [0, 1]                — adapters MUST normalise per sensor
  2. img is uint8, shape (H, W, 3)
  3. K is float, shape (3, 3), positive fx/fy
  4. dist is None XOR a 4-vector (Kannala k1..k4) when is_fisheye=True
  5. pts_cam shape (N, 3), z > 0 for all stored points
  6. z_cam == pts_cam[:, 2]            — bit-equal
  7. uv_full shape (N, 2), in image bounds
  8. project(pts_cam, K [, dist]) ≈ uv_full to within 1e-3 px
     (build-time projection invariant — catches K-vs-dist mismatches)
  9. intensity / is_obj shape (N,), float32
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_REPROJ_TOL_PX = 1e-3
_INTENSITY_TOL = 1e-6


@dataclass
class CalibFrame:
    img: np.ndarray             # (H, W, 3) uint8 RGB
    K: np.ndarray               # (3, 3) float64
    is_fisheye: bool
    dist: np.ndarray | None     # (4,) Kannala k1..k4, or None for pinhole
    pts_cam: np.ndarray         # (N, 3) float32 — camera-frame XYZ
    intensity: np.ndarray       # (N,) float32 in [0, 1]
    uv_full: np.ndarray         # (N, 2) float32 — projection in image px
    z_cam: np.ndarray           # (N,) float32 — pts_cam[:, 2]
    is_obj: np.ndarray          # (N,) float32 — cuboid membership flag
    cuboids: list[dict] = field(default_factory=list)
    scene_id: str = ''
    frame_id: int = -1
    cam_id: str = ''
    extras: dict[str, Any] = field(default_factory=dict)

    # ─────────────────────────── validation ───────────────────────────
    def validate(self, *, strict: bool = True) -> None:
        """Strict invariant check. Raises on any violation when `strict`,
        otherwise prints warnings and returns."""
        errors: list[str] = []

        # 1. img
        if self.img.dtype != np.uint8 or self.img.ndim != 3 or self.img.shape[-1] != 3:
            errors.append(f'img must be (H,W,3) uint8, got '
                           f'{self.img.shape}/{self.img.dtype}')
        H, W = self.img.shape[:2]

        # 2. K
        if self.K.shape != (3, 3):
            errors.append(f'K must be (3,3), got {self.K.shape}')
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            errors.append(f'K has non-positive fx/fy: '
                           f'fx={self.K[0,0]} fy={self.K[1,1]}')

        # 3. dist
        if self.is_fisheye:
            if self.dist is None or self.dist.shape != (4,):
                errors.append(f'is_fisheye=True requires dist (4,), '
                               f'got {None if self.dist is None else self.dist.shape}')
        else:
            if self.dist is not None:
                errors.append('is_fisheye=False but dist is provided')

        # 4. pts_cam / intensity / uv / z shapes
        N = self.pts_cam.shape[0]
        if self.pts_cam.shape != (N, 3):
            errors.append(f'pts_cam must be (N,3), got {self.pts_cam.shape}')
        if self.intensity.shape != (N,):
            errors.append(f'intensity shape {self.intensity.shape} != ({N},)')
        if self.uv_full.shape != (N, 2):
            errors.append(f'uv_full shape {self.uv_full.shape} != ({N},2)')
        if self.z_cam.shape != (N,):
            errors.append(f'z_cam shape {self.z_cam.shape} != ({N},)')
        if self.is_obj.shape != (N,):
            errors.append(f'is_obj shape {self.is_obj.shape} != ({N},)')

        # 5. intensity must be float32 AND already normalised to [0,1]. The
        #    whole point of the new pipeline is that adapters apply the
        #    per-sensor normalisation once at data-creation time so the
        #    cache stores values that consumers can use directly without
        #    re-normalising. (This is the contract the dataset reader will
        #    move to once we've rebuilt the caches.)
        if self.intensity.dtype != np.float32:
            errors.append(f'intensity dtype must be float32, got '
                           f'{self.intensity.dtype}')
        if self.intensity.size:
            i_min, i_max = float(self.intensity.min()), float(self.intensity.max())
            if i_min < -_INTENSITY_TOL or i_max > 1.0 + _INTENSITY_TOL:
                errors.append(f'intensity not in [0,1]: '
                               f'min={i_min:.4f} max={i_max:.4f} '
                               '(adapter must normalise per-sensor)')

        # 6. z_cam == pts_cam[:, 2]
        if N > 0 and not np.array_equal(self.z_cam, self.pts_cam[:, 2]):
            d = float(np.abs(self.z_cam - self.pts_cam[:, 2]).max())
            errors.append(f'z_cam != pts_cam[:,2] (max diff {d:.3e})')

        # 7. z > 0
        if N > 0 and float(self.z_cam.min()) <= 0:
            errors.append(f'pts_cam includes z<=0 (min z={self.z_cam.min():.3f}) — '
                           'adapter must drop behind-camera points before validate()')

        # 8. uv in image bounds
        if N > 0:
            u_min, u_max = float(self.uv_full[:, 0].min()), float(self.uv_full[:, 0].max())
            v_min, v_max = float(self.uv_full[:, 1].min()), float(self.uv_full[:, 1].max())
            if u_min < 0 or u_max >= W or v_min < 0 or v_max >= H:
                errors.append(f'uv_full out of image bounds: '
                               f'u∈[{u_min:.1f},{u_max:.1f}] v∈[{v_min:.1f},{v_max:.1f}] '
                               f'image {W}x{H}')

        # 9. reprojection consistency (uv_full == project(pts_cam, K, dist))
        if N > 0 and not errors:  # skip if shapes already wrong
            from scripts.util.projection import project_pinhole, project_kannala
            if self.is_fisheye:
                uv_recomp = project_kannala(self.pts_cam,
                                              self.K.astype(np.float64),
                                              self.dist.astype(np.float64))
            else:
                uv_recomp = project_pinhole(self.pts_cam,
                                              self.K.astype(np.float64))
            d = float(np.abs(self.uv_full - uv_recomp).max())
            if d > _REPROJ_TOL_PX:
                errors.append(f'uv_full mismatches project(pts_cam,K[,dist]): '
                               f'max diff {d:.3e} px (tol {_REPROJ_TOL_PX:.0e})')

        if errors:
            msg = 'CalibFrame.validate() failed:\n  - ' + '\n  - '.join(errors)
            if strict:
                raise ValueError(msg)
            print('WARN:', msg)

    # ───────────────────────────── helpers ────────────────────────────
    @property
    def n_pts(self) -> int:
        return self.pts_cam.shape[0]

    @property
    def hw(self) -> tuple[int, int]:
        return self.img.shape[0], self.img.shape[1]

    def __repr__(self) -> str:
        return (f'CalibFrame(scene={self.scene_id!r} frame={self.frame_id} '
                f'cam={self.cam_id!r} {self.hw[1]}x{self.hw[0]}px '
                f'fisheye={self.is_fisheye} N_pts={self.n_pts})')
