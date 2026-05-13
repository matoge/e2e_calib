"""End-to-end calibration inference: image + LiDAR → (correction, residuals).

Wraps the canonical sliding-tile inference + Gauss-Newton BA solver used
in scripts/ba/ba_multicam_corr.py so the serving layer doesn't reinvent
the pipeline. Single instance is GPU-resident and thread-safe under
torch.no_grad() (each request gets its own tensor allocations; the
model itself is read-only after load).
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image

# Reuse the canonical pipeline functions (no fork).
from scripts.ba.ba_multicam_corr import (
    infer_tiles, build_model as _ba_build_model,
    resolve_dof_list, solve_dofs, delta_to_dict, DOF_JAC,
)

from .model_loader import LoadedModel
from .schemas import (
    CalibRequest, CalibResponse, CalibSequenceRequest, CalibSequenceResponse,
    Correction,
)


DEFAULT_BA = dict(
    tile_size=256, model_input_size=128, tile_stride=128,
    max_pts_per_tile=256, min_pts_per_tile=16,
    dof=['omega_x', 'omega_y', 'tx', 'ty', 'df_common'],
    damping=1e-3,
)


@dataclass
class PipelineResult:
    correction: dict       # DoF name → value (degrees / meters / px)
    n_pts: int
    n_tiles: int
    elapsed_ms: float


class CalibInferencePipeline:
    """Thread-safe inference wrapper around a single loaded checkpoint."""

    def __init__(self, loaded: LoadedModel, device: str | None = None,
                 ba_defaults: dict | None = None):
        self.device = torch.device(device or
                                     ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.loaded = loaded
        self.ba = dict(ba_defaults or DEFAULT_BA)
        self.model = self._build_model(loaded)

    # --------------------------------------------------------------- model
    def _build_model(self, loaded: LoadedModel):
        cfg = dict(loaded.model_cfg)
        cfg['ckpt'] = str(loaded.path)
        # ba_multicam_corr.build_model handles state_dict loading + .eval().
        m = _ba_build_model(cfg, self.device)
        return m

    @property
    def version(self) -> str:
        return self.loaded.version

    # ------------------------------------------------------------ inference
    def infer(self, req: CalibRequest) -> PipelineResult:
        t0 = time.time()
        img = self._decode_image(req)
        pts_w = np.asarray(req.pts_world, dtype=np.float64).reshape(-1, 3)
        K     = np.asarray(req.K,         dtype=np.float64).reshape(3, 3)
        T_cw  = np.asarray(req.T_cw,      dtype=np.float64).reshape(4, 4)

        # World → cam, then project. Mirror scripts.ba.ba_multicam_corr.load_frame.
        homo = np.column_stack([pts_w, np.ones(len(pts_w))])
        pts_c = (T_cw @ homo.T).T[:, :3]
        z = pts_c[:, 2].astype(np.float32)
        uv = (K @ pts_c.T).T[:, :2] / np.maximum(z[:, None], 1e-6)
        uv = uv.astype(np.float32)

        K_f32 = K.astype(np.float32)
        ba_cfg = dict(self.ba)
        if req.dof:
            ba_cfg['dof'] = list(req.dof)

        # Optional intensity passthrough. The current infer_tiles signature
        # doesn't read intensity (model takes uvd-only or [uvd, i] inside);
        # leave as-is here — the model's PointMLP consumes [..., 4] when
        # use_intensity=True via dist_uvd shape, which infer_tiles builds
        # from `z` + uv. Adding intensity end-to-end will require a small
        # extension to infer_tiles' bucket builder.

        res = infer_tiles(self.model, img, uv, z, K_f32, ba_cfg, self.device)
        if res is None:
            return PipelineResult(correction={}, n_pts=0, n_tiles=0,
                                   elapsed_ms=(time.time() - t0) * 1e3)
        UV, PAR, Z = res

        dof_names = resolve_dof_list(ba_cfg)
        delta = solve_dofs(UV, PAR, Z, K_f32, dof_names, damping=ba_cfg['damping'])
        dof_vals = delta_to_dict(delta, dof_names)
        n_tiles = self._estimate_n_tiles(img, ba_cfg)
        return PipelineResult(
            correction=dof_vals,
            n_pts=int(len(UV)),
            n_tiles=n_tiles,
            elapsed_ms=(time.time() - t0) * 1e3,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _decode_image(req: CalibRequest) -> np.ndarray:
        try:
            import turbojpeg as _tj
            _TJ = _tj.TurboJPEG()
            data = base64.b64decode(req.image_b64)
            try:
                arr = np.asarray(_TJ.decode(data, pixel_format=_tj.TJPF_RGB))
                return arr
            except Exception:
                pass  # not jpeg; fall through to PIL
        except ImportError:
            pass
        data = base64.b64decode(req.image_b64)
        return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))

    def _estimate_n_tiles(self, img: np.ndarray, ba_cfg: dict) -> int:
        H, W = img.shape[:2]
        T = ba_cfg['tile_size']; S = ba_cfg.get('tile_stride', 128)
        if T >= W or T >= H:
            return 1
        return ((W - T) // S + 1) * ((H - T) // S + 1)


    # -------------------------------------------------------- sequence mode
    def infer_sequence(self, req: CalibSequenceRequest) -> "SequenceResult":
        """Pool multiple frames' per-pt residuals into ONE BA solve.

        Same math as scripts/ba/ba_multicam_corr.py's multi-scene loop:
        each frame's `infer_tiles` returns (uv, par, z); we concatenate
        across frames and solve once. Per-frame timings are summed, not
        averaged, so the reported elapsed_ms is the wall the server spent.
        """
        t0 = time.time()
        ba_cfg = dict(self.ba)
        if req.dof:
            ba_cfg['dof'] = list(req.dof)

        pooled_uv: list[np.ndarray] = []
        pooled_par: list[np.ndarray] = []
        pooled_z:  list[np.ndarray] = []
        n_tiles_total = 0
        K_ref: np.ndarray | None = None

        for fr in req.frames:
            img = self._decode_image(fr)
            pts_w = np.asarray(fr.pts_world, dtype=np.float64).reshape(-1, 3)
            K     = np.asarray(fr.K,         dtype=np.float64).reshape(3, 3)
            T_cw  = np.asarray(fr.T_cw,      dtype=np.float64).reshape(4, 4)
            if K_ref is None:
                K_ref = K
            homo = np.column_stack([pts_w, np.ones(len(pts_w))])
            pts_c = (T_cw @ homo.T).T[:, :3]
            z = pts_c[:, 2].astype(np.float32)
            uv = (K @ pts_c.T).T[:, :2] / np.maximum(z[:, None], 1e-6)
            uv = uv.astype(np.float32)
            K_f32 = K.astype(np.float32)
            res = infer_tiles(self.model, img, uv, z, K_f32, ba_cfg, self.device)
            if res is None:
                continue
            UV, PAR, Z = res
            pooled_uv.append(UV); pooled_par.append(PAR); pooled_z.append(Z)
            n_tiles_total += self._estimate_n_tiles(img, ba_cfg)

        if not pooled_uv:
            return SequenceResult(correction={}, n_frames=len(req.frames),
                                   n_pts=0, n_tiles_total=n_tiles_total,
                                   elapsed_ms=(time.time() - t0) * 1e3)

        UV_all  = np.concatenate(pooled_uv,  axis=0)
        PAR_all = np.concatenate(pooled_par, axis=0)
        Z_all   = np.concatenate(pooled_z,   axis=0)
        K_f32   = K_ref.astype(np.float32)

        dof_names = resolve_dof_list(ba_cfg)
        delta = solve_dofs(UV_all, PAR_all, Z_all, K_f32, dof_names,
                            damping=ba_cfg['damping'])
        dof_vals = delta_to_dict(delta, dof_names)
        return SequenceResult(
            correction=dof_vals, n_frames=len(req.frames),
            n_pts=int(len(UV_all)), n_tiles_total=n_tiles_total,
            elapsed_ms=(time.time() - t0) * 1e3,
        )


@dataclass
class SequenceResult:
    correction: dict
    n_frames: int
    n_pts: int
    n_tiles_total: int
    elapsed_ms: float


def correction_response(result: PipelineResult,
                         model_version: str) -> CalibResponse:
    """Format the pipeline output for the public API. Unit/key mapping
    is centralized here to keep the wire schema stable across DoF
    permutations on the BA side."""
    c = Correction(
        omega_x_deg = result.correction.get('omega_x'),
        omega_y_deg = result.correction.get('omega_y'),
        omega_z_deg = result.correction.get('omega_z'),
        tx_m        = result.correction.get('tx'),
        ty_m        = result.correction.get('ty'),
        tz_m        = result.correction.get('tz'),
        dfx_px      = result.correction.get('dfx'),
        dfy_px      = result.correction.get('dfy'),
        df_common_px= result.correction.get('df_common'),
    )
    return CalibResponse(
        correction=c, n_pts_pooled=result.n_pts, n_tiles=result.n_tiles,
        elapsed_ms=round(result.elapsed_ms, 2),
        model_version=model_version,
    )


def sequence_response(result: SequenceResult,
                       model_version: str) -> CalibSequenceResponse:
    c = Correction(
        omega_x_deg = result.correction.get('omega_x'),
        omega_y_deg = result.correction.get('omega_y'),
        omega_z_deg = result.correction.get('omega_z'),
        tx_m        = result.correction.get('tx'),
        ty_m        = result.correction.get('ty'),
        tz_m        = result.correction.get('tz'),
        dfx_px      = result.correction.get('dfx'),
        dfy_px      = result.correction.get('dfy'),
        df_common_px= result.correction.get('df_common'),
    )
    return CalibSequenceResponse(
        correction=c, n_frames=result.n_frames,
        n_pts_pooled=result.n_pts, n_tiles_total=result.n_tiles_total,
        elapsed_ms=round(result.elapsed_ms, 2),
        model_version=model_version,
    )
