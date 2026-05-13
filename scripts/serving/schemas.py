"""Pydantic I/O schemas for the calibration server.

Wire format mirrors the canonical BA inputs (image + lidar points in
world frame + cam intrinsics + cam extrinsic). All arrays travel as
flattened lists of floats; clients decode lengths from the `shape`
fields so we never have to commit to a specific binary layout (e.g.
msgpack / arrow) in the public API. For high-QPS clients we can later
add an optional `multipart/octet-stream` body that skips the JSON
encoding entirely — the pipeline class is agnostic.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CalibRequest(BaseModel):
    """One frame's worth of input. Same convention as load_frame() and the
    BA tool — points are in WORLD frame, extrinsic is T_cw (world→cam)."""

    image_b64: str = Field(
        ...,
        description="Base64-encoded JPEG or PNG of the camera frame. "
                    "Decoded via TurboJPEG/PIL on the server side.",
    )
    pts_world: List[float] = Field(
        ...,
        description="Flat (N*3,) list of lidar points in WORLD coordinates "
                    "(meters). Reshaped to (N, 3) server-side.",
    )
    intensity: Optional[List[float]] = Field(
        None,
        description="Per-point intensity, shape (N,). Required when the "
                    "model was trained with use_intensity=True; ignored for "
                    "legacy pixel-only models.",
    )
    K: List[float] = Field(
        ...,
        description="Flat row-major 3x3 intrinsic. fx, 0, cx, 0, fy, cy, 0,0,1.",
        min_length=9, max_length=9,
    )
    T_cw: List[float] = Field(
        ...,
        description="Flat row-major 4x4 world→cam transform. "
                    "Convention matches T_cw = inv(T_wc) used in the BA tool.",
        min_length=16, max_length=16,
    )
    image_size: Optional[List[int]] = Field(
        None,
        description="Optional (W, H) pair. If omitted, decoded from the "
                    "image bytes.",
        min_length=2, max_length=2,
    )
    dof: Optional[List[str]] = Field(
        None,
        description="DoF subset for the BA solve (overrides server default). "
                    "Valid names: omega_x, omega_y, omega_z, tx, ty, tz, "
                    "dfx, dfy, df_common.",
    )


class Correction(BaseModel):
    """Solved per-cam correction vector. Units: angles=degrees, t=meters,
    df=pixels. Empty when no DoF set was solved."""
    omega_x_deg: Optional[float] = None
    omega_y_deg: Optional[float] = None
    omega_z_deg: Optional[float] = None
    tx_m:        Optional[float] = None
    ty_m:        Optional[float] = None
    tz_m:        Optional[float] = None
    dfx_px:      Optional[float] = None
    dfy_px:      Optional[float] = None
    df_common_px: Optional[float] = None


class CalibResponse(BaseModel):
    correction: Correction
    n_pts_pooled: int = Field(
        ...,
        description="Number of LiDAR pivots that passed BA filters (tile-in, "
                    "z>0.5, in_image). Sanity check that the request actually "
                    "had enough overlap to solve.",
    )
    n_tiles: int = Field(..., description="Sliding-window tiles processed.")
    elapsed_ms: float = Field(..., description="Server-side inference wall.")
    model_version: str = Field(..., description="Active model checkpoint tag.")


class HealthResponse(BaseModel):
    status: str
    model_version: str
    cuda_available: bool


class CalibSequenceRequest(BaseModel):
    """A list of frames from the SAME camera (single cam id, identical K).
    The server runs per-frame inference and pools every pivot into ONE
    global BA solve — the same multi-frame averaging used in
    scripts/ba/ba_multicam_corr.py with N scenes × N frames. K and dof
    are taken from the first frame; per-frame T_cw / image / lidar pts
    can differ.
    """
    frames: List[CalibRequest] = Field(
        ..., min_length=1,
        description="Frames pooled into one BA. Order doesn't affect the result.",
    )
    dof: Optional[List[str]] = Field(
        None,
        description="Override DoF for the global solve (per-frame `dof` ignored).",
    )


class CalibSequenceResponse(BaseModel):
    correction: Correction
    n_frames: int
    n_pts_pooled: int
    n_tiles_total: int
    elapsed_ms: float
    model_version: str
