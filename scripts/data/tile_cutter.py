"""tile_cutter — CalibFrame → list of tile inst dicts.

The single source of truth for "how a frame is split into tiles" — used
both by the cache build pipeline (so LMDB tiles are produced from
adapters) and by online sliding inference (so CaaaS sees the same tile
geometry the model was trained on).

Tile origins come from scripts.util.tile_layout.make_tile_starts.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch
from PIL import Image

from scripts.data.calib_frame import CalibFrame
from scripts.util.tile_layout import make_tile_starts


DEFAULT_TILE_W = 512
DEFAULT_TILE_H = 512
DEFAULT_STRIDE = 384
DEFAULT_PAD_PX = 64
DEFAULT_JPG_QUALITY = 90


def _crop_image(img: np.ndarray, tu0: int, tv0: int, w: int, h: int,
                 jpg_quality: int = DEFAULT_JPG_QUALITY) -> tuple[bytes, int, int]:
    """Cut a (h, w) tile out of `img`, encode to JPEG, return (bytes, IW, IH).

    Prefers TurboJPEG (matches the build_*_v3 cache builders) so the JPEG
    bytes are byte-identical with what the legacy pipeline produced; falls
    back to PIL when TurboJPEG isn't installed.
    """
    crop = img[tv0:tv0 + h, tu0:tu0 + w]
    crop = np.ascontiguousarray(crop)
    try:
        import turbojpeg as _tj
        _TJ = _tj.TurboJPEG()
        buf = _TJ.encode(crop, quality=jpg_quality, pixel_format=_tj.TJPF_RGB)
        return bytes(buf), int(crop.shape[1]), int(crop.shape[0])
    except Exception:
        b = io.BytesIO()
        Image.fromarray(crop).save(b, format='JPEG', quality=jpg_quality)
        return b.getvalue(), int(crop.shape[1]), int(crop.shape[0])


def frame_to_tiles(cf: CalibFrame, *,
                    tile_w: int = DEFAULT_TILE_W,
                    tile_h: int = DEFAULT_TILE_H,
                    stride: int = DEFAULT_STRIDE,
                    pad_px: int = DEFAULT_PAD_PX,
                    y_start: int = 0,
                    jpg_quality: int = DEFAULT_JPG_QUALITY,
                    min_pts: int = 0) -> list[dict[str, Any]]:
    """Split one validated CalibFrame into a list of per-tile inst dicts.

    Each returned dict has the same shape that build_*_v3 currently writes
    to the LMDB cache (`pts`, `uv_full`, `z_cam`, `is_obj`, `intensity`,
    `K_full`, `distortion?`, `is_fisheye?`, `cam_pos`, `R_gt`, `T_gt`,
    `IH`, `IW`, `tile_u0`, `tile_v0`, `tile_id`, `jpg_bytes`, `scene`,
    `frame`, `cam`).

    `pad_px` widens the per-tile point selection so points just outside
    the tile boundary still influence the bucket / sub-grid selection at
    training time (matches build's behaviour).

    Tiles with fewer than `min_pts` LiDAR points inside (counted on the
    padded box, since that's what later builds the bucket) are dropped.
    """
    cf.validate()  # cheap — single source of truth for the adapter's contract
    H, W = cf.hw

    x_starts = make_tile_starts(W, tile_w, stride)
    y_starts = make_tile_starts(H, tile_h, stride, axis_start=y_start)

    out: list[dict[str, Any]] = []
    tile_id = 0
    for tv0 in y_starts:
        for tu0 in x_starts:
            in_pad = ((cf.uv_full[:, 0] >= tu0 - pad_px) &
                      (cf.uv_full[:, 0] <  tu0 + tile_w + pad_px) &
                      (cf.uv_full[:, 1] >= tv0 - pad_px) &
                      (cf.uv_full[:, 1] <  tv0 + tile_h + pad_px))
            if int(in_pad.sum()) < min_pts:
                tile_id += 1
                continue

            jpg_bytes, IW_t, IH_t = _crop_image(
                cf.img, tu0, tv0, tile_w, tile_h, jpg_quality)

            uv_t = cf.uv_full[in_pad]
            in_box = ((uv_t[:, 0] >= tu0) & (uv_t[:, 0] < tu0 + tile_w)
                      & (uv_t[:, 1] >= tv0) & (uv_t[:, 1] < tv0 + tile_h)
                      ).astype(np.float32)

            # Tile-side records keep the SAME parent K / dist (uv lives in
            # PARENT-image coords, exactly like build_*_v3); the (tile_u0,
            # tile_v0) origin is recorded so consumers can shift uv into
            # tile-local coords on the fly.
            inst: dict[str, Any] = dict(
                pts        = torch.from_numpy(cf.pts_cam[in_pad].copy()),
                uv_full    = torch.from_numpy(uv_t.copy()),
                z_cam      = torch.from_numpy(cf.z_cam[in_pad].copy()),
                is_obj     = torch.from_numpy(cf.is_obj[in_pad].copy()),
                in_box     = torch.from_numpy(in_box),
                intensity  = torch.from_numpy(cf.intensity[in_pad].copy()),
                K_full     = torch.from_numpy(cf.K.astype(np.float32)),
                cam_pos    = torch.zeros(3, dtype=torch.float32),
                R_gt       = torch.eye(3, dtype=torch.float32),
                T_gt       = torch.eye(4, dtype=torch.float32),
                jpg_bytes  = jpg_bytes,
                IH         = IH_t,
                IW         = IW_t,
                tile_u0    = int(tu0),
                tile_v0    = int(tv0),
                tile_id    = int(tile_id),
                scene      = cf.scene_id,
                frame      = int(cf.frame_id),
                cam        = cf.cam_id,
            )
            if cf.is_fisheye:
                inst['is_fisheye'] = True
                inst['distortion'] = torch.from_numpy(cf.dist.astype(np.float32))
            if cf.cuboids:
                inst['cuboids'] = cf.cuboids
            out.append(inst)
            tile_id += 1
    return out
