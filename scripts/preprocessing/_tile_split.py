"""Helper: cut a single full-image inst into N sliding-window tiles.

Image size (IW × IH) varies per dataset / camera, so we compute tile origins
on the fly. Reusable from build_pandaset_full_v3 / build_nuscenes_v3 /
build_waymo_v3.
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DEFAULT_TILE   = 512
DEFAULT_STRIDE = 384   # 128 px overlap with tile=512
DEFAULT_PAD    = 64    # ~10 % of tile side
DEFAULT_QUAL   = 90


def make_tile_starts(span: int, tile: int, stride: int, axis_start: int = 0) -> list[int]:
    """Return tile origins along an axis covering [axis_start, span).

    Adds a right-edge tile only if the natural-strided last start is at least
    `stride/2` short of the edge — otherwise the natural last is too close
    to the edge to warrant a duplicate, and we just shift it to the edge.
    Examples:
      span=1920, tile=512, stride=384, start=0   → [0, 384, 768, 1152, 1408] (5)
      span=1080, tile=512, stride=384, start=200 → [200, 568] (2, top sky skipped)
      span= 900, tile=512, stride=384, start=0   → [0, 388]   (2, NS-height short)
    """
    if span < tile + axis_start:
        return [max(0, axis_start)]
    starts = list(range(axis_start, span - tile + 1, stride))
    if not starts:
        return [span - tile]
    edge = span - tile
    if edge - starts[-1] >= stride // 2:
        starts.append(edge)
    elif starts[-1] != edge:
        starts[-1] = edge
    return starts


def cut_inst_to_tiles(*, jpg_bytes: bytes, IW: int, IH: int,
                       pts_vis: np.ndarray, uv_vis: np.ndarray,
                       z_vis: np.ndarray, is_obj_vis: np.ndarray,
                       common_inst: dict, tile_w: int = DEFAULT_TILE,
                       tile_h: int = DEFAULT_TILE,
                       stride: int = DEFAULT_STRIDE,
                       pad_px: int = DEFAULT_PAD,
                       y_start: int = 0,
                       jpg_quality: int = DEFAULT_QUAL,
                       out_dir: Path = None,
                       gid_base: int = 0) -> list[str]:
    """Decode the parent JPEG once, slice into tiles, save each as inst .pt.

    `common_inst` carries the per-frame fields (cam_pos, R_gt, T_gt, K_full,
    cuboids, scene, cam, frame) that are inherited by every tile from this
    frame.

    Returns: list of saved tile filenames.
    """
    try:
        import turbojpeg as _tj
        _TJ = _tj.TurboJPEG()
        img_full_arr = np.asarray(_TJ.decode(jpg_bytes, pixel_format=_tj.TJPF_RGB))
        def _encode(arr):
            return _TJ.encode(arr, quality=jpg_quality, pixel_format=_tj.TJPF_RGB)
    except Exception:
        img_full_arr = np.asarray(Image.open(io.BytesIO(jpg_bytes)).convert('RGB'))
        def _encode(arr, q=jpg_quality):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format='JPEG', quality=q)
            return buf.getvalue()

    x_starts = make_tile_starts(IW, tile_w, stride)
    y_starts = make_tile_starts(IH, tile_h, stride, axis_start=y_start)

    out_files = []
    tile_id = 0
    gid = gid_base
    for ty in y_starts:
        for tx in x_starts:
            in_pad = ((uv_vis[:, 0] >= tx - pad_px) & (uv_vis[:, 0] < tx + tile_w + pad_px) &
                      (uv_vis[:, 1] >= ty - pad_px) & (uv_vis[:, 1] < ty + tile_h + pad_px))
            pts_t    = pts_vis[in_pad]
            uv_t     = uv_vis[in_pad]
            z_t      = z_vis[in_pad]
            is_obj_t = is_obj_vis[in_pad]
            in_box_t = ((uv_t[:, 0] >= tx) & (uv_t[:, 0] < tx + tile_w) &
                        (uv_t[:, 1] >= ty) & (uv_t[:, 1] < ty + tile_h)).astype(np.float32)

            tile_arr = img_full_arr[ty:ty + tile_h, tx:tx + tile_w].copy()
            tile_jpg = _encode(tile_arr)

            inst = dict(common_inst)
            inst.update(dict(
                jpg_bytes = tile_jpg,
                IH        = int(tile_h),
                IW        = int(tile_w),
                tile_u0   = int(tx),
                tile_v0   = int(ty),
                tile_id   = int(tile_id),
                pts       = torch.from_numpy(pts_t),
                uv_full   = torch.from_numpy(uv_t),
                z_cam     = torch.from_numpy(z_t),
                is_obj    = torch.from_numpy(is_obj_t),
                in_box    = torch.from_numpy(in_box_t),
            ))
            fname = f'{gid:08d}_t{tile_id}.pt'
            torch.save(inst, out_dir / fname)
            out_files.append(fname)
            tile_id += 1
    return out_files
