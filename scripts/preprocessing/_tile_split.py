"""Helper: cut a single full-image inst into N sliding-window tiles.

Image size (IW × IH) varies per dataset / camera, so we compute tile origins
on the fly. Reusable from build_pandaset_full_v3 / build_nuscenes_v3 /
build_waymo_v3.

Tile-start computation lives in scripts.util.tile_layout so the cache
build and the online sliding inference (infer_tiles) split frames the
exact same way. Do NOT reintroduce a local copy — re-export only.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# scripts/util on path even when imported from the bare preprocessing package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.util.tile_layout import make_tile_starts  # noqa: F401, E402

DEFAULT_TILE   = 512
DEFAULT_STRIDE = 384   # 128 px overlap with tile=512
DEFAULT_PAD    = 64    # ~10 % of tile side
DEFAULT_QUAL   = 90


def cut_inst_to_tiles(*, jpg_bytes: bytes = None,
                       img_full_arr: np.ndarray = None,
                       IW: int, IH: int,
                       pts_vis: np.ndarray, uv_vis: np.ndarray,
                       z_vis: np.ndarray, is_obj_vis: np.ndarray,
                       intensity_vis: np.ndarray | None = None,
                       is_radar_vis: np.ndarray | None = None,
                       extra_per_point: dict[str, np.ndarray] | None = None,
                       common_inst: dict, tile_w: int = DEFAULT_TILE,
                       tile_h: int = DEFAULT_TILE,
                       stride: int = DEFAULT_STRIDE,
                       pad_px: int = DEFAULT_PAD,
                       y_start: int = 0,
                       y_end: int | None = None,
                       jpg_quality: int = DEFAULT_QUAL,
                       out_dir: Path = None,
                       gid_base: int = 0) -> list[str]:
    """Slice a parent image into tiles, save each as inst .pt.

    Pass EITHER `img_full_arr` (preferred, no decode loss) OR `jpg_bytes`
    (back-compat). When jpg_bytes is provided we still TJ-decode → encode,
    but builders that originally have a numpy frame should pass img_full_arr
    so the JPEG round-trip happens only once at tile-write time.

    `common_inst` carries the per-frame fields (cam_pos, R_gt, T_gt, K_full,
    cuboids, scene, cam, frame) that are inherited by every tile from this
    frame.

    Returns: list of saved tile filenames.
    """
    try:
        import turbojpeg as _tj
        _TJ = _tj.TurboJPEG()
        if img_full_arr is None:
            assert jpg_bytes is not None, 'pass img_full_arr or jpg_bytes'
            img_full_arr = np.asarray(_TJ.decode(jpg_bytes, pixel_format=_tj.TJPF_RGB))
        def _encode(arr):
            return _TJ.encode(arr, quality=jpg_quality, pixel_format=_tj.TJPF_RGB)
    except Exception:
        if img_full_arr is None:
            assert jpg_bytes is not None, 'pass img_full_arr or jpg_bytes'
            img_full_arr = np.asarray(Image.open(io.BytesIO(jpg_bytes)).convert('RGB'))
        def _encode(arr, q=jpg_quality):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format='JPEG', quality=q)
            return buf.getvalue()

    x_starts = make_tile_starts(IW, tile_w, stride)
    y_span = IH if y_end is None else min(int(y_end), IH)
    y_starts = make_tile_starts(y_span, tile_h, stride, axis_start=y_start)

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
            if intensity_vis is not None:
                inst['intensity'] = torch.from_numpy(intensity_vis[in_pad].astype(np.float32))
            if is_radar_vis is not None:
                inst['is_radar'] = torch.from_numpy(is_radar_vis[in_pad].astype(np.float32))
            if extra_per_point:
                for k, arr in extra_per_point.items():
                    inst[k] = torch.from_numpy(np.ascontiguousarray(arr[in_pad]))
            fname = f'{gid:08d}_t{tile_id}.pt'
            torch.save(inst, out_dir / fname)
            out_files.append(fname)
            tile_id += 1
    return out_files
