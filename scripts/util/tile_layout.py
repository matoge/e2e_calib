"""Tile layout — tile geometry for building tiled caches.

The cache build pipeline (scripts/preprocessing/_tile_split.py /
build_*_v3) routes through this function. (Online sliding inference,
scripts/ba/ba_multicam_corr.infer_tiles, used to share it; that inference
path was removed on 2026-10-07. Inference now cuts windows inside the
dataset, scripts/inference/infer_calib.py.)

Conventions:
  * `span`        full image side in pixels
  * `tile`        tile side in pixels (square tiles assumed elsewhere)
  * `stride`      step between tile starts
  * `axis_start`  optional offset (used to skip the y-axis sky band on
                  driving datasets)

A right-edge tile is added only when the natural-strided last start is
≥ stride/2 short of the edge — otherwise the natural last is "close
enough" and we shift it to the edge instead of adding a duplicate.
"""
from __future__ import annotations


def make_tile_starts(span: int, tile: int, stride: int,
                      axis_start: int = 0) -> list[int]:
    """Return tile origins along one axis covering [axis_start, span).

    Examples:
      span=1920, tile=512, stride=384, start=0   → [0, 384, 768, 1152, 1408] (5)
      span=1080, tile=512, stride=384, start=200 → [200, 568] (2, top sky skipped)
      span= 900, tile=512, stride=384, start=0   → [0, 388]   (2, NS-height short)
      span= 512, tile=512, stride=384, start=0   → [0]        (1, exactly fits)
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
