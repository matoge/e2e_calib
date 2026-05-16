"""CalibData — thin hierarchical view over the V3 LMDB tile caches.

The build_*_v3 pipelines flatten everything into one big
{key → blob} dict per cache. The hierarchy that's actually meaningful
to humans (and to BA) — scene → frame → tile → crop — only survives
inside the filename:
    <scene_short><frame:04d>_t<tile_id>.pt
This wrapper rebuilds that hierarchy on top of the existing cache so
callers can ask for "all tiles of frame F in scene S" or "the next
N frames in scene S" without touching LMDB / fname parsing.

Conventions match every build_*_v3 currently in tree:
    fname  ::= <scene_short>(<frame:04d>)?_t<tile_id>.pt
    scene_short = first 8 hex chars of md5(scene_name)
    frame  is the integer in `inst['frame']`
    tile_id is sequential within a (scene, frame)

The wrapper is intentionally thin — anything that already works on
PandaSetCalibDatasetFull (`ds._load_inst(idx)`, `ds[idx]`, etc.)
keeps working unchanged. CalibData just adds new entry points:

    cd = CalibData('/cache/kamikado_v3_tiled', exp='km_wv_wm_dgx2_n2_img128_v2')
    cd.scenes()                       # → list[str]   ('kamikado/<scene>')
    cd.frames(scene)                  # → list[int]   sorted
    cd.tiles(scene, frame)            # → list[int]   tile_id
    cd.tile(scene, frame, tile_id)    # → dict (the inst as _load_inst returns)
    cd.frame_tiles(scene, frame)      # → list[dict]  every tile of that frame
    cd.crop(idx, *, seed=None)        # → dataset sample tuple (training-style)

`crop` is a passthrough to PandaSetCalibDatasetFull.__getitem__ so the
training augmentation pipeline stays the single source of truth for
the lowest level of the hierarchy.

Multi-cache use:
    cd = CalibData(['/cache/kamikado_v3_tiled', '/cache/woven_v3_tile'])
    cd.scenes()  # → ['kamikado/<...>', 'woven/<seq>', ...]

Each cache becomes a top-level "dataset" tag (the cache dir's basename
stripped of the trailing _v3_* suffix).
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# NOTE: keep imports lightweight — CalibData is meant to be importable
# from the CaaaS Flask app, midtrain_vis, BA scripts, and notebooks
# without dragging in CUDA / torch unless the user actually asks for a
# tile/crop. Heavy imports happen inside method bodies.


_TILE_FNAME_RE = re.compile(r'^(?P<scene_id>[0-9a-f]{6,16})'
                            r'(?P<frame>\d{4})'
                            r'_t(?P<tile>\d+)\.pt$')


def _parse_fname(fn: str) -> tuple[str, int, int] | None:
    """Filename → (scene_short, frame_idx, tile_id) or None on mismatch."""
    m = _TILE_FNAME_RE.match(fn)
    if m is None:
        return None
    return m['scene_id'], int(m['frame']), int(m['tile'])


def _cache_tag(cache_path: str | Path) -> str:
    """Cache dir → short tag ('kamikado_v3_tiled' → 'kamikado')."""
    name = Path(cache_path).name
    return re.sub(r'_v3.*$', '', name)


class CalibData:
    """Hierarchical view over one or more V3 tile caches.

    See module docstring for the hierarchy and entry points. The cache(s)
    must already exist; this class never writes.
    """

    def __init__(self, caches: str | Path | Iterable[str | Path],
                 exp: str | None = None):
        self.caches = ([Path(caches)] if isinstance(caches, (str, Path))
                        else [Path(c) for c in caches])
        self.exp = exp
        self._ds_per_cache: dict[Path, object] = {}
        # (cache_idx, scene_short) → scene_name placeholder; we don't have
        # a reverse map without scanning meta.pt, but consumers that only
        # need to group tiles by frame don't need the full scene string.
        self._index: dict[str, dict[int, dict[int, tuple[Path, str]]]] = {}
        for c in self.caches:
            self._build_index(c)

    # ───────────────────────── index build ─────────────────────────
    def _build_index(self, cache: Path):
        """Scan filenames once at construction so all later queries are O(1)."""
        tag = _cache_tag(cache)
        per_scene: dict[str, dict[int, dict[int, tuple[Path, str]]]] = \
            defaultdict(lambda: defaultdict(dict))
        # Build scans inst/<fname>.pt OR LMDB keys; both are the same fname set.
        # Prefer reading the LMDB key list (cheaper than 100k+ stat calls).
        meta = cache / 'meta.pt'
        if not meta.is_file():
            raise FileNotFoundError(f'no meta.pt at {cache} — not a V3 cache')
        import torch
        m = torch.load(meta, weights_only=False, map_location='cpu')
        # The build scripts store {'train': [...], 'val': [...]} in meta.pt.
        # We index everything; split filtering is the caller's job.
        all_fnames: list[str] = []
        for split in ('train', 'val'):
            if split in m:
                all_fnames.extend(m[split])
        for fn in all_fnames:
            parsed = _parse_fname(Path(fn).name)
            if parsed is None:
                continue
            scene_short, frame_idx, tile_id = parsed
            scene_key = f'{tag}/{scene_short}'
            per_scene[scene_key][frame_idx][tile_id] = (cache, fn)
        self._index.update(per_scene)

    # ─────────────────────── hierarchy queries ─────────────────────
    def scenes(self) -> list[str]:
        return sorted(self._index.keys())

    def frames(self, scene: str) -> list[int]:
        return sorted(self._index[scene].keys())

    def tiles(self, scene: str, frame: int) -> list[int]:
        return sorted(self._index[scene][frame].keys())

    # ────────────────────────── data load ──────────────────────────
    def _ds_for_cache(self, cache: Path):
        """Materialize the heavy PandaSetCalibDatasetFull lazily, per cache.

        Imported lazily so just listing scenes() doesn't need torch.
        """
        if cache not in self._ds_per_cache:
            from scripts.inference.infer_pipeline import make_ds
            ds, c = make_ds(self.exp, str(cache), split='val', oversample=1)
            self._ds_per_cache[cache] = (ds, c)
        return self._ds_per_cache[cache]

    def tile(self, scene: str, frame: int, tile_id: int) -> dict:
        """Load one tile's inst dict (same shape as ds._load_inst(idx))."""
        cache, fn = self._index[scene][frame][tile_id]
        ds, _ = self._ds_for_cache(cache)
        # PandaSetCalibDatasetFull indexes by self.fnames position, not by
        # filename → look up the position. fnames is small enough (~10k)
        # that a dict comprehension on first miss is fine; cache it.
        if not hasattr(ds, '_fname_to_idx'):
            ds._fname_to_idx = {f: i for i, f in enumerate(ds.fnames)}
        return ds._load_inst(ds._fname_to_idx[fn])

    def frame_tiles(self, scene: str, frame: int) -> list[dict]:
        """Every tile of a (scene, frame) as a list of inst dicts."""
        return [self.tile(scene, frame, t) for t in self.tiles(scene, frame)]

    def crop(self, scene: str, frame: int, tile_id: int, *,
              seed: int | None = None):
        """Training-style sample (perturbed crop) for one tile.

        Returns the same tuple shape as PandaSetCalibDatasetFull[idx].
        """
        cache, fn = self._index[scene][frame][tile_id]
        ds, _ = self._ds_for_cache(cache)
        if not hasattr(ds, '_fname_to_idx'):
            ds._fname_to_idx = {f: i for i, f in enumerate(ds.fnames)}
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
        return ds[ds._fname_to_idx[fn]]
