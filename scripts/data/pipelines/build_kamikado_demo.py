"""Smallest possible build pipeline — turn ONE kamikado scene into a list
of tile inst dicts via the new adapter + tile_cutter, then count them.

The point isn't to write LMDB yet (that's Phase 1 step 4), it's to get a
DAG live in ClearML's /pipelines page so the rest of the project can
copy-paste this skeleton and add real components.

Run remotely (so it appears in ClearML):
    python -m scripts.data.pipelines.build_kamikado_demo \
        --scene /home/hfunaya/cache/kamikado/scenes/points_ip664_D_..._3000_3020 \
        --max-frames 3 \
        --queue dgx2_cpu

Run locally for smoke (no ClearML enqueue):
    python -m scripts.data.pipelines.build_kamikado_demo --scene ... --local
"""
# NOTE: do NOT add `from __future__ import annotations` here — ClearML
# pickles the @component function bodies into a tmp file with imports
# prepended above the source, which makes __future__ no longer the
# first non-comment line and triggers SyntaxError.

import argparse
import sys
from pathlib import Path

# IMPORTANT: the @component decorator pickles the function body and ships
# it to the worker. ClearML expects a clean import; do NOT do path
# manipulation at module top-level because the decorator runs on import.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


from clearml import PipelineDecorator  # noqa: E402


@PipelineDecorator.component(
    return_values=['n_frames'],
    cache=True,
    packages=['numpy', 'pillow', 'scipy', 'torch'],
)
def list_scene_frames(scene_dir: str, max_frames: int = -1) -> int:
    from pathlib import Path
    from scripts.data.adapters.kamikado import list_frames
    fs = list_frames(Path(scene_dir))
    if max_frames > 0:
        fs = fs[:max_frames]
    print(f'[list_scene_frames] {scene_dir}: {len(fs)} frames')
    return len(fs)


@PipelineDecorator.component(
    return_values=['n_tiles_total'],
    cache=True,
    packages=['numpy', 'pillow', 'scipy', 'torch'],
)
def adapt_and_tile(scene_dir: str, n_frames: int) -> int:
    """Per-frame adapter → tile_cutter, return total tile count."""
    from pathlib import Path
    from scripts.data.adapters.kamikado import load_frame, list_frames, TILE_LAYOUT
    from scripts.data.tile_cutter import frame_to_tiles
    fs = list_frames(Path(scene_dir))[:n_frames]
    total = 0
    for fr in fs:
        cf = load_frame(Path(scene_dir), fr)
        tiles = frame_to_tiles(cf, **TILE_LAYOUT)
        total += len(tiles)
        print(f'  frame {fr}: {len(tiles)} tiles, scene={cf.scene_id}')
    print(f'[adapt_and_tile] {scene_dir}: {total} total tiles across {len(fs)} frames')
    return total


@PipelineDecorator.component(
    return_values=['n_viz'],
    cache=False,  # viz output is the whole point; never skip
    packages=['numpy', 'pillow', 'scipy', 'torch', 'matplotlib'],
)
def viz_debug_samples(scene_dir: str, n_frames: int,
                       tiles_per_frame: int = 2) -> int:
    """Render GT projection on a few tiles and upload to ClearML as
    Debug samples so the pipeline run is inspectable from the UI."""
    import io as _io
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL
    from clearml import Task

    from scripts.data.adapters.kamikado import load_frame, list_frames
    from scripts.data.tile_cutter import frame_to_tiles

    task = Task.current_task()
    logger = task.get_logger() if task is not None else None

    fs = list_frames(Path(scene_dir))[:n_frames]
    n_uploaded = 0
    for fr in fs:
        cf = load_frame(Path(scene_dir), fr)
        tiles = frame_to_tiles(cf)
        if not tiles:
            print(f'  frame {fr}: no tiles, skip')
            continue
        for ti, t in enumerate(tiles[:tiles_per_frame]):
            tile_img = np.asarray(_PIL.open(_io.BytesIO(t['jpg_bytes']))
                                   .convert('RGB'))
            # uv_full is parent-image coords; tile-local = uv - (tu0, tv0).
            uv = t['uv_full'].numpy().astype(np.float32) \
                  - np.array([t['tile_u0'], t['tile_v0']], dtype=np.float32)
            H, W = tile_img.shape[:2]
            keep = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H))
            uv = uv[keep]
            z  = t['z_cam'].numpy().astype(np.float32)[keep]

            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=120)
            ax.imshow(tile_img)
            sc = ax.scatter(uv[:, 0], uv[:, 1], c=z, s=4, cmap='viridis',
                              vmin=0, vmax=80)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
            ax.set_title(f'frame={fr} tile_id={t["tile_id"]} '
                          f'origin=({t["tile_u0"]},{t["tile_v0"]}) '
                          f'N_pts={len(uv)}', fontsize=8)
            plt.colorbar(sc, ax=ax, label='z [m]', fraction=0.04)
            buf = _io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            arr = np.asarray(_PIL.open(buf).convert('RGB'))
            if logger is not None:
                logger.report_image(
                    title='gt_projection',
                    series=f'frame_{fr}',
                    iteration=t['tile_id'],
                    image=arr)
            n_uploaded += 1
            print(f'  viz frame={fr} tile_id={t["tile_id"]}: '
                   f'{len(uv)} pts drawn')
    print(f'[viz_debug_samples] uploaded {n_uploaded} debug images')
    return n_uploaded


@PipelineDecorator.component(
    return_values=['lmdb_path', 'n_tiles'],
    cache=False,
    packages=['numpy', 'pillow', 'scipy', 'torch', 'lmdb'],
)
def write_lmdb(scene_dir: str, n_frames: int, out_dir: str,
                val_every: int = 5) -> tuple:
    """Adapt → tile_cut → write LMDB cache for the demo scene.

    val_every: every N-th frame goes into the val split (rest into train).
    """
    from pathlib import Path
    from scripts.data.adapters.kamikado import load_frame, list_frames, TILE_LAYOUT
    from scripts.data.tile_cutter import frame_to_tiles
    from scripts.data.lmdb_writer import write_lmdb_cache

    fs = list_frames(Path(scene_dir))[:n_frames]

    def _gen():
        for fi, fr in enumerate(fs):
            cf = load_frame(Path(scene_dir), fr)
            split = 'val' if (fi % val_every == 0) else 'train'
            for t in frame_to_tiles(cf, **TILE_LAYOUT):
                yield split, t

    res = write_lmdb_cache(out_dir, _gen(), cam='fcm', is_fisheye=True,
                            overwrite=True)
    print(f'[write_lmdb] {res}')
    return res['lmdb_path'], res['n_train'] + res['n_val']


@PipelineDecorator.component(
    return_values=['ok'],
    cache=False,
    packages=['numpy', 'torch', 'lmdb'],
)
def golden_compare(new_lmdb_path: str,
                    ref_cache: str = '/cache/kamikado_v3_tiled') -> bool:
    """Compare new LMDB tile contents to ref cache by (scene, frame, tile_id).

    The fname (gid) scheme differs between the legacy build script (sequential
    gid) and the new lmdb_writer (md5(scene)[:4]+frame), so we can't byte-
    compare on key. Instead, walk both caches, parse the header.scene /
    header.frame / header.tile_u0 / header.tile_v0 — the tuple
    (scene, frame, tile_u0, tile_v0) uniquely identifies a tile. Then
    compare the raw bodies (offsets+payload, the part AFTER the header
    pickle) which IS gid-independent.
    """
    import lmdb, pickle, struct
    from pathlib import Path
    HDR_LEN_FMT = '<Q'
    HDR_LEN_SIZE = struct.calcsize(HDR_LEN_FMT)

    def _index(env_path):
        env = lmdb.open(str(env_path), readonly=True, lock=False, subdir=True)
        idx: dict[tuple, bytes] = {}
        with env.begin() as t:
            for k, v in t.cursor():
                if k.startswith(b'__cubs__/'):
                    continue
                hdr_len = struct.unpack(HDR_LEN_FMT, v[:HDR_LEN_SIZE])[0]
                hdr = pickle.loads(v[HDR_LEN_SIZE:HDR_LEN_SIZE + hdr_len])
                key = (str(hdr.get('scene', '')), int(hdr.get('frame', -1)),
                       int(hdr.get('tile_u0', 0)), int(hdr.get('tile_v0', 0)))
                # Body bytes (offsets + payload). gid-independent.
                body = v[HDR_LEN_SIZE + hdr_len:]
                idx[key] = body
        env.close()
        return idx

    new_idx = _index(Path(new_lmdb_path))
    ref_idx = _index(Path(ref_cache) / 'data.lmdb')
    overlap = set(new_idx) & set(ref_idx)
    print(f'[golden_compare] new={len(new_idx)} ref={len(ref_idx)} overlap={len(overlap)}')
    n_match = 0; n_diff = 0
    for key in overlap:
        if new_idx[key] == ref_idx[key]:
            n_match += 1
        else:
            n_diff += 1
            if n_diff <= 3:
                print(f'  DIFF key={key} '
                      f'len_new={len(new_idx[key])} len_ref={len(ref_idx[key])}')
    print(f'[golden_compare] match={n_match} diff={n_diff} '
          f'(of {len(overlap)} overlapping tiles)')
    return n_diff == 0 and n_match > 0


@PipelineDecorator.component(
    return_values=['dataset_id', 'version'],
    cache=False,
    packages=['clearml'],
)
def register_dataset(lmdb_out_dir: str, n_tiles: int,
                       dataset_name: str = 'kamikado_v3_tiled',
                       dataset_project: str = 'e2e_calib/cache',
                       parent_datasets: list = None) -> tuple:
    """Snapshot the new LMDB cache as a versioned ClearML Dataset.

    parent_datasets: list of previous dataset_id to inherit from. If None,
    auto-fetches the latest dataset of the same name as the parent.
    """
    from clearml import Dataset

    parents = parent_datasets
    if parents is None:
        try:
            prev = Dataset.get(dataset_name=dataset_name,
                                dataset_project=dataset_project)
            parents = [prev.id]
        except Exception:
            parents = None

    ds = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        parent_datasets=parents,
    )
    ds.add_files(lmdb_out_dir, verbose=False)
    ds.upload(verbose=False, show_progress=False)
    ds.finalize()
    print(f'[register_dataset] {dataset_name} v={ds.tags or "?"} '
          f'id={ds.id} parents={parents} n_tiles={n_tiles}')
    return ds.id, str(ds.tags)


@PipelineDecorator.pipeline(
    name='build_kamikado_demo',
    project='e2e_calib/data',
    version='0.3.0',
)
def pipeline(scene_dir: str, max_frames: int, lmdb_out_dir: str,
              ref_cache: str = '/cache/kamikado_v3_tiled',
              register: bool = False):
    n = list_scene_frames(scene_dir, max_frames)
    total = adapt_and_tile(scene_dir, n)
    n_viz = viz_debug_samples(scene_dir, n, tiles_per_frame=2)
    lmdb_path, n_tiles = write_lmdb(scene_dir, n, lmdb_out_dir)
    ok = golden_compare(lmdb_path, ref_cache)
    if register:
        ds_id, ver = register_dataset(lmdb_out_dir, n_tiles)
        print(f'pipeline done: tiles={total} viz={n_viz} '
              f'lmdb={lmdb_path} (n={n_tiles}) golden_ok={ok} '
              f'dataset_id={ds_id} version={ver}')
    else:
        print(f'pipeline done: tiles={total} viz={n_viz} '
              f'lmdb={lmdb_path} (n={n_tiles}) golden_ok={ok}  '
              '(--register skipped)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--max-frames', type=int, default=3)
    ap.add_argument('--queue', default='dgx2',
                    help='ClearML execution queue for component tasks')
    ap.add_argument('--local', action='store_true',
                    help='run pipeline locally instead of enqueueing on ClearML')
    ap.add_argument('--lmdb-out',
                    default='/tmp/kamikado_demo_lmdb',
                    help='where to write the demo LMDB (must be writable)')
    ap.add_argument('--ref-cache', default='/cache/kamikado_v3_tiled',
                    help='production cache to byte-compare against')
    ap.add_argument('--register', action='store_true',
                    help='also register the new LMDB as a versioned ClearML Dataset')
    args = ap.parse_args()

    if args.local:
        PipelineDecorator.run_locally()
    else:
        PipelineDecorator.set_default_execution_queue(args.queue)
    pipeline(args.scene, args.max_frames, args.lmdb_out, args.ref_cache,
             register=args.register)


if __name__ == '__main__':
    main()
