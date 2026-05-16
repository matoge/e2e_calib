"""build_caches — single Pipeline that rebuilds kamikado / woven / waymo
caches end-to-end via the new CalibFrame + tile_cutter + lmdb_writer path.

Each dataset is a thin component because it picks a different adapter.
The shape (load_frame → frame_to_tiles → write_lmdb_cache → register
ClearML Dataset) is identical across all three.

Run on the DGX2 controller:
    python -m scripts.data.pipelines.build_caches \\
        --datasets kamikado,waymo \\
        --kamikado-root /home/hfunaya/cache/kamikado/scenes \\
        --waymo-dir     /mnt/.../waymo/training \\
        --woven-root    /home/hfunaya/woven_sequence \\
        --out-root      /raid/home/hfunaya/cache_v4 \\
        --max-frames-per-scene 10 \\
        --queue dgx2 --local

Notes:
- Adapter raw paths must be visible to the worker. With --local that's
  the controller (this DGX2). Without --local you need raw mounted on
  whichever queue's worker picks the task up.
- --max-frames-per-scene caps the per-scene frame count so smoke runs
  finish in minutes; pass -1 for full builds.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from clearml import PipelineDecorator  # noqa: E402


# ─── Per-dataset adapter components. Each yields (split, tile_inst). ────

@PipelineDecorator.component(
    return_values=['out_dir'],
    cache=False,
    packages=['numpy', 'pillow', 'scipy', 'torch', 'lmdb'],
)
def build_kamikado_cache(raw_root: str, out_dir: str,
                          max_frames_per_scene: int, val_every: int) -> str:
    from pathlib import Path
    from scripts.data.adapters.kamikado import (load_frame, list_frames,
                                                  TILE_LAYOUT)
    from scripts.data.tile_cutter import frame_to_tiles
    from scripts.data.lmdb_writer import write_lmdb_cache

    raw = Path(raw_root)
    scenes = sorted(p for p in raw.iterdir() if p.is_dir())
    print(f'[kamikado] {len(scenes)} scenes under {raw}')

    def _gen():
        n_global = 0
        for scene in scenes:
            fs = list_frames(scene)
            if max_frames_per_scene > 0:
                fs = fs[:max_frames_per_scene]
            for fi, fr in enumerate(fs):
                cf = load_frame(scene, fr)
                split = 'val' if (n_global % val_every == 0) else 'train'
                for t in frame_to_tiles(cf, **TILE_LAYOUT, min_pts=0):
                    yield split, t
                n_global += 1
            print(f'  scene {scene.name}: {len(fs)} frames consumed')

    res = write_lmdb_cache(out_dir, _gen(), cam='fcm', is_fisheye=True,
                            overwrite=True, map_size_gb=80)
    print(f'[build_kamikado_cache] {res}')
    return out_dir


@PipelineDecorator.component(
    return_values=['out_dir'],
    cache=False,
    packages=['numpy', 'pillow', 'scipy', 'torch', 'lmdb'],
)
def build_woven_cache(raw_root: str, out_dir: str,
                       max_frames_per_scene: int, val_every: int) -> str:
    from pathlib import Path
    from scripts.data.adapters.woven import (load_frame, list_frames,
                                                TILE_LAYOUT)
    from scripts.data.tile_cutter import frame_to_tiles
    from scripts.data.lmdb_writer import write_lmdb_cache

    raw = Path(raw_root)
    # Assume raw structure: raw_root/<dataset>/<date>/<seq>/.
    seqs = sorted({p.parent for p in raw.rglob('tss4_fcm') if p.is_dir()})
    print(f'[woven] {len(seqs)} sequences under {raw}')

    def _gen():
        n_global = 0
        for seq in seqs:
            try:
                fs = list_frames(seq)
            except Exception as e:
                print(f'  WARN seq {seq.name}: {e}'); continue
            if max_frames_per_scene > 0:
                fs = fs[:max_frames_per_scene]
            for fr in fs:
                try:
                    cf = load_frame(seq, fr)
                except Exception as e:
                    print(f'  WARN frame {fr}: {e}'); continue
                split = 'val' if (n_global % val_every == 0) else 'train'
                for t in frame_to_tiles(cf, **TILE_LAYOUT, min_pts=0):
                    yield split, t
                n_global += 1
            print(f'  seq {seq.name}: consumed')

    res = write_lmdb_cache(out_dir, _gen(), cam='tss4_fcm', is_fisheye=True,
                            overwrite=True, map_size_gb=120)
    print(f'[build_woven_cache] {res}')
    return out_dir


@PipelineDecorator.component(
    return_values=['out_dir'],
    cache=False,
    packages=['numpy', 'pillow', 'scipy', 'torch', 'lmdb', 'pandas',
              'pyarrow'],
)
def build_waymo_cache(out_dir: str, max_segs: int,
                       max_frames_per_seg: int, val_every: int) -> str:
    from scripts.data.adapters.waymo import (load_frame, list_frames,
                                                list_segs, TILE_LAYOUT)
    from scripts.data.tile_cutter import frame_to_tiles
    from scripts.data.lmdb_writer import write_lmdb_cache

    segs = list_segs()
    if max_segs > 0:
        segs = segs[:max_segs]
    print(f'[waymo] {len(segs)} segments')

    def _gen():
        n_global = 0
        for seg in segs:
            fs = list_frames(seg)
            if max_frames_per_seg > 0:
                fs = fs[:max_frames_per_seg]
            for ts in fs:
                # Front camera only for now (cam_id=1).
                try:
                    cf = load_frame(seg, ts, cam_id=1)
                except Exception as e:
                    print(f'  WARN seg={seg} ts={ts}: {e}'); continue
                split = 'val' if (n_global % val_every == 0) else 'train'
                for t in frame_to_tiles(cf, **TILE_LAYOUT, min_pts=0):
                    yield split, t
                n_global += 1
            print(f'  seg {seg}: consumed')

    res = write_lmdb_cache(out_dir, _gen(), cam='1', is_fisheye=False,
                            overwrite=True, map_size_gb=200)
    print(f'[build_waymo_cache] {res}')
    return out_dir


# ─── ClearML Dataset registration (per cache, parent inheritance). ──────

@PipelineDecorator.component(
    return_values=['dataset_id'],
    cache=False,
    packages=['clearml'],
)
def register_dataset_step(out_dir: str, dataset_name: str,
                            dataset_project: str = 'e2e_calib/cache') -> str:
    from clearml import Dataset
    parents = None
    try:
        prev = Dataset.get(dataset_name=dataset_name,
                            dataset_project=dataset_project)
        parents = [prev.id]
    except Exception:
        pass
    ds = Dataset.create(dataset_name=dataset_name,
                         dataset_project=dataset_project,
                         parent_datasets=parents)
    ds.add_files(out_dir, verbose=False)
    ds.upload(verbose=False, show_progress=False)
    ds.finalize()
    print(f'[register_dataset_step] {dataset_name} → id={ds.id}')
    return ds.id


@PipelineDecorator.pipeline(
    name='build_caches',
    project='e2e_calib/data',
    version='0.1.0',
)
def pipeline(datasets: str, kamikado_root: str, woven_root: str,
              waymo_dir: str, out_root: str,
              max_frames_per_scene: int, val_every: int):
    todo = [d.strip() for d in datasets.split(',') if d.strip()]
    results = {}
    if 'kamikado' in todo:
        out = f'{out_root}/kamikado_v3_tiled'
        out = build_kamikado_cache(kamikado_root, out,
                                     max_frames_per_scene, val_every)
        ds_id = register_dataset_step(out, 'kamikado_v3_tiled')
        results['kamikado'] = ds_id
    if 'woven' in todo:
        out = f'{out_root}/woven_v3_tile'
        out = build_woven_cache(woven_root, out,
                                  max_frames_per_scene, val_every)
        ds_id = register_dataset_step(out, 'woven_v3_tile')
        results['woven'] = ds_id
    if 'waymo' in todo:
        out = f'{out_root}/waymo_v3_tiled_i'
        out = build_waymo_cache(out, max_segs=2,
                                  max_frames_per_seg=max_frames_per_scene,
                                  val_every=val_every)
        ds_id = register_dataset_step(out, 'waymo_v3_tiled_i')
        results['waymo'] = ds_id
    print(f'pipeline done: {results}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='kamikado',
                    help='comma list: kamikado, woven, waymo')
    ap.add_argument('--kamikado-root',
                    default='/home/hfunaya/cache/kamikado/scenes')
    ap.add_argument('--woven-root',
                    default='/home/hfunaya/woven_sequence')
    ap.add_argument('--waymo-dir', default='',
                    help='set WAYMO_DIR env at the worker if needed')
    ap.add_argument('--out-root',
                    default='/raid/home/hfunaya/cache_v4')
    ap.add_argument('--max-frames-per-scene', type=int, default=-1)
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--queue', default='dgx2')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    if args.waymo_dir:
        import os
        os.environ['WAYMO_DIR'] = args.waymo_dir

    if args.local:
        PipelineDecorator.run_locally()
    else:
        PipelineDecorator.set_default_execution_queue(args.queue)
    pipeline(args.datasets, args.kamikado_root, args.woven_root,
             args.waymo_dir, args.out_root,
             args.max_frames_per_scene, args.val_every)


if __name__ == '__main__':
    main()
