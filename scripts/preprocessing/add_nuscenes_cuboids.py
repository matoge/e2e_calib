"""Post-processor: add annotations/cuboids/{fi:02d}.pkl.gz to converted nuScenes PS scenes.

nuScenes sample_annotation.json stores boxes in WORLD frame already
(translation = global xyz, rotation = global heading quat). Frame index fi
maps to the fi-th sample in scene-token order, exactly like
nuscenes_to_pandaset.py walks first_sample_token → next.
"""
from __future__ import annotations
import argparse, gzip, json, math, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _q_to_yaw(qw, qx, qy, qz):
    s = 2.0 * (qw * qz + qx * qy)
    c = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s, c)


_CTX_CACHE = {}

def _load_meta(meta_dir: Path):
    if str(meta_dir) in _CTX_CACHE:
        return _CTX_CACHE[str(meta_dir)]
    scenes  = {s['name']: s for s in json.load(open(meta_dir / 'scene.json'))}
    samples = {s['token']: s for s in json.load(open(meta_dir / 'sample.json'))}
    annos = json.load(open(meta_dir / 'sample_annotation.json'))
    annos_by_sample = {}
    for a in annos:
        annos_by_sample.setdefault(a['sample_token'], []).append(a)
    ctx = dict(scenes=scenes, samples=samples, annos=annos_by_sample)
    _CTX_CACHE[str(meta_dir)] = ctx
    return ctx


def process_one(scene_name: str, ps_scene_dir: Path, meta_dir: Path):
    out_dir = ps_scene_dir / 'annotations' / 'cuboids'
    if out_dir.is_dir() and any(out_dir.glob('*.pkl.gz')):
        return f'[skip] {scene_name}'

    ctx = _load_meta(meta_dir)
    if scene_name not in ctx['scenes']:
        return f'[skip-no-scene] {scene_name}'
    scn = ctx['scenes'][scene_name]
    sample_toks = []
    tok = scn['first_sample_token']
    while tok:
        sample_toks.append(tok)
        tok = ctx['samples'][tok]['next']

    n_ps_frames = len(list((ps_scene_dir / 'lidar').glob('*.pkl')))
    if n_ps_frames == 0:
        return f'[skip-empty-ps] {scene_name}'

    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(n_ps_frames):
        if fi >= len(sample_toks):
            df = pd.DataFrame()
        else:
            stok = sample_toks[fi]
            rows = []
            for a in ctx['annos'].get(stok, []):
                tx, ty, tz = a['translation']
                # nuScenes size = [width, length, height] = [y, x, z] in box-local
                w, l, h = a['size']
                qw, qx, qy, qz = a['rotation']
                yaw = _q_to_yaw(qw, qx, qy, qz)
                rows.append({
                    'yaw':          float(yaw),
                    'position.x':   float(tx),
                    'position.y':   float(ty),
                    'position.z':   float(tz),
                    'dimensions.x': float(l),  # length
                    'dimensions.y': float(w),  # width
                    'dimensions.z': float(h),  # height
                    'category':     a.get('category_name', 'unknown'),
                })
            df = pd.DataFrame(rows, columns=[
                'yaw', 'position.x', 'position.y', 'position.z',
                'dimensions.x', 'dimensions.y', 'dimensions.z', 'category'
            ])
        with gzip.open(out_dir / f'{fi:02d}.pkl.gz', 'wb') as f:
            pickle.dump(df, f)
    return f'[ok]   {scene_name}  frames={n_ps_frames}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-meta', default='/mnt/nvme6t/nuscenes/data/v1.0-trainval')
    ap.add_argument('--dst', default='/mnt/nvme6t/nuscenes_ps')
    ap.add_argument('--workers', type=int, default=1, help='Process pool size — set >1 carefully (meta is ~1GB)')
    ap.add_argument('--limit',   type=int, default=0)
    args = ap.parse_args()

    meta = Path(args.src_meta)
    dst  = Path(args.dst)
    ps_scenes = sorted([d for d in dst.iterdir() if d.is_dir()])
    if args.limit:
        ps_scenes = ps_scenes[:args.limit]
    print(f'processing {len(ps_scenes)} scenes  ({args.workers} workers)')

    if args.workers <= 1:
        # serial — load meta once
        for i, ps in enumerate(ps_scenes):
            r = process_one(ps.name, ps, meta)
            print(f'  [{i+1}/{len(ps_scenes)}] {r}', flush=True)
    else:
        tasks = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ps in ps_scenes:
                tasks.append(ex.submit(process_one, ps.name, ps, meta))
            for i, fut in enumerate(as_completed(tasks)):
                print(f'  [{i+1}/{len(tasks)}] {fut.result()}', flush=True)


if __name__ == '__main__':
    main()
