"""V3 FULL LMDB cache for kamikado scenes (FCM fisheye), tile-free.

Inputs (one directory per scene, flat layout):
    SCENE/
      calib.calib          — JSON intrinsics + fisheye coeffs + extrinsics V→S
      image_N.png          — raw parent image (3840×2160, fisheye)
      points_V_N.txt       — lidar pts in VEHICLE frame, `x y z intensity`

Output:
    OUT/data.lmdb/         — single LMDB, one key per frame (`{gid:08d}.pt`)
    OUT/meta.pt            — {train: [fnames], val: [...], cam, is_fisheye}

Notes:
- Skips the tile loop entirely; each frame becomes ONE inst, packed straight
  into LMDB via ShardWriter (per-worker shard, then merge_shards).
- No `inst/` dir is created — saves 100k× small `.pt` open() overhead at
  training time and keeps inode pressure off the filesystem.
- pack format identical to convert_tile_cache_to_lmdb output, so the existing
  PandaSetCalibDatasetFull reader works unchanged.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

# A handful of kamikado PNGs are truncated (uploader interrupted). Tolerate by
# loading what we can; truncated frames usually still decode > 90% of pixels.
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing.lmdb_writer import ShardWriter, merge_shards  # noqa: E402
from scripts.util.projection import project_lidar_into_image  # noqa: E402


DEFAULT_SRC = Path('/home/hfunaya/raw/kamikado/scenes')
DEFAULT_OUT = Path('/raid/home/hfunaya/cache/kamikado_v3_full')


def _frame_idx_from_image(p: Path) -> int:
    m = re.search(r'_(\d+)\.png$', p.name)
    return int(m.group(1)) if m else -1


def _load_calib(scene: Path):
    with open(scene / 'calib.calib') as f:
        data = json.load(f)
    K = np.array(data['calibration']['intrinsics']['camera_model']
                  ['pinhole_parameters']['matrix_image_camera']['matrix']).T
    K = np.ascontiguousarray(K, dtype=np.float32)
    dist = np.asarray(data['calibration']['intrinsics']['distortion_model']
                       ['generic_fisheye_parameters']['coefficients'],
                       dtype=np.float32)
    quat = data['calibration']['extrinsics']['transform_VS']['so3']
    from scipy.spatial.transform import Rotation
    R_VS = Rotation.from_quat([quat['x'], quat['y'], quat['z'], quat['w']]).as_matrix()
    t_VS = np.asarray(data['calibration']['extrinsics']['transform_VS']
                       ['translation']['matrix'][0], dtype=np.float64)
    T_VS = np.eye(4); T_VS[:3, :3] = R_VS; T_VS[:3, 3] = t_VS
    T_SV = np.linalg.inv(T_VS)
    return K, dist, T_VS.astype(np.float32), T_SV.astype(np.float32)


def _read_points_V(p: Path) -> np.ndarray:
    arr = np.loadtxt(p, comments='#', usecols=(0, 1, 2, 3), dtype=np.float32)
    return np.atleast_2d(arr)


def _png_to_arr(png_path: Path) -> tuple[np.ndarray, int, int]:
    img = Image.open(png_path).convert('RGB')
    arr = np.asarray(img)
    H, W = arr.shape[:2]
    return arr, int(W), int(H)


def _double_scan_frames(scene: Path, ratio: float) -> set[int]:
    counts = {}
    for p in scene.glob('points_V_*.txt'):
        try:
            i = int(p.stem.split('_')[-1])
        except ValueError:
            continue
        counts[i] = sum(1 for _ in open(p))
    if not counts:
        return set()
    vals = sorted(counts.values())
    med = vals[len(vals) // 2]
    return {i for i, n in counts.items() if n >= ratio * med}


def _iter_scene_frames(scene: Path, dbl_ratio: float):
    skip = _double_scan_frames(scene, dbl_ratio)
    images = sorted(scene.glob('image_*.png'), key=_frame_idx_from_image)
    kept = []
    for p in images:
        i = _frame_idx_from_image(p)
        if i < 0 or i in skip:
            continue
        if not (scene / f'points_V_{i}.txt').exists():
            continue
        kept.append(i)
    return kept, skip


def process_one_frame(scene_dir: Path, frame_idx: int, K, dist, T_SV,
                       jpg_q: int):
    """Build one inst dict (no I/O), or return None on skip."""
    img_path = scene_dir / f'image_{frame_idx}.png'
    pts_path = scene_dir / f'points_V_{frame_idx}.txt'
    if not img_path.exists() or not pts_path.exists():
        return None
    img_arr, IW, IH = _png_to_arr(img_path)
    pts_V = _read_points_V(pts_path)
    if pts_V.size == 0:
        return None
    _, pts_vis, uv_vis, z_vis, intensity_vis = project_lidar_into_image(
        pts_V, K, T_SV, IW, IH, is_fisheye=True, dist=dist, z_min=0.5)
    if len(pts_vis) < 64:
        return None
    is_obj_vis = np.zeros(len(pts_vis), dtype=np.float32)

    import io as _io
    _b = _io.BytesIO()
    Image.fromarray(img_arr).save(_b, format='JPEG', quality=jpg_q)
    inst = dict(
        cam_pos    = torch.zeros(3, dtype=torch.float32),
        R_gt       = torch.eye(3, dtype=torch.float32),
        T_gt       = torch.eye(4, dtype=torch.float32),
        K_full     = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)),
        distortion = torch.from_numpy(dist.astype(np.float32)),
        is_fisheye = True,
        cuboids    = [],
        scene      = scene_dir.name, cam='fcm', frame=int(frame_idx),
        jpg_bytes  = _b.getvalue(),
        IH=IH, IW=IW,
        pts        = torch.from_numpy(pts_vis),
        uv_full    = torch.from_numpy(uv_vis),
        z_cam      = torch.from_numpy(z_vis),
        is_obj     = torch.from_numpy(is_obj_vis),
        intensity  = torch.from_numpy(intensity_vis),
    )
    return inst


def worker_process(args_tuple):
    """Per-worker entrypoint: reads its own task list, writes its own shard."""
    (worker_id, tasks, shard_path, jpg_q, shard_map_gb) = args_tuple
    sw = ShardWriter(shard_path, map_size_gb=shard_map_gb)
    written = 0
    failed = 0
    fnames_local = []  # [(scene, fname)]
    for (scene_str, frame_idx, K, dist, T_SV, gid) in tasks:
        try:
            inst = process_one_frame(Path(scene_str), int(frame_idx),
                                      K, dist, T_SV, jpg_q)
        except Exception:
            inst = None
            failed += 1
        if inst is None:
            continue
        fname = f'{int(gid):08d}.pt'
        sw.put_inst(fname, inst)
        fnames_local.append((str(inst['scene']), fname))
        written += 1
    sw.close()
    return worker_id, written, failed, fnames_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(DEFAULT_SRC),
                    help='dir containing one subdir per scene')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-frames-per-scene', type=int, default=None)
    ap.add_argument('--val-frac', type=float, default=0.15,
                    help='fraction of scenes used for val')
    ap.add_argument('--double-scan-ratio', type=float, default=1.4)
    ap.add_argument('--parent-jpg-q', type=int, default=95)
    ap.add_argument('--shard-map-gb', type=int, default=40,
                    help='per-worker shard LMDB map_size; final merge map_size auto-scales')
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = sorted(p for p in src.iterdir() if p.is_dir())
    print(f'{len(scenes)} scenes under {src}', flush=True)

    # Build flat task list (scene, frame, K, dist, T_SV, gid) deterministically.
    tasks: list[tuple] = []
    scene_of_frame: list[tuple[str, int]] = []  # parallel to tasks
    GID_PER_FRAME = 1  # FULL: one frame → one inst → one gid
    gid_cursor = 0
    for scene in scenes:
        try:
            K, dist, T_VS, T_SV = _load_calib(scene)
        except Exception as e:
            print(f'  [{scene.name}] calib load failed: {e}', flush=True)
            continue
        frames, skipped = _iter_scene_frames(scene, args.double_scan_ratio)
        if args.max_frames_per_scene:
            frames = frames[:args.max_frames_per_scene]
        print(f'  [{scene.name}] {len(frames)} frames '
              f'(skipped {len(skipped)} double-scan)', flush=True)
        for f in frames:
            tasks.append((str(scene), f, K, dist, T_SV, gid_cursor))
            scene_of_frame.append((scene.name, f))
            gid_cursor += GID_PER_FRAME

    print(f'total frames to process: {len(tasks)}', flush=True)
    if not tasks:
        return

    # Split tasks across workers (round-robin) so shards stay balanced.
    nw = max(1, args.workers)
    shard_dir = out / '_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    per_worker = [[] for _ in range(nw)]
    for i, t in enumerate(tasks):
        per_worker[i % nw].append(t)
    worker_args = [
        (wid, per_worker[wid], str(shard_dir / f'shard_{wid:02d}.lmdb'),
         args.parent_jpg_q, args.shard_map_gb)
        for wid in range(nw)
    ]

    all_fnames: list[tuple[str, str]] = []
    if nw == 1:
        wid, w, f, fns = worker_process(worker_args[0])
        all_fnames.extend(fns)
        print(f'  worker {wid}: wrote {w} failed {f}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(worker_process, wa): wa for wa in worker_args}
            done = 0
            total_fail = 0
            for fut in as_completed(futs):
                wid, w, f, fns = fut.result()
                all_fnames.extend(fns)
                total_fail += f
                done += 1
                print(f'  [{done}/{nw}] worker {wid}: wrote {w} failed {f}', flush=True)
            print(f'total failed frames: {total_fail}', flush=True)

    # Merge shards into final LMDB.
    final_lmdb = out / 'data.lmdb'
    shard_paths = sorted(p for p in shard_dir.glob('shard_*.lmdb') if p.is_dir())
    print(f'merging {len(shard_paths)} shards → {final_lmdb} ...', flush=True)
    merge_map_gb = max(80, args.shard_map_gb * len(shard_paths) + 20)
    n_written, n_cubs = merge_shards(shard_paths, final_lmdb,
                                      map_size_gb=merge_map_gb)
    print(f'merged: {n_written} insts + {n_cubs} cuboid keys', flush=True)

    # Cleanup shards (optional — keep for now to allow incremental rebuild).
    # import shutil; shutil.rmtree(shard_dir)

    # Scene-level train/val split.
    val_scenes = set(s.name for s in scenes[:max(1, int(len(scenes) * args.val_frac))])
    # Re-derive (scene, fname) by scanning tasks order using the gid encoding.
    train_files: list[str] = []
    val_files: list[str] = []
    for scene_name, fname in all_fnames:
        if scene_name in val_scenes:
            val_files.append(fname)
        else:
            train_files.append(fname)
    train_files.sort(); val_files.sort()
    meta = {'train': train_files, 'val': val_files,
            'cam': 'fcm', 'is_fisheye': True}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(train_files)} val={len(val_files)} '
          f'(val scenes = {sorted(val_scenes)})', flush=True)


if __name__ == '__main__':
    main()
