"""V3 tile cache for kamikado-san's woven scenes (FCM fisheye).

Input (one directory per scene, flat layout):
    SCENE/
      calib.calib          — JSON: intrinsics + fisheye coeffs + extrinsics V→S
      image_N.png          — raw parent image (3840×2160, PNG, fisheye)
      points_V_N.txt       — lidar pts in VEHICLE frame, `x y z intensity` lines

Output (same schema as build_zod_v3.py so downstream training + LMDB
conversion work unchanged):
    CACHE/inst/{scene_short}{frame:04d}_t{tile_id}.pt
    CACHE/meta.pt  {train, val, cam, is_fisheye}

Notes:
- No 2D/3D annotations in this dataset: `cuboids = []`, `is_obj = 0`.
- Frames whose points_V_*.txt line count is ≥ 1.4× the scene median are
  assumed to contain a double LiDAR sweep (observed on several captures)
  and are skipped.
- K_full stored is the parent-image intrinsic. In reality the effective
  fx/fy drifts per-tile near the fisheye edges; keeping the parent K is a
  first-order approximation and can be refined later by inserting a
  tile-centered rectified K.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing._tile_split import cut_inst_to_tiles  # noqa: E402
from scripts.util.projection import project_lidar_into_image  # noqa: E402


DEFAULT_SRC = Path('/mnt/datadisk3/tmpoc_kamikado/scenes')
DEFAULT_OUT = Path('/mnt/datadisk3/tmpoc_kamikado/cache/kamikado_v3_tiled')


def _short_scene_id(scene_name: str) -> str:
    """Stable 8-hex scene prefix used in tile filenames (<= 8 chars)."""
    import hashlib
    return hashlib.md5(scene_name.encode()).hexdigest()[:8]


def _frame_idx_from_image(p: Path) -> int:
    m = re.search(r'_(\d+)\.png$', p.name)
    return int(m.group(1)) if m else -1


def _load_calib(scene: Path):
    """Return (K, dist4, T_VS, T_SV, IW_hint, IH_hint).

    Matches viewer.py::get_camera_params exactly. `matrix_image_camera.matrix`
    is stored transposed in the JSON, so we .T it back. Fisheye distortion is
    Kannala-Brandt 4-coef (k1..k4).
    """
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
    """Load `x y z intensity` comment-hash text file. Returns (N, 4) float32
    with intensity in column 3. Caller separates xyz from intensity so the
    `pts` tensor stored in cache keeps the (N, 3) shape every downstream
    consumer expects, and intensity is carried on a separate `intensity`
    key.
    """
    arr = np.loadtxt(p, comments='#', usecols=(0, 1, 2, 3), dtype=np.float32)
    return np.atleast_2d(arr)


def _png_to_arr(png_path: Path) -> tuple[np.ndarray, int, int]:
    """PNG → (H, W, 3) uint8 numpy array. No JPEG round-trip; tiles are
    encoded straight from this array by cut_inst_to_tiles, so the JPEG
    quality loss only happens once at tile-write time (was twice when
    we used to pre-encode the parent into JPEG_q=95)."""
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


def process_frame(args_tuple):
    (scene_dir, frame_idx, K, dist, T_SV, out_dir, gid_start,
     tile_layout, jpg_q) = args_tuple
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_path = scene_dir / f'image_{frame_idx}.png'
        pts_path = scene_dir / f'points_V_{frame_idx}.txt'
        if not img_path.exists() or not pts_path.exists():
            return frame_idx, 0
        # PNG → numpy directly. cut_inst_to_tiles encodes each tile to JPEG
        # exactly once (was twice: PNG→JPEG-95 then re-encode at TJ.encode).
        img_arr, IW, IH = _png_to_arr(img_path)

        pts_V = _read_points_V(pts_path)   # (N, 4) xyzi
        if pts_V.size == 0:
            return frame_idx, 0

        # Shared LiDAR→cam→fisheye-projection pipeline (matches CaaaS).
        _, pts_vis, uv_vis, z_vis, intensity_vis = project_lidar_into_image(
            pts_V, K, T_SV, IW, IH,
            is_fisheye=True, dist=dist, z_min=0.5)
        if len(pts_vis) < 64:
            return frame_idx, 0
        is_obj_vis = np.zeros(len(pts_vis), dtype=np.float32)

        common_inst = dict(
            cam_pos    = torch.zeros(3, dtype=torch.float32),
            R_gt       = torch.eye(3, dtype=torch.float32),
            T_gt       = torch.eye(4, dtype=torch.float32),
            K_full     = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)),
            distortion = torch.from_numpy(dist.astype(np.float32)),
            is_fisheye = True,
            cuboids    = [],
            scene = scene_dir.name, cam = 'fcm', frame = int(frame_idx),
        )

        if tile_layout is None:
            import io as _io
            _b = _io.BytesIO()
            Image.fromarray(img_arr).save(_b, format='JPEG', quality=jpg_q)
            inst = dict(common_inst)
            inst.update(dict(
                jpg_bytes = _b.getvalue(),
                IH=IH, IW=IW,
                pts       = torch.from_numpy(pts_vis),
                uv_full   = torch.from_numpy(uv_vis),
                z_cam     = torch.from_numpy(z_vis),
                is_obj    = torch.from_numpy(is_obj_vis),
                intensity = torch.from_numpy(intensity_vis),
            ))
            torch.save(inst, inst_dir / f'{gid_start:08d}.pt')
            return frame_idx, 1

        tw, th, st, pad, y0, q = tile_layout
        tile_files = cut_inst_to_tiles(
            img_full_arr=img_arr, IW=IW, IH=IH,
            pts_vis=pts_vis, uv_vis=uv_vis, z_vis=z_vis,
            is_obj_vis=is_obj_vis,
            extra_per_point={'intensity': intensity_vis},
            common_inst=common_inst,
            tile_w=tw, tile_h=th, stride=st, pad_px=pad,
            y_start=y0, jpg_quality=q,
            out_dir=inst_dir, gid_base=gid_start)
        return frame_idx, len(tile_files)
    except Exception as e:
        return frame_idx, -1


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
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384)
    ap.add_argument('--tile-pad',     type=int, default=64)
    ap.add_argument('--tile-y-start', type=int, default=600,
                    help='2160-tall image — skip top ~600 px of sky')
    ap.add_argument('--tile-jpg-q',   type=int, default=95)
    ap.add_argument('--parent-jpg-q', type=int, default=95,
                    help='quality for PNG → JPEG re-encode before tiling')
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / 'inst').mkdir(parents=True, exist_ok=True)

    tile_layout = None
    if args.tile:
        tile_layout = (args.tile_w, args.tile_h, args.tile_stride,
                       args.tile_pad, args.tile_y_start, args.tile_jpg_q)
        print(f'TILE mode: tile={args.tile_w}×{args.tile_h} '
              f'stride={args.tile_stride} pad={args.tile_pad} '
              f'y_start={args.tile_y_start}', flush=True)

    scenes = sorted(p for p in src.iterdir() if p.is_dir())
    print(f'{len(scenes)} scenes under {src}', flush=True)

    tasks = []
    scene_of_gid: dict[int, str] = {}
    gid_cursor = 0
    GID_PER_FRAME = 100  # one frame → up to 100 tiles; matches zod build
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
            tasks.append((str(scene), f, K, dist, T_SV, str(out),
                           gid_cursor, tile_layout, args.parent_jpg_q))
            scene_of_gid[gid_cursor] = scene.name
            gid_cursor += GID_PER_FRAME

    print(f'total frames to process: {len(tasks)}', flush=True)
    if not tasks:
        return

    written = 0
    if args.workers <= 1:
        for a in tasks:
            _, n = process_frame(a)
            written += max(0, n)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_frame, a): a for a in tasks}
            done = 0
            for fut in as_completed(futs):
                _, n = fut.result()
                written += max(0, n)
                done += 1
                if done % 40 == 0 or done == len(tasks):
                    print(f'  [{done}/{len(tasks)}] written={written}',
                          flush=True)
    print(f'done: {written} tile insts → {out}', flush=True)

    # Scene-level train/val split (deterministic: first `val_frac` scenes go
    # to val). Tiles inherit their parent scene's split via the `scene` field.
    inst_dir = out / 'inst'
    val_scenes = set(s.name for s in scenes[:max(1, int(len(scenes) * args.val_frac))])
    train_files, val_files = [], []
    for f in sorted(p.name for p in inst_dir.glob('*.pt')):
        try:
            inst = torch.load(inst_dir / f, weights_only=False)
            if str(inst.get('scene', '')) in val_scenes:
                val_files.append(f)
            else:
                train_files.append(f)
        except Exception:
            train_files.append(f)
    meta = {'train': train_files, 'val': val_files,
            'cam': 'fcm', 'is_fisheye': True}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(train_files)} val={len(val_files)} '
          f'(val scenes = {sorted(val_scenes)})', flush=True)


if __name__ == '__main__':
    main()
