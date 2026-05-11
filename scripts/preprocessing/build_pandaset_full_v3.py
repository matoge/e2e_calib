"""V3 cache: full-image (decoded JPG) + per-frame metadata.

No crop pre-baked. Per (scene, cam, frame) we save:
  jpg_bytes: raw JPEG bytes (~200KB, 30x smaller than decoded uint8)
  IH, IW:    image dims
  pts:       (N_vis, 3) float32                — visible (in-image) lidar pts (world)
  cam_pos:   (3,)                              — camera world position
  R_gt:      (3, 3)                            — cam-to-world rotation
  T_gt:      (4, 4)                            — world-to-cam (inv pose)
  K_full:    (3, 3)                            — intrinsic
  uv_full:   (N_vis, 2) float32                — GT-projected (u,v) in full-image px
  z_cam:     (N_vis,)   float32                — GT camera-frame depth
  is_obj:    (N_vis,)   float32                — 1.0 if pt lies in ANY cuboid else 0
  cuboids:   list[dict] of useful labels       — kept for debugging / refined masks
  scene/cam/frame                              — provenance

NOTE: uv_full/z_cam/is_obj were moved from __getitem__ to build time on
2026-05 after profiling: is_obj on 13k pts × 160 cubs was 412 ms/call,
dominating SPS at 90 workers. Pre-computing → __getitem__ drops to ~20ms.

Storage (front_camera, 103 scenes, ~80 frames each):
  ~80 frames × 6.2 MB image + ~0.4 MB lidar ≈ 7 MB/frame
  total ≈ 103 × 80 × 7 MB ≈ 58 GB

Crop is decided at __getitem__ time → no center bias from cache.
"""
import argparse, sys, os, gzip, pickle, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets.pandaset import (USEFUL_LABELS, _quat_pos_to_mat, _project)
from datasets.pandaset_full import _is_obj_per_point


def _load_pkl(path: Path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    return pickle.load(open(path, 'rb'))


_TILE_LAYOUT = None  # set by main() when --tile; None = full-frame mode


def _process_scene(args_tuple):
    if len(args_tuple) == 9:
        (scene_root, scene_name, cam_name, out_dir, min_pts, gid_start, stride,
         tile_layout, frame_filter) = args_tuple
    else:
        (scene_root, scene_name, cam_name, out_dir, min_pts, gid_start, stride,
         tile_layout) = args_tuple
        frame_filter = None  # None = accept all
    sc_dir = Path(scene_root) / scene_name
    cam_dir = sc_dir / 'camera' / cam_name
    if not cam_dir.exists():
        return scene_name, []
    poses = json.load(open(cam_dir / 'poses.json'))
    intr  = json.load(open(cam_dir / 'intrinsics.json'))
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]], dtype=np.float64)

    cb_dir = sc_dir / 'annotations' / 'cuboids'
    ld_dir = sc_dir / 'lidar'
    # PandaSet ships BOTH 00.pkl and 00.pkl.gz (identical content).
    # If we sorted both together, cubs[44]=22.pkl while image fi=44 → frame mismatch.
    # Prefer .pkl; fall back to .pkl.gz only when .pkl missing.
    def _frame_files(d: Path) -> list[Path]:
        by_stem = {}
        for f in list(d.glob('*.pkl')) + list(d.glob('*.pkl.gz')):
            stem = f.name.split('.', 1)[0]  # '44' from '44.pkl' or '44.pkl.gz'
            if stem not in by_stem or f.suffix == '.pkl':
                by_stem[stem] = f
        return [by_stem[k] for k in sorted(by_stem.keys(), key=int)]

    cubs = _frame_files(cb_dir)
    lids = _frame_files(ld_dir)
    n = min(len(poses), len(cubs), len(lids))

    IW, IH = 1920, 1080
    out_files = []
    gid = gid_start

    for fi in range(0, n, stride):
        # frame_filter, when set, is a set of allowed frame indices for this scene
        if frame_filter is not None and fi not in frame_filter:
            continue
        cp = poses[fi]
        pose_mat = _quat_pos_to_mat(cp['heading'], cp['position'])
        cam_pos = np.array([cp['position']['x'], cp['position']['y'], cp['position']['z']],
                           dtype=np.float32)

        df = _load_pkl(lids[fi])
        if 'd' in df.columns:
            df = df[df['d'] == 0]
        pts_world = df[['x','y','z']].values.astype(np.float32)
        uv, z = _project(pts_world, pose_mat, K)
        vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
        if vis.sum() < min_pts:
            continue
        pts_vis = pts_world[vis]

        img_path = cam_dir / f'{fi:02d}.jpg'
        if not img_path.exists():
            continue
        # Store JPEG bytes inline (~200KB) instead of decoded raw uint8 (~6MB).
        # 30× cache shrink: 50GB → ~1.6GB. Decode is done at __getitem__ time
        # via TurboJPEG (~10ms/sample full-decode, faster with crop region).
        jpg_bytes = img_path.read_bytes()
        # Read dims from header without decoding pixels
        with Image.open(img_path) as _im:
            IW, IH = _im.size  # (W, H) — flip to (H, W) below

        cubs_df = _load_pkl(cubs[fi])
        if 'label' in cubs_df.columns:
            cubs_df = cubs_df[cubs_df['label'].isin(USEFUL_LABELS)]
        cub_list = []
        for _, obj in cubs_df.iterrows():
            cub_list.append(dict(
                pos  = np.array([obj['position.x'], obj['position.y'], obj['position.z']], dtype=np.float32),
                dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']], dtype=np.float32),
                yaw  = float(obj['yaw']),
                label= str(obj['label']),
            ))

        R_gt = Rotation.from_quat([cp['heading']['x'], cp['heading']['y'],
                                    cp['heading']['z'], cp['heading']['w']
                                    ]).as_matrix().astype(np.float32)
        T_gt = np.linalg.inv(pose_mat).astype(np.float32)

        # Pre-compute per-point caches used at __getitem__ time.
        # Without these, each __getitem__ call spends ~412ms in is_obj
        # on 13k pts × 160 cubs — dominating DataLoader SPS by 10x+.
        uv_vis = uv[vis].astype(np.float32)                        # (N_vis, 2)
        z_vis  = z[vis].astype(np.float32)                         # (N_vis,)
        is_obj_vis = _is_obj_per_point(pts_vis, cub_list)          # (N_vis,) float32

        K_t = K.astype(np.float32)
        common_inst = dict(
            cam_pos  = torch.from_numpy(cam_pos),
            R_gt     = torch.from_numpy(R_gt),
            T_gt     = torch.from_numpy(T_gt),
            K_full   = torch.from_numpy(K_t),
            cuboids  = cub_list,
            scene    = scene_name,
            cam      = cam_name,
            frame    = int(fi),
        )

        if tile_layout is None:
            # Full-frame mode (legacy): single inst per frame
            inst = dict(common_inst)
            inst.update(dict(
                jpg_bytes = jpg_bytes,
                IH        = int(IH),
                IW        = int(IW),
                pts      = torch.from_numpy(pts_vis),
                uv_full  = torch.from_numpy(uv_vis),
                z_cam    = torch.from_numpy(z_vis),
                is_obj   = torch.from_numpy(is_obj_vis),
            ))
            fname = f'{gid:08d}.pt'
            torch.save(inst, Path(out_dir) / fname)
            out_files.append(fname)
            gid += 1
        else:
            # Tile mode: N tiles per frame, each tile a self-contained inst.
            # Decode parent JPEG once via TurboJPEG (fall back to PIL), then
            # crop each tile + re-encode small JPEG.
            tw, th, x_starts, y_starts, pad_px, jpg_q = tile_layout
            try:
                import turbojpeg as _tj
                _TJ = _tj.TurboJPEG()
                img_full_arr = np.asarray(_TJ.decode(jpg_bytes,
                                                       pixel_format=_tj.TJPF_RGB))
                _encode = lambda arr: _TJ.encode(arr, quality=jpg_q,
                                                   pixel_format=_tj.TJPF_RGB)
            except Exception:
                img_full_arr = np.asarray(Image.open(img_path).convert('RGB'))
                def _encode(arr, q=jpg_q):
                    import io as _io
                    buf = _io.BytesIO()
                    Image.fromarray(arr).save(buf, format='JPEG', quality=q)
                    return buf.getvalue()

            tile_id = 0
            for ty in y_starts:
                for tx in x_starts:
                    # Keep tile + pad_px ring of pts. in_box=1 → strict tile (eligible
                    # crop pivots), 0 → padding ring (frustum context only). K_full is
                    # NOT shifted because distortion (Waymo KB / NS / ZOD) is centered
                    # on the original principal point — shifting cx/cy would corrupt
                    # the distortion model. Dataset subtracts (tile_u0, tile_v0) at
                    # access time, after the (un)distort projection step.
                    in_pad = ((uv_vis[:, 0] >= tx - pad_px) & (uv_vis[:, 0] < tx + tw + pad_px) &
                              (uv_vis[:, 1] >= ty - pad_px) & (uv_vis[:, 1] < ty + th + pad_px))
                    pts_t    = pts_vis[in_pad]
                    uv_t     = uv_vis[in_pad]                         # parent-image coords
                    z_t      = z_vis[in_pad]
                    is_obj_t = is_obj_vis[in_pad]
                    in_box_t = ((uv_t[:, 0] >= tx) & (uv_t[:, 0] < tx + tw) &
                                (uv_t[:, 1] >= ty) & (uv_t[:, 1] < ty + th)).astype(np.float32)

                    tile_arr = img_full_arr[ty:ty + th, tx:tx + tw].copy()
                    tile_jpg = _encode(tile_arr)

                    inst = dict(common_inst)
                    inst.update(dict(
                        jpg_bytes = tile_jpg,
                        IH        = int(th),
                        IW        = int(tw),
                        tile_u0   = int(tx),                          # parent-image origin
                        tile_v0   = int(ty),
                        tile_id   = int(tile_id),
                        pts       = torch.from_numpy(pts_t),
                        uv_full   = torch.from_numpy(uv_t),           # parent-image coords
                        z_cam     = torch.from_numpy(z_t),
                        is_obj    = torch.from_numpy(is_obj_t),
                        in_box    = torch.from_numpy(in_box_t),       # 1.0 = strict tile
                    ))
                    fname = f'{gid:08d}_t{tile_id}.pt'
                    torch.save(inst, Path(out_dir) / fname)
                    out_files.append(fname)
                    tile_id += 1
            gid += 1

    return scene_name, out_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',    default='/mnt/nvme6t/pandaset')
    ap.add_argument('--out',     default='/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full')
    ap.add_argument('--cam',     default='front_camera')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val-frac',type=float, default=0.15)
    ap.add_argument('--seed',    type=int,   default=42)
    ap.add_argument('--min-pts', type=int,   default=8)
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1, help='frame stride (5 → 2Hz)')
    # Tile mode (sliding-window): build N tiles/frame instead of 1 full frame.
    # All units are full-image (1920×1080) px.
    ap.add_argument('--tile', action='store_true',
                     help='enable sliding-window tile output (5×2=10 tiles/frame)')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384, help='step between tile origins (overlap = tile_w - stride)')
    ap.add_argument('--tile-pad',     type=int, default=64,  help='padding for pts/cuboid filter on tile edge (~10%% of tile side)')
    ap.add_argument('--tile-y-start', type=int, default=200, help='skip this many top-image px (sky)')
    ap.add_argument('--tile-jpg-q',   type=int, default=90)
    args = ap.parse_args()

    tile_layout = None
    if args.tile:
        IW, IH = 1920, 1080
        tw, th, st = args.tile_w, args.tile_h, args.tile_stride
        x_starts = list(range(0, IW - tw + 1, st))
        if x_starts[-1] != IW - tw: x_starts.append(IW - tw)
        ys = []
        y = args.tile_y_start
        while y + th <= IH:
            ys.append(y); y += st
        if not ys or ys[-1] != IH - th: ys.append(IH - th)
        tile_layout = (tw, th, x_starts, ys, args.tile_pad, args.tile_jpg_q)
        print(f'TILE mode: {len(x_starts)}×{len(ys)} = {len(x_starts)*len(ys)} tiles/frame, '
              f'tile={tw}×{th}, stride={st}, pad={args.tile_pad}, '
              f'y_start={args.tile_y_start}, q={args.tile_jpg_q}', flush=True)
        print(f'  x_starts={x_starts}\n  y_starts={ys}', flush=True)

    out_dir = Path(args.out)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    # Accept any subdir as a scene name. PS uses '001'/'002' (digits),
    # AV2 uses UUIDs ('00a6ffc1-...'), and converted layouts vary.
    # Skip known non-scene helper dirs.
    SKIP_DIRS = {'_proj_cache', '__pycache__', '.git', '.cache'}
    scenes = sorted([p.name for p in root.iterdir()
                      if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith('.')])
    import random
    rng = random.Random(args.seed); rng.shuffle(scenes)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_scenes = set(scenes[:n_val])
    print(f'cam={args.cam} scenes={len(scenes)} val={n_val} train={len(scenes)-n_val} workers={args.workers}', flush=True)

    # estimate gid stride: ~80 frames per scene, conservative buffer
    gid_stride = 200
    train_files, val_files = [], []
    t0 = time.time()
    if args.workers <= 1:
        for si, sc in enumerate(scenes):
            _, fnames = _process_scene((str(root), sc, args.cam, str(inst_dir), args.min_pts, si * gid_stride, args.stride, tile_layout))
            (val_files if sc in val_scenes else train_files).extend(fnames)
            print(f'[{si+1}/{len(scenes)}] {sc} +{len(fnames)} ({time.time()-t0:.0f}s)', flush=True)
    else:
        argv = [(str(root), sc, args.cam, str(inst_dir), args.min_pts, si * gid_stride, args.stride, tile_layout)
                for si, sc in enumerate(scenes)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_scene, a): a[1] for a in argv}
            done = 0
            for fut in as_completed(futures):
                sc, fnames = fut.result()
                (val_files if sc in val_scenes else train_files).extend(fnames)
                done += 1
                print(f'[{done}/{len(scenes)}] {sc} +{len(fnames)} ({time.time()-t0:.0f}s)', flush=True)

    meta = {'train': sorted(train_files), 'val': sorted(val_files), 'cam': args.cam}
    torch.save(meta, out_dir / 'meta.pt')
    print(f'\nsaved meta.pt: train={len(meta["train"])} val={len(meta["val"])} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
