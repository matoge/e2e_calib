"""V3 full-image cache for nuScenes.

Schema matches build_pandaset_full_v3 / build_waymo_v3:
  jpg_bytes : raw JPEG bytes inline (~120KB)
  IH, IW    : ints
  pts       : (N, 3) float32 — lidar pts in WORLD frame (LIDAR_TOP @ keyframe)
  cam_pos   : (3,)            — camera world position
  R_gt      : (3, 3)          — cam→world rotation
  T_gt      : (4, 4)          — world→cam (inv pose)
  K_full    : (3, 3)          — pinhole intrinsics
  cuboids   : list[{pos, dims, yaw}]  — annotated 3D boxes for this sample (world frame)
  scene, cam, frame           — provenance

nuScenes key frames are at 2Hz natively. --stride 2 → 1Hz.

Usage:
  python build_nuscenes_v3.py --max-scenes 5         # smoke
  python build_nuscenes_v3.py                        # all scenes, 2Hz, front cam
  python build_nuscenes_v3.py --cams CAM_FRONT,CAM_BACK --stride 1
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed


NS_ROOT  = Path('/mnt/nvme6t/nuscenes/data')
META_DIR = NS_ROOT / 'v1.0-trainval'

CAM_CHANNELS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT')


def _quat_mat(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _T_sensor_to_world(cal, ep) -> np.ndarray:
    R_s = _quat_mat(cal['rotation']); t_s = np.array(cal['translation'])
    R_e = _quat_mat(ep['rotation']);  t_e = np.array(ep['translation'])
    T = np.eye(4); T[:3, :3] = R_e; T[:3, 3] = t_e
    S = np.eye(4); S[:3, :3] = R_s; S[:3, 3] = t_s
    return (T @ S).astype(np.float64)


_CTX = {}


def _worker_init(meta_dir: str):
    meta = Path(meta_dir)
    _CTX['samples'] = {s['token']: s for s in json.load(open(meta / 'sample.json'))}
    _CTX['cal']     = {c['token']: c for c in json.load(open(meta / 'calibrated_sensor.json'))}
    _CTX['ego']     = {e['token']: e for e in json.load(open(meta / 'ego_pose.json'))}
    _CTX['scenes']  = {s['token']: s for s in json.load(open(meta / 'scene.json'))}
    _CTX['inst']    = {i['token']: i for i in json.load(open(meta / 'instance.json'))}
    _CTX['cat']     = {c['token']: c for c in json.load(open(meta / 'category.json'))}
    sd_by = {}
    for d in json.load(open(meta / 'sample_data.json')):
        if not d['is_key_frame']:
            continue
        parts = d['filename'].split('/')
        if len(parts) < 2:
            continue
        ch = parts[1]
        if (ch in CAM_CHANNELS or ch == 'LIDAR_TOP'
                or ch.startswith('RADAR_')):
            sd_by.setdefault(d['sample_token'], {})[ch] = d
    _CTX['sd_by'] = sd_by
    ann_by = {}
    for a in json.load(open(meta / 'sample_annotation.json')):
        ann_by.setdefault(a['sample_token'], []).append(a)
    _CTX['ann_by'] = ann_by


# nuScenes annotation categories worth treating as "obj"
_OBJ_PREFIXES = ('vehicle.', 'human.', 'animal', 'static_object.bicycle_rack')


def _ann_to_cuboid(ann) -> dict:
    """nuScenes annotation → {pos, dims, yaw} in world frame."""
    pos = np.array(ann['translation'], dtype=np.float32)        # (3,) world
    size = np.array(ann['size'], dtype=np.float32)              # (w, l, h)
    # PandaSet dims convention is (length_x, width_y, height_z) in object frame
    # nuScenes size is (w, l, h); object x-axis = forward → swap to (l, w, h).
    dims = np.array([size[1], size[0], size[2]], dtype=np.float32)
    q = ann['rotation']  # [w, x, y, z]
    R = _quat_mat(q)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return dict(pos=pos, dims=dims, yaw=yaw)


def _convert_scene(args_tuple):
    (scene_token, out_dir, data_root, cams_keep, stride, max_frames, gid_start,
     tile_layout) = args_tuple
    scene = _CTX['scenes'][scene_token]
    scene_name = scene['name']
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    sample_tok = scene['first_sample_token']
    samples = []
    while sample_tok:
        sd_map = _CTX['sd_by'].get(sample_tok, {})
        if 'LIDAR_TOP' in sd_map and any(ch in sd_map for ch in cams_keep):
            samples.append((sample_tok, sd_map))
        sample_tok = _CTX['samples'][sample_tok]['next']
    if stride > 1:
        samples = samples[::stride]
    if max_frames:
        samples = samples[:max_frames]

    written = 0
    gid = gid_start
    for fi, (sample_tok, sd_map) in enumerate(samples):
        # lidar in world
        lid = sd_map['LIDAR_TOP']
        T_lid2w = _T_sensor_to_world(_CTX['cal'][lid['calibrated_sensor_token']],
                                      _CTX['ego'][lid['ego_pose_token']])
        bin_path = Path(data_root) / lid['filename']
        try:
            pts_lid = np.frombuffer(bin_path.read_bytes(),
                                     dtype=np.float32).reshape(-1, 5)[:, :3]
        except Exception:
            continue
        h = np.hstack([pts_lid, np.ones((len(pts_lid), 1), dtype=pts_lid.dtype)])
        pts_world = (T_lid2w @ h.T).T[:, :3].astype(np.float32)
        is_radar_lid = np.zeros(len(pts_world), dtype=np.float32)

        # 5 radars in world frame. Continental ARS 408 → essentially 2D (z≈0
        # in sensor frame), but the world transform plants them at ~ego height.
        # Useful for the residual net: sparse but velocity/RCS-rich; treat them
        # as additional pts that flow through the same crop/frustum logic.
        radar_chs = ('RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                     'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT')
        radar_pts_list = []
        try:
            from nuscenes.utils.data_classes import RadarPointCloud
            RadarPointCloud.disable_filters()  # raw, no dyn_prop / RCS filter
            for rch in radar_chs:
                sd_r = sd_map.get(rch)
                if sd_r is None: continue
                T_r2w = _T_sensor_to_world(_CTX['cal'][sd_r['calibrated_sensor_token']],
                                            _CTX['ego'][sd_r['ego_pose_token']])
                rp = RadarPointCloud.from_file(str(Path(data_root) / sd_r['filename']))
                pts_r = rp.points[:3].T.astype(np.float32)             # (N,3) sensor frame
                if len(pts_r) == 0: continue
                hr = np.hstack([pts_r, np.ones((len(pts_r), 1), dtype=np.float32)])
                pts_r_w = (T_r2w @ hr.T).T[:, :3].astype(np.float32)
                radar_pts_list.append(pts_r_w)
        except Exception:
            pass  # devkit missing or radar PCD unreadable; skip silently
        if radar_pts_list:
            pts_radar_world = np.concatenate(radar_pts_list, axis=0)
            pts_world = np.concatenate([pts_world, pts_radar_world], axis=0)
            is_radar_lid = np.concatenate([is_radar_lid,
                                            np.ones(len(pts_radar_world), dtype=np.float32)])

        # cuboids for this sample (world frame)
        cubs = []
        for ann in _CTX['ann_by'].get(sample_tok, []):
            inst_tok = ann['instance_token']
            cat_tok = _CTX['inst'][inst_tok]['category_token']
            cat_name = _CTX['cat'][cat_tok]['name']
            if not any(cat_name.startswith(p) for p in _OBJ_PREFIXES):
                continue
            c = _ann_to_cuboid(ann)
            c['label'] = cat_name
            cubs.append(c)

        # per camera
        for ch in cams_keep:
            sd_cam = sd_map.get(ch)
            if sd_cam is None: continue
            cal_cam = _CTX['cal'][sd_cam['calibrated_sensor_token']]
            ep_cam  = _CTX['ego'][sd_cam['ego_pose_token']]
            T_cam2w = _T_sensor_to_world(cal_cam, ep_cam)
            T_gt = np.linalg.inv(T_cam2w).astype(np.float32)
            R_gt = T_cam2w[:3, :3].astype(np.float32)
            cam_pos = T_cam2w[:3, 3].astype(np.float32)
            K = np.array(cal_cam['camera_intrinsic'], dtype=np.float32)

            jpg_path = Path(data_root) / sd_cam['filename']
            if not jpg_path.exists(): continue
            try:
                jpg_bytes = jpg_path.read_bytes()
                with Image.open(jpg_path) as _im:
                    IW, IH = _im.size
            except Exception:
                continue

            # Filter pts to in-image (visible)
            homo = np.column_stack([pts_world, np.ones(len(pts_world))])
            pcam = (T_gt @ homo.T)[:3].T
            z = pcam[:, 2]
            uv = ((K @ pcam.T)[:2] / np.maximum(pcam[:, 2:].T, 1e-6)).T
            vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
            if vis.sum() < 16: continue
            pts_vis    = pts_world[vis]
            uv_vis     = uv[vis].astype(np.float32)
            z_vis      = z[vis].astype(np.float32)
            is_radar_v = is_radar_lid[vis].astype(np.float32)

            # is_obj on visible pts (cuboid membership in world frame).
            # Radar pts technically can fall inside a cuboid too — keep the
            # same test; if it surprises later we can mask radar out here.
            from datasets.pandaset_full import _is_obj_per_point
            is_obj_vis = _is_obj_per_point(pts_vis, cubs).astype(np.float32)

            common_inst = dict(
                cam_pos = torch.from_numpy(cam_pos),
                R_gt    = torch.from_numpy(R_gt),
                T_gt    = torch.from_numpy(T_gt),
                K_full  = torch.from_numpy(K),
                cuboids = cubs,
                scene = scene_name, cam = ch, frame = int(fi),
            )

            if tile_layout is None:
                inst = dict(common_inst)
                inst.update(dict(
                    jpg_bytes = jpg_bytes,
                    IH = int(IH), IW = int(IW),
                    pts      = torch.from_numpy(pts_vis),
                    uv_full  = torch.from_numpy(uv_vis),
                    z_cam    = torch.from_numpy(z_vis),
                    is_obj   = torch.from_numpy(is_obj_vis),
                    is_radar = torch.from_numpy(is_radar_v),
                ))
                torch.save(inst, inst_dir / f'{gid:08d}.pt')
                gid += 1
                written += 1
            else:
                from scripts.preprocessing._tile_split import cut_inst_to_tiles
                tw, th, st, pad, y0, q = tile_layout
                tile_files = cut_inst_to_tiles(
                    jpg_bytes=jpg_bytes, IW=int(IW), IH=int(IH),
                    pts_vis=pts_vis, uv_vis=uv_vis, z_vis=z_vis,
                    is_obj_vis=is_obj_vis, is_radar_vis=is_radar_v,
                    common_inst=common_inst,
                    tile_w=tw, tile_h=th, stride=st, pad_px=pad,
                    y_start=y0, jpg_quality=q,
                    out_dir=inst_dir, gid_base=gid)
                gid += 1
                written += len(tile_files)

    return scene_name, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',         default='/mnt/nvme6t/e2e_calib_cache/nuscenes_v3_full')
    ap.add_argument('--data-root',   default=str(NS_ROOT))
    ap.add_argument('--meta-dir',    default=str(META_DIR))
    ap.add_argument('--cams',        default='CAM_FRONT', help='comma list e.g. CAM_FRONT,CAM_BACK')
    ap.add_argument('--workers',     type=int, default=8)
    ap.add_argument('--max-scenes',  type=int, default=None)
    ap.add_argument('--max-frames',  type=int, default=None)
    ap.add_argument('--stride',      type=int, default=1, help='1 → 2Hz (native), 2 → 1Hz')
    ap.add_argument('--val-frac',    type=float, default=0.15)
    ap.add_argument('--seed',        type=int, default=42)
    # Tile mode: build N tiles/frame instead of 1 full frame per cam.
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384)
    ap.add_argument('--tile-pad',     type=int, default=64)
    ap.add_argument('--tile-y-start', type=int, default=0,  help='NS images are 900 tall — keep 0')
    ap.add_argument('--tile-jpg-q',   type=int, default=90)
    args = ap.parse_args()

    tile_layout = None
    if args.tile:
        tile_layout = (args.tile_w, args.tile_h, args.tile_stride,
                       args.tile_pad, args.tile_y_start, args.tile_jpg_q)
        print(f'TILE mode: tile={args.tile_w}×{args.tile_h} stride={args.tile_stride} '
              f'pad={args.tile_pad} y_start={args.tile_y_start}', flush=True)

    cams_keep = tuple(args.cams.split(','))
    for c in cams_keep:
        assert c in CAM_CHANNELS, f'unknown cam: {c} (must be in {CAM_CHANNELS})'

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load minimum meta in main proc to enumerate scenes; full meta in workers.
    scenes = json.load(open(Path(args.meta_dir) / 'scene.json'))
    scenes_sorted = sorted(scenes, key=lambda s: s['name'])
    if args.max_scenes:
        scenes_sorted = scenes_sorted[:args.max_scenes]
    print(f'cams={cams_keep}  scenes={len(scenes_sorted)}  stride={args.stride}  '
          f'workers={args.workers}  out={out_dir}', flush=True)

    gid_stride = 4000  # ~40 frames × 6 cams = 240 per scene, buffer
    argv = [(s['token'], str(out_dir), args.data_root, cams_keep, args.stride,
             args.max_frames, i * gid_stride, tile_layout)
            for i, s in enumerate(scenes_sorted)]

    t0 = time.time()
    written_total = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                              initializer=_worker_init,
                              initargs=(args.meta_dir,)) as ex:
        futs = {ex.submit(_convert_scene, a): a for a in argv}
        done = 0
        for fut in as_completed(futs):
            name, n = fut.result()
            written_total += n
            done += 1
            print(f'[{done}/{len(argv)}] {name}: +{n}  total={written_total} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    # train/val split (object-level on scene name)
    fnames = sorted(p.name for p in (out_dir / 'inst').glob('*.pt'))
    import random
    rng = random.Random(args.seed); rng.shuffle(scenes_sorted)
    n_val = max(1, int(len(scenes_sorted) * args.val_frac))
    val_scenes = set(s['name'] for s in scenes_sorted[:n_val])

    train_files, val_files = [], []
    for f in fnames:
        # decode scene from inst on disk for accuracy
        inst = torch.load(out_dir / 'inst' / f, weights_only=False)
        if inst.get('scene') in val_scenes:
            val_files.append(f)
        else:
            train_files.append(f)
    torch.save({'train': train_files, 'val': val_files}, out_dir / 'meta.pt')
    print(f'saved meta.pt: train={len(train_files)} val={len(val_files)}  → {out_dir}',
          flush=True)


if __name__ == '__main__':
    main()
