"""V3 tile cache for TSS4 (woven_sequence vehicle 248) slow-drive captures.

Source: /mnt/ecp-perception/woven_sequence/adas-data_01/20230612_001946/
        sequence=248_*  (31 sequences × 50 frames = 1550 frames)
Calib:  /home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_26/
        recalibration.json [vehicle_id="248"].fcm  (KB4 fisheye, refined)

Output cache schema matches build_kamikado_v3 exactly so the existing
PandaSetCalibDatasetFull loader + train_ps_v3_ddp pipeline can ingest
it without changes:
    inst .pt with keys
      jpg_bytes, IH, IW, K_full, distortion, is_fisheye=True,
      cam_pos=0, R_gt=I, T_gt=I, scene, cam='fcm', frame,
      pts (N,3) cam-frame, uv_full (N,2), z_cam (N,), is_obj (N,)=0,
      intensity (N,), tile_u0/v0/id, in_box, cuboids=[]

Notes:
- T_cam_lidar = R_to_rdf @ inv(R_cam_to_veh) @ I  (LiDAR is rear_axle frame
  = vehicle frame), tvec = -R_to_rdf @ mp.  Matches the projector that the
  overlay PNGs (docs/assets/2026-05-24_tss4_overlay/*.jpg) were rendered
  with so any drift visible there will reproduce in the cache and the
  trainer can correct it.
- Y-band crop (top 15% / bottom 15% dropped) is enforced via tile y_start
  / y_end. X is full width (KB4 distortion is symmetric so periphery
  must come back later).
- Turning frames (|yawrate| ≥ --max-yawrate, default 0.05 rad/s ≈ 2.9°/s)
  are dropped at frame level; on this slow-drive set that keeps 1426/1550
  frames. Reason: pose-conditioned LiDAR sweep timing on turning is
  noisier and would introduce label noise into a calib trainer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocessing._tile_split import cut_inst_to_tiles  # noqa: E402
from scripts.util.projection import project_lidar_into_image  # noqa: E402


DEFAULT_SRC = Path('/mnt/ecp-perception/woven_sequence/adas-data_01/20230612_001946')
DEFAULT_RECALIB = Path(
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_26/recalibration.json'
)
DEFAULT_OUT = Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled')
VEHICLE_ID = '248'

# rear_axle (x=fwd, y=left, z=up) → camera RDF (x=right, y=down, z=fwd)
# This matches scripts/_debug/_tss4_calib_overlay.build_K_D_RT.
_R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def load_calib(recalib_path: Path, vehicle_id: str = VEHICLE_ID):
    data = json.loads(recalib_path.read_text())[vehicle_id]
    fcm = data['fcm']
    kb = fcm['kb']
    fx = fy = float(kb['focal_length'])
    cx, cy = fcm['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.asarray([kb['k1'], kb['k2'], kb['k3'], kb['k4']],
                      dtype=np.float64)
    IW, IH = fcm['resolution']

    mp_fcm = np.asarray(fcm['mp'], dtype=np.float64).reshape(3, 1)
    roll, pitch, yaw = fcm['rot']  # rot_order zyx
    R_cam_to_veh = Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    R_total = _R_TO_RDF @ np.linalg.inv(R_cam_to_veh)
    tvec = (-_R_TO_RDF @ mp_fcm).flatten()

    poslv = data.get('poslv')
    if poslv is not None:
        mp_p = np.asarray(poslv['mp'], dtype=np.float64)
        roll_p, pitch_p, yaw_p = poslv['rot']
        R_p = Rotation.from_euler('zyx', [yaw_p, pitch_p, roll_p]).as_matrix()
        R_rear2cam = R_total @ R_p
        t_rear2cam = (R_total @ mp_p) + tvec
    else:
        R_rear2cam = R_total
        t_rear2cam = tvec

    T_cam_lidar = np.eye(4, dtype=np.float64)
    T_cam_lidar[:3, :3] = R_rear2cam
    T_cam_lidar[:3, 3] = t_rear2cam
    return (K.astype(np.float32), dist.astype(np.float32),
            T_cam_lidar.astype(np.float32), int(IW), int(IH))


def _list_seq_frames(seq_dir: Path):
    cam = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
    lid = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
    n = min(len(cam), len(lid))
    return cam[:n], lid[:n]


def process_frame(args_tuple):
    (seq_name, frame_idx, cam_path, lid_path, K, dist, T_cam_lidar,
     IW, IH, max_yawrate, out_dir, gid_start, tile_layout,
     parent_pad_px) = args_tuple
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)
    try:
        d = np.load(lid_path)
        yawrate = float(d['yawrate'])
        speed = float(d['speed'])
        if abs(yawrate) >= max_yawrate:
            return seq_name, frame_idx, 0, 'turning'

        img = np.asarray(Image.open(cam_path).convert('RGB'))
        if img.shape[:2] != (IH, IW):
            return seq_name, frame_idx, 0, f'bad-shape {img.shape}'

        # rear_axle frame XYZ + intensity
        pts_xyzi = np.stack([d['xs'], d['ys'], d['zs'],
                             d['intensity'].astype(np.float32)], axis=-1)
        pts_xyzi = pts_xyzi.astype(np.float32)
        # match overlay pre-filter
        keep_pre = pts_xyzi[:, 0] > -10.0
        pts_xyzi = pts_xyzi[keep_pre]

        _keep, pts_cam, uv, z, intensity = project_lidar_into_image(
            pts_xyzi, K, T_cam_lidar, IW, IH,
            is_fisheye=True, dist=dist, z_min=0.5, pad_px=parent_pad_px)
        if len(pts_cam) < 64:
            return seq_name, frame_idx, 0, 'few-pts'

        is_obj = np.zeros(len(pts_cam), dtype=np.float32)

        common_inst = dict(
            cam_pos    = torch.zeros(3, dtype=torch.float32),
            R_gt       = torch.eye(3, dtype=torch.float32),
            T_gt       = torch.eye(4, dtype=torch.float32),
            K_full     = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)),
            distortion = torch.from_numpy(dist.astype(np.float32)),
            is_fisheye = True,
            cuboids    = [],
            scene      = seq_name,
            cam        = 'tss4_fcm',
            frame      = int(frame_idx),
            speed      = float(speed),
            yawrate    = float(yawrate),
        )

        tw, th, st, pad, y0, y1, q = tile_layout
        tile_files = cut_inst_to_tiles(
            img_full_arr=img, IW=IW, IH=IH,
            pts_vis=pts_cam, uv_vis=uv, z_vis=z,
            is_obj_vis=is_obj,
            extra_per_point={'intensity': intensity.astype(np.float32)},
            common_inst=common_inst,
            tile_w=tw, tile_h=th, stride=st, pad_px=pad,
            y_start=y0, y_end=y1, jpg_quality=q,
            out_dir=inst_dir, gid_base=gid_start)
        return seq_name, frame_idx, len(tile_files), 'ok'
    except Exception as e:
        return seq_name, frame_idx, -1, f'err: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(DEFAULT_SRC))
    ap.add_argument('--recalib', default=str(DEFAULT_RECALIB))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--vehicle', default=VEHICLE_ID)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--max-yawrate', type=float, default=0.05,
                    help='|yawrate| threshold (rad/s); frames above are dropped')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--tile-w', type=int, default=512)
    ap.add_argument('--tile-h', type=int, default=512)
    ap.add_argument('--tile-stride', type=int, default=384)
    ap.add_argument('--tile-pad', type=int, default=128,
                    help='pixels of inter-tile pad kept on each tile (so points '
                         'just outside a tile are still in inst[\'pts\'])')
    ap.add_argument('--parent-pad', type=int, default=128,
                    help='pixels of OUTSIDE-parent pad kept by '
                         'project_lidar_into_image. Allows points that GT '
                         'calib projects just outside the parent image '
                         '(e.g. u<0 or u>=IW) to still be cached, so that '
                         'after a δ̂ correction they can re-enter visible tiles.')
    ap.add_argument('--y-frac-keep', type=float, default=0.70,
                    help='fraction of vertical extent kept around image centre')
    ap.add_argument('--tile-jpg-q', type=int, default=92)
    ap.add_argument('--max-frames-per-seq', type=int, default=None,
                    help='smoke knob; None = all frames')
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / 'inst').mkdir(parents=True, exist_ok=True)

    K, dist, T_cam_lidar, IW, IH = load_calib(Path(args.recalib), args.vehicle)
    print(f'calib: fx={K[0,0]:.2f} cc=({K[0,2]:.2f},{K[1,2]:.2f}) '
          f'k={dist.tolist()}  IW×IH={IW}×{IH}', flush=True)

    keep = float(args.y_frac_keep)
    margin = (1.0 - keep) / 2.0
    y_start = int(round(IH * margin))
    y_end = int(round(IH * (1.0 - margin)))
    print(f'Y-band: keep {keep*100:.0f}% → y∈[{y_start},{y_end}] '
          f'(IH={IH})', flush=True)

    tile_layout = (args.tile_w, args.tile_h, args.tile_stride, args.tile_pad,
                   y_start, y_end, args.tile_jpg_q)

    seqs = sorted([p for p in src.iterdir()
                   if p.is_dir() and p.name.startswith(f'sequence={args.vehicle}_')])
    print(f'sequences: {len(seqs)}', flush=True)

    tasks = []
    GID_PER_FRAME = 100
    gid_cursor = 0
    for seq in seqs:
        cam_files, lid_files = _list_seq_frames(seq)
        if args.max_frames_per_seq:
            cam_files = cam_files[:args.max_frames_per_seq]
            lid_files = lid_files[:args.max_frames_per_seq]
        for fi, (cp, lp) in enumerate(zip(cam_files, lid_files)):
            tasks.append((seq.name, fi, str(cp), str(lp),
                          K, dist, T_cam_lidar, IW, IH,
                          float(args.max_yawrate),
                          str(out), gid_cursor, tile_layout,
                          int(args.parent_pad)))
            gid_cursor += GID_PER_FRAME
    print(f'frames queued: {len(tasks)}', flush=True)

    t0 = time.time()
    n_kept = n_turn = n_few = n_err = 0
    written = 0
    if args.workers <= 1:
        for a in tasks:
            _, _, n, status = process_frame(a)
            if status == 'ok':
                n_kept += 1; written += n
            elif status == 'turning':
                n_turn += 1
            elif status == 'few-pts':
                n_few += 1
            else:
                n_err += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_frame, a): a for a in tasks}
            done = 0
            for fut in as_completed(futs):
                _, _, n, status = fut.result()
                if status == 'ok':
                    n_kept += 1; written += n
                elif status == 'turning':
                    n_turn += 1
                elif status == 'few-pts':
                    n_few += 1
                else:
                    n_err += 1
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f'  [{done}/{len(tasks)}] kept={n_kept} '
                          f'turn={n_turn} few={n_few} err={n_err} '
                          f'tiles={written}  ({time.time()-t0:.0f}s)',
                          flush=True)

    print(f'done: kept_frames={n_kept} (turn_drop={n_turn}, few_pts={n_few}, '
          f'err={n_err})  tiles={written}  ({time.time()-t0:.0f}s)', flush=True)

    # Scene-level train/val split: deterministic first val_frac sequences →
    # val. Tiles inherit via `scene` field.
    inst_dir = out / 'inst'
    val_seqs = set(s.name for s in seqs[:max(1, int(len(seqs) * args.val_frac))])
    train_files, val_files = [], []
    for f in sorted(p.name for p in inst_dir.glob('*.pt')):
        try:
            inst = torch.load(inst_dir / f, weights_only=False)
            if str(inst.get('scene', '')) in val_seqs:
                val_files.append(f)
            else:
                train_files.append(f)
        except Exception:
            train_files.append(f)
    meta = {'train': train_files, 'val': val_files,
            'cam': 'tss4_fcm', 'is_fisheye': True}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt: train={len(train_files)} val={len(val_files)} '
          f'(val_seqs={sorted(val_seqs)})', flush=True)


if __name__ == '__main__':
    main()
