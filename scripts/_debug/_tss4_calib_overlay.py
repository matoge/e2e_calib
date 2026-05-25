"""TSS4 (vehicle 248) slow-drive calib check.

Project vls128_rear_axle LiDAR onto tss4_fcm camera using **recalibration.json**
(loom's refined KB-fisheye calib, NOT setting-248.json which is stale) and
tile the overlay so we can see centre/periphery drift.

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/_debug/_tss4_calib_overlay.py

Outputs to docs/assets/2026-05-24_tss4_overlay/.

Calibration source = ~/git/loom/backend/assets/woven_sequence/llinking_26/
  recalibration.json [vehicle_id="248"].fcm  (matches loom woven_sequence.py
  get_camera_params + get_rtotal):
    kb.focal_length  : fx = fy
    kb.k1..k4        : KB distortion coefs (forward poly, 4 coef)
    cc               : principal point
    mp               : camera position in rear_axle frame [x,y,z]
    rot, rot_order=zyx: vehicle->camera (roll, pitch, yaw)
    resolution       : [3840,1952]

LiDAR npz vls128_rear_axle: xs/ys/zs already in rear_axle frame
(`coordinate_system='rear_axle'`).

Pipeline (mirrors loom/backend/woven_sequence.py:get_rtotal + project_points_fisheye):
  1. R_camera_to_vehicle = Rotation.from_euler('zyx', [yaw,pitch,roll])
  2. R_to_rdf = [[0,-1,0],[0,0,-1],[1,0,0]]   # cam->RDF (right-down-fwd)
  3. R_total = R_to_rdf @ inv(R_camera_to_vehicle)
     mp_total = -R_to_rdf @ mp
  4. p_cam = R_total @ p_rear_axle + mp_total      (z>0 = in front)
  5. KB project via cv2.fisheye.projectPoints(K=[[fx,0,cx],[0,fy,cy]], D=[k1,k2,k3,k4])
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SEQ_ROOT = Path('/mnt/ecp-perception/woven_sequence/adas-data_01/20230612_001946')
RECALIB_PATH = Path(
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_26/recalibration.json'
)
VEHICLE_ID = '248'
OUT_DIR = ROOT / 'docs/assets/2026-05-24_tss4_overlay'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TILES_X = 4
N_TILES_Y = 2
N_FRAMES = 0  # 0 = all frames in the sequence
DOT_SIZE = 1
WRITE_TILES = False  # full.jpg only — for sequential / video mode
# Sibling sequences to render. None = pick up ALL sequence=248_* under SEQ_ROOT
# (sorted by start timestamp), Empty list = first only.
SEQUENCES: list[str] | None = None
# Match loom/backend/project_points.py:create_projected_image's pre-filter
# (drops points behind the rear-axle origin by >10m).
REAR_X_CUT = -10.0


def load_recalib(vehicle_id: str = VEHICLE_ID) -> dict:
    return json.loads(RECALIB_PATH.read_text())[vehicle_id]


def build_K_D_RT(calib: dict):
    """Build K, D, R_rear2cam, t_rear2cam EXACTLY as loom/backend/project_points.py
    does (the loom reference projector that produces projected_images/*.png).

    Pipeline:
      fcm: K from kb.focal_length, dist=[kb.k1..k4], mp/rot define R_total/tvec
      R_total   = R_to_rdf @ inv(R_camera_to_vehicle('zyx', [yaw,pitch,roll]))
      tvec      = -R_to_rdf @ fcm.mp
      poslv (if present): adds an additional rear_axle->poslv rotation+offset:
        R_rear2cam = R_total @ R_poslv
        t_rear2cam = R_total @ poslv.mp + tvec
    """
    fcm = calib['fcm']
    kb = fcm['kb']
    fx = fy = float(kb['focal_length'])
    cx, cy = fcm['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.asarray([kb['k1'], kb['k2'], kb['k3'], kb['k4']], dtype=np.float64)
    IW, IH = fcm['resolution']

    mp_fcm = np.asarray(fcm['mp'], dtype=np.float64).reshape(3, 1)
    roll, pitch, yaw = fcm['rot']  # rot_order=zyx
    R_cam_to_veh = Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    R_to_rdf = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_total = R_to_rdf @ np.linalg.inv(R_cam_to_veh)
    tvec = (-R_to_rdf @ mp_fcm).flatten()

    poslv = calib.get('poslv')
    if poslv is not None:
        mp_p = np.asarray(poslv['mp'], dtype=np.float64)
        roll_p, pitch_p, yaw_p = poslv['rot']
        R_p = Rotation.from_euler('zyx', [yaw_p, pitch_p, roll_p]).as_matrix()
        R_rear2cam = R_total @ R_p
        t_rear2cam = (R_total @ mp_p) + tvec
    else:
        R_rear2cam = R_total
        t_rear2cam = tvec

    return K, D, R_rear2cam, t_rear2cam.reshape(3, 1), int(IW), int(IH)


def project(pts_w: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, D: np.ndarray):
    pts_cam = (R @ pts_w.T + t).T  # (N,3)
    z = pts_cam[:, 2]
    valid = z > 0.5
    if not valid.any():
        return np.empty((0, 2)), np.empty((0,)), np.array([], dtype=int)
    pcv = pts_cam[valid].reshape(-1, 1, 3).astype(np.float64)
    uv, _ = cv2.fisheye.projectPoints(pcv, np.zeros((3, 1)), np.zeros((3, 1)), K, D)
    uv = uv.reshape(-1, 2)
    return uv, z[valid], np.where(valid)[0]


def color_for_depth(z: float, zmax: float = 80.0) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, z / zmax))
    # near=red, mid=yellow, far=cyan
    if t < 0.5:
        s = t / 0.5
        return (255, int(255 * s), 0)
    s = (t - 0.5) / 0.5
    return (int(255 * (1 - s)), 255, int(255 * s))


def draw_dots(arr: np.ndarray, uv: np.ndarray, z: np.ndarray, size: int = DOT_SIZE):
    H, W, _ = arr.shape
    for (u, v), zv in zip(uv, z):
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < W and 0 <= iv < H):
            continue
        col = color_for_depth(float(zv))
        for du in range(-size, size + 1):
            for dv in range(-size, size + 1):
                x, y = iu + du, iv + dv
                if 0 <= x < W and 0 <= y < H:
                    arr[y, x] = col


def stamp_text(arr: np.ndarray, lines: list[str], anchor=(8, 8), font_size=18):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    line_h = font_size + 4
    box_w = int(max(font.getlength(t) for t in lines)) + pad * 2
    box_h = line_h * len(lines) + pad * 2
    x0, y0 = anchor
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    for i, t in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * line_h), t, font=font, fill=(255, 255, 255))
    arr[...] = np.asarray(img)


def make_tile_grid(arr: np.ndarray, n_tx: int, n_ty: int, color=(0, 255, 0), thick=3):
    H, W, _ = arr.shape
    for i in range(1, n_tx):
        x = i * W // n_tx
        arr[:, max(0, x - thick // 2): x + (thick + 1) // 2] = color
    for j in range(1, n_ty):
        y = j * H // n_ty
        arr[max(0, y - thick // 2): y + (thick + 1) // 2, :] = color


def crop_tiles(arr: np.ndarray, n_tx: int, n_ty: int) -> list[tuple[str, np.ndarray]]:
    H, W, _ = arr.shape
    tiles = []
    for j in range(n_ty):
        for i in range(n_tx):
            y0, y1 = j * H // n_ty, (j + 1) * H // n_ty
            x0, x1 = i * W // n_tx, (i + 1) * W // n_tx
            tile = arr[y0:y1, x0:x1].copy()
            tiles.append((f'tile_{j}_{i}', tile))
    return tiles


def render_one(seq_dir: Path, frame_idx: int, save_prefix: str, calib: dict):
    K, D, R, t, IW, IH = build_K_D_RT(calib)

    cam_dir = seq_dir / 'tss4_fcm'
    lid_dir = seq_dir / 'vls128_rear_axle'
    cam_files = sorted(cam_dir.glob('*.jpg'))
    lid_files = sorted(lid_dir.glob('*.npz'))
    if frame_idx >= len(cam_files):
        print(f'[{save_prefix}] frame_idx {frame_idx} >= {len(cam_files)}')
        return
    cam_path = cam_files[frame_idx]
    lid_path = lid_files[frame_idx]

    img = np.asarray(Image.open(cam_path).convert('RGB')).copy()
    if img.shape[:2] != (IH, IW):
        print(f'[{save_prefix}] WARN image {img.shape} != setting {IH}x{IW}')
    d = np.load(lid_path)
    pts_w_all = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
    speed = float(d['speed'])
    pre_mask = pts_w_all[:, 0] > REAR_X_CUT  # match loom pre-filter
    pts_w_sub = pts_w_all[pre_mask]
    uv, z, _ = project(pts_w_sub, R, t, K, D)
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    uv_in = uv[in_bounds]; z_in = z[in_bounds]
    print(f'[{save_prefix}] frame={frame_idx} cam={cam_path.name} lid_pts={len(pts_w_all)} '
          f'after_pre={len(pts_w_sub)}  projected_in_image={len(uv_in)}  speed={speed:.2f}m/s')

    full = img.copy()
    draw_dots(full, uv_in, z_in, size=DOT_SIZE)
    make_tile_grid(full, N_TILES_X, N_TILES_Y, color=(0, 255, 0), thick=3)
    stamp_text(full, [
        f'TSS4 (veh 248) slow drive  speed={speed:.2f} m/s',
        f'frame={frame_idx}  cam={cam_path.stem}',
        f'lidar pts in image: {len(uv_in)} / {len(pts_w_sub)} (pre-filt) / {len(pts_w_all)} raw',
        f'res {IW}x{IH}  fc={K[0,0]:.1f}  cc=({K[0,2]:.1f},{K[1,2]:.1f})',
        'red=near  yellow=mid  cyan=far  green=tile boundary',
    ], font_size=22)

    Image.fromarray(full).save(OUT_DIR / f'{save_prefix}_full.jpg', quality=88)
    if WRITE_TILES:
        tiles = crop_tiles(full, N_TILES_X, N_TILES_Y)
        for name, tile in tiles:
            Image.fromarray(tile).save(OUT_DIR / f'{save_prefix}_{name}.jpg', quality=92)
        tile_h, tile_w, _ = tiles[0][1].shape
        pad = 8
        canvas_h = N_TILES_Y * tile_h + (N_TILES_Y - 1) * pad
        canvas_w = N_TILES_X * tile_w + (N_TILES_X - 1) * pad
        canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
        for j in range(N_TILES_Y):
            for i in range(N_TILES_X):
                tile = tiles[j * N_TILES_X + i][1]
                y0 = j * (tile_h + pad)
                x0 = i * (tile_w + pad)
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        Image.fromarray(canvas).save(OUT_DIR / f'{save_prefix}_tiles_grid.jpg', quality=92)


def main():
    calib = load_recalib(VEHICLE_ID)
    fcm = calib['fcm']
    print(f'recalib veh={VEHICLE_ID}  kb.focal_length={fcm["kb"]["focal_length"]:.3f}  '
          f'cc={fcm["cc"]}  mp={fcm["mp"]}  rot(zyx)={fcm["rot"]}')

    if SEQUENCES is None:
        # ALL sibling sequence=248_* under SEQ_ROOT, sorted by start timestamp
        seq_dirs = sorted([p for p in SEQ_ROOT.iterdir()
                           if p.is_dir() and p.name.startswith(f'sequence={VEHICLE_ID}_')])
    elif not SEQUENCES:
        seq_dirs = sorted([p for p in SEQ_ROOT.iterdir()
                           if p.is_dir() and p.name.startswith('sequence=')])[:1]
    else:
        seq_dirs = [SEQ_ROOT / s for s in SEQUENCES]

    global_k = 0
    for s_idx, seq in enumerate(seq_dirs):
        if not seq.is_dir():
            print(f'[skip] {seq.name} missing', file=sys.stderr)
            continue
        cam_files = sorted((seq / 'tss4_fcm').glob('*.jpg'))
        print(f'\n=== seq{s_idx:02d} {seq.name}  ({len(cam_files)} frames) ===')
        if N_FRAMES <= 0 or N_FRAMES >= len(cam_files):
            idxs = list(range(len(cam_files)))
        else:
            idxs = np.linspace(0, len(cam_files) - 1, N_FRAMES).round().astype(int).tolist()
        for k, idx in enumerate(idxs):
            render_one(seq, idx,
                       save_prefix=f'seq{s_idx:02d}_frame_{global_k:04d}_idx{idx:03d}',
                       calib=calib)
            global_k += 1


if __name__ == '__main__':
    main()
