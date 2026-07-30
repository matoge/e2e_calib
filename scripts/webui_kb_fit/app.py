"""TSS4 KB4 fit WebUI.

Konva-based zoom/pan canvas:
  - background: 1 frame from sequence=248_*
  - overlay: LiDAR points projected with current K/D/R/t (color-by-depth)
  - left-click → drag → click: add user residual arrow (start=current projected
    pos, end=true pos)  [TODO]
  - right-click on arrow: delete                                 [TODO]
  - "Fit" button: GN on k1..k4 only, anchored by central 80%      [TODO]

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/webui_kb_fit/app.py
  → http://localhost:5005

Now (this file): backend serves /api/frame (jpeg b64 + projected uvz)
                 + /api/calib (current K, D, R, t, IW, IH).
                 Frontend = Konva canvas with image + dots, mouse-wheel zoom,
                 drag pan. No fit yet.
"""
from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from scipy.spatial.transform import Rotation

RECALIB_PATH = Path(
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_26/recalibration.json'
)
SEQ_ROOT = Path('/mnt/ecp-perception/woven_sequence/llinking_29/20230612_001946')
ITER1_INST = Path('/raid/home/hfunaya/cache_v5/tss4_v3_full_iter1/inst/00000000.pt')
VEHICLE_ID = '248'
REAR_X_CUT = -10.0  # match loom pre-filter

app = Flask(__name__)

# In-memory state. Mutated by /api/fit and /api/reset_kb.
_recalib = json.loads(RECALIB_PATH.read_text())


def _load_iter1_kb4() -> dict | None:
    """Load iter1 K/D from cache_v5 inst pt and synthesize an fcm dict
    compatible with the rest of the code.  Drops KB k5..k10 + tangential
    + dfx/dfy asymmetry: this UI fits KB4 only and the small extras would
    just sit unused.

    iter1 inst has T_gt=I, cam_pos=0  → world → cam transform is already
    baked into the LMDB; we still re-use loom's R_total/tvec from
    recalibration.json for projection (LiDAR comes in rear-axle).  So the
    only thing we adopt from iter1 is fx, cy, cx and KB k1..k4.
    """
    if not ITER1_INST.is_file():
        return None
    import torch
    inst = torch.load(ITER1_INST, weights_only=False)
    K = inst['K_full'].numpy()
    D = inst['distortion'].numpy()
    fx = 0.5 * (float(K[0, 0]) + float(K[1, 1]))   # symmetrize for KB radial
    cx = float(K[0, 2]); cy = float(K[1, 2])
    fcm = copy.deepcopy(_recalib[VEHICLE_ID]['fcm'])
    fcm['kb']['focal_length'] = fx
    fcm['kb']['k1'] = float(D[0])
    fcm['kb']['k2'] = float(D[1])
    fcm['kb']['k3'] = float(D[2])
    fcm['kb']['k4'] = float(D[3])
    fcm['cc'] = [cx, cy]
    return fcm


_iter1_fcm = None  # disable: use recalibration.json baseline only
print('[init] iter1 fcm available:', _iter1_fcm is not None)
if _iter1_fcm is not None:
    print(f'       fx={_iter1_fcm["kb"]["focal_length"]:.2f}  cc={_iter1_fcm["cc"]}')
    print(f'       k1..k4 = {[_iter1_fcm["kb"][f"k{i}"] for i in (1,2,3,4)]}')

_PERSIST_PATH = Path('/tmp/_webui_kb_fit_state.json')
_state: dict[str, Any] = {
    'fcm': copy.deepcopy(_iter1_fcm or _recalib[VEHICLE_ID]['fcm']),
    'poslv': copy.deepcopy(_recalib[VEHICLE_ID].get('poslv')),
}
# Init reference for /api/reset_kb:
_init_fcm = copy.deepcopy(_state['fcm'])
# Resume persisted fcm (rot/mp/kb) across server restarts.
if _PERSIST_PATH.is_file():
    try:
        saved = json.loads(_PERSIST_PATH.read_text())
        _state['fcm'].update(saved)
        print(f'[init] resumed _state from {_PERSIST_PATH}')
    except Exception as e:
        print(f'[init] could not resume {_PERSIST_PATH}: {e}')


def _persist():
    try:
        _PERSIST_PATH.write_text(json.dumps(_state['fcm']))
    except Exception as e:
        print(f'[persist] failed: {e}')
# Per-frame LiDAR cache: keep last loaded (seq, idx) → pts_w (rear-axle).
_lidar_cache: dict[tuple[str, int], np.ndarray] = {}


def _build_K_D_RT(fcm: dict, poslv: dict | None):
    kb = fcm['kb']
    fx = fy = float(kb['focal_length'])
    cx, cy = fcm['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.asarray([kb['k1'], kb['k2'], kb['k3'], kb['k4']], dtype=np.float64)
    IW, IH = fcm['resolution']

    mp_fcm = np.asarray(fcm['mp'], dtype=np.float64).reshape(3, 1)
    roll, pitch, yaw = fcm['rot']
    R_cam_to_veh = Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    R_to_rdf = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    R_total = R_to_rdf @ np.linalg.inv(R_cam_to_veh)
    tvec = (-R_to_rdf @ mp_fcm).flatten()

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


def _list_seqs() -> list[str]:
    if not SEQ_ROOT.is_dir():
        return []
    return sorted(p.name for p in SEQ_ROOT.iterdir()
                  if p.is_dir() and p.name.startswith(f'sequence={VEHICLE_ID}_'))


def _project(pts_w: np.ndarray, R: np.ndarray, t: np.ndarray,
             K: np.ndarray, D: np.ndarray):
    pts_cam = (R @ pts_w.T + t).T
    z = pts_cam[:, 2]
    valid = z > 0.5
    if not valid.any():
        return np.empty((0, 2)), np.empty((0,))
    pcv = pts_cam[valid].reshape(-1, 1, 3).astype(np.float64)
    uv, _ = cv2.fisheye.projectPoints(pcv, np.zeros((3, 1)), np.zeros((3, 1)), K, D)
    return uv.reshape(-1, 2), z[valid]


RAW_SEQ_ROOT = Path('/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/'
                    '20230612_001946')

# Woven canary sequences (each dir holds setting-<ipXXX>.json + tss4_fcm/).
CANARY_SEQ_ROOTS = [
    Path('/raid/home/hfunaya/woven_canary_local/canary_unilab/test01'),
]


def _canary_options() -> list[dict]:
    """Enumerate (vehicle, seq_dir, sample_jpg) for the crop_designer
    dropdown. Vehicle = ipXXX parsed from `sequence=<vehicle>-...`.
    """
    opts = []
    for root in CANARY_SEQ_ROOTS:
        if not root.is_dir():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not (seq_dir.is_dir() and seq_dir.name.startswith('sequence=')):
                continue
            head = seq_dir.name[len('sequence='):]
            vehicle = head.split('-')[0]           # 'ip607-lidar0-...' → 'ip607'
            setting = seq_dir / f'setting-{vehicle}.json'
            fcm_dir = seq_dir / 'tss4_fcm'
            if not (setting.is_file() and fcm_dir.is_dir()):
                continue
            jpgs = sorted(fcm_dir.glob('*.jpg'))
            if not jpgs:
                continue
            opts.append({
                'source': 'canary',
                'label': f'{vehicle} — {seq_dir.name[:60]}',
                'vehicle': vehicle,
                'seq_dir': str(seq_dir),
                'sample_jpg': str(jpgs[0]),
            })
    # TSS4 default (hard-coded in original /api/rect_image).
    tss4_jpg = (RAW_SEQ_ROOT / 'sequence=248_20230612_001946_'
                '1686533186104-1686533191007/tss4_fcm/0000_1686533186104.jpg')
    if tss4_jpg.is_file():
        opts.insert(0, {
            'source': 'tss4',
            'label': '248 (TSS4 raw / 20230612_001946)',
            'vehicle': '248',
            'seq_dir': '',
            'sample_jpg': str(tss4_jpg),
        })
    return opts


@app.route('/')
def index():
    seqs = _list_seqs()
    return render_template('index.html', seqs=seqs)


@app.route('/crop_designer')
def crop_designer():
    return render_template('crop_designer.html')


@app.route('/api/crop_options')
def api_crop_options():
    return jsonify({'options': _canary_options()})


@app.route('/api/rect_image')
def api_rect_image():
    """Return the balance=1.0 stretched rectified image as a downscaled
    JPEG b64 + the K_rect for the original resolution.

    Query params (optional):
      source   = 'canary' | 'tss4'     (default 'tss4' = backwards compat)
      vehicle  = 'ipXXX' or '247/248/249'
      seq_dir  = when source=canary, the sequence directory
    """
    import base64
    src = request.args.get('source', 'tss4')
    if src == 'canary':
        seq_dir = Path(request.args.get('seq_dir', ''))
        vehicle = request.args.get('vehicle', '')
        setting = seq_dir / f'setting-{vehicle}.json'
        if not setting.is_file():
            return jsonify({'error': f'no setting: {setting}'}), 404
        raw = json.loads(setting.read_text())
        entry = raw[0] if isinstance(raw, list) else raw
        fcm = entry['fcm']
        jpgs = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
        if not jpgs:
            return jsonify({'error': f'no jpgs under {seq_dir}/tss4_fcm'}), 404
        src_jpg = jpgs[0]
    else:
        vehicle = request.args.get('vehicle', '248')
        src_jpg = (RAW_SEQ_ROOT / 'sequence=248_20230612_001946_'
                   '1686533186104-1686533191007/tss4_fcm/0000_1686533186104.jpg')
        if not src_jpg.is_file():
            return jsonify({'error': f'no src jpg: {src_jpg}'}), 404
        fcm = json.loads(RECALIB_PATH.read_text())[vehicle]['fcm']
    fx = fy = float(fcm['kb']['focal_length'])
    cx0, cy0 = fcm['cc']
    W, H = int(fcm['resolution'][0]), int(fcm['resolution'][1])
    K = np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1]], dtype=np.float64)
    D = np.asarray([fcm['kb'][f'k{i}'] for i in (1, 2, 3, 4)],
                   dtype=np.float64).reshape(4, 1)
    # Fixed stretched canvas width; height scales with input aspect so the
    # KB rectify doesn't crop out FOV for tall (2160-line) canary jpgs.
    W_out = 10000
    H_out = int(round(W_out * H / W))
    K_b1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=1.0, fov_scale=1.0,
        new_size=(W_out, H_out))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_b1, (W_out, H_out), cv2.CV_16SC2)
    img = cv2.imread(str(src_jpg))
    rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    # Downscale for display so the browser doesn't choke on a 10000-wide image
    scale = 0.18
    prev = cv2.resize(rect, None, fx=scale, fy=scale)
    ok, buf = cv2.imencode('.jpg', prev, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jsonify({
        'img_b64': base64.b64encode(buf.tobytes()).decode('ascii'),
        'preview_W': prev.shape[1], 'preview_H': prev.shape[0],
        'orig_W': W_out, 'orig_H': H_out,
        'scale': scale,
        'K_rect': K_b1.tolist(),
        'K_rect_fx': float(K_b1[0, 0]),
        'K_rect_cx': float(K_b1[0, 2]),
        'K_rect_cy': float(K_b1[1, 2]),
    })


@app.route('/api/save_crops', methods=['POST'])
def api_save_crops():
    """Save user-drawn crop boxes (in original 10000x5083 coords) to a JSON
    file under _outputs/. Each box has: name, x0, y0, x1, y1, fx_scale.
    """
    payload = request.get_json(force=True)
    boxes = payload.get('boxes', [])
    out_dir = Path('/home/hfunaya/git/e2e_calib/scripts/webui_kb_fit/_outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / 'crop_boxes.json'
    out_p.write_text(json.dumps({'boxes': boxes}, indent=2))
    return jsonify({'wrote': str(out_p), 'n_boxes': len(boxes)})


@app.route('/lidar3d')
def lidar3d():
    seqs = sorted(p.name for p in RAW_SEQ_ROOT.iterdir()
                  if p.is_dir() and p.name.startswith('sequence=248_'))
    return render_template('lidar3d.html', seqs=seqs)


_DESKEW = None


def _get_deskew():
    """lazy import: vls128_azi_offsets / lidar_deskew utilities."""
    global _DESKEW
    if _DESKEW is None:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib.util as _u
        spec = _u.spec_from_file_location(
            'vls128_azi_offsets',
            str(Path(__file__).resolve().parent / 'vls128_azi_offsets.py'))
        m_az = _u.module_from_spec(spec); spec.loader.exec_module(m_az)
        spec = _u.spec_from_file_location(
            'lidar_deskew',
            str(Path(__file__).resolve().parent / 'lidar_deskew.py'))
        m_dk = _u.module_from_spec(spec); spec.loader.exec_module(m_dk)
        _DESKEW = (m_az, m_dk)
    return _DESKEW


@app.route('/api/lidar_raw')
def api_lidar_raw():
    """Return vls128 point cloud for one frame, optionally deskewed.

    Query:
      seq: sequence name under RAW_SEQ_ROOT
      idx: frame index (0..n-1)
      sub: subsample stride (default 1)
      deskew: 0|1 (default 0). When 1, points are returned in rear_axle
              frame at camera-capture time, motion-compensated using POSLV
              pose + per-laser AziOffset.
    """
    seq = request.args.get('seq', '')
    idx = int(request.args.get('idx', '0'))
    sub_step = int(request.args.get('sub', '1'))
    do_deskew = int(request.args.get('deskew', '0'))

    seq_dir = RAW_SEQ_ROOT / seq
    npz_files = sorted((seq_dir / 'vls128').glob('*.npz'))
    if not npz_files:
        return jsonify({'error': f'no vls128 npz under {seq_dir}'}), 404
    idx = max(0, min(idx, len(npz_files) - 1))
    z = np.load(npz_files[idx])
    xs = z['xs'].astype(np.float64)
    ys = z['ys'].astype(np.float64)
    zs = z['zs'].astype(np.float64)
    intensity = z['intensity']
    pt_off = z['point_time_offset_us'].astype(np.int32)
    layer = z['layer_id'].astype(np.int16)
    az_npz = z['azimuth'].astype(np.float64)
    sweep_end_s = float(z['timestamp_millisecond'])
    coord_sys = str(z['coordinate_system'])

    # azimuth-based per-point sweep time (used for both color and deskew).
    # Step 1: base time from npz azimuth (= disk angle as recorded).
    # Step 2: subtract per-laser AziOffset converted to time (= the laser
    #   beam fires AziOffset_rad / 2π * 100ms earlier/later than the disk
    #   angle; we subtract because a laser pointing +AziOffset ahead means
    #   the disk had to be that much behind when the beam reached its target).
    m_az, m_dk = _get_deskew()
    R_vls, mp_vls, R_poslv, mp_poslv, phaselock = m_dk.load_calib()
    SWEEP = m_dk.SWEEP_PERIOD_S
    sweep_start_s = sweep_end_s - SWEEP

    az_shift = np.mod((az_npz - phaselock), 2 * np.pi)            # base angle
    t_base_s = sweep_start_s + (az_shift / (2 * np.pi)) * SWEEP

    # AziOffset in radians → equivalent time delta in seconds
    rad_to_sec = SWEEP / (2 * np.pi)
    t_pt_az = t_base_s - m_az.AZI_OFFSET_RAD[layer] * rad_to_sec  # sec

    # offset within sweep, allowed to go slightly negative or > SWEEP because
    # of the per-laser correction (= the wrap region the user mentioned).
    pt_off_az = ((t_pt_az - sweep_start_s) * 1e6).astype(np.int32)

    if do_deskew:
        # Prefer 200Hz POS.parquet over 10Hz csv when available.
        pq = seq_dir / 'poslv' / 'POS.parquet'
        csv = seq_dir / 'poslv' / 'poslv.csv'
        ts_a, e, n, u, roll, pitch, head = m_dk.load_poslv_poses(pq if pq.is_file() else csv)
        t_pt = t_pt_az
        cam_unix_ms = float(npz_files[idx].stem.split('_')[1])
        t_cam = cam_unix_ms / 1000.0
        # sensor → rear_axle (per-point but no time-warp yet)
        p_s = np.stack([xs, ys, zs], axis=-1)
        p_rear_pt = m_dk.sensor_to_rear_axle(p_s, R_vls, mp_vls, R_poslv, mp_poslv)
        # warp via world poses
        T_pt  = m_dk.interp_pose(t_pt, ts_a, e, n, u, roll, pitch, head)
        T_cam = m_dk.interp_pose(np.asarray([t_cam]), ts_a, e, n, u, roll, pitch, head)[0]
        ones = np.ones((len(p_rear_pt), 1))
        ph = np.concatenate([p_rear_pt, ones], axis=-1)
        p_world = np.einsum('mij,mj->mi', T_pt, ph)
        T_cam_inv = np.linalg.inv(T_cam)
        p_rear_cam = (T_cam_inv @ p_world.T).T[:, :3]
        xs = p_rear_cam[:, 0]; ys = p_rear_cam[:, 1]; zs = p_rear_cam[:, 2]
        coord_sys = 'rear_axle (deskewed @ t_cam)'
        # also report displacement statistics
        disp = np.linalg.norm(p_rear_cam - p_rear_pt, axis=1)
        disp_stats = {'max_m': float(disp.max()),
                      'mean_m': float(disp.mean()),
                      'p99_m': float(np.percentile(disp, 99))}
    else:
        # NO deskew, but still convert to rear_axle so the two views share frames.
        p_s = np.stack([xs, ys, zs], axis=-1)
        p_rear = m_dk.sensor_to_rear_axle(p_s, R_vls, mp_vls, R_poslv, mp_poslv)
        xs = p_rear[:, 0]; ys = p_rear[:, 1]; zs = p_rear[:, 2]
        coord_sys = 'rear_axle (no deskew)'
        disp_stats = None

    # use azimuth-based per-point time (correct) instead of the fake
    # within-layer-index pt_off in the npz.
    pt_off = pt_off_az
    # raw sensor-frame azimuth (= arctan2(y_sensor, x_sensor)) with the
    # per-laser AziOffset BAKED-IN, exactly as recorded in the npz.
    az_raw_send = az_npz.astype(np.float32)
    if sub_step > 1:
        xs = xs[::sub_step]; ys = ys[::sub_step]; zs = zs[::sub_step]
        intensity = intensity[::sub_step]
        pt_off = pt_off[::sub_step]
        layer = layer[::sub_step]
        az_raw_send = az_raw_send[::sub_step]

    return jsonify({
        'seq': seq, 'idx': idx, 'n_frames': len(npz_files),
        'cam_name': npz_files[idx].name,
        'coordinate_system': coord_sys,
        'deskewed': bool(do_deskew),
        'deskew_stats': disp_stats,
        'sweep_end_ts_sec': sweep_end_s,
        'speed_mps':  float(z['speed']),
        'yawrate_rps': float(z['yawrate']),
        'n_points_total': int(len(z['xs'])),
        'n_points_returned': int(len(xs)),
        'xs': xs.astype(np.float32).tolist(),
        'ys': ys.astype(np.float32).tolist(),
        'zs': zs.astype(np.float32).tolist(),
        'intensity': intensity.astype(np.float32).tolist(),
        'pt_off_us': pt_off.tolist(),
        'layer': layer.tolist(),
        'az_raw': az_raw_send.tolist(),
    })


@app.route('/api/calib')
def api_calib():
    fcm = _state['fcm']
    rot = list(fcm.get('rot', [0.0, 0.0, 0.0]))
    while len(rot) < 3:
        rot.append(0.0)
    return jsonify({
        'fx': float(fcm['kb']['focal_length']),
        'cx': float(fcm['cc'][0]),
        'cy': float(fcm['cc'][1]),
        'k1': float(fcm['kb']['k1']),
        'k2': float(fcm['kb']['k2']),
        'k3': float(fcm['kb']['k3']),
        'k4': float(fcm['kb']['k4']),
        # rad は内部用、UI 表示は deg。
        'roll_rad':  float(rot[0]),
        'pitch_rad': float(rot[1]),
        'yaw_rad':   float(rot[2]),
        'roll':  float(np.rad2deg(rot[0])),
        'pitch': float(np.rad2deg(rot[1])),
        'yaw':   float(np.rad2deg(rot[2])),
        'mp_x_m': float(fcm.get('mp', [0,0,0])[0]),
        'mp_y_m': float(fcm.get('mp', [0,0,0])[1]),
        'mp_z_m': float(fcm.get('mp', [0,0,0])[2]),
        'IW': int(fcm['resolution'][0]),
        'IH': int(fcm['resolution'][1]),
    })


@app.route('/api/frame')
def api_frame():
    seq = request.args.get('seq', '')
    idx = int(request.args.get('idx', '0'))
    seq_dir = SEQ_ROOT / seq
    cam_files = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
    lid_files = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
    if not cam_files:
        return jsonify({'error': f'no cam files under {seq_dir}'}), 404
    idx = max(0, min(idx, len(cam_files) - 1))
    cam_p = cam_files[idx]
    lid_p = lid_files[idx]

    img = Image.open(cam_p).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    d = np.load(lid_p)
    pts_w_all = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
    pts_w = pts_w_all[pts_w_all[:, 0] > REAR_X_CUT]
    _lidar_cache[(seq, idx)] = pts_w
    K, D, R, t, IW, IH = _build_K_D_RT(_state['fcm'], _state['poslv'])
    uv, z = _project(pts_w, R, t, K, D)
    # 画像 + 外周 10% に LiDAR を残す。folded fisheye 領域に矢印を引けるように。
    pad_u = 0.10 * IW; pad_v = 0.10 * IH
    in_b = ((uv[:, 0] >= -pad_u) & (uv[:, 0] < IW + pad_u) &
            (uv[:, 1] >= -pad_v) & (uv[:, 1] < IH + pad_v))
    uv = uv[in_b]; z = z[in_b]
    return jsonify({
        'img_b64': img_b64,
        'IW': int(IW), 'IH': int(IH),
        'pad_u': float(pad_u), 'pad_v': float(pad_v),
        'n_frames': len(cam_files),
        'cam_name': cam_p.name,
        'uv': uv.round(2).tolist(),
        'z':  z.round(3).tolist(),
    })


@app.route('/api/reproject')
def api_reproject():
    """Re-project the LiDAR points of (seq, idx) under current K/D/R/t.
    Falls back to disk read if not in cache (e.g. after server restart)."""
    seq = request.args.get('seq', '')
    idx = int(request.args.get('idx', '0'))
    pts_w = _lidar_cache.get((seq, idx))
    if pts_w is None:
        seq_dir = SEQ_ROOT / seq
        lid_files = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
        if not lid_files:
            return jsonify({'error': f'no lidar files under {seq_dir}'}), 404
        idx = max(0, min(idx, len(lid_files) - 1))
        d = np.load(lid_files[idx])
        pts_w_all = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
        pts_w = pts_w_all[pts_w_all[:, 0] > REAR_X_CUT]
        _lidar_cache[(seq, idx)] = pts_w
    K, D, R, t, IW, IH = _build_K_D_RT(_state['fcm'], _state['poslv'])
    uv, z = _project(pts_w, R, t, K, D)
    pad_u = 0.10 * IW; pad_v = 0.10 * IH
    in_b = ((uv[:, 0] >= -pad_u) & (uv[:, 0] < IW + pad_u) &
            (uv[:, 1] >= -pad_v) & (uv[:, 1] < IH + pad_v))
    uv = uv[in_b]; z = z[in_b]
    return jsonify({'uv': uv.round(2).tolist(), 'z': z.round(3).tolist()})


def _kb_distort(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    t2 = theta * theta
    return theta * (1 + k[0]*t2 + k[1]*t2**2 + k[2]*t2**3 + k[3]*t2**4)


def _project_kb(uv_n: np.ndarray, fx: float, cx: float, cy: float,
                k: np.ndarray) -> np.ndarray:
    """Project normalized image-plane (X/Z, Y/Z) → pixel via KB.
    uv_n: (N,2). Returns (N,2) pixel coords."""
    x, y = uv_n[:, 0], uv_n[:, 1]
    r = np.sqrt(x*x + y*y) + 1e-12
    theta = np.arctan(r)
    td = _kb_distort(theta, k)
    scale = td / r
    return np.stack([fx * scale * x + cx, fx * scale * y + cy], axis=-1)


@app.route('/api/fit_kb', methods=['POST'])
def api_fit_kb():
    """GN fit on k1..k4 only.

    Targets:
      - user arrows: (u_curr, v_curr) → (u_true, v_true), residual = current
        projected pixel - target pixel for the SAME 3-D ray (we recover the
        ray from u_curr by inverse-projecting through current K/D, which gives
        the (X/Z, Y/Z) direction; then we want the new K/D to land that
        direction at u_true).
      - anchors: uniform grid in central 80%; for each anchor we sample the
        same ray (X/Z, Y/Z) at u_anchor, and require new K/D to keep it at
        u_anchor (residual = 0).

    Decision variable: dk = (dk1..dk4). Update k ← k + dk.
    """
    payload = request.get_json(force=True)
    arrows = payload.get('arrows', [])  # [{u0,v0,u1,v1}]
    # Anchor placement: θ-uniform on [0, θ_anchor_max_deg]; n_anchor_r radii;
    # n_anchor_phi azimuths.  Replaces the old rectangular grid.
    n_anchor_r   = int(payload.get('n_anchor_r', 6))
    n_anchor_phi = int(payload.get('n_anchor_phi', 16))
    theta_anchor_max_deg = float(payload.get('theta_anchor_max_deg', 50.0))
    n_iter = int(payload.get('n_iter', 8))
    lam = float(payload.get('lam', 1e-6))
    # Soft monotonicity:  dt_d/dθ > 0 on [0, θ_max] with weight w_mono.
    w_mono = float(payload.get('w_mono', 1e3))
    n_mono = int(payload.get('n_mono', 80))
    theta_max_deg = float(payload.get('theta_max_deg', 65.0))
    # Stay-near-current ridge on (k - k_init) (k_init = recalibration.json).
    w_ridge = float(payload.get('w_ridge', 50.0))

    fcm = _state['fcm']; kb = fcm['kb']
    fx = float(kb['focal_length']); cx, cy = float(fcm['cc'][0]), float(fcm['cc'][1])
    k = np.asarray([kb['k1'], kb['k2'], kb['k3'], kb['k4']], dtype=np.float64)
    IW, IH = int(fcm['resolution'][0]), int(fcm['resolution'][1])

    # Build anchor in θ-uniform circles (radial fisheye).  For each θ_i the
    # corresponding pixel radius is r_i = fx · t_d(θ_i) under CURRENT k; we
    # then pick n_phi azimuth samples around the principal point.
    theta_anchor = np.linspace(
        np.deg2rad(2.0),  # skip θ=0 (degenerate)
        np.deg2rad(theta_anchor_max_deg),
        n_anchor_r,
    )
    phi_anchor = np.linspace(0, 2*np.pi, n_anchor_phi, endpoint=False)
    Th, Ph = np.meshgrid(theta_anchor, phi_anchor)
    Th = Th.ravel(); Ph = Ph.ravel()
    td_now = _kb_distort(Th, k)
    r_pix  = fx * td_now
    Ua = cx + r_pix * np.cos(Ph)
    Va = cy + r_pix * np.sin(Ph)
    in_b = (Ua >= 0) & (Ua < IW) & (Va >= 0) & (Va < IH)
    anchors = np.stack([Ua[in_b], Va[in_b]], axis=-1)

    # Each "observation" needs: ray (X/Z, Y/Z) AND target pixel.
    # For arrows:  ray ← inverse(u0, v0, current k);  target ← (u1, v1)
    # For anchors: ray ← inverse(u, v, current k);    target ← (u, v)
    def rays_from_uv(uv: np.ndarray, k_curr: np.ndarray) -> np.ndarray:
        # cv2.fisheye.undistortPoints needs (N,1,2) float64.
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
        D = k_curr.reshape(1, 4).astype(np.float64)
        rays = cv2.fisheye.undistortPoints(
            uv.reshape(-1, 1, 2).astype(np.float64), K, D
        ).reshape(-1, 2)
        return rays  # (N, 2) = (X/Z, Y/Z)

    # Stack obs
    obs_list = []
    if arrows:
        a = np.asarray([[a['u0'], a['v0'], a['u1'], a['v1']] for a in arrows],
                       dtype=np.float64)
        obs_list.append((rays_from_uv(a[:, :2], k), a[:, 2:4],
                         np.full(len(a), 1.0)))  # weight 1.0
    obs_list.append((rays_from_uv(anchors, k), anchors.copy(),
                     np.full(len(anchors), 0.5)))  # anchors lighter (×0.5)

    rays = np.concatenate([o[0] for o in obs_list], axis=0)
    targets = np.concatenate([o[1] for o in obs_list], axis=0)
    weights = np.concatenate([o[2] for o in obs_list], axis=0)

    # Init reference (for ridge term) = whatever Reset KB returns to.
    k_init = np.asarray([
        float(_init_fcm['kb'][f'k{i}']) for i in (1, 2, 3, 4)
    ], dtype=np.float64)

    # Monotonicity sample grid (constant across iters).
    theta_mono = np.linspace(0.05, np.deg2rad(theta_max_deg), n_mono)
    tm2 = theta_mono**2; tm4 = tm2*tm2; tm6 = tm4*tm2; tm8 = tm4*tm4
    # dtd/dθ at θ_mono (linear in k):  1 + 3k1 θ² + 5k2 θ⁴ + 7k3 θ⁶ + 9k4 θ⁸
    A_mono = np.stack([3*tm2, 5*tm4, 7*tm6, 9*tm8], axis=-1)  # (n_mono, 4)
    b_mono = np.ones_like(theta_mono)                         # constant 1

    history = []
    for it in range(n_iter):
        # Forward + analytic Jacobian wrt k = [k1..k4]
        x, y = rays[:, 0], rays[:, 1]
        r = np.sqrt(x*x + y*y) + 1e-12
        theta = np.arctan(r)
        t2 = theta * theta
        t3 = theta * t2; t5 = t3*t2; t7 = t5*t2; t9 = t7*t2
        td = theta + k[0]*t3 + k[1]*t5 + k[2]*t7 + k[3]*t9
        scale = td / r
        u_pred = fx * scale * x + cx
        v_pred = fx * scale * y + cy
        # d(td)/dk_i = θ^(2i+1)
        dtd = np.stack([t3, t5, t7, t9], axis=-1)        # (N, 4)
        # d(scale)/dk = dtd / r
        dscale = dtd / r[:, None]
        Ju = fx * x[:, None] * dscale                    # (N, 4)
        Jv = fx * y[:, None] * dscale                    # (N, 4)
        # Residual = pred - target
        ru = u_pred - targets[:, 0]
        rv = v_pred - targets[:, 1]
        # Stack 2N×4
        J = np.concatenate([Ju, Jv], axis=0)
        r_ = np.concatenate([ru, rv], axis=0)
        w = np.concatenate([weights, weights], axis=0)
        H = J.T @ (w[:, None] * J)
        g = J.T @ (w * r_)

        # Soft monotonicity penalty (one-sided): only active where slope ≤ 0.
        # slope(θ_i) = b + A·k.  We want slope > 0; add residual = slope when
        # slope < 0 with weight w_mono.
        slope = b_mono + A_mono @ k                         # (n_mono,)
        active = slope < 0.0
        if active.any():
            J_m = A_mono[active]                            # (n_a, 4)
            r_m = slope[active]                             # (n_a,)
            H += w_mono * (J_m.T @ J_m)
            g += w_mono * (J_m.T @ r_m)

        # Ridge toward k_init.
        H += w_ridge * np.eye(4)
        g += w_ridge * (k - k_init)

        H += lam * np.eye(4)
        try:
            dk = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        # Backtracking: never accept a step that creates a non-monotone region.
        accepted = False
        for bt in range(8):
            k_try = k + dk
            slope_try = b_mono + A_mono @ k_try
            if (slope_try > 0).all():
                k = k_try; accepted = True; break
            dk = dk * 0.5
        if not accepted:
            break

        rms = float(np.sqrt(np.mean((w * r_)**2)))
        n_violate = int(((b_mono + A_mono @ k) <= 0).sum())
        history.append({'iter': it, 'rms': rms,
                        'k': k.tolist(),
                        'step_norm': float(np.linalg.norm(dk)),
                        'slope_min': float((b_mono + A_mono @ k).min()),
                        'n_violate': n_violate})
        if np.linalg.norm(dk) < 1e-9:
            break

    # commit
    fcm['kb']['k1'], fcm['kb']['k2'], fcm['kb']['k3'], fcm['kb']['k4'] = \
        float(k[0]), float(k[1]), float(k[2]), float(k[3])
    _persist()
    return jsonify({'k': k.tolist(), 'history': history,
                    'n_arrows': len(arrows), 'n_anchors': int(anchors.shape[0])})


@app.route('/api/nudge_rot', methods=['POST'])
def api_nudge_rot():
    """fcm['rot'] = [roll, pitch, yaw] (deg, zyx-Euler) を ±delta_deg ずらす。
    payload: {axis: 'yaw'|'pitch'|'roll', delta_deg: float}
    Returns updated rot list and current K (unchanged here, only extrinsic)."""
    payload = request.get_json(force=True)
    axis = payload.get('axis')
    delta = float(payload.get('delta_deg', 0.0))
    if axis not in ('roll', 'pitch', 'yaw'):
        return jsonify({'error': f'bad axis {axis}'}), 400
    rot = list(_state['fcm'].get('rot', [0.0, 0.0, 0.0]))
    while len(rot) < 3:
        rot.append(0.0)
    idx_map = {'roll': 0, 'pitch': 1, 'yaw': 2}
    # fcm['rot'] は radian、ボタン入力は deg なので変換してから加算。
    rot[idx_map[axis]] = float(rot[idx_map[axis]]) + np.deg2rad(delta)
    _state['fcm']['rot'] = rot
    _persist()
    return jsonify({'rot': rot, 'axis': axis, 'delta_deg': delta,
                    'rot_deg': [float(np.rad2deg(r)) for r in rot]})


@app.route('/api/nudge_mp', methods=['POST'])
def api_nudge_mp():
    """fcm['mp'] = [x, y, z] (m) を ±delta_m ずらす。
    payload: {axis: 'x'|'y'|'z', delta_m: float}"""
    payload = request.get_json(force=True)
    axis = payload.get('axis')
    delta = float(payload.get('delta_m', 0.0))
    if axis not in ('x', 'y', 'z'):
        return jsonify({'error': f'bad axis {axis}'}), 400
    mp = list(_state['fcm'].get('mp', [0.0, 0.0, 0.0]))
    while len(mp) < 3:
        mp.append(0.0)
    idx_map = {'x': 0, 'y': 1, 'z': 2}
    mp[idx_map[axis]] = float(mp[idx_map[axis]]) + delta
    _state['fcm']['mp'] = mp
    _persist()
    return jsonify({'mp': mp, 'axis': axis, 'delta_m': delta})


@app.route('/api/reset_kb', methods=['POST'])
def api_reset_kb():
    """Reset to iter1 (preferred) else recalibration.json."""
    _state['fcm'] = copy.deepcopy(_init_fcm)
    _persist()
    src = _state['fcm']['kb']
    return jsonify({
        'k': [float(src['k1']), float(src['k2']),
              float(src['k3']), float(src['k4'])],
        'fx': float(src['focal_length']),
        'cc': list(_state['fcm']['cc']),
    })


def _render_overlay_jpg(seq: str, idx: int, fcm: dict, poslv: dict | None) -> bytes:
    """投影 overlay を JPEG bytes で返す。/api/save の evidence 用。"""
    seq_dir = SEQ_ROOT / seq
    cam_files = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
    lid_files = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
    idx = max(0, min(idx, len(cam_files) - 1))
    img = np.asarray(Image.open(cam_files[idx]).convert('RGB')).copy()
    pts_w = _lidar_cache.get((seq, idx))
    if pts_w is None:
        d = np.load(lid_files[idx])
        pts_w = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
        pts_w = pts_w[pts_w[:, 0] > REAR_X_CUT]
    K, D, R, t, IW, IH = _build_K_D_RT(fcm, poslv)
    uv, z = _project(pts_w, R, t, K, D)
    in_b = (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    uv = uv[in_b]; z = z[in_b]
    over = img.astype(np.float32); a = 0.55
    for (u, v), zv in zip(uv, z):
        iu, iv = int(round(u)), int(round(v))
        zn = max(0.0, min(1.0, float(zv) / 80.0))
        if zn < 0.5:
            s = zn / 0.5; c = np.array([255, 255*s, 0], dtype=np.float32)
        else:
            s = (zn - 0.5) / 0.5; c = np.array([255*(1-s), 255, 255*s], dtype=np.float32)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                x, y = iu+du, iv+dv
                if 0 <= x < IW and 0 <= y < IH:
                    over[y, x] = (1-a)*over[y, x] + a*c
    out = np.clip(over, 0, 255).astype(np.uint8)
    pil = Image.fromarray(out)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


@app.route('/api/save', methods=['POST'])
def api_save():
    """Save current fit as a ClearML task with evidence (overlay before/after,
    arrows, K/D before/after, why)."""
    payload = request.get_json(force=True)
    seq = payload.get('seq', '')
    idx = int(payload.get('idx', 0))
    arrows = payload.get('arrows', [])
    fit_params = payload.get('fit_params', {})
    why = (payload.get('why') or '').strip()
    if len(why) < 50:
        return jsonify({'error': 'why must be ≥ 50 chars'}), 400

    fcm_after = copy.deepcopy(_state['fcm'])
    fcm_before = copy.deepcopy(_init_fcm)
    src_a = fcm_after['kb']; src_b = fcm_before['kb']
    cc_a = fcm_after['cc']; cc_b = fcm_before['cc']
    rot_a = list(fcm_after.get('rot',  [0.0, 0.0, 0.0]))
    rot_b = list(fcm_before.get('rot', [0.0, 0.0, 0.0]))
    while len(rot_a) < 3: rot_a.append(0.0)
    while len(rot_b) < 3: rot_b.append(0.0)
    mp_a = list(fcm_after.get('mp',  [0.0, 0.0, 0.0]))
    mp_b = list(fcm_before.get('mp', [0.0, 0.0, 0.0]))
    while len(mp_a) < 3: mp_a.append(0.0)
    while len(mp_b) < 3: mp_b.append(0.0)
    delta = {
        'd_fx': float(src_a['focal_length']) - float(src_b['focal_length']),
        'd_cx': float(cc_a[0]) - float(cc_b[0]),
        'd_cy': float(cc_a[1]) - float(cc_b[1]),
        'd_k1': float(src_a['k1']) - float(src_b['k1']),
        'd_k2': float(src_a['k2']) - float(src_b['k2']),
        'd_k3': float(src_a['k3']) - float(src_b['k3']),
        'd_k4': float(src_a['k4']) - float(src_b['k4']),
        'd_roll_deg':  float(np.rad2deg(rot_a[0] - rot_b[0])),
        'd_pitch_deg': float(np.rad2deg(rot_a[1] - rot_b[1])),
        'd_yaw_deg':   float(np.rad2deg(rot_a[2] - rot_b[2])),
        'd_mp_x_cm': float((mp_a[0] - mp_b[0]) * 100.0),
        'd_mp_y_cm': float((mp_a[1] - mp_b[1]) * 100.0),
        'd_mp_z_cm': float((mp_a[2] - mp_b[2]) * 100.0),
    }

    # ── ClearML Task ──
    try:
        from clearml import Task
    except Exception as e:
        return jsonify({'error': f'clearml not installed: {e}'}), 500

    import time as _time
    ts = _time.strftime('%Y%m%d_%H%M%S')
    task_name = f'manual_fit_{seq[:40]}_f{idx:03d}_{ts}'
    task = Task.init(
        project_name='e2e_calib/calib_manual_fit',
        task_name=task_name,
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.set_comment(why)
    task.connect({
        'seq': seq, 'frame_idx': idx, 'n_arrows': len(arrows),
        'fit_params': fit_params,
        'baseline': {
            'fx': src_b['focal_length'], 'cx': cc_b[0], 'cy': cc_b[1],
            'k1': src_b['k1'], 'k2': src_b['k2'],
            'k3': src_b['k3'], 'k4': src_b['k4'],
            'roll_deg':  float(np.rad2deg(rot_b[0])),
            'pitch_deg': float(np.rad2deg(rot_b[1])),
            'yaw_deg':   float(np.rad2deg(rot_b[2])),
            'mp_x_m': float(mp_b[0]),
            'mp_y_m': float(mp_b[1]),
            'mp_z_m': float(mp_b[2]),
        },
        'after': {
            'fx': src_a['focal_length'], 'cx': cc_a[0], 'cy': cc_a[1],
            'k1': src_a['k1'], 'k2': src_a['k2'],
            'k3': src_a['k3'], 'k4': src_a['k4'],
            'roll_deg':  float(np.rad2deg(rot_a[0])),
            'pitch_deg': float(np.rad2deg(rot_a[1])),
            'yaw_deg':   float(np.rad2deg(rot_a[2])),
            'mp_x_m': float(mp_a[0]),
            'mp_y_m': float(mp_a[1]),
            'mp_z_m': float(mp_a[2]),
        },
        'delta': delta,
    })
    task.upload_artifact('arrows', arrows)
    task.upload_artifact('recalibration_after_fcm', fcm_after)

    out_dir = Path('/home/hfunaya/git/e2e_calib/scripts/webui_kb_fit/_outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    before_path = out_dir / f'{task_name}_before.jpg'
    after_path = out_dir / f'{task_name}_after.jpg'
    before_path.write_bytes(_render_overlay_jpg(seq, idx, fcm_before, _state.get('poslv')))
    after_path.write_bytes(_render_overlay_jpg(seq, idx, fcm_after, _state.get('poslv')))
    task.upload_artifact('overlay_before', str(before_path))
    task.upload_artifact('overlay_after', str(after_path))

    task.close()
    return jsonify({
        'task_id': task.id, 'task_name': task_name,
        'task_url': f'http://172.16.200.185:8082/projects/*/experiments/{task.id}',
        'before_jpg': str(before_path), 'after_jpg': str(after_path),
        'delta': delta,
    })


@app.route('/api/cml_history')
def api_cml_history():
    """List recent ClearML manual_fit tasks (newest first), with key params."""
    try:
        from clearml import Task
    except Exception as e:
        return jsonify({'error': f'clearml not installed: {e}'}), 500
    n = int(request.args.get('n', 30))
    kind = request.args.get('kind', 'all')   # 'manual' | 'gn' | 'all'
    tasks = Task.get_tasks(
        project_name='e2e_calib/calib_manual_fit',
        task_filter={'order_by': ['-last_update']},
    )
    def _is_kind(t):
        is_manual = t.name.startswith('manual_fit_')
        is_gn = (t.name == 'gn_12dof_perpoint'
                 or t.name.startswith('gn_12dof'))
        if kind == 'manual':
            return is_manual
        if kind == 'gn':
            return is_gn
        return is_manual or is_gn
    tasks = [t for t in tasks if _is_kind(t)]
    tasks = tasks[:n * 3]
    out = []
    for t in tasks:
        try:
            params = t.get_parameters_as_dict() or {}
            after = params.get('General', {}).get('after', params.get('after', {}))
            delta = params.get('General', {}).get('delta', params.get('delta', {}))
            seq = params.get('General', {}).get('seq', params.get('seq', ''))
            frame_idx = params.get('General', {}).get('frame_idx', params.get('frame_idx', 0))
            arts = t.artifacts or {}
            has_fcm = 'recalibration_after_fcm' in arts
            has_gn = 'gn_12dof_result_json' in arts
            if not (has_fcm or has_gn):
                continue
            if len(out) >= n:
                break
            out.append({
                'task_id': t.id,
                'task_name': t.name,
                'kind': 'gn' if has_gn else ('manual' if has_fcm else 'other'),
                'last_update': str(t.data.last_update) if t.data and t.data.last_update else '',
                'comment': (t.comment or '')[:200],
                'seq': seq,
                'frame_idx': frame_idx,
                'after_fx': after.get('fx') if isinstance(after, dict) else None,
                'after_cx': after.get('cx') if isinstance(after, dict) else None,
                'after_cy': after.get('cy') if isinstance(after, dict) else None,
                'has_fcm_artifact': has_fcm,
                'has_gn_artifact': has_gn,
            })
        except Exception as ex:
            out.append({'task_id': t.id, 'error': str(ex)})
    return jsonify({'tasks': out})


@app.route('/api/cml_load/<task_id>', methods=['POST'])
def api_cml_load(task_id):
    """Load `recalibration_after_fcm` artifact from a ClearML task into _state."""
    try:
        from clearml import Task
    except Exception as e:
        return jsonify({'error': f'clearml not installed: {e}'}), 500
    try:
        t = Task.get_task(task_id=task_id)
    except Exception as e:
        return jsonify({'error': f'task not found: {e}'}), 404
    arts = t.artifacts or {}
    if 'recalibration_after_fcm' in arts:
        fcm = arts['recalibration_after_fcm'].get()
        if isinstance(fcm, str):
            fcm = json.loads(fcm)
        _state['fcm'] = fcm
        kind = 'manual'
    elif 'gn_12dof_result_json' in arts:
        gn = arts['gn_12dof_result_json'].get()
        if isinstance(gn, str):
            gn = json.loads(gn)
        # Apply GN delta on top of CURRENT _state['fcm'] (R_fcm @ mp_fcm convention,
        # mirrors scripts/_debug/_bake_gn_into_recalib.py).
        import numpy as _np
        from scipy.spatial.transform import Rotation as _R
        fcm = copy.deepcopy(_state['fcm'])
        omega = _np.asarray(gn['omega_rad'], dtype=_np.float64)
        dt = _np.asarray(gn['dt_m'], dtype=_np.float64)
        df = float(gn['df']); dc = float(gn['dc'])
        dk = _np.asarray(gn['dk'], dtype=_np.float64)
        fl0 = float(fcm['kb']['focal_length'])
        cx0, cy0 = fcm['cc']
        k0 = _np.array([fcm['kb'][f'k{i}'] for i in (1, 2, 3, 4)],
                        dtype=_np.float64)
        mp0 = _np.asarray(fcm['mp'], dtype=_np.float64)
        roll0, pitch0, yaw0 = fcm['rot']
        R_cv0 = _R.from_euler('zyx', [yaw0, pitch0, roll0]).as_matrix()
        R_to_rdf = _np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]],
                               dtype=_np.float64)
        R_extra = _R.from_rotvec(omega).as_matrix()
        R_cv_new = R_cv0 @ R_to_rdf.T @ R_extra.T @ R_to_rdf
        yaw_n, pitch_n, roll_n = _R.from_matrix(R_cv_new).as_euler('zyx')
        fcm['rot'] = [float(roll_n), float(pitch_n), float(yaw_n)]
        mp_new = R_to_rdf.T @ R_extra @ R_to_rdf @ mp0 - R_to_rdf.T @ dt
        fcm['mp'] = mp_new.tolist()
        fcm['kb']['focal_length'] = fl0 + df
        if 'kb3' in fcm and 'focal_length' in fcm['kb3']:
            fcm['kb3']['focal_length'] = float(fcm['kb3']['focal_length']) + df
        if 'fc' in fcm:
            fcm['fc'] = [float(fcm['fc'][0]) + df, float(fcm['fc'][1]) + df]
        cc_new = [cx0 + dc, cy0 + dc]
        fcm['cc'] = cc_new
        fcm['distortion_center'] = list(cc_new)
        k_new = (k0 + dk).tolist()
        for i, ki in enumerate(k_new, start=1):
            fcm['kb'][f'k{i}'] = float(ki)
        _state['fcm'] = fcm
        kind = 'gn'
    else:
        return jsonify({'error': 'no recalibration_after_fcm or gn_12dof_result_json artifact'}), 400
    try:
        _persist_state()
    except Exception:
        pass
    return jsonify({'ok': True, 'task_id': task_id, 'kind': kind,
                     'fcm_keys': list(fcm.keys())})


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5007')), debug=False)
