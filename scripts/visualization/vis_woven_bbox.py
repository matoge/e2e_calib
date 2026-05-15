"""WovenSequence の OBJECT bbox + box 内 LiDAR 点 可視化。

  col 0: 画像 + 3D bbox の 8 角投影 + LiDAR 点 (Kannala 投影)
         box 内点 = 緑、それ以外 = 灰
  col 1: rear_axle BEV (x: forward, y: left) で LiDAR 点 + box 上面 4 角 + box 内点

verify フィルタ:
  edit_history.user が --users で指定された集合に入っている box だけ採用
  (デフォ: hiroyuki.funaya / yolo / automation_yolo)

データ:
  - 画像: tss4_fcm/<frame_id>.jpg                3840x2160 fisheye
  - LiDAR (rear_axle FLU): vls128_rear_axle/<frame_id>.npz  (xs, ys, zs, intensity)
  - アノテ: saved_annotations/<frame_id>.json
  - calib: setting-ip*.json  fcm.{mp, rot, fc, cc, kb}  (Kannala-Brandt)

簡略仮定:
  - vls128 の mp/rot は 0 (vehicle FLU = LiDAR)
  - camera_delay (33ms) のための pose 補間は省略 (静止物体ほぼ一致、動的物体は若干ズレる)
"""
import argparse, json, sys, pathlib, glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from PIL import Image

# 動作確認済みの投影パイプラインをそのまま import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing.build_woven_sequence_v3 import (   # noqa: E402
    _load_setting,
    _camera_calib_fcm,
    _load_metadata,
    _get_poses,
    _camera_delay_ms_for_frame,
    _pose_at_camera_time,
    _load_pts_intensity,
    _lidar_to_cam_at_camera_time,
    _project_kannala,
)


DEFAULT_SEQ = ('/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_27/'
               'tf_long2/sequence=ip654_1337941440921107425_'
               '16943630305775105398_1749030654176-1749030664176')

# 8 corner index → 12 edge pairs (cube wireframe)
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom-ish quad
    (4, 5), (5, 6), (6, 7), (7, 4),  # top-ish quad
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

LABEL_COLOR = {
    'car':              '#1e6fff',
    'truck':            '#0fa550',
    'bus':              '#0fa550',
    'sign':             '#c13c14',
    'traffic_light_bulb':  '#ff8c00',
    'traffic_body':     '#ff8c00',
    'lanemarker_edge':  '#9966cc',
    'pedestrian':       '#e0008c',
}


DEFAULT_VERIFIED_USERS = {'hiroyuki.funaya', 'yolo', 'automation_yolo'}


def _last_user(edit_history) -> str | None:
    if isinstance(edit_history, dict):
        return edit_history.get('user')
    if isinstance(edit_history, list) and edit_history:
        last = edit_history[-1]
        if isinstance(last, dict):
            return last.get('user')
    return None


def _load_boxes(seq: Path, frame_id: str, allowed_users: set | None):
    """returns list of dict(center, size, yaw, label, corners_uv (8,2) or None)
    allowed_users が None なら全 box、そうでなければ edit_history.user でフィルタ"""
    p = seq / 'saved_annotations' / f'{frame_id}.json'
    if not p.exists():
        return []
    d = json.load(open(p))
    out = []
    for det in d.get('details', []) or []:
        if det.get('type') != 'box':
            continue
        u = _last_user(det.get('edit_history'))
        if allowed_users is not None and u not in allowed_users:
            continue
        attrs = det.get('attributes') or {}
        cf = attrs.get('3dbb_rear_axle')   # box-内点フィルタは lidar と同じ rear_axle 系で
        if not cf:
            continue
        center = np.asarray(cf['center_meter'], dtype=np.float32)
        size   = np.asarray(cf['size_meter'],   dtype=np.float32)
        direction = cf.get('direction')
        if isinstance(direction, (list, tuple)) and len(direction) >= 2:
            yaw = float(np.arctan2(float(direction[1]), float(direction[0])))
        else:
            yaw = 0.0
        corners_uv = attrs.get('projected_3d_corners')
        corners_uv = (np.asarray(corners_uv, dtype=np.float32)
                      if corners_uv is not None else None)
        out.append({
            'center':     center,
            'size':       size,
            'yaw':        yaw,
            'label':      str(det.get('label', '')),
            'object_id':  det.get('object_id'),
            'corners_uv': corners_uv,
            'user':       u,
            'not_in_lidar': bool(det.get('not_in_lidar', False)),
        })
    return out


def _points_in_box(pts: np.ndarray, box: dict) -> np.ndarray:
    """rear_axle FLU 上で 1 box 内点判定。pts (N,3) → (N,) bool"""
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
    local = (R @ (pts - box['center'][None, :]).T).T
    half = box['size'] * 0.5
    return np.all(np.abs(local) <= half[None, :], axis=1)


def _bbox_top_polygon(box: dict) -> np.ndarray:
    """rear_axle BEV で box 上面 4角 (xy plane, z 無視) を返す (4,2)"""
    l, w, _ = box['size']
    half = np.array([[ l/2,  w/2],
                     [ l/2, -w/2],
                     [-l/2, -w/2],
                     [-l/2,  w/2]], dtype=np.float32)
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
    return half @ R2.T + box['center'][:2][None, :]


def _draw_box_on_image(ax, corners_uv: np.ndarray, color: str, lw: float = 1.2):
    if corners_uv is None or len(corners_uv) != 8:
        return
    segs = [(corners_uv[i], corners_uv[j]) for i, j in CUBE_EDGES]
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=0.85)
    ax.add_collection(lc)


def _color_for(label: str, default='#888888'):
    return LABEL_COLOR.get(label, default)


def make_grid(seq: Path, frame_ids: list, out_path: Path,
              bev_xlim=(0, 80), bev_ylim=(-30, 30),
              max_pts_show=80000, label_filter=None,
              allowed_users: set | None = None,
              point_size_img=1.0):
    n = len(frame_ids)
    fig, axes = plt.subplots(n, 2, figsize=(18, 5.5 * n), dpi=110)
    fig.patch.set_facecolor('#f6f4ed')
    if n == 1:
        axes = axes[None, :]

    calib = _load_fcm_calib(seq)
    W, H = calib['resolution']

    for ri, fid in enumerate(frame_ids):
        # --- load ---
        img = np.asarray(Image.open(seq / 'tss4_fcm' / f'{fid}.jpg'))
        pts, inten = _load_lidar(seq, fid)
        boxes = _load_boxes(seq, fid, allowed_users)
        if label_filter:
            boxes = [b for b in boxes if b['label'] in label_filter]

        # box 内 mask (どれかの box に入った点)
        in_any = np.zeros(len(pts), dtype=bool)
        for b in boxes:
            in_any |= _points_in_box(pts, b)

        # --- LiDAR を画像に投影 (Kannala) ---
        pts_cam = _veh_to_cam(pts, calib)
        uv, valid = _project_kannala(pts_cam, calib['K'], calib['dist'])
        in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                       & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        # --- col 0: image + 8-corner box wireframe + LiDAR overlay ---
        ax = axes[ri, 0]
        ax.imshow(img)
        # 灰: box外, 緑: box内
        m_out = in_img & (~in_any)
        m_in  = in_img & in_any
        if m_out.any():
            ax.scatter(uv[m_out, 0], uv[m_out, 1], s=point_size_img,
                       c='#aab', alpha=0.35, linewidths=0)
        if m_in.any():
            ax.scatter(uv[m_in, 0], uv[m_in, 1], s=point_size_img * 2.5,
                       c='#0fa550', alpha=0.95, linewidths=0)
        for b in boxes:
            _draw_box_on_image(ax, b['corners_uv'], _color_for(b['label']),
                               lw=1.6)
            if b['corners_uv'] is not None and len(b['corners_uv']) == 8:
                u, v = b['corners_uv'].mean(0)
                ax.text(u, v, str(b.get('object_id', '')),
                        color='white', fontsize=6,
                        bbox=dict(facecolor=_color_for(b['label']), alpha=0.6,
                                  edgecolor='none', pad=1.0))
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_title(f'{fid}   ({len(boxes)} verified boxes, '
                     f'{int(m_in.sum())} pts in-box-on-img)',
                     fontsize=10, loc='left')
        ax.set_xticks([]); ax.set_yticks([])

        # --- col 1: rear_axle BEV ---
        ax = axes[ri, 1]
        ax.set_facecolor('#fff')
        # 全点 (subsample for speed)
        N = len(pts)
        if N > max_pts_show:
            idx = np.random.RandomState(0).choice(N, max_pts_show, replace=False)
            ps = pts[idx]; im = inten[idx]; ia = in_any[idx]
        else:
            ps, im, ia = pts, inten, in_any
        # forward = x, left = y → (x→horizontal, y→vertical, but plot x as vertical
        # to make "forward = up" feel natural)
        ax.scatter(ps[~ia, 1], ps[~ia, 0],
                   s=0.4, c='#aab', alpha=0.45, linewidths=0)
        ax.scatter(ps[ia, 1],  ps[ia, 0],
                   s=2.0, c='#0fa550', alpha=0.95, linewidths=0)
        # boxes (top polygon)
        for b in boxes:
            poly = _bbox_top_polygon(b)
            ax.add_patch(MplPolygon(poly[:, [1, 0]], closed=True,
                                    fill=False,
                                    edgecolor=_color_for(b['label']),
                                    linewidth=1.0, alpha=0.9))
        ax.set_xlim(bev_ylim[1], bev_ylim[0])  # y_left  (反転で左=右側)
        ax.set_ylim(bev_xlim[0], bev_xlim[1])  # x_forward
        ax.set_aspect('equal')
        ax.set_xlabel('y (left, m)'); ax.set_ylabel('x (fwd, m)')
        ax.set_title('rear_axle BEV  (green = in-box)', fontsize=10, loc='left')
        ax.grid(alpha=0.25)

        print(f'  {fid}: {len(boxes):3d} boxes, '
              f'in_box pts {int(in_any.sum()):6d} / {N}')

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    plt.close(fig)
    print(f'saved → {out_path}')


def _pick_frame_ids(seq: Path, n: int, seed: int = 0):
    jpgs = sorted((seq / 'tss4_fcm').glob('*.jpg'))
    if not jpgs:
        raise SystemExit(f'no images in {seq}/tss4_fcm')
    if n >= len(jpgs):
        return [p.stem for p in jpgs]
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(jpgs), n, replace=False))
    return [jpgs[i].stem for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default=DEFAULT_SEQ)
    ap.add_argument('--out', default='vis_woven_bbox.png')
    ap.add_argument('-n', '--n-frames', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames', nargs='+', default=None,
                    help='specific frame ids (e.g. 0050_1749030625599869952)')
    ap.add_argument('--labels', nargs='+', default=None,
                    help='only draw boxes whose label is in this set '
                         '(default: all)')
    ap.add_argument('--users', nargs='+',
                    default=sorted(DEFAULT_VERIFIED_USERS),
                    help='only show boxes whose last edit_history.user is in '
                         f'this set. Default: {sorted(DEFAULT_VERIFIED_USERS)}. '
                         'Pass "--users all" to disable filter.')
    ap.add_argument('--bev-fwd-min', type=float, default=0.0)
    ap.add_argument('--bev-fwd-max', type=float, default=80.0)
    ap.add_argument('--bev-side',     type=float, default=30.0,
                    help='BEV ±side (m) on the y axis')
    args = ap.parse_args()

    seq = Path(args.seq)
    if args.frames:
        frame_ids = args.frames
    else:
        frame_ids = _pick_frame_ids(seq, args.n_frames, args.seed)
    print(f'sequence: {seq.name}')
    print(f'frames: {frame_ids}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(args.users) == 1 and args.users[0].lower() == 'all':
        allowed_users = None
    else:
        allowed_users = set(args.users)
    print(f'verified-user filter: {allowed_users}')

    make_grid(seq, frame_ids, out_path,
              bev_xlim=(args.bev_fwd_min, args.bev_fwd_max),
              bev_ylim=(-args.bev_side, args.bev_side),
              label_filter=set(args.labels) if args.labels else None,
              allowed_users=allowed_users)


if __name__ == '__main__':
    main()
