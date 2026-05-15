"""LiDAR 投影 + 3D box (8角ワイヤフレーム) + box 内点を緑色。

- 投影は build_woven_sequence_v3.py のパイプラインをそのまま使う (自前で書かない)
- box の画像描画は annotation の `projected_3d_corners` を使う (これも自前で投影しない)
- box 内点判定は rear_axle FLU 系で yaw 逆回転 + axis-aligned (build_woven_sequence_v3
  の _is_obj_per_point と同じロジック)
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing.build_woven_sequence_v3 import (
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


SEQ = Path('/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_27/'
           'tf_long2/sequence=ip654_1337941440921107425_'
           '16943630305775105398_1749030654176-1749030664176')

VERIFIED_USERS = {'hiroyuki.funaya', 'yolo', 'automation_yolo'}

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

LABEL_COLOR = {
    'car':                '#1e6fff',
    'truck':              '#0fa550',
    'bus':                '#0fa550',
    'sign':               '#c13c14',
    'traffic_body':       '#ff8c00',
    'traffic_light_bulb': '#ff8c00',
    'lanemarker_edge':    '#9966cc',
    'pedestrian':         '#e0008c',
}


def last_user(eh):
    if isinstance(eh, dict): return eh.get('user')
    if isinstance(eh, list) and eh:
        last = eh[-1]
        if isinstance(last, dict): return last.get('user')
    return None


def load_boxes(seq, fid, allowed_users):
    p = seq / 'saved_annotations' / f'{fid}.json'
    if not p.exists(): return []
    d = json.load(open(p))
    out = []
    for det in d.get('details', []) or []:
        if det.get('type') != 'box': continue
        if allowed_users is not None \
           and last_user(det.get('edit_history')) not in allowed_users:
            continue
        attrs = det.get('attributes') or {}
        cf = attrs.get('3dbb_rear_axle')
        if not cf: continue
        center = np.asarray(cf['center_meter'], dtype=np.float32)
        size   = np.asarray(cf['size_meter'],   dtype=np.float32)
        d_ = cf.get('direction')
        yaw = (float(np.arctan2(float(d_[1]), float(d_[0])))
               if isinstance(d_, (list, tuple)) and len(d_) >= 2 else 0.0)
        cuv = attrs.get('projected_3d_corners')
        cuv = np.asarray(cuv, dtype=np.float32) if cuv is not None else None
        out.append(dict(
            center=center, size=size, yaw=yaw,
            label=str(det.get('label', '')),
            object_id=det.get('object_id'),
            corners_uv=cuv,
        ))
    return out


def points_in_box(pts, box):
    c, s = np.cos(box['yaw']), np.sin(box['yaw'])
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
    local = (R @ (pts - box['center']).T).T
    return np.all(np.abs(local) <= box['size'] * 0.5, axis=1)


def draw_box(ax, corners_uv, color, lw=1.6):
    if corners_uv is None or len(corners_uv) != 8: return
    segs = [(corners_uv[i], corners_uv[j]) for i, j in CUBE_EDGES]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=0.9))


def main():
    out_path = Path('/home/hfunaya/git/e2e_calib/out/vis_woven_proj_only.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setting = _load_setting(SEQ)
    K, dist, R_cam_from_veh, t_cam_in_veh, W, H, delay_default = \
        _camera_calib_fcm(setting)
    metadata = _load_metadata(SEQ)
    frame_ids, poses = _get_poses(metadata)
    print(f'frames in metadata: {len(frame_ids)}')

    sample_idxs = [int(len(frame_ids) * f) for f in (0.2, 0.4, 0.6, 0.8)]
    fids = [frame_ids[i] for i in sample_idxs]
    print(f'sampled frames: {fids}')

    fig, axes = plt.subplots(len(fids), 1, figsize=(14, 8 * len(fids)), dpi=110)
    if len(fids) == 1: axes = [axes]
    fig.patch.set_facecolor('#f6f4ed')

    for ri, (idx, fid) in enumerate(zip(sample_idxs, fids)):
        img_path = SEQ / 'tss4_fcm' / f'{fid}.jpg'
        if not img_path.exists():
            print(f'  skip {fid}: no image'); continue
        img = np.asarray(Image.open(img_path))

        pts_veh, intensity = _load_pts_intensity(SEQ, fid)
        delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
        pose_curr = poses[fid]
        pose_camera = _pose_at_camera_time(poses, frame_ids, idx, delay_ms)
        pts_cam = _lidar_to_cam_at_camera_time(
            pts_veh, pose_curr, pose_camera, R_cam_from_veh, t_cam_in_veh)
        uv = _project_kannala(pts_cam, K, dist)
        z = pts_cam[:, 2]

        # box (VERIFY filter)
        boxes = load_boxes(SEQ, fid, VERIFIED_USERS)

        # box 内点 (rear_axle FLU = pts_veh)
        in_any = np.zeros(len(pts_veh), dtype=bool)
        for b in boxes:
            in_any |= points_in_box(pts_veh, b)

        valid = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                          & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        m_out = valid & (~in_any)
        m_in  = valid & in_any

        ax = axes[ri]
        ax.imshow(img)
        # 灰: box外 (depth カラーじゃなく落ち着いた灰)
        if m_out.any():
            ax.scatter(uv[m_out, 0], uv[m_out, 1],
                       c=z[m_out], s=0.5, cmap='turbo',
                       alpha=0.55, linewidths=0, vmin=0, vmax=80)
        # 緑: box内
        if m_in.any():
            ax.scatter(uv[m_in, 0], uv[m_in, 1],
                       c='#0fa550', s=2.0, alpha=0.95, linewidths=0)

        for b in boxes:
            color = LABEL_COLOR.get(b['label'], '#888')
            draw_box(ax, b['corners_uv'], color)
            if b['corners_uv'] is not None and len(b['corners_uv']) == 8:
                u, v = b['corners_uv'].mean(0)
                ax.text(u, v, str(b.get('object_id', '')),
                        color='white', fontsize=6,
                        bbox=dict(facecolor=color, alpha=0.6,
                                  edgecolor='none', pad=1.0))

        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{fid}   verified_boxes={len(boxes)}  '
                     f'in-box pts={int(m_in.sum())}  '
                     f'all-img pts={int(valid.sum())}/{len(uv)}  '
                     f'delay={delay_ms:.1f}ms',
                     fontsize=10, loc='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    plt.close(fig)
    print(f'saved → {out_path}')


if __name__ == '__main__':
    main()
