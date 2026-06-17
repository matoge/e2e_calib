"""Bake combined per-frame masks (dashboard polygon AND SAM3 dynamic instance
masks with per-instance dilation) into a directory of PNGs.

Output: <out_dir>/<jpg_stem>.png  (uint8 0/255, white=keep, black=skip)

The trainer's _read_mask reads these directly. white = keep (loss in),
black = skip (loss zeroed and Gaussian init dropped).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np


DEFAULT_DASH = Path('/home/hfunaya/git/e2e_calib/scripts/webui_kb_fit/'
                     '_outputs/_dashboard_polygons.json')
DEFAULT_INST_DIR = Path('/home/hfunaya/git/e2e_calib/scripts/webui_kb_fit/'
                         '_outputs/sam3_pylon_seq/inst')


def build_dashboard_mask(W: int, H: int, dash_json: Path) -> np.ndarray:
    """Returns (H,W) uint8 with 1=keep, 0=skip from polygon list."""
    if not dash_json.is_file():
        return np.ones((H, W), dtype=np.uint8)
    d = json.loads(dash_json.read_text())
    m = np.full((H, W), 255, np.uint8)
    for poly in d.get('polygons', []):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(m, [pts], 0)
    return (m > 127).astype(np.uint8)


def build_dyn_mask_per_inst(inst_npy: Path, dilate_frac: float) -> np.ndarray:
    """SAM3 per-instance dilation. Returns (H,W) uint8 with 1=keep, 0=skip."""
    inst = np.load(inst_npy)
    H, W = inst.shape
    skip = np.zeros((H, W), dtype=np.uint8)
    ids = [int(i) for i in np.unique(inst) if i != 0]
    for iid in ids:
        m = (inst == iid).astype(np.uint8)
        if m.sum() == 0:
            continue
        ys, xs = np.where(m > 0)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        short = min(h, w)
        r = max(2, int(round(short * dilate_frac)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        skip |= cv2.dilate(m, k, iterations=1)
    return (skip == 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-dir', type=Path, required=True,
                    help='woven_sequence/.../sequence=...')
    ap.add_argument('--out-dir', type=Path, required=True,
                    help='where to write <stem>.png')
    ap.add_argument('--dash-json', type=Path, default=DEFAULT_DASH)
    ap.add_argument('--inst-dir', type=Path, default=DEFAULT_INST_DIR,
                    help='dir of SAM3 inst/<idx>_<unixms>.npy files (one per frame)')
    ap.add_argument('--dilate-frac', type=float, default=0.025)
    ap.add_argument('--no-dashboard', action='store_true',
                    help='ignore dashboard polygons; mask = SAM3 dynamic only')
    ap.add_argument('--undistort-crop-half', action='store_true',
                    help='apply fisheye undistort + crop[2500..7500, 2069..3469] '
                          '+ halve to 2500x700 (matches woven_pandaset_pylon/001_half)')
    ap.add_argument('--undistort-crop-8k-half', action='store_true',
                    help='wider crop[1000..9000, 2069..3469] + halve to 4000x700 '
                          '(matches woven_pandaset_pylon/001_8k_half, HFoV 118deg)')
    ap.add_argument('--vehicle', default='248')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cam_files = sorted((args.seq_dir / 'tss4_fcm').glob('*.jpg'))
    if len(cam_files) == 0:
        raise SystemExit(f'no jpgs under {args.seq_dir}/tss4_fcm')
    img0 = cv2.imread(str(cam_files[0]))
    H, W = img0.shape[:2]
    print(f'[seq] {len(cam_files)} frames  {W}x{H}')

    if args.no_dashboard:
        m_dash = np.ones((H, W), dtype=np.uint8)
        print('[dash] disabled (--no-dashboard)')
    else:
        m_dash = build_dashboard_mask(W, H, args.dash_json)
        print(f'[dash] kept {m_dash.mean():.2%} ({args.dash_json.name})')

    # optional undistort + crop + halve to 2500x700 (matches 001_half)
    map1 = map2 = None
    crop_xyxy = None
    out_size = None
    if args.undistort_crop_half or args.undistort_crop_8k_half:
        import json as _json
        rec = _json.loads(open(
            '/home/hfunaya/git/loom/backend/assets/woven_sequence/'
            'llinking_26/recalibration.json').read())[args.vehicle]['fcm']
        K_fis = np.array([[rec['kb']['focal_length'], 0, rec['cc'][0]],
                          [0, rec['kb']['focal_length'], rec['cc'][1]],
                          [0, 0, 1]], dtype=np.float64)
        D_fis = np.array([rec['kb'][f'k{i}'] for i in (1, 2, 3, 4)],
                          dtype=np.float64).reshape(4, 1)
        RECT_W, RECT_H = 10000, 5083
        K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_fis, D_fis, (W, H), np.eye(3), balance=1.0, fov_scale=1.0,
            new_size=(RECT_W, RECT_H))
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_fis, D_fis, np.eye(3), K_rect, (RECT_W, RECT_H), cv2.CV_16SC2)
        if args.undistort_crop_8k_half:
            crop_xyxy = (1000, 2069, 9000, 3469)
            out_size = (4000, 700)
            print(f'[undistort+crop+halve 8k] output {out_size[0]}x{out_size[1]}')
        else:
            crop_xyxy = (2500, 2069, 7500, 3469)
            out_size = (2500, 700)
            print(f'[undistort+crop+halve] enabled, output {out_size[0]}x{out_size[1]}')

    inst_files = sorted(args.inst_dir.glob('*.npy'))
    if len(inst_files) != len(cam_files):
        print(f'[warn] inst count {len(inst_files)} != cam {len(cam_files)} '
              f'-- matching by index')
    inst_by_idx = {int(p.name.split('_')[0]): p for p in inst_files}

    n_total = len(cam_files)
    kept_sum = 0
    for i, jpg in enumerate(cam_files):
        if i in inst_by_idx:
            m_dyn = build_dyn_mask_per_inst(inst_by_idx[i], args.dilate_frac)
        else:
            print(f'[skip] no inst for frame {i}, dyn mask = all keep')
            m_dyn = np.ones((H, W), dtype=np.uint8)
        m = ((m_dash & m_dyn) * 255).astype(np.uint8)
        if map1 is not None:
            m_rect = cv2.remap(m, map1, map2, interpolation=cv2.INTER_NEAREST)
            x0, y0, x1, y1 = crop_xyxy
            m_crop = m_rect[y0:y1, x0:x1]
            m = cv2.resize(m_crop, out_size, interpolation=cv2.INTER_NEAREST)
        out_p = args.out_dir / f'{jpg.stem}.png'
        cv2.imwrite(str(out_p), m)
        kept = (m > 0).mean()
        kept_sum += kept
        if i == 0 or (i + 1) % 10 == 0:
            print(f'  [{i+1}/{n_total}] {jpg.stem}  kept={kept:.2%}')
    print(f'[done] mean kept = {kept_sum/n_total:.2%}  → {args.out_dir}')


if __name__ == '__main__':
    main()
