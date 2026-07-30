"""Undistort tss4_fcm KB4 fisheye → pinhole using recalibration.json (= the
manual-fit baked-in K/D).  balance=1.0 keeps the full FOV (output image
gets larger than the input — that's the price for not losing pixels).

Output:
    <out_dir>/<stem>.jpg        rectified pinhole (size W'×H')
    <out_dir>/K_rect.json       {fx, fy, cx, cy, width, height}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np


RECALIB = Path('/home/hfunaya/git/loom/backend/assets/woven_sequence/'
               'llinking_26/recalibration.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-jpgs-dir', type=Path, required=True,
                    help='dir holding the input .jpg fisheye images')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--balance', type=float, default=1.0,
                    help='0=keep only valid (no black), 1=full FOV (large)')
    ap.add_argument('--fov-scale', type=float, default=1.0)
    ap.add_argument('--out-w', type=int, default=None,
                    help='override rectified width (default: same as input)')
    ap.add_argument('--out-h', type=int, default=None)
    args = ap.parse_args()

    fcm = json.loads(RECALIB.read_text())[args.vehicle]['fcm']
    fx = fy = float(fcm['kb']['focal_length'])
    cx, cy = fcm['cc']
    W_in, H_in = int(fcm['resolution'][0]), int(fcm['resolution'][1])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([fcm['kb']['k1'], fcm['kb']['k2'],
                  fcm['kb']['k3'], fcm['kb']['k4']], dtype=np.float64).reshape(4, 1)
    print(f'[K]\n{K}\n[D] {D.flatten()}\nin size = ({W_in},{H_in})')

    W_out = args.out_w or W_in
    H_out = args.out_h or H_in

    K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W_in, H_in), np.eye(3),
        balance=args.balance, fov_scale=args.fov_scale,
        new_size=(W_out, H_out))
    print(f'[K_rect]\n{K_rect}\nout size = ({W_out},{H_out})')

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_rect, (W_out, H_out), cv2.CV_16SC2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpgs = sorted(Path(args.src_jpgs_dir).glob('*.jpg'))
    print(f'[run] {len(jpgs)} → {args.out_dir}')
    for i, p in enumerate(jpgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rect = cv2.remap(img, map1, map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        cv2.imwrite(str(args.out_dir / p.name), rect, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i == 0 or (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(jpgs)}]')

    (args.out_dir / 'K_rect.json').write_text(json.dumps({
        'fx': float(K_rect[0, 0]), 'fy': float(K_rect[1, 1]),
        'cx': float(K_rect[0, 2]), 'cy': float(K_rect[1, 2]),
        'width': int(W_out), 'height': int(H_out),
        'balance': args.balance, 'fov_scale': args.fov_scale,
        'src_K': K.tolist(), 'src_D': D.flatten().tolist(),
        'src_size': [W_in, H_in],
    }, indent=2))
    print(f'[done] K_rect.json written, fx={K_rect[0,0]:.2f} cx={K_rect[0,2]:.2f}')


if __name__ == '__main__':
    main()
