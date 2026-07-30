"""Undistort all jpgs in a sequence to a stretched (balance=1.0,
new_size=10000x5083) rectified pinhole image, then crop a user-specified
region.  Output:
    <out_dir>/<stem>.jpg   cropped pinhole jpg
    <out_dir>/K_rect.json  K matrix for the cropped image
                            (so SplatAD can use it directly)
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
    ap.add_argument('--src-jpgs-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--rect-W', type=int, default=10000,
                    help='balance=1.0 stretched canvas width')
    ap.add_argument('--rect-H', type=int, default=5083)
    ap.add_argument('--crop-x0', type=int, required=True)
    ap.add_argument('--crop-y0', type=int, required=True)
    ap.add_argument('--crop-x1', type=int, required=True)
    ap.add_argument('--crop-y1', type=int, required=True)
    args = ap.parse_args()

    fcm = json.loads(RECALIB.read_text())[args.vehicle]['fcm']
    fx = fy = float(fcm['kb']['focal_length'])
    cx0, cy0 = fcm['cc']
    W_in, H_in = int(fcm['resolution'][0]), int(fcm['resolution'][1])
    K = np.array([[fx, 0, cx0], [0, fy, cy0], [0, 0, 1]], dtype=np.float64)
    D = np.array([fcm['kb']['k1'], fcm['kb']['k2'],
                  fcm['kb']['k3'], fcm['kb']['k4']], dtype=np.float64).reshape(4, 1)
    print(f'[in] K=\n{K}\nD={D.flatten()}\n size=({W_in},{H_in})')

    K_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W_in, H_in), np.eye(3), balance=1.0, fov_scale=1.0,
        new_size=(args.rect_W, args.rect_H))
    print(f'[K_rect (full)] fx={K_rect[0,0]:.2f} cx={K_rect[0,2]:.2f} cy={K_rect[1,2]:.2f}')

    # remap to full stretched
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_rect, (args.rect_W, args.rect_H), cv2.CV_16SC2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jpgs = sorted(Path(args.src_jpgs_dir).glob('*.jpg'))
    print(f'[run] {len(jpgs)} jpgs → crop x[{args.crop_x0},{args.crop_x1}] y[{args.crop_y0},{args.crop_y1}]')

    for i, p in enumerate(jpgs):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        crop = rect[args.crop_y0:args.crop_y1, args.crop_x0:args.crop_x1]
        cv2.imwrite(str(args.out_dir / p.name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i == 0 or (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(jpgs)}]')

    # K for the cropped image: same fx/fy, principal point shifted
    fx_c = float(K_rect[0, 0]); fy_c = float(K_rect[1, 1])
    cx_c = float(K_rect[0, 2]) - args.crop_x0
    cy_c = float(K_rect[1, 2]) - args.crop_y0
    W_c = args.crop_x1 - args.crop_x0
    H_c = args.crop_y1 - args.crop_y0
    out_K = {
        'fx': fx_c, 'fy': fy_c, 'cx': cx_c, 'cy': cy_c,
        'width': int(W_c), 'height': int(H_c),
        'src_K': K.tolist(), 'src_D': D.flatten().tolist(),
        'src_size': [W_in, H_in],
        'rect_size_full': [args.rect_W, args.rect_H],
        'crop_xyxy': [args.crop_x0, args.crop_y0, args.crop_x1, args.crop_y1],
        'balance': 1.0,
    }
    (args.out_dir / 'K_rect.json').write_text(json.dumps(out_K, indent=2))
    print(f'[done] {args.out_dir}/  W×H = {W_c}×{H_c}  fx={fx_c:.1f} cx={cx_c:.1f} cy={cy_c:.1f}')


if __name__ == '__main__':
    main()
