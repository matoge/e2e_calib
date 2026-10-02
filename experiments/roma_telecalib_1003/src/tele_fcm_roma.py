"""TELE <-> FCM extrinsic calib via RoMaV2 dense warp.

Pipeline (smoke, one frame):
    1. Load setting-*.json → fc, cc, kb (KB4) for both cameras.
    2. KB4 undistort both to a common virtual pinhole (crop FCM center,
       keep TELE full) at a shared output K.
    3. RoMaV2 match on the two undistorted PNGs → warp (A→B, [-1,1]²) +
       certainty.
    4. Save visualization: side-by-side images + correspondence lines on
       pixels where certainty > args.conf_thresh (0.3 default).
    5. Save kpts (A, B) + certainty to .npz for downstream E-mat solve.

Run (host-side, no docker needed if romav2+torch installed locally):
    python src/tele_fcm_roma.py \\
        --seq /path/to/sequence=.../ \\
        --frame 10 \\
        --out-dir results/frame0010 \\
        --conf 0.3
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / 'RoMaV2' / 'src'))
from romav2 import RoMaV2


def _load_setting(seq: Path):
    s = next(seq.glob('setting-*.json'))
    cfg = json.load(open(s))
    return cfg[0] if isinstance(cfg, list) and cfg else cfg


def _cam_block(setting, name):
    """fc / cc / kb from setting-*.json  → (K (3,3), D (4,))"""
    c = setting[name]
    fx, fy = c['fc']
    cx, cy = c['cc']
    kb = c['kb']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([kb['k1'], kb['k2'], kb['k3'], kb['k4']], dtype=np.float64)
    return K, D


def _kb_undistort(img_bgr, K, D, K_new, out_wh):
    """OpenCV fisheye (KB4) undistort to pinhole K_new of size out_wh=(w,h)."""
    w, h = out_wh
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    return cv2.remap(img_bgr, map1, map2, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)


def _draw_warp_viz(im_a_bgr, im_b_bgr, kptsA, kptsB, cert, conf_thr, out_path,
                     max_points=4000):
    """Dense side-by-side viz of ALL cert>thr correspondences (up to max_points
    for drawability). Dots only — no connecting lines (too noisy at density)."""
    H, W = im_a_bgr.shape[:2]
    canvas = np.concatenate([im_a_bgr, im_b_bgr], axis=1)
    kA = kptsA.cpu().numpy() if hasattr(kptsA, 'cpu') else np.asarray(kptsA)
    kB = kptsB.cpu().numpy() if hasattr(kptsB, 'cpu') else np.asarray(kptsB)
    c  = cert.cpu().numpy() if hasattr(cert, 'cpu') else np.asarray(cert)
    m = c > conf_thr
    kA, kB, c = kA[m], kB[m], c[m]
    if len(kA) > max_points:
        idx = np.random.RandomState(0).choice(len(kA), max_points, replace=False)
        kA, kB, c = kA[idx], kB[idx], c[idx]
    # Dots only (dense). Color = certainty (red→green).
    for (ua, va), (ub, vb), ci in zip(kA, kB, c):
        colr = (0, int(255*ci), int(255*(1-ci)))
        cv2.circle(canvas, (int(ua), int(va)), 1, colr, -1)
        cv2.circle(canvas, (int(ub)+W, int(vb)), 1, colr, -1)
    cv2.putText(canvas, f'cert>{conf_thr:.2f}  dense N={len(kA)}',
                 (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                 (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--frame', type=int, default=10,
                    help='frame index (0-based, matches `0000_<ts>.jpg` ordering)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--out-wh', type=int, nargs=2, default=[1920, 1200],
                    help='undistorted output size (same for both cams)')
    ap.add_argument('--fcm-focal',  type=float, default=3000.0,
                    help='FCM virtual pinhole focal (lower = wider FOV; keep > tele '
                         'focal / rig-zoom-ratio so FCM covers TELE overlap).')
    ap.add_argument('--tele-focal', type=float, default=4500.0,
                    help='TELE virtual pinhole focal. TELE native=~7376 so 4500 '
                         'crops a bit without black borders.')
    args = ap.parse_args()

    seq = Path(args.seq)
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    setting = _load_setting(seq)
    K_fcm,  D_fcm  = _cam_block(setting, 'fcm')
    K_tele, D_tele = _cam_block(setting, 'tele')

    fcm_files  = sorted((seq / 'tss4_fcm').glob('*.jpg'))
    tele_files = sorted((seq / 'tss4_tele').glob('*.jpg'))
    im_fcm_path  = fcm_files[args.frame]
    im_tele_path = tele_files[args.frame]
    im_fcm_bgr  = cv2.imread(str(im_fcm_path))
    im_tele_bgr = cv2.imread(str(im_tele_path))
    print(f'[in] fcm  {im_fcm_bgr.shape}  {im_fcm_path.name}')
    print(f'[in] tele {im_tele_bgr.shape} {im_tele_path.name}')

    # Per-cam virtual pinhole K: FCM wider (smaller focal) covers TELE FOV + margin.
    w, h = args.out_wh
    K_v_fcm  = np.array([[args.fcm_focal,  0, w/2],
                          [0, args.fcm_focal,  h/2],
                          [0, 0, 1]], dtype=np.float64)
    K_v_tele = np.array([[args.tele_focal, 0, w/2],
                          [0, args.tele_focal, h/2],
                          [0, 0, 1]], dtype=np.float64)

    im_fcm_u  = _kb_undistort(im_fcm_bgr,  K_fcm,  D_fcm,  K_v_fcm,  (w, h))
    im_tele_u = _kb_undistort(im_tele_bgr, K_tele, D_tele, K_v_tele, (w, h))
    cv2.imwrite(str(outd / 'fcm_undist.jpg'),  im_fcm_u)
    cv2.imwrite(str(outd / 'tele_undist.jpg'), im_tele_u)
    print(f'[undist] fcm→{im_fcm_u.shape}  tele→{im_tele_u.shape}  saved')

    # RoMaV2 match on the undistorted pair
    torch.set_float32_matmul_precision('highest')     # required by RoMaV2 forward
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RoMaV2()
    model.apply_setting('precise')
    preds = model.match(str(outd / 'fcm_undist.jpg'),
                         str(outd / 'tele_undist.jpg'))
    H_a, W_a = im_fcm_u.shape[:2]
    H_b, W_b = im_tele_u.shape[:2]
    # Save dense per-pixel confidence heatmaps (overlap_AB and overlap_BA).
    def _overlap_heatmap(overlap, bg_bgr, out_path):
        arr = overlap[0].cpu().numpy() if hasattr(overlap, 'cpu') else np.asarray(overlap[0])
        arr = np.clip(arr, 0.0, 1.0)
        arr8 = (arr * 255).astype(np.uint8)
        # Resize heatmap to source image size
        arr8 = cv2.resize(arr8, (bg_bgr.shape[1], bg_bgr.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        hm = cv2.applyColorMap(arr8, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(bg_bgr, 0.5, hm, 0.5, 0)
        cv2.imwrite(str(out_path), blend)
    _overlap_heatmap(preds['overlap_AB'], im_fcm_u,  outd / 'conf_AB.jpg')
    _overlap_heatmap(preds['overlap_BA'], im_tele_u, outd / 'conf_BA.jpg')
    # Sample dense correspondences
    matches, overlaps, prec_AB, prec_BA = model.sample(preds, 5000)
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_a, W_a, H_b, W_b)
    # overlaps = per-sample confidence
    print(f'[roma] sampled {len(kptsA)} pts  overlap mean={overlaps.mean().item():.3f}  '
          f'cert>{args.conf}: {int((overlaps > args.conf).sum())}')

    # Save viz + raw
    _draw_warp_viz(im_fcm_u, im_tele_u, kptsA, kptsB, overlaps,
                   args.conf, outd / 'warp_viz.jpg')
    np.savez(outd / 'match.npz',
             kptsA=kptsA.cpu().numpy(), kptsB=kptsB.cpu().numpy(),
             overlaps=overlaps.cpu().numpy(),
             K_v_fcm=K_v_fcm, K_v_tele=K_v_tele,
             fcm_src_K=K_fcm, fcm_src_D=D_fcm,
             tele_src_K=K_tele, tele_src_D=D_tele,
             fcm_undist_wh=(w, h), tele_undist_wh=(w, h))
    print(f'[out] {outd}/')
    print(f'  fcm_undist.jpg / tele_undist.jpg / warp_viz.jpg / match.npz')


if __name__ == '__main__':
    main()
