"""Smoke: POST /api/calibrate with image+pts+K from a cache sample.

Pulls one val frame from the kamikado tile cache via the dataset, packs
it as raw multipart (image bytes + pts .npy + K JSON + dist JSON +
is_fisheye=1), and hits the local CaaaS endpoint to confirm the new
sliding-tile + BA path behaves end-to-end.
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path
import urllib.request
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image

from scripts.inference.infer_pipeline import make_ds


CAAAS_EXP = 'km_wv_wm_dgx2_n2_img128_v2'
CACHE = '/cache/kamikado_v3_tiled'


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    bound = '----CaaaSSmoke' + str(np.random.randint(1e9))
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{bound}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                  .encode())
        out.write(str(v).encode()); out.write(b'\r\n')
    for k, (fname, buf) in files.items():
        out.write(f'--{bound}\r\nContent-Disposition: form-data; name="{k}"; '
                  f'filename="{fname}"\r\nContent-Type: application/octet-stream\r\n\r\n'
                  .encode())
        out.write(buf); out.write(b'\r\n')
    out.write(f'--{bound}--\r\n'.encode())
    return out.getvalue(), f'multipart/form-data; boundary={bound}'


def main():
    ds, c = make_ds(CAAAS_EXP, CACHE, split='val', oversample=1)
    inst = ds._load_inst(0)
    print(f'sample: scene={inst.get("scene")} frame={inst.get("frame")} '
          f'IH×IW={inst["IH"]}×{inst["IW"]}  fisheye={inst.get("is_fisheye", False)}')

    full_jpg = bytes(inst['jpg_bytes'])
    img = np.asarray(Image.open(io.BytesIO(full_jpg)).convert('RGB'))
    K  = inst['K_full'].numpy().astype(np.float64)
    # Tile-local K shift (uv_full is in PARENT coords; cached image IS the tile)
    tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
    if tu0 or tv0:
        K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
    dist = inst['distortion'].numpy().astype(np.float64) \
        if 'distortion' in inst else None

    # Reconstruct LiDAR-frame (N,4) input from the cached cam-frame pts +
    # intensity. Since the cache stores cam-frame pts, T_cam_lidar = I works.
    pts_cam = inst['pts'].numpy().astype(np.float32)  # (N, 3)
    intensity = (inst['intensity'].numpy().astype(np.float32)
                  if 'intensity' in inst else np.zeros(len(pts_cam), np.float32))
    pts4 = np.column_stack([pts_cam, intensity])  # cam-frame "lidar" + i

    # Pack request
    pts_buf = io.BytesIO(); np.save(pts_buf, pts4); pts_npy = pts_buf.getvalue()
    fields = dict(
        K=json.dumps(K.tolist()),
        T_cam_lidar=json.dumps(np.eye(4).tolist()),
        exp=CAAAS_EXP,
        is_fisheye='1' if inst.get('is_fisheye', False) else '0',
        huber_k='0', n_iter='1', sigma_max='0',
    )
    if inst.get('is_fisheye', False) and dist is not None:
        fields['dist'] = json.dumps(dist.tolist())
    files = dict(
        image=('frame.jpg', full_jpg),
        pts=('pts.npy', pts_npy),
    )
    body, ctype = _multipart(fields, files)

    req = urllib.request.Request(
        'http://localhost:5006/api/calibrate',
        data=body, headers={'Content-Type': ctype}, method='POST')
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())

    print('=== response ===')
    print(f'  ok               = {d.get("ok")}')
    print(f'  elapsed_ms       = {d.get("elapsed_ms")}')
    print(f'  n_input_pts      = {d.get("n_input_pts")}')
    print(f'  n_in_image       = {d.get("n_in_image")}')
    print(f'  n_pool           = {d.get("n_pool")}')
    print(f'  n_pool_after_filter = {d.get("n_pool_after_filter")}')
    if d.get('ok'):
        print('  delta_pred       =', [round(x, 4) for x in d['delta_pred']])
        print('  sigma_pred       =', [round(x, 4) for x in d['sigma_pred']])
        ub = np.array(d['uv_before']); ua = np.array(d['uv_after'])
        print(f'  reprojection shift mean={np.linalg.norm(ua-ub, axis=1).mean():.2f}px '
              f'max={np.linalg.norm(ua-ub, axis=1).max():.2f}px')
    else:
        print(f'  error = {d.get("error")}')


if __name__ == '__main__':
    main()
