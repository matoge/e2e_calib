"""YRP + zoom aug の可視化 smoke (Task #229).

kamikado FULL の 1 frame を取り、`scripts.util.yrp_zoom_aug` で生成した
homography を画像と LiDAR uv の両方に適用して整合を確認する。

4 ケースを contact sheet で出す:
  1. zoom=1, YRP=(+5, 0, 0)  pure yaw           hint regime
  2. zoom=1, YRP=(0, +5, 0)  pure pitch         hint regime
  3. zoom=1, YRP=(0, 0, +2)  pure roll          hint regime
  4. zoom=2, pivot=任意,    YRP=(+3, -3, +1)   drop  regime

各ケース:
  原画像 + LiDAR uv (depth colormap) | warped 画像 + warped LiDAR uv
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.util.yrp_zoom_aug import (  # noqa: E402
    compose_H, apply_H_image, apply_H_uv,
)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def colorize_by_z(z, vmax=None):
    z = z.cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
    if vmax is None:
        vmax = float(np.percentile(z, 95))
    norm = np.clip(z / max(vmax, 1e-3), 0, 1)
    cmap = plt.cm.turbo
    rgb = (cmap(1.0 - norm)[:, :3] * 255).astype(np.uint8)
    return rgb


def draw_points(img_rgb, uv, colors, radius=3):
    out = Image.fromarray(img_rgb).convert('RGB')
    draw = ImageDraw.Draw(out)
    H, W = img_rgb.shape[:2]
    for (u, v), c in zip(uv, colors):
        if 0 <= u < W and 0 <= v < H:
            draw.ellipse((u - radius, v - radius, u + radius, v + radius),
                         fill=tuple(int(x) for x in c))
    return np.asarray(out)


def label_panel(img_np, text):
    out = Image.fromarray(img_np).convert('RGB')
    draw = ImageDraw.Draw(out)
    pad = 6
    draw.rectangle((0, 0, max(700, len(text) * 11), 32),
                   fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 0))
    return np.asarray(out)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inst_path = Path('/home/hfunaya/cache_v5/kamikado_v3_full/inst/00056800.pt')
    out_dir = (REPO / 'scripts/_debug/_outputs/_yrp_zoom_smoke')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'loading {inst_path}')
    inst = torch.load(inst_path, map_location='cpu', weights_only=False)

    img = np.asarray(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    H_img, W_img = img.shape[:2]
    K = inst['K_full'].numpy().astype(np.float64)
    uv = inst['uv_full'].numpy()           # (N, 2) float32
    z  = inst['z_cam'].numpy()             # (N,)
    print(f'image {H_img}x{W_img}, K=\n{K}')
    print(f'lidar pts: {len(uv)}, z range [{z.min():.1f}, {z.max():.1f}]')

    # Stride LiDAR for visibility (3840×2160 with 62k pts is dense).
    STRIDE = 12
    uv_s = uv[::STRIDE]
    z_s  = z[::STRIDE]
    in_img = ((uv_s[:, 0] >= 0) & (uv_s[:, 0] < W_img) &
              (uv_s[:, 1] >= 0) & (uv_s[:, 1] < H_img) &
              (z_s > 0))
    uv_s = uv_s[in_img]
    z_s  = z_s[in_img]
    colors = colorize_by_z(z_s, vmax=80.0)
    print(f'  visualized: {len(uv_s)} pts')

    img_t = torch.from_numpy(img).float().permute(2, 0, 1).to(device) / 255.0

    # Cases
    cases = [
        dict(label='C1 zoom=1 yaw=+5°    hint',
             yrp=(5.0, 0.0, 0.0), zoom=1.0,
             pivot=(W_img / 2, H_img / 2)),
        dict(label='C2 zoom=1 pitch=+5°  hint',
             yrp=(0.0, 5.0, 0.0), zoom=1.0,
             pivot=(W_img / 2, H_img / 2)),
        dict(label='C3 zoom=1 roll=+2°   hint',
             yrp=(0.0, 0.0, 2.0), zoom=1.0,
             pivot=(W_img / 2, H_img / 2)),
        dict(label='C4 zoom=2 pivot=(0.7W,0.3H) yrp=(+3,-3,+1)  drop',
             yrp=(3.0, -3.0, 1.0), zoom=2.0,
             pivot=(W_img * 0.7, H_img * 0.3)),
    ]

    # Original panel (shared across cases)
    orig_panel = draw_points(img, uv_s, colors, radius=3)
    orig_panel = label_panel(orig_panel, 'ORIG (LiDAR uv, depth colormap)')

    rows = []
    for c in cases:
        H, R = compose_H(c['yrp'], c['zoom'], c['pivot'], K)
        warped_t = apply_H_image(img_t, H, padding_mode='border').clamp(0, 1)
        warped_np = (warped_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        uv_warp = apply_H_uv(uv_s, H)
        # depth invariant under our t=0 + 2D similarity contract.
        warped_panel = draw_points(warped_np, uv_warp, colors, radius=3)
        warped_panel = label_panel(warped_panel, c['label'])
        sbs = np.concatenate([orig_panel, warped_panel], axis=1)
        # Persist per-case as well so individual can be inspected
        out_path = out_dir / f'{c["label"][:2].lower()}.jpg'
        Image.fromarray(sbs).save(out_path, quality=85)
        # Compute residual to check that warping is internally consistent
        # (NOT a model residual — we merely sanity-check that uv_warp is
        # finite and inside-image fraction is reasonable).
        in_warp = ((uv_warp[:, 0] >= 0) & (uv_warp[:, 0] < W_img) &
                   (uv_warp[:, 1] >= 0) & (uv_warp[:, 1] < H_img))
        print(f'  {c["label"]}: in_warp_frac={in_warp.mean():.2f}  '
              f'|max uv_warp|=({uv_warp[:,0].max():.1f}, {uv_warp[:,1].max():.1f})  '
              f'wrote {out_path}')
        rows.append(sbs)

    # Contact sheet
    h, w, _ = rows[0].shape
    canvas = np.zeros((h * len(rows), w, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        canvas[i * h:(i + 1) * h] = r
    contact = out_dir / 'contact.jpg'
    # Downscale for 4×2160 = 8640 px tall canvas
    contact_img = Image.fromarray(canvas)
    scale = 1600 / contact_img.height
    new_size = (int(contact_img.width * scale), int(contact_img.height * scale))
    contact_img.resize(new_size, Image.BILINEAR).save(contact, quality=80)
    print(f'\nwrote {contact.resolve()}')


if __name__ == '__main__':
    main()
