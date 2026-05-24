"""Pair-mode visualisation on the new full-frame PandaSet cache.

Drives PandaSetCalibDatasetFull(pair_mode=True) on
  /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full
and saves four panels per sample (frame_A full, frame_B full, crop_A,
crop_B) plus a side-by-side composite. Every dot/box has a baked-in legend.

Color contract (must match across all panels):
  YELLOW : own lidar projected via own GT pose (= calib target uvd for that frame)
  GREEN  : inst_A lidar projected via the *perturbed* δ_A pose (model input on A)
  MAGENTA: inst_A lidar warped via T_gt_B  (= cross-frame target for the network)
  BLUE   : pivot
            – on frame_A: chosen pts_A[i] projected via T_gt_A
            – on frame_B: same pts_A[i] projected via T_gt_B (crop center on B)
            – on the crop panels: image center (pivot is centred by construction)
  RED bbox / CYAN bbox : the cs×cs window cropped out of frame_A / frame_B

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_pair_overlay_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull, decode_inst_img  # noqa: E402

CACHE_DIR = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
OUT_DIR = ROOT / 'docs/assets/2026-05-24_pair_overlay_full'
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 256
MIN_CROP = 256
MAX_CROP = 512
N_SAMPLES = 4
SEED = 7

YELLOW  = np.array([255, 220,   0], dtype=np.uint8)
GREEN   = np.array([ 30, 220,  60], dtype=np.uint8)
MAGENTA = np.array([255,  60, 220], dtype=np.uint8)
BLUE    = np.array([ 60, 130, 255], dtype=np.uint8)
RED     = np.array([240,  60,  60], dtype=np.uint8)
CYAN    = np.array([ 30, 220, 220], dtype=np.uint8)
WHITE   = np.array([245, 245, 245], dtype=np.uint8)


def world_to_cam(pts_w, R_cam_world, cam_pos):
    return (pts_w - cam_pos[None, :]) @ R_cam_world


def project_pinhole(pts_cam, K):
    z = np.maximum(pts_cam[:, 2], 1e-6)
    u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=-1)


def draw_dot(arr, u, v, color, size=2):
    H, W, _ = arr.shape
    iu, iv = int(round(u)), int(round(v))
    for du in range(-size, size + 1):
        for dv in range(-size, size + 1):
            x, y = iu + du, iv + dv
            if 0 <= x < W and 0 <= y < H:
                arr[y, x] = color


def draw_line(arr, u0, v0, u1, v1, color):
    """Bresenham line, 1px thick. Skips out-of-bounds pixels."""
    H, W, _ = arr.shape
    x0, y0 = int(round(u0)), int(round(v0))
    x1, y1 = int(round(u1)), int(round(v1))
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            arr[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy


def draw_rect(arr, u0, v0, cs, color, thick=2):
    H, W, _ = arr.shape
    u1, v1 = u0 + cs - 1, v0 + cs - 1
    for t in range(thick):
        for u in range(max(0, u0 - t), min(W, u1 + t + 1)):
            for v in (v0 - t, v1 + t):
                if 0 <= v < H:
                    arr[v, u] = color
        for v in range(max(0, v0 - t), min(H, v1 + t + 1)):
            for u in (u0 - t, u1 + t):
                if 0 <= u < W:
                    arr[v, u] = color


def stamp_legend(arr, lines, anchor=(8, 8), pad=6, font_size=14):
    """Burn a small legend box into the top-left of arr."""
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    line_h = font_size + 4
    box_w = max(font.getlength(t) for _, t in lines) + pad * 2
    box_h = line_h * len(lines) + pad * 2
    x0, y0 = anchor
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    for i, (color, text) in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * line_h), text, font=font,
                  fill=tuple(int(c) for c in color))
    arr[...] = np.asarray(img)


def render_one(ds, pair_idx, save_prefix):
    """Re-run pivot + crop logic so we can draw on the full image. Builds the
    same window the dataset would emit and additionally overlays the cross-frame
    target (inst_A lidar warped by T_gt_B) which is what the network must learn
    to match in cross-frame mode."""
    rng_state = np.random.get_state()
    i_A, i_B = ds.pair_index[pair_idx % len(ds.pair_index)]
    inst_A = ds._load_inst(i_A)
    inst_B = ds._load_inst(i_B)
    K_A = inst_A['K_full'].numpy()
    K_B = inst_B['K_full'].numpy()
    R_gt_A = inst_A['R_gt'].numpy(); cp_A = inst_A['cam_pos'].numpy()
    R_gt_B = inst_B['R_gt'].numpy(); cp_B = inst_B['cam_pos'].numpy()
    pts_w_A = inst_A['pts'].numpy()
    pts_w_B = inst_B['pts'].numpy()
    IH_A = int(inst_A['IH']); IW_A = int(inst_A['IW'])
    IH_B = int(inst_B['IH']); IW_B = int(inst_B['IW'])

    np.random.set_state(rng_state)
    sample = ds[pair_idx]
    if sample is None:
        print(f'[{save_prefix}] pair returned None')
        return
    A12, B12, dpose = sample
    img_A_crop, true_A, dist_A, vfp_A, *_, cs_A_t, _ = A12
    img_B_crop, true_B, dist_B, vfp_B, *_, cs_B_t, _ = B12
    cs = int(float(cs_A_t))
    assert cs == int(float(cs_B_t))
    last_A = dict(ds._last_pair_A)
    last_B = dict(ds._last_pair_B)
    u0_A, v0_A = int(last_A['u0']), int(last_A['v0'])
    u0_B, v0_B = int(last_B['u0']), int(last_B['v0'])
    pu_A, pv_A = float(last_A['pivot_uv'][0]), float(last_A['pivot_uv'][1])
    pu_B, pv_B = float(last_B['pivot_uv'][0]), float(last_B['pivot_uv'][1])
    pivot_world_idx = int(last_B['pivot_world_idx'])
    R_off_A = np.asarray(last_A['R_off'])
    cp_off_A = np.asarray(last_A['cp_off'])
    S = float(IMG_SIZE)

    full_A = decode_inst_img(inst_A).permute(1, 2, 0).numpy().copy()
    full_B = decode_inst_img(inst_B).permute(1, 2, 0).numpy().copy()

    # Sub-sample for full-frame overlay. Sweep is ~100k points → must be SPARSE
    # on B side or the magenta/green soup is unreadable. Use one shared index
    # set so MAGENTA and GREEN dots are paired (same world point on B).
    N_SUB = 350        # same density on both frames
    all_idx = np.arange(len(pts_w_A))
    shared_sub = np.random.choice(all_idx, size=min(N_SUB, len(all_idx)), replace=False)
    sub_idx_A = shared_sub
    sub_idx_B = shared_sub

    # ── Frame A overlays
    pts_cam_A_gt = world_to_cam(pts_w_A, R_gt_A, cp_A)
    z_A_full = pts_cam_A_gt[:, 2]
    uv_A = project_pinhole(pts_cam_A_gt, K_A)
    valid_A = ((z_A_full > 0.5) & (uv_A[:, 0] >= 0) & (uv_A[:, 0] < IW_A)
               & (uv_A[:, 1] >= 0) & (uv_A[:, 1] < IH_A))
    for k in sub_idx_A[valid_A[sub_idx_A]]:
        draw_dot(full_A, uv_A[k, 0], uv_A[k, 1], YELLOW, size=1)
    draw_rect(full_A, u0_A, v0_A, cs, RED, thick=4)
    draw_dot(full_A, pu_A, pv_A, BLUE, size=8)

    # ── Frame B overlays
    # MAGENTA target: A lidar warped by T_gt_B (= what calib-perfect cross-frame gives)
    pts_cam_AB = world_to_cam(pts_w_A, R_gt_B, cp_B)
    z_AB = pts_cam_AB[:, 2]
    uv_AB = project_pinhole(pts_cam_AB, K_B)
    valid_AB = ((z_AB > 0.5) & (uv_AB[:, 0] >= 0) & (uv_AB[:, 0] < IW_B)
                & (uv_AB[:, 1] >= 0) & (uv_AB[:, 1] < IH_B))
    # GREEN input: A lidar warped to B with BOTH calib drift AND pose drift
    # composed (built by the dataset, see _last_pair_B['R_off']/['cp_off']).
    R_off_B_drift = np.asarray(last_B['R_off'])
    cp_off_B_drift = np.asarray(last_B['cp_off'])
    pts_cam_AB_pert = world_to_cam(pts_w_A, R_off_B_drift, cp_off_B_drift)
    z_AB_pert = pts_cam_AB_pert[:, 2]
    uv_AB_pert = project_pinhole(pts_cam_AB_pert, K_B)
    valid_AB_pert = ((z_AB_pert > 0.5) & (uv_AB_pert[:, 0] >= 0) & (uv_AB_pert[:, 0] < IW_B)
                     & (uv_AB_pert[:, 1] >= 0) & (uv_AB_pert[:, 1] < IH_B))
    # Same shared index → magenta and green pair up; draw the WHITE drift line
    # FIRST so the dots paint over the line endpoints (visible vector field).
    pair_mask_full = valid_AB[sub_idx_B] & valid_AB_pert[sub_idx_B]
    for k in sub_idx_B[pair_mask_full]:
        draw_line(full_B, uv_AB[k, 0], uv_AB[k, 1],
                          uv_AB_pert[k, 0], uv_AB_pert[k, 1], WHITE)
    for k in sub_idx_B[valid_AB[sub_idx_B]]:
        draw_dot(full_B, uv_AB[k, 0], uv_AB[k, 1], MAGENTA, size=2)
    for k in sub_idx_B[valid_AB_pert[sub_idx_B]]:
        draw_dot(full_B, uv_AB_pert[k, 0], uv_AB_pert[k, 1], GREEN, size=2)
    draw_rect(full_B, u0_B, v0_B, cs, CYAN, thick=4)
    draw_dot(full_B, pu_B, pv_B, BLUE, size=8)
    # Pivot identity: build_one_pair picked world pt index pivot_world_idx
    # whose A-projection is the A-pivot and whose T_gt_B projection is the
    # crop-B center (up to clip + int rounding when the pivot is near image
    # edge). Compute the residual as a sanity stamp.
    pivot_resid = float(np.linalg.norm(uv_AB[pivot_world_idx] - np.array([pu_B, pv_B])))

    # ── Crop panels (S=IMG_SIZE)
    crop_A_arr = img_A_crop.permute(1, 2, 0).numpy().copy()
    crop_B_arr = img_B_crop.permute(1, 2, 0).numpy().copy()
    # On A: yellow = own GT (true_A), green = perturbed (dist_A).
    # Rows are paired: (true_A[r], dist_A[r]) is the same lidar point under
    # GT vs perturbed pose. Draw a thin line first so dots paint on top.
    n_pair = min(true_A.shape[0], dist_A.shape[0])
    for r in range(n_pair):
        ut, vt = float(true_A[r, 0]), float(true_A[r, 1])
        ud, vd = float(dist_A[r, 0]), float(dist_A[r, 1])
        if not (0 <= ut < IMG_SIZE and 0 <= vt < IMG_SIZE):
            continue
        if not (0 <= ud < IMG_SIZE and 0 <= vd < IMG_SIZE):
            continue
        draw_line(crop_A_arr, ut, vt, ud, vd, WHITE)
    for r in range(true_A.shape[0]):
        u, v = float(true_A[r, 0]), float(true_A[r, 1])
        if 0 <= u < IMG_SIZE and 0 <= v < IMG_SIZE:
            draw_dot(crop_A_arr, u, v, YELLOW, size=1)
    for r in range(dist_A.shape[0]):
        u, v = float(dist_A[r, 0]), float(dist_A[r, 1])
        if 0 <= u < IMG_SIZE and 0 <= v < IMG_SIZE:
            draw_dot(crop_A_arr, u, v, GREEN, size=1)
    # On B: MAGENTA = cross-frame target, GREEN = input Q (A lidar with the
    # same calib drift, warped via Δpose_AB). Lines pair the same world point.
    # Sparsify hard: pick at most ~80 paired points so the drift vector field
    # is actually readable.
    in_crop = (valid_AB & valid_AB_pert
               & (uv_AB[:, 0] >= u0_B) & (uv_AB[:, 0] < u0_B + cs)
               & (uv_AB[:, 1] >= v0_B) & (uv_AB[:, 1] < v0_B + cs)
               & (uv_AB_pert[:, 0] >= u0_B) & (uv_AB_pert[:, 0] < u0_B + cs)
               & (uv_AB_pert[:, 1] >= v0_B) & (uv_AB_pert[:, 1] < v0_B + cs))
    crop_idx = np.where(in_crop)[0]
    N_CROP_B = 80
    if len(crop_idx) > N_CROP_B:
        crop_idx = np.random.choice(crop_idx, size=N_CROP_B, replace=False)
    for k in crop_idx:
        ut = (uv_AB[k, 0] - u0_B) * S / cs
        vt = (uv_AB[k, 1] - v0_B) * S / cs
        ud = (uv_AB_pert[k, 0] - u0_B) * S / cs
        vd = (uv_AB_pert[k, 1] - v0_B) * S / cs
        draw_line(crop_B_arr, ut, vt, ud, vd, WHITE)
    for k in crop_idx:
        ul = (uv_AB[k, 0] - u0_B) * S / cs
        vl = (uv_AB[k, 1] - v0_B) * S / cs
        draw_dot(crop_B_arr, ul, vl, MAGENTA, size=2)
    for k in crop_idx:
        ul = (uv_AB_pert[k, 0] - u0_B) * S / cs
        vl = (uv_AB_pert[k, 1] - v0_B) * S / cs
        draw_dot(crop_B_arr, ul, vl, GREEN, size=2)
    # Pivot on crops = centre by construction.
    draw_dot(crop_A_arr, IMG_SIZE / 2, IMG_SIZE / 2, BLUE, size=4)
    draw_dot(crop_B_arr, IMG_SIZE / 2, IMG_SIZE / 2, BLUE, size=4)

    # ── Legends
    leg_full_A = [
        (YELLOW, 'YELLOW : A lidar via T_gt_A (sparse)'),
        (RED,    'RED box: crop_A window'),
        (BLUE,   'BLUE   : pivot'),
    ]
    leg_full_B = [
        (MAGENTA, 'MAGENTA: A lidar via T_gt_B (target)'),
        (GREEN,   'GREEN  : A lidar via PERT calib + dpose (Q)'),
        (WHITE,   'WHITE  : drift line target<->Q'),
        (CYAN,    'CYAN bx: crop_B window'),
        (BLUE,    f'BLUE   : pivot (resid={pivot_resid:.2f}px)'),
    ]
    leg_crop_A = [
        (YELLOW, 'YEL: T_gt_A target'),
        (GREEN,  'GRN: pert Q'),
        (WHITE,  'WHT: pair line'),
        (BLUE,   'BLU: pivot'),
    ]
    leg_crop_B = [
        (MAGENTA, 'MAG: T_gt_B target'),
        (GREEN,   'GRN: calib+pose Q'),
        (WHITE,   'WHT: pair line'),
        (BLUE,    'BLU: pivot'),
    ]
    stamp_legend(full_A, leg_full_A, font_size=18)
    stamp_legend(full_B, leg_full_B, font_size=18)
    stamp_legend(crop_A_arr, leg_crop_A, font_size=11)
    stamp_legend(crop_B_arr, leg_crop_B, font_size=11)

    # ── Composite (top: full_A | full_B; bottom: crop_A | crop_B)
    pad = 8
    H = max(full_A.shape[0], full_B.shape[0])
    Wf = full_A.shape[1] + pad + full_B.shape[1]
    canvas_top = np.full((H, Wf, 3), 32, dtype=np.uint8)
    canvas_top[:full_A.shape[0], :full_A.shape[1]] = full_A
    canvas_top[:full_B.shape[0], full_A.shape[1] + pad:full_A.shape[1] + pad + full_B.shape[1]] = full_B
    Hc = IMG_SIZE
    Wc = IMG_SIZE * 2 + pad
    canvas_bot = np.full((Hc, Wc, 3), 32, dtype=np.uint8)
    canvas_bot[:, :IMG_SIZE] = crop_A_arr
    canvas_bot[:, IMG_SIZE + pad:IMG_SIZE * 2 + pad] = crop_B_arr
    spacer = np.full((pad, max(Wf, Wc), 3), 32, dtype=np.uint8)
    top_pad = np.full((H, max(Wf, Wc), 3), 32, dtype=np.uint8)
    top_pad[:, :Wf] = canvas_top
    bot_pad = np.full((Hc, max(Wf, Wc), 3), 32, dtype=np.uint8)
    bot_pad[:, :Wc] = canvas_bot
    composite = np.concatenate([top_pad, spacer, bot_pad], axis=0)
    out_path = OUT_DIR / f'{save_prefix}_composite.png'
    Image.fromarray(composite).save(out_path)
    Image.fromarray(full_A).save(OUT_DIR / f'{save_prefix}_frame_A_full.png')
    Image.fromarray(full_B).save(OUT_DIR / f'{save_prefix}_frame_B_full.png')
    Image.fromarray(crop_A_arr).save(OUT_DIR / f'{save_prefix}_crop_A.png')
    Image.fromarray(crop_B_arr).save(OUT_DIR / f'{save_prefix}_crop_B.png')

    print(f'[{save_prefix}] scene={inst_A.get("scene")} cam={inst_A.get("cam")} '
          f'frame_A={inst_A.get("frame")} frame_B={inst_B.get("frame")}  '
          f'cs={cs} vfp_A={float(vfp_A):.1f} vfp_B={float(vfp_B):.1f}  '
          f'Δpose_AB t=({dpose[0]:+.3f},{dpose[1]:+.3f},{dpose[2]:+.3f})  '
          f'ypr=({dpose[3]:+.3f},{dpose[4]:+.3f},{dpose[5]:+.3f})  '
          f'→ {out_path.relative_to(ROOT)}')


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE_DIR, split='train',
        img_size=IMG_SIZE, min_crop_px=MIN_CROP, max_crop_px=MAX_CROP,
        max_offset_m=0.20, max_rot_deg=1.0,
        pose_frame='orig', oversample=1, frame_stride=1,
        grid_n=16, k_per_cell=8,
        preload=False,
        pair_mode=True, pair_stride=1,
    )
    print(f'pair_index size = {len(ds.pair_index)}')
    for k in range(N_SAMPLES):
        pair_idx = int(np.random.randint(0, len(ds.pair_index)))
        try:
            render_one(ds, pair_idx, save_prefix=f'sample_{k:02d}')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'sample {k} failed: {e}')
    print(f'\nwrote {N_SAMPLES} composites to {OUT_DIR.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
