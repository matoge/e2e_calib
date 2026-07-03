"""Render a lateral-sweep video from a pose_opt ckpt.

Generates camera trajectory: forward along the original POSLV path while
sinusoidally sliding ±2 m in the vehicle's lateral (y) axis. With a learned
GS scene, this produces a "neighbour-lane drive-by" view that is impossible
without 3D reconstruction working — strong qualitative proof.

Usage (inside e2e-calib-splatkb:v1-examples container):
    python render_lateral_sweep.py \\
        --ckpt /raid/_splat_kb/woven_pinhole_pose_0613_0355/ckpts/ckpt_29999_rank0.pt \\
        --pandaset /raid/home/hfunaya/woven_pandaset_pylon/001_half \\
        --seq /mnt/ecp-perception/.../sequence=248_... \\
        --out /raid/_splat_kb/_lateral_sweep.mp4 \\
        --n-frames 200 --amp-y 2.0 --cycles 2.0
"""
from __future__ import annotations
import argparse, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
import cv2
import imageio.v3 as iio

sys.path.insert(0, '/host_e2e_calib/scripts/splatad_kb')
from woven_parser_pinhole import WovenParserPinhole
from gsplat.rendering import rasterization


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


_IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def apply_delta(c2w: torch.Tensor, d9: torch.Tensor) -> torch.Tensor:
    dx, drot = d9[:3], d9[3:9]
    rot = rotation_6d_to_matrix(drot + _IDENTITY_6D.to(d9.device).to(d9.dtype))
    T = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    T[:3, :3] = rot
    T[:3, 3] = dx
    return c2w @ T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--pandaset', type=Path, required=True)
    ap.add_argument('--seq', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-frames', type=int, default=200)
    ap.add_argument('--amp-y', type=float, default=0.20,
                    help='lateral amplitude in metres (e.g. 0.20 = ±20 cm)')
    ap.add_argument('--cycles', type=float, default=2.0,
                    help='number of full sin cycles over the trajectory')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--fixed-frame', type=int, default=None,
                    help='if set, hold base camera at this train frame index '
                          'and only oscillate laterally (no forward motion)')
    args = ap.parse_args()

    device = torch.device('cuda')

    parser = WovenParserPinhole(args.pandaset, args.seq, vehicle=args.vehicle)
    n = len(parser.image_names)
    test_every = parser.test_every
    train_idx_list = [i for i in range(n) if i % test_every != 0]

    ck = torch.load(str(args.ckpt), map_location='cpu', weights_only=False)
    splats = ck['splats']
    means = splats['means'].to(device)
    quats = splats['quats'].to(device)
    scales = torch.exp(splats['scales']).to(device)
    opacities = torch.sigmoid(splats['opacities']).to(device).squeeze(-1)
    colors = torch.cat([splats['sh0'].to(device), splats['shN'].to(device)], 1)

    pose_delta = None
    if 'pose_adjust' in ck:
        pose_delta = ck['pose_adjust']['embeds.weight'].to(device)
        print(f'[ckpt] pose_adjust loaded ({tuple(pose_delta.shape)})')

    K = torch.from_numpy(parser.Ks_dict[0]).float().to(device).unsqueeze(0)
    W, H = parser.imsize_dict[0]

    # base trajectory: linearly march through the 50 cam_t2w along time
    c2w_all = torch.from_numpy(parser.camtoworlds).float().to(device)  # (50,4,4)

    # for the sweep, parameter t in [0,1]; sample at n_frames
    ts = np.linspace(0.0, 1.0, args.n_frames)

    tmp = tempfile.mkdtemp(prefix='lateral_sweep_',
                              dir=str(args.out.parent))
    print(f'[render] tmp dir = {tmp}')

    for fi, t in enumerate(ts):
        # base camera pose
        if args.fixed_frame is not None:
            c2w_base = c2w_all[args.fixed_frame].clone()
            idx_f = float(args.fixed_frame)
        else:
            idx_f = t * (n - 1)
            i0 = int(np.floor(idx_f)); i1 = min(i0 + 1, n - 1)
            w1 = idx_f - i0; w0 = 1.0 - w1
            c2w_base = c2w_all[i0] * w0 + c2w_all[i1] * w1
            # re-orthogonalize R via SVD
            R = c2w_base[:3, :3]
            U, S, Vt = torch.linalg.svd(R)
            c2w_base[:3, :3] = U @ Vt

        # apply nearest-train delta if pose_opt
        if pose_delta is not None:
            ref_idx = int(round(idx_f))
            train_arr = np.asarray(train_idx_list)
            diffs = np.abs(train_arr - ref_idx)
            order = np.argsort(diffs)[:2]
            d0, d1 = diffs[order[0]], diffs[order[1]]
            if d0 + d1 == 0:
                interp = pose_delta[order[0]]
            else:
                w0_d = d1 / (d0 + d1); w1_d = d0 / (d0 + d1)
                interp = pose_delta[order[0]] * w0_d + pose_delta[order[1]] * w1_d
            c2w_base = apply_delta(c2w_base, interp)

        # add lateral offset along the camera's local "right" axis
        # cam coords are PS-cam: x=right, y=down, z=forward.
        # so "right in world" = c2w[:,:3] @ [1,0,0]
        right_world = c2w_base[:3, 0]
        offset = float(args.amp_y * np.sin(2 * np.pi * args.cycles * t))
        c2w = c2w_base.clone()
        c2w[:3, 3] = c2w[:3, 3] + right_world * offset

        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=3, packed=False, camera_model='pinhole',
        )
        img = out[0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        # annotate offset
        canvas = img.copy()
        cv2.putText(canvas, f't={t:.2f}  Δy={offset:+.2f}m',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        iio.imwrite(f'{tmp}/{fi:04d}.png', canvas)
        if fi % 20 == 0:
            print(f'  [{fi}/{args.n_frames}] t={t:.2f} Δy={offset:+.2f}m')

    print(f'[ffmpeg] -> {args.out}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(args.fps),
        '-i', f'{tmp}/%04d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        str(args.out),
    ], check=True)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
