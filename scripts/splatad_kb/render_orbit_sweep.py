"""Render an orbit (front/back/left/right) sweep video from a pose_opt ckpt.

Camera position traces a circle in the cam-local x/z plane (right-forward),
holding orientation fixed. amp = circle radius in metres.
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
    ap.add_argument('--n-frames', type=int, default=180)
    ap.add_argument('--amp', type=float, default=0.30,
                    help='orbit radius in metres')
    ap.add_argument('--cycles', type=float, default=2.0)
    ap.add_argument('--fixed-frame', type=int, default=1,
                    help='if --march not set, hold base camera at this frame')
    ap.add_argument('--march', action='store_true',
                    help='walk along POSLV path while orbiting (overrides --fixed-frame)')
    ap.add_argument('--yaw-amp-deg', type=float, default=0.0,
                    help='peak yaw oscillation amplitude (deg), about cam-y axis')
    ap.add_argument('--yaw-cycles', type=float, default=2.0)
    ap.add_argument('--fps', type=int, default=30)
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
    c2w_all = torch.from_numpy(parser.camtoworlds).float().to(device)

    def base_at(idx_f: float):
        i0 = int(np.floor(idx_f)); i1 = min(i0 + 1, n - 1)
        w1 = idx_f - i0; w0 = 1.0 - w1
        c2w_b = c2w_all[i0] * w0 + c2w_all[i1] * w1
        R = c2w_b[:3, :3]
        U, S, Vt = torch.linalg.svd(R)
        c2w_b = c2w_b.clone()
        c2w_b[:3, :3] = U @ Vt
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
            c2w_b = apply_delta(c2w_b, interp)
        return c2w_b

    ts = np.linspace(0.0, 1.0, args.n_frames)
    tmp = tempfile.mkdtemp(prefix='orbit_sweep_', dir=str(args.out.parent))
    print(f'[render] tmp dir = {tmp}')

    for fi, t in enumerate(ts):
        if args.march:
            idx_f = t * (n - 1)
        else:
            idx_f = float(args.fixed_frame)
        c2w_base = base_at(idx_f)
        right_world   = c2w_base[:3, 0]
        forward_world = c2w_base[:3, 2]
        ang = 2 * np.pi * args.cycles * t
        dx_amt = args.amp * np.cos(ang)
        dz_amt = args.amp * np.sin(ang)
        c2w = c2w_base.clone()
        c2w[:3, 3] = (c2w_base[:3, 3]
                      + right_world * float(dx_amt)
                      + forward_world * float(dz_amt))
        yaw_deg = 0.0
        if args.yaw_amp_deg != 0.0:
            yaw_deg = args.yaw_amp_deg * float(np.sin(2 * np.pi * args.yaw_cycles * t))
            theta = np.radians(yaw_deg)
            cy_t = np.cos(theta); sy_t = np.sin(theta)
            # cam-local Ry: rotate about cam +y (down). Multiply on the right
            # so the rotation is applied in cam frame (orientation only).
            Ry = torch.tensor([[ cy_t, 0.0, sy_t, 0.0],
                                [ 0.0,  1.0, 0.0,  0.0],
                                [-sy_t, 0.0, cy_t, 0.0],
                                [ 0.0,  0.0, 0.0,  1.0]],
                               dtype=c2w.dtype, device=c2w.device)
            c2w = c2w @ Ry

        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=3, packed=False, camera_model='pinhole',
        )
        img = out[0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        canvas = img.copy()
        cv2.putText(canvas,
                    f'right={dx_amt:+.2f}m  fwd={dz_amt:+.2f}m  ang={np.degrees(ang)%360:6.1f}deg',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        iio.imwrite(f'{tmp}/{fi:04d}.png', canvas)
        if fi % 20 == 0:
            print(f'  [{fi}/{args.n_frames}] right={dx_amt:+.2f} fwd={dz_amt:+.2f}')

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
