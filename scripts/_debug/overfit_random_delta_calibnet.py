"""1-image random-δ overfit using the production CalibNetDepth backbone.

Same data pipeline as overfit_1image_random_delta.py, but the network is the
real `models.model_depth.CalibNetDepth` (per-point query + cross-attn). The 5
output channels are reused as [Δuv (2), L (3 = lower-tri Cholesky of W)].

Two losses available — flip DIRECT_DUV_SUPERVISE:
  True : MSE(duv_pred, duv_oracle)        — sanity that the per-point head can
                                            regress Δuv at all under random δ.
  False: MSE(δ̂, δ_target) via 1-step KB BA — the actual graduation gate.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'rd', str(REPO / 'scripts' / '_debug' / 'overfit_1image_random_delta.py'))
rd = importlib.util.module_from_spec(spec); _s.modules['rd'] = rd
spec.loader.exec_module(rd)

from scripts.ba.ba_kb_jac import project_kb
from scripts.ba.ba_torch import solve_kb
from models.model_depth import CalibNetDepth

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DOF_ACTIVE   = ['omega_x', 'omega_y']
DELTA_RANGE  = 1.0
BATCH_SIZE   = 8
LR           = 1e-4
N_ITER       = 600
LOG_EVERY    = 20
DIRECT_DUV_SUPERVISE = True       # Δuv first; flip to False once it's < 5 px.
IMG_SIZE_NET = 128                # Resize the 512 tile to 128 to match production
                                   # (CalibNetDepth's PE / deformable / CNN strides
                                   # are all tuned for img_size=128).
FREEZE_W     = True                # use W = I; isolates Δuv signal from W-hack.

HELDOUT_2D = [( 1.0, 1.5), (-0.8, 1.2), (0.6, -0.7)]


def build_W_from_L(L_raw, sigma_floor=0.2):
    floor = 1.0 / sigma_floor
    l11 = F.softplus(L_raw[..., 0]) + floor
    l21 = L_raw[..., 1]
    l22 = F.softplus(L_raw[..., 2]) + floor
    z = torch.zeros_like(l11)
    L = torch.stack([
        torch.stack([l11, z],   dim=-1),
        torch.stack([l21, l22], dim=-1),
    ], dim=-2)
    return L @ L.transpose(-1, -2)


def main():
    torch.manual_seed(7); np.random.seed(7)
    rng = np.random.RandomState(7)

    cf = rd.load_frame(rd.m.SCENE, rd.m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + rd.m.DV
    u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
    u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
    img_tile_np = img_full[v0:v1, u0:u1].copy()
    TILE = rd.m.TILE
    SCALE = IMG_SIZE_NET / TILE                            # 128/512 = 0.25

    # Resize the tile once — feed the network a 128×128 image but keep all
    # geometry (uv, depth, K, dist) in the original 512-pixel space; we just
    # scale uv by SCALE on the way INTO the net and Δuv by 1/SCALE on the way
    # OUT.  K does NOT need to change because we never reproject in net space.
    img_tile_small = (
        F.interpolate(
            torch.from_numpy(img_tile_np.transpose(2, 0, 1)[None]),
            size=(IMG_SIZE_NET, IMG_SIZE_NET),
            mode='bilinear', align_corners=False,
        )[0].numpy()
    )

    K_t    = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)
    dist_t = torch.from_numpy(dist).float().unsqueeze(0).to(DEVICE)

    model = CalibNetDepth(
        d=128, img_size=IMG_SIZE_NET, in_channels=3,
        n_layers=2, self_first=False, kv_self_attn=False,
        use_convnext=True, use_intensity=False,
        convnext_n_blocks=2,
    ).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model: {n_param/1e6:.2f}M params  (CalibNetDepth, ConvNeXt, n_layers=2)')

    rgb_t_b = torch.from_numpy(img_tile_small[None]).to(DEVICE).expand(
        BATCH_SIZE, -1, -1, -1).contiguous()

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    K_t_b    = K_t.expand(BATCH_SIZE, -1, -1)
    dist_t_b = dist_t.expand(BATCH_SIZE, -1)

    print(f'\n  {"iter":>4}  {"loss":>10}  {"|δ̂-tgt|":>10}  {"|Δuv-orcl|":>10}  '
           f'{"N_max":>6}  {"N_min":>6}')

    for it in range(N_ITER + 1):
        per_b = []
        for _ in range(BATCH_SIZE):
            while True:
                a = rng.uniform(-DELTA_RANGE, DELTA_RANGE)
                b = rng.uniform(-DELTA_RANGE, DELTA_RANGE)
                d = rd.make_delta6(a, b)
                uv_p, z_p, pc_in = rd.perturb_pool(pts_cam, K, dist, d,
                                                    u0, v0, u1, v1)
                if len(uv_p) >= 50:
                    break
            uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
            duv_o = project_kb(pc_in, K, dist) - uv_p
            per_b.append(dict(d=d, uv_p=uv_p, uv_l=uv_l, z=z_p, duv_o=duv_o))

        N_max = max(len(b['uv_p']) for b in per_b)
        uv_p_b   = np.zeros((BATCH_SIZE, N_max, 2), np.float32)
        uv_l_b   = np.zeros((BATCH_SIZE, N_max, 2), np.float32)
        z_b      = np.zeros((BATCH_SIZE, N_max), np.float32)
        duv_o_b  = np.zeros((BATCH_SIZE, N_max, 2), np.float32)
        valid_b  = np.zeros((BATCH_SIZE, N_max), bool)
        for bi, b in enumerate(per_b):
            n = len(b['uv_p'])
            uv_p_b[bi, :n]  = b['uv_p']
            uv_l_b[bi, :n]  = b['uv_l']
            z_b[bi, :n]     = b['z']
            duv_o_b[bi, :n] = b['duv_o']
            valid_b[bi, :n] = True

        # CalibNetDepth wants distorted_uvd with U,V in *pixels of the input
        # image*, so scale uv from 512-tile space to 128-net space.
        uv_l_net = uv_l_b * SCALE
        d_norm_b = z_b[..., None] / 50.0
        uvd_in = np.concatenate([uv_l_net, d_norm_b], axis=-1)            # (B,N,3)
        uvd_t   = torch.from_numpy(uvd_in).float().to(DEVICE)
        valid_t = torch.from_numpy(valid_b).to(DEVICE)
        uv_p_t  = torch.from_numpy(uv_p_b).to(DEVICE)
        uv_l_t  = torch.from_numpy(uv_l_b).to(DEVICE)
        z_t     = torch.from_numpy(z_b).to(DEVICE)
        duv_o_t = torch.from_numpy(duv_o_b).to(DEVICE)
        delta_target = torch.stack([
            torch.from_numpy(-b['d'][:2]).float() for b in per_b
        ], dim=0).to(DEVICE)

        # CalibNetDepth's kp-mask convention: True == padding.
        kp_mask = ~valid_t

        opt.zero_grad()
        out = model(rgb_t_b, uvd_t, key_padding_mask=kp_mask)              # (B,N,5)
        duv = out[..., :2] / SCALE                                         # → 512-space px
        L_raw = out[..., 2:5]
        if FREEZE_W:
            W = torch.eye(2, device=DEVICE, dtype=duv.dtype).expand(
                BATCH_SIZE, N_max, 2, 2)
        else:
            W = build_W_from_L(L_raw)

        if DIRECT_DUV_SUPERVISE:
            sq = (duv - duv_o_t).pow(2).sum(dim=-1)
            loss = sq[valid_t].mean()
            with torch.no_grad():
                delta_hat, _ = solve_kb(
                    uv_p_t, duv, W, z_t, K_t_b, dist_t_b, DOF_ACTIVE,
                    valid=valid_t, n_iter=1, damping=1e-3,
                )
        else:
            delta_hat, _ = solve_kb(
                uv_p_t, duv, W, z_t, K_t_b, dist_t_b, DOF_ACTIVE,
                valid=valid_t, n_iter=1, damping=1e-3,
            )
            loss = ((delta_hat - delta_target) ** 2).mean()

        loss.backward(); opt.step()

        if it % LOG_EVERY == 0 or it == N_ITER:
            with torch.no_grad():
                derr = (delta_hat - delta_target).abs().max().item()
                err = (duv - duv_o_t).norm(dim=-1)
                duv_err = err[valid_t].mean().item()
            n_min = min(len(b['uv_p']) for b in per_b)
            print(f'  {it:>4}  {loss.item():>10.5f}  {derr:>10.4f}  '
                   f'{duv_err:>10.3f}  {N_max:>6d}  {n_min:>6d}')

    # ─── Held-out evaluation ─────────────────────────────────────────
    print('\n=== held-out evaluation ===')
    print(f'  {"δ_pert":<28}  {"|δ̂-tgt|":>10}  {"|Δuv-orcl|":>10}')
    for omx, omy in HELDOUT_2D:
        d_pert = rd.make_delta6(omx, omy)
        uv_p, z_p, pc_in = rd.perturb_pool(pts_cam, K, dist, d_pert,
                                            u0, v0, u1, v1)
        uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
        duv_o = project_kb(pc_in, K, dist) - uv_p

        d_norm = z_p[..., None] / 50.0
        uvd = np.concatenate([uv_l * SCALE, d_norm], axis=-1)
        uvd_t = torch.from_numpy(uvd[None]).float().to(DEVICE)
        valid_t = torch.ones(1, len(uv_p), dtype=torch.bool, device=DEVICE)

        with torch.no_grad():
            out = model(rgb_t_b[:1], uvd_t, key_padding_mask=~valid_t)
            duv_pred = out[..., :2] / SCALE
            L_raw_pred = out[..., 2:5]
            if FREEZE_W:
                W = torch.eye(2, device=DEVICE, dtype=duv_pred.dtype).expand(
                    1, len(uv_p), 2, 2)
            else:
                W = build_W_from_L(L_raw_pred)
            uv_p_t = torch.from_numpy(uv_p[None]).float().to(DEVICE)
            z_t    = torch.from_numpy(z_p[None]).float().to(DEVICE)
            delta_hat, _ = solve_kb(
                uv_p_t, duv_pred, W, z_t, K_t, dist_t, DOF_ACTIVE,
                n_iter=1, damping=1e-3,
            )
            target = torch.tensor([[-omx, -omy]], device=DEVICE)
            derr = (delta_hat - target).abs().max().item()
            duv_err = (duv_pred[0].cpu().numpy() - duv_o)
            duv_err_mean = float(np.linalg.norm(duv_err, axis=1).mean())

        tag = f'(ωx={omx:+.1f}, ωy={omy:+.1f})'
        print(f'  {tag:<28}  {derr:>10.4f}  {duv_err_mean:>10.3f}')


if __name__ == '__main__':
    main()
