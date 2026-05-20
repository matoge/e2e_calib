"""Sanity: SAME network as overfit_1image_random_delta.py, but δ_pert is FIXED.
If this can't drive |Δuv-oracle| → 0, the random-δ pipeline isn't the bug.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'rd', str(REPO / 'scripts' / '_debug' / 'overfit_1image_random_delta.py'))
rd = importlib.util.module_from_spec(spec); _s.modules['rd'] = rd
spec.loader.exec_module(rd)

from scripts.ba.ba_kb_jac import project_kb
from scripts.ba.ba_torch import solve_kb

DEVICE = rd.DEVICE
N_ITER = 600
LOG_EVERY = 20
BATCH_SIZE = 32
LR = 1e-4
FIXED_DELTA = (1.0, 1.5)


def main():
    torch.manual_seed(7); np.random.seed(7)
    cf = rd.load_frame(rd.m.SCENE, rd.m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + rd.m.DV
    u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
    u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
    img_tile_np = img_full[v0:v1, u0:u1].copy()

    K_t = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)
    dist_t = torch.from_numpy(dist).float().unsqueeze(0).to(DEVICE)

    model = rd.TileToTokenHeadCoord(d=96, n_blocks=1, n_heads=4).to(DEVICE)
    print(f'model: {sum(p.numel() for p in model.parameters())/1e3:.1f}k params')

    rgb_t_b = torch.from_numpy(
        img_tile_np.transpose(2, 0, 1)[None]).to(DEVICE).expand(
        BATCH_SIZE, -1, -1, -1).contiguous()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── Fix δ once and pre-build all the per-batch tensors ─────────
    omx, omy = FIXED_DELTA
    d_pert = rd.make_delta6(omx, omy)
    uv_p, z_p, pc_in = rd.perturb_pool(pts_cam, K, dist, d_pert,
                                        u0, v0, u1, v1)
    uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
    duv_o = project_kb(pc_in, K, dist) - uv_p
    print(f'δ=({omx:+.1f},{omy:+.1f})  N={len(uv_p)}  '
           f'|Δuv_oracle| mean={np.linalg.norm(duv_o, axis=1).mean():.2f} px')

    H_t = W_t = rd.m.TILE
    dep_one = rd.render_depth(uv_l, z_p, H_t, W_t)[None]   # (1, 1, H, W)
    dep_t_b = torch.from_numpy(dep_one).to(DEVICE).expand(
        BATCH_SIZE, -1, -1, -1).contiguous()

    uv_p_t = torch.from_numpy(uv_p).float().unsqueeze(0).to(DEVICE)
    uv_l_t = torch.from_numpy(uv_l).float().unsqueeze(0).to(DEVICE)
    z_t    = torch.from_numpy(z_p).float().unsqueeze(0).to(DEVICE)
    duv_o_t = torch.from_numpy(duv_o).float().unsqueeze(0).to(DEVICE)
    valid_t = torch.ones(1, len(uv_p), dtype=torch.bool, device=DEVICE)

    K_t_b    = K_t
    dist_t_b = dist_t

    print(f'\n  {"iter":>4}  {"loss":>10}  {"|Δuv-orcl|":>10}  {"|δ̂-tgt|":>10}')
    for it in range(N_ITER + 1):
        opt.zero_grad()
        duv_map, _ = model(rgb_t_b[:1], dep_t_b[:1])
        duv = rd.gather_token_batched(duv_map, uv_l_t, rd.m.TILE, valid_t)
        sq = (duv - duv_o_t).pow(2).sum(dim=-1)
        loss = sq[valid_t].mean()
        loss.backward(); opt.step()

        if it % LOG_EVERY == 0:
            with torch.no_grad():
                err = (duv - duv_o_t).norm(dim=-1)[valid_t].mean().item()
                W = torch.eye(2, device=DEVICE, dtype=duv.dtype).expand(
                    1, len(uv_p), 2, 2)
                delta_hat, _ = solve_kb(
                    uv_p_t, duv, W, z_t, K_t, dist_t, rd.DOF_ACTIVE,
                    valid=valid_t, n_iter=1, damping=1e-3,
                )
                target = torch.tensor([-omx, -omy], device=DEVICE)
                derr = (delta_hat[0] - target).abs().max().item()
            print(f'  {it:>4}  {loss.item():>10.4f}  {err:>10.3f}  {derr:>10.4f}')


if __name__ == '__main__':
    main()
