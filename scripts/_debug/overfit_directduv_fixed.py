"""Sanity: direct Δuv supervision with FIXED δ_pert (no BA, no random).

Strict ablation of overfit_directduv_supervise.py: same network and input
encoding, but δ_pert is held constant. If THIS doesn't drop to ~0,
the network architecture is broken. If it does, the random-δ failure
is "moving target / lr too high", not architecture.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'rd', str(REPO / 'scripts' / '_debug' / 'overfit_1image_random_delta.py'))
rd = importlib.util.module_from_spec(spec); _s.modules['rd'] = rd
spec.loader.exec_module(rd)

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import project_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
DEVICE = rd.DEVICE
N_ITER = 600
LR = 1e-3
LOG_EVERY = 20
DELTA_FIXED = rd.make_delta6(1.0, 1.5)         # = baseline δ_pert


def main():
    torch.manual_seed(7); np.random.seed(7)

    cf = load_frame(rd.m.SCENE, rd.m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + rd.m.DV
    u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
    u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
    img_tile_np = img_full[v0:v1, u0:u1].copy()

    # Static pool (fixed δ).
    uv_pert_np, z_pert_np, pts_cam_in = rd.perturb_pool(
        pts_cam, K, dist, DELTA_FIXED, u0, v0, u1, v1)
    uv_local_np = uv_pert_np.copy()
    uv_local_np[:, 0] -= u0; uv_local_np[:, 1] -= v0
    duv_oracle_np = project_kb(pts_cam_in, K, dist) - uv_pert_np
    print(f'pts in tile: {len(uv_pert_np)}')
    print(f'oracle |Δuv|: mean {np.linalg.norm(duv_oracle_np, axis=1).mean():.1f} px')

    img_t = rd.render_input(img_tile_np, uv_local_np, z_pert_np)
    uv_l_t = torch.from_numpy(uv_local_np).float().to(DEVICE)
    duv_oracle_t = torch.from_numpy(duv_oracle_np).float().to(DEVICE)

    model = rd.TileToTokenHeadCoord(d=96, n_blocks=2, in_ch=3).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'model: {n_param/1e3:.1f}k params')
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    log_iter, log_loss, log_err = [], [], []
    print(f'\n  {"iter":>4}  {"loss":>10}  {"|Δuv-orcl|":>12}')
    for it in range(N_ITER + 1):
        opt.zero_grad()
        duv_map, _ = model(img_t)
        duv = rd.m.gather_token(duv_map, uv_l_t, rd.m.TILE)
        loss = ((duv - duv_oracle_t) ** 2).mean()
        loss.backward(); opt.step()

        if it % LOG_EVERY == 0 or it == N_ITER:
            with torch.no_grad():
                err = (duv - duv_oracle_t).norm(dim=-1).mean().item()
            log_iter.append(it); log_loss.append(loss.item()); log_err.append(err)
            print(f'  {it:>4}  {loss.item():>10.4f}  {err:>12.3f}')

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(log_iter, log_loss, '-', lw=2, label='per-pt MSE [px²]')
    ax.semilogy(log_iter, log_err, '--', lw=1.5, label='|Δuv-oracle| mean [px]')
    ax.set_xlabel('Adam iter'); ax.set_ylabel('value (log)')
    ax.set_title('Direct Δuv supervision · FIXED δ_pert\n'
                   '32×32 token, 16-window Swin self-attn, 1/z heatmap input')
    ax.grid(which='both', alpha=0.3); ax.legend()
    fig.tight_layout()
    out_path = OUT / 'overfit_directduv_fixed.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
