"""Sanity: can the SAME network learn Δuv directly from per-token supervision?

Cuts BA out of the picture entirely. Network input = same as
overfit_1image_random_delta.py (RGB + 1/z heatmap + CoordConv).
Loss = ‖duv − duv_oracle‖² over the in-tile points (per-point MSE,
gathered with the same nearest-cell scheme as the BA run).

If THIS doesn't drop to ~0 even with random δ_pert per step, the
problem is the network (receptive field, capacity, …), not BA.
If it drops fast → network is fine, BA-loss path is the bottleneck.
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


def main():
    torch.manual_seed(7); np.random.seed(7)
    rng = np.random.RandomState(7)

    cf = load_frame(rd.m.SCENE, rd.m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + rd.m.DV
    u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
    u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
    img_tile_np = img_full[v0:v1, u0:u1].copy()

    model = rd.TileToTokenHeadCoord(d=96, n_blocks=2, in_ch=3).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'model: {n_param/1e3:.1f}k params  (same as random-δ run)')
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    log_iter, log_loss = [], []
    print(f'\n  {"iter":>4}  {"loss":>10}  {"|Δuv-orcl|":>12}  {"N":>5}')
    for it in range(N_ITER + 1):
        delta_pert = rd.sample_delta(rng)
        uv_pert_np, z_pert_np, pts_cam_in = rd.perturb_pool(
            pts_cam, K, dist, delta_pert, u0, v0, u1, v1)
        if len(uv_pert_np) < 50:
            continue
        uv_local_np = uv_pert_np.copy()
        uv_local_np[:, 0] -= u0; uv_local_np[:, 1] -= v0
        duv_oracle_np = project_kb(pts_cam_in, K, dist) - uv_pert_np

        img_t = rd.render_input(img_tile_np, uv_local_np, z_pert_np)
        uv_l_t = torch.from_numpy(uv_local_np).float().to(DEVICE)
        duv_oracle_t = torch.from_numpy(duv_oracle_np).float().to(DEVICE)

        opt.zero_grad()
        duv_map, _ = model(img_t)
        duv = rd.m.gather_token(duv_map, uv_l_t, rd.m.TILE)        # (N, 2)
        loss = ((duv - duv_oracle_t) ** 2).mean()
        loss.backward(); opt.step()

        if it % LOG_EVERY == 0 or it == N_ITER:
            with torch.no_grad():
                duv_err = (duv - duv_oracle_t).norm(dim=-1).mean().item()
            log_iter.append(it); log_loss.append(loss.item())
            print(f'  {it:>4}  {loss.item():>10.4f}  {duv_err:>12.3f}  '
                   f'{len(uv_pert_np):>5d}')

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.semilogy(log_iter, log_loss, '-', lw=2)
    ax.set_xlabel('Adam iter'); ax.set_ylabel('per-pt MSE  [px²]')
    ax.set_title('Direct Δuv supervision (no BA in loss)\n'
                   'random δ_pert per step, RGB+1/z heatmap+CoordConv input')
    ax.grid(which='both', alpha=0.3)
    fig.tight_layout()
    out_path = OUT / 'overfit_directduv_supervise.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
