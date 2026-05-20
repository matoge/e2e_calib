"""1-image overfit baseline: ConvNeXt → per-token (Δuv, L) → KB BA → 6-DoF
pose loss. Verify that the gradient pipeline (network → ba_torch → loss →
backward) actually trains: a single (image, δ_pert) pair should reach
‖δ̂ − δ_target‖² ≈ 0 in a few hundred Adam steps.

Setup:
  • Image: 512×512 crop of `points_ip664_D_20260226_224648_d005_3000_3020/
    image_0.png`, centred at (W/2, H/2 + 300) — same as Fig 6.
  • Perturbation: fixed δ_pert ∈ R^6 (~1° rot + ~0.3 m trans).  The image
    is GT-pose; we project ALL frame LiDAR pts under the perturbed
    calibration. The pts whose perturbed projection lands inside the tile
    form the BA pool.
  • Network: 32×32-token ConvNeXt (stem stride 4 + one downsample = /16
    on a 512 tile) + per-token Δuv head (2) + per-token Cholesky head
    (3 → lower-triangular L; W = LLᵀ).
  • Each LiDAR point pulls its (Δuv, L) from the nearest token cell.
  • BA: scripts.ba.ba_torch.solve_kb, 6-DoF, n_iter=2.
  • Pose loss: per-DoF scale (deg→1, m→10 so the two contribute equally
    at unit perturbation), then mean-squared.
  • Optimiser: Adam lr=1e-3, 200 iter.

Output:
  docs/assets/2026-05-19_diffba/overfit_loss_curve.png
  Console: per-iter loss + δ̂ trajectory.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import project_kb            # numpy reference for setup
from scripts.ba.ba_torch import solve_kb                # torch / autograd

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
OUT.mkdir(parents=True, exist_ok=True)

SCENE = Path.home() / 'cache' / 'kamikado' / 'scenes' / \
        'points_ip664_D_20260226_224648_d005_3000_3020'
FRAME = 0
TILE = 512
DV = 300          # tile centre = (W/2, H/2 + DV)
TOKEN_GRID = 32   # 32×32 token grid → 16-px cell

# Same δ_pert as the visible Fig 6 (~1° rot, ~0.3 m trans).
DELTA_PERT = np.array([1.0, 1.5, 0.5,        # ω_x, ω_y, ω_z  [deg]
                        0.20, -0.30, 0.40],   # tx, ty, tz     [m]
                       dtype=np.float64)
DOF = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
# Loss scale: rotations in deg, translations ×10 so a 1° error and a 0.1 m
# error contribute equally. Net effect: the loss treats 1° ≡ 0.1 m.
LOSS_SCALE = torch.tensor([1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=torch.float32)

LR = 1e-3
N_ITER = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 7


# ─── Network: tiny ConvNeXt-ish backbone → per-token (Δuv, L) ──────────
class _ConvNeXtBlock(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, d * expansion)
        self.pw2 = nn.Linear(d * expansion, d)

    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + h.permute(0, 3, 1, 2)


class TileToTokenHead(nn.Module):
    """512×512 RGB → 32×32 token of dim D, then per-token (Δuv, L)."""
    def __init__(self, d=96, n_blocks=2):
        super().__init__()
        # stride-4 patchify + stride-4 → /16 = 32×32 on a 512 tile.
        self.stem = nn.Sequential(
            nn.Conv2d(3, d // 2, 4, stride=4),
            nn.GELU(),
            nn.Conv2d(d // 2, d, 4, stride=4),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ConvNeXtBlock(d) for _ in range(n_blocks)])
        self.head_duv = nn.Conv2d(d, 2, 1)            # Δu, Δv
        self.head_L   = nn.Conv2d(d, 3, 1)            # l11, l21, l22  (lower tri)

    def forward(self, img):
        x = self.stem(img)
        x = self.blocks(x)
        duv = self.head_duv(x)                        # (B, 2, 32, 32)
        L_raw = self.head_L(x)                        # (B, 3, 32, 32)
        return duv, L_raw


def gather_token(token_map, uv_in_tile, tile_size):
    """token_map: (1, C, G, G); uv_in_tile: (N, 2) ∈ [0, tile_size).
    Returns (N, C) by nearest-neighbour cell lookup."""
    G = token_map.shape[-1]
    cell_size = tile_size / G
    gx = torch.clamp((uv_in_tile[:, 0] / cell_size).long(), 0, G - 1)
    gy = torch.clamp((uv_in_tile[:, 1] / cell_size).long(), 0, G - 1)
    flat = token_map[0].permute(1, 2, 0).reshape(G * G, -1)   # (G², C)
    return flat[gy * G + gx]


def build_W_from_L(L_raw, sigma_floor=0.2):
    """L_raw: (..., 3) → W: (..., 2, 2) PSD, with diagonal ≥ 1/sigma_floor²
    (i.e. cap σ at sigma_floor px to avoid runaway).
    L = [[l11, 0], [l21, l22]] with l11, l22 = softplus(.) + 1/sigma_floor."""
    floor = 1.0 / sigma_floor
    l11 = F.softplus(L_raw[..., 0]) + floor
    l21 = L_raw[..., 1]
    l22 = F.softplus(L_raw[..., 2]) + floor
    z = torch.zeros_like(l11)
    L = torch.stack([
        torch.stack([l11, z],   dim=-1),
        torch.stack([l21, l22], dim=-1),
    ], dim=-2)
    return L @ L.transpose(-1, -2)            # (..., 2, 2)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)

    # ─── Data prep ────────────────────────────────────────────────────
    cf = load_frame(SCENE, FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64)
    dist = np.asarray(cf.dist, dtype=np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    print(f'frame {W_full}×{H_full}  pts {len(pts_cam)}  fx={K[0,0]:.1f}')

    cu, cv = W_full // 2, H_full // 2 + DV
    u0 = cu - TILE // 2; v0 = cv - TILE // 2
    u1 = u0 + TILE;       v1 = v0 + TILE

    # Apply δ_pert to camera pose (perturbed extrinsic). Project ALL frame
    # LiDAR pts under the perturbed calibration; keep those whose perturbed
    # uv lands inside the tile.
    R_pert = Rotation.from_rotvec(np.deg2rad(DELTA_PERT[:3])).as_matrix()
    pts_cam_pert = pts_cam @ R_pert.T + DELTA_PERT[3:6]
    uv_pert_all = project_kb(pts_cam_pert, K, dist)
    z_pert_all = pts_cam_pert[:, 2]
    in_tile = ((uv_pert_all[:, 0] >= u0) & (uv_pert_all[:, 0] < u1)
                & (uv_pert_all[:, 1] >= v0) & (uv_pert_all[:, 1] < v1)
                & (z_pert_all > 0.5))
    uv_pert = uv_pert_all[in_tile]                 # parent-px
    z_pert = z_pert_all[in_tile]
    pts_cam_in = pts_cam[in_tile]
    print(f'in-tile-after-pert pts: {len(uv_pert)} / {len(pts_cam)}')

    # uv in tile-local coords (for token gather + visualisation).
    uv_local = uv_pert.copy()
    uv_local[:, 0] -= u0
    uv_local[:, 1] -= v0

    # ─── Tensor packing ───────────────────────────────────────────────
    img_t = torch.from_numpy(img_full[v0:v1, u0:u1].copy()).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,512,512)
    uv_p_t = torch.from_numpy(uv_pert).float().to(device)               # parent-px (N,2)
    uv_l_t = torch.from_numpy(uv_local).float().to(device)              # tile-local (N,2)
    z_t   = torch.from_numpy(z_pert).float().to(device)                 # (N,)
    K_t   = torch.from_numpy(K).float().unsqueeze(0).to(device)         # (1,3,3)
    dist_t = torch.from_numpy(dist).float().unsqueeze(0).to(device)     # (1,4)

    # δ_target = -δ_pert (BA convention: δ̂ undoes the perturbation by
    # rotating uv_pert's back-projection to land at uv_GT = uv_pert + duv*).
    delta_target = torch.tensor(-DELTA_PERT, dtype=torch.float32, device=device)
    loss_scale = LOSS_SCALE.to(device)

    # ─── Model ────────────────────────────────────────────────────────
    model = TileToTokenHead(d=96, n_blocks=2).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'model: {n_param/1e3:.1f}k params')
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Ground-truth Δuv (so we can monitor the per-pt error during training).
    uv_gt_all = project_kb(pts_cam_in, K, dist)
    duv_oracle = torch.from_numpy(uv_gt_all - uv_pert).float().to(device)

    # ─── Training loop ────────────────────────────────────────────────
    losses = []
    delta_hat_hist = []
    duv_err_hist = []
    print(f'\n  {"iter":>4}  {"loss":>10}  {"|δ̂-tgt|":>10}  {"|Δuv-oracle|":>14}')
    for it in range(N_ITER + 1):
        opt.zero_grad()
        duv_map, L_map = model(img_t)                          # (1,2,32,32),(1,3,32,32)
        # Per-pt look-up via token gather.
        duv = gather_token(duv_map,  uv_l_t, TILE)             # (N, 2)
        L_raw = gather_token(L_map.contiguous(), uv_l_t, TILE) # (N, 3)
        W = build_W_from_L(L_raw)                              # (N, 2, 2)

        # Add batch dim for ba_torch.
        delta_hat, _ = solve_kb(
            uv_p_t.unsqueeze(0), duv.unsqueeze(0), W.unsqueeze(0),
            z_t.unsqueeze(0), K_t, dist_t, DOF,
            n_iter=2, damping=1e-3,
        )                                                       # (1, 6)
        delta_hat = delta_hat[0]                                # (6,)

        # Pose loss: scaled MSE of (δ̂ − δ_target).
        loss = (((delta_hat - delta_target) * loss_scale) ** 2).mean()
        loss.backward()
        opt.step()

        if it % 10 == 0 or it == N_ITER:
            with torch.no_grad():
                derr = (delta_hat - delta_target).abs().max().item()
                duv_err = (duv - duv_oracle).norm(dim=-1).mean().item()
            losses.append((it, loss.item()))
            delta_hat_hist.append(delta_hat.detach().cpu().numpy().copy())
            duv_err_hist.append(duv_err)
            print(f'  {it:>4}  {loss.item():>10.5f}  {derr:>10.4f}  {duv_err:>14.4f}')

    # ─── Final report ────────────────────────────────────────────────
    print('\n  Final:')
    print(f'    {"":<12}' + ''.join(f'{n:>10s}' for n in DOF))
    print(f'    {"target":<12}' + ''.join(f'{v:>+10.4f}' for v in delta_target.cpu().numpy()))
    print(f'    {"δ̂":<12}'    + ''.join(f'{v:>+10.4f}' for v in delta_hat.detach().cpu().numpy()))
    print(f'    {"residual":<12}'
           + ''.join(f'{v:>+10.4f}' for v in (delta_hat.detach().cpu().numpy()
                                                - delta_target.cpu().numpy())))

    # ─── Plot loss curve + δ̂ evolution ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    iters_l = [it for it, _ in losses]
    vals_l = [v for _, v in losses]
    axes[0].semilogy(iters_l, vals_l, '-', lw=2, label='pose loss')
    axes[0].semilogy(iters_l, duv_err_hist, '--', lw=1.5, label='|Δuv − oracle| (px)')
    axes[0].set_xlabel('Adam iteration'); axes[0].set_ylabel('value (log)')
    axes[0].set_title('Overfit: 1 image + fixed δ_pert')
    axes[0].grid(which='both', alpha=0.3); axes[0].legend()

    delta_hat_arr = np.array(delta_hat_hist)
    for j, n in enumerate(DOF):
        axes[1].plot(iters_l, delta_hat_arr[:, j], '-', label=n)
        axes[1].axhline(delta_target.cpu().numpy()[j], color=f'C{j}', ls=':', alpha=0.5)
    axes[1].set_xlabel('Adam iteration')
    axes[1].set_ylabel('δ̂  (deg / m)')
    axes[1].set_title('δ̂ trajectory  (dashed = δ_target)')
    axes[1].grid(alpha=0.3); axes[1].legend(loc='best', fontsize=7, ncol=2)
    fig.tight_layout()
    out_path = OUT / 'overfit_loss_curve.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
