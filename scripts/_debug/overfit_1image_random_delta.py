"""1-image overfit, RANDOM δ per step, depth-heatmap input + CoordConv stem.

This is the "graduation" sanity test from the diff-BA design notes:
  - N=1 image, single tile.
  - Each step: sample random δ_pert (ω_x, ω_y in ±1°, others 0), project
    LiDAR under δ_pert, render a 1/z heatmap on top of the RGB tile (3ch).
  - Stem also receives 2 normalised xy channels (CoordConv) so the
    translation-equivariant ConvNeXt stack can express the location-
    dependent KB Jacobian.
  - BA = closed-form 1 GN step (n_iter=1), 2-DoF.
  - Pass criterion: held-out |Δuv − oracle| ≈ 0  → graduate to real run.

Output:
  docs/assets/2026-05-19_diffba/overfit_1image_random_delta.png
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

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'overfit', str(REPO / 'scripts' / '_debug' / 'overfit_one_image_ba.py'))
m = importlib.util.module_from_spec(spec); _s.modules['overfit'] = m
spec.loader.exec_module(m)

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import project_kb, kb_jacobian as kb_jac_np
from scripts.ba.ba_torch import solve_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DOF_ACTIVE = ['omega_x', 'omega_y']
DELTA_RANGE_DEG = 1.0
BATCH_SIZE = 32          # 32 random δ per step (averages the moving-target noise)
LR = 1e-4
FREEZE_W = True          # if True, ignore L head and use W = I (isolates Δuv signal
                         # from the "W-hack" failure mode where the net would
                         # rather shrink rows than predict Δuv)
DIRECT_DUV_SUPERVISE = True  # bypass BA: loss = MSE(duv_pred, duv_oracle).
N_ITER = 600
LOG_EVERY = 20
INVZ_NEAR = 1.0          # 1/z normalisation: clip z ∈ [INVZ_NEAR, INVZ_FAR]
INVZ_FAR = 60.0
DOT_RAD = 1              # half-side of the 1/z stamp (3×3 px square)

HELDOUT_2D = [
    ( 1.0,  1.5),
    (-0.8,  1.2),
    ( 0.6, -0.7),
]


# ─── Network: 32×32 token (/16), simplest possible ────────────────────
TOKEN_GRID = 32        # 512 / 16 = 32×32, cell = 16 px (≈ 1 token / point)


def _coord_concat(x):
    B, _, H, W = x.shape
    ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    coord = torch.stack([gx, gy], dim=0)[None].expand(B, -1, -1, -1)
    return torch.cat([x, coord], dim=1)


def _stem(in_ch, d):
    return nn.Sequential(
        nn.Conv2d(in_ch + 2, d // 2, 4, stride=4),       # /4
        nn.GELU(),
        nn.Conv2d(d // 2, d, 4, stride=4),               # /16 → 32×32
        nn.GELU(),
    )


class TileToTokenHeadCoord(nn.Module):
    """Two-encoder + cross-attention.  RGB encoder and depth encoder are
    independent ConvNeXt streams (CoordConv at each stem); the depth
    tokens then cross-attend over the RGB tokens once.  Heads sit on the
    depth side (Δuv, L). Pos embed = CoordConv only (no learned PE)."""
    def __init__(self, d=96, n_blocks=1, n_heads=4, grid=TOKEN_GRID):
        super().__init__()
        self.stem_rgb = _stem(3, d)
        self.stem_dep = _stem(1, d)
        self.blocks_rgb = nn.Sequential(*[m._ConvNeXtBlock(d) for _ in range(n_blocks)])
        self.blocks_dep = nn.Sequential(*[m._ConvNeXtBlock(d) for _ in range(n_blocks)])
        self.pe_q  = nn.Parameter(torch.zeros(1, grid * grid, d))
        self.pe_kv = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.normal_(self.pe_q,  std=0.02)
        nn.init.normal_(self.pe_kv, std=0.02)
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.xattn   = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d),
        )
        self.head_duv = nn.Conv2d(d, 2, 1)
        self.head_L   = nn.Conv2d(d, 3, 1)

    def forward(self, img_rgb, img_dep):
        rgb = self.blocks_rgb(self.stem_rgb(_coord_concat(img_rgb)))   # (B, d, G, G)
        dep = self.blocks_dep(self.stem_dep(_coord_concat(img_dep)))   # (B, d, G, G)
        B, d, G, _ = rgb.shape
        rgb_t = rgb.permute(0, 2, 3, 1).reshape(B, G * G, d)
        dep_t = dep.permute(0, 2, 3, 1).reshape(B, G * G, d)
        q  = self.norm_q(dep_t + self.pe_q)
        kv = self.norm_kv(rgb_t + self.pe_kv)
        att, _ = self.xattn(q, kv, kv, need_weights=False)
        dep_t = dep_t + att
        dep_t = dep_t + self.ffn(self.norm_ffn(dep_t))
        x = dep_t.reshape(B, G, G, d).permute(0, 3, 1, 2).contiguous()
        return self.head_duv(x), self.head_L(x)


def make_delta6(omx, omy):
    return np.array([omx, omy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def sample_delta(rng):
    a = rng.uniform(-DELTA_RANGE_DEG, DELTA_RANGE_DEG)
    b = rng.uniform(-DELTA_RANGE_DEG, DELTA_RANGE_DEG)
    return make_delta6(a, b)


def perturb_pool(pts_cam, K, dist, delta_pert, u0, v0, u1, v1):
    R = Rotation.from_rotvec(np.deg2rad(delta_pert[:3])).as_matrix()
    pts_p = pts_cam @ R.T + delta_pert[3:6]
    uv = project_kb(pts_p, K, dist)
    z = pts_p[:, 2]
    keep = ((uv[:, 0] >= u0) & (uv[:, 0] < u1)
              & (uv[:, 1] >= v0) & (uv[:, 1] < v1) & (z > 0.5))
    return uv[keep], z[keep], pts_cam[keep]


def render_depth(uv_local, z_pert, H_t, W_t):
    """Stamp 1/z (DOT_RAD=1, fixed *100 scale) on a blank H×W grid → (1,H,W)
    numpy float32. NOT alpha-blended on RGB — depth is its own channel for
    the two-encoder variant."""
    invz = np.zeros((H_t, W_t), dtype=np.float32)
    invz_pt = 1.0 / np.clip(z_pert, INVZ_NEAR, INVZ_FAR)
    uvi = uv_local.round().astype(int)
    for du in range(-DOT_RAD, DOT_RAD + 1):
        for dv in range(-DOT_RAD, DOT_RAD + 1):
            ui = np.clip(uvi[:, 0] + du, 0, W_t - 1)
            vi = np.clip(uvi[:, 1] + dv, 0, H_t - 1)
            np.maximum.at(invz, (vi, ui), invz_pt)
    return np.clip(invz * 100.0, 0, 1)[None]      # (1, H, W)


def render_input(img_tile, uv_local, z_pert):
    """RGB α-blended with 1/z heatmap → (1, 3, H, W) tensor on DEVICE."""
    H_t, W_t = img_tile.shape[:2]
    invz = render_depth(uv_local, z_pert, H_t, W_t)[0]
    cmap = plt.get_cmap('magma')
    rgba = cmap(invz)[..., :3].astype(np.float32)
    a = ((invz > 0).astype(np.float32) * 0.65)[..., None]
    blended = a * rgba + (1 - a) * img_tile
    blended = np.clip(blended, 0, 1).astype(np.float32)
    return torch.from_numpy(blended).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def gather_token_batched(token_map, uv_in_tile, tile_size, valid):
    """Bilinear sample of token_map at uv_in_tile (sub-pixel)."""
    B, C, G, _ = token_map.shape
    # uv_in_tile is in pixel units in [0, tile_size). Map to grid_sample's
    # normalised [-1, 1] referencing the centres of the G×G token cells.
    cell = tile_size / G
    gx_f = (uv_in_tile[..., 0] - cell / 2) / (tile_size - cell) * 2 - 1
    gy_f = (uv_in_tile[..., 1] - cell / 2) / (tile_size - cell) * 2 - 1
    gx_f = torch.where(valid, gx_f, torch.zeros_like(gx_f))
    gy_f = torch.where(valid, gy_f, torch.zeros_like(gy_f))
    grid = torch.stack([gx_f, gy_f], dim=-1).unsqueeze(2)        # (B, N, 1, 2)
    sampled = F.grid_sample(token_map, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1).contiguous()      # (B, N, C)


def J_from_lin_pt(uv_pert, z_pert, delta_h_full, K, dist):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X0 = (uv_pert[:,0] - cx) * z_pert / fx
    Y0 = (uv_pert[:,1] - cy) * z_pert / fy
    Z0 = z_pert
    R_h = Rotation.from_rotvec(np.deg2rad(delta_h_full[:3])).as_matrix()
    pts_lin = np.stack([X0, Y0, Z0], axis=1) @ R_h.T + delta_h_full[3:6]
    Ju, Jv = kb_jac_np(pts_lin[:,0], pts_lin[:,1], pts_lin[:,2], K, dist, DOF_ACTIVE)
    N = len(uv_pert)
    J = np.zeros((2*N, 2)); J[0::2] = Ju; J[1::2] = Jv
    return J, N


def main():
    torch.manual_seed(7); np.random.seed(7)
    rng = np.random.RandomState(7)

    cf = load_frame(m.SCENE, m.FRAME)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + m.DV
    u0 = cu - m.TILE // 2; v0 = cv - m.TILE // 2
    u1 = u0 + m.TILE;       v1 = v0 + m.TILE
    img_tile_np = img_full[v0:v1, u0:u1].copy()

    K_t = torch.from_numpy(K).float().unsqueeze(0).to(DEVICE)
    dist_t = torch.from_numpy(dist).float().unsqueeze(0).to(DEVICE)
    loss_scale = torch.tensor([1.0, 1.0], dtype=torch.float32, device=DEVICE)

    model = TileToTokenHeadCoord(d=96, n_blocks=3, n_heads=4).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'model: {n_param/1e3:.1f}k params  (RGB enc + Depth enc + cross-attn)')

    rgb_t_b = torch.from_numpy(
        img_tile_np.transpose(2, 0, 1)[None]).to(DEVICE).expand(
        BATCH_SIZE, -1, -1, -1).contiguous()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    log_iter, log_loss, log_duv_err = [], [], []
    print(f'\n  {"iter":>4}  {"loss":>10}  {"|δ̂-tgt|":>10}  '
           f'{"|Δuv-orcl|":>10}  {"N_max":>6}  {"N_min":>6}')
    K_t_b    = K_t.expand(BATCH_SIZE, -1, -1)
    dist_t_b = dist_t.expand(BATCH_SIZE, -1)
    for it in range(N_ITER + 1):
        # ─── sample BATCH_SIZE different random δ_pert ───────────────
        per_b = []
        for _ in range(BATCH_SIZE):
            while True:
                d = sample_delta(rng)
                uv_p, z_p, pc_in = perturb_pool(pts_cam, K, dist, d,
                                                  u0, v0, u1, v1)
                if len(uv_p) >= 50:
                    break
            uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
            duv_o = project_kb(pc_in, K, dist) - uv_p
            per_b.append(dict(d=d, uv_p=uv_p, uv_l=uv_l, z=z_p, duv_o=duv_o))

        N_max = max(len(b['uv_p']) for b in per_b)
        # pack with zero-pad + valid mask.
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

        # depth tensor: 1ch raw 1/z stamp, different per batch element.
        H_t = W_t = m.TILE
        dep_b = np.stack([
            render_depth(b['uv_l'], b['z'], H_t, W_t) for b in per_b
        ], axis=0)                                          # (B, 1, H, W)
        dep_t_b   = torch.from_numpy(dep_b).to(DEVICE)
        uv_p_t_b  = torch.from_numpy(uv_p_b).to(DEVICE)
        uv_l_t_b  = torch.from_numpy(uv_l_b).to(DEVICE)
        z_t_b     = torch.from_numpy(z_b).to(DEVICE)
        valid_t_b = torch.from_numpy(valid_b).to(DEVICE)
        duv_o_t_b = torch.from_numpy(duv_o_b).to(DEVICE)
        delta_target_b = torch.stack([
            torch.from_numpy(-b['d'][:2]).float() for b in per_b
        ], dim=0).to(DEVICE)            # (B, 2)

        opt.zero_grad()
        duv_map, L_map = model(rgb_t_b, dep_t_b)
        duv = gather_token_batched(duv_map, uv_l_t_b, m.TILE, valid_t_b)
        if FREEZE_W:
            W = torch.eye(2, device=DEVICE, dtype=duv.dtype).expand(
                BATCH_SIZE, N_max, 2, 2)
        else:
            L_raw = gather_token_batched(
                L_map.contiguous(), uv_l_t_b, m.TILE, valid_t_b)
            W = m.build_W_from_L(L_raw)

        if DIRECT_DUV_SUPERVISE:
            sq = (duv - duv_o_t_b).pow(2).sum(dim=-1)             # (B, N)
            loss = sq[valid_t_b].mean()
            with torch.no_grad():
                delta_hat, _ = solve_kb(
                    uv_p_t_b, duv, W, z_t_b, K_t_b, dist_t_b, DOF_ACTIVE,
                    valid=valid_t_b, n_iter=1, damping=1e-3,
                )
        else:
            delta_hat, _ = solve_kb(
                uv_p_t_b, duv, W, z_t_b, K_t_b, dist_t_b, DOF_ACTIVE,
                valid=valid_t_b, n_iter=1, damping=1e-3,
            )
            loss = (((delta_hat - delta_target_b) * loss_scale) ** 2).mean()
        loss.backward(); opt.step()

        if it % LOG_EVERY == 0 or it == N_ITER:
            with torch.no_grad():
                derr = (delta_hat - delta_target_b).abs().max().item()
                err = (duv - duv_o_t_b).norm(dim=-1)
                duv_err = err[valid_t_b].mean().item()
            log_iter.append(it); log_loss.append(loss.item())
            log_duv_err.append(duv_err)
            n_min = min(len(b['uv_p']) for b in per_b)
            print(f'  {it:>4}  {loss.item():>10.5f}  {derr:>10.4f}  '
                   f'{duv_err:>10.3f}  {N_max:>6d}  {n_min:>6d}')

    # ─── Held-out evaluation ─────────────────────────────────────────
    print('\n=== held-out evaluation (RMS px) ===')
    print(f'  {"δ_pert":<28}  {"|δ̂-tgt|":>10}  {"|Δuv-orcl|":>10}  '
           f'{"learn row":>10}  {"learn null":>10}  {"orcl null":>10}  {"cos row":>9}')
    eval_results = []
    for omx, omy in HELDOUT_2D:
        delta_pert = make_delta6(omx, omy)
        uv_pert_np, z_pert_np, pts_cam_in = perturb_pool(
            pts_cam, K, dist, delta_pert, u0, v0, u1, v1)
        uv_local_np = uv_pert_np.copy()
        uv_local_np[:, 0] -= u0; uv_local_np[:, 1] -= v0
        duv_oracle_np = project_kb(pts_cam_in, K, dist) - uv_pert_np

        rgb_t = torch.from_numpy(
            img_tile_np.transpose(2, 0, 1)[None]).to(DEVICE)
        dep_t_one = torch.from_numpy(
            render_depth(uv_local_np, z_pert_np, m.TILE, m.TILE)[None]
        ).to(DEVICE)
        uv_p_t = torch.from_numpy(uv_pert_np).float().to(DEVICE)
        uv_l_t = torch.from_numpy(uv_local_np).float().to(DEVICE)
        z_t    = torch.from_numpy(z_pert_np).float().to(DEVICE)
        delta_target = torch.from_numpy(-delta_pert[:2]).float().to(DEVICE)

        with torch.no_grad():
            duv_map, L_map = model(rgb_t, dep_t_one)
            duv = m.gather_token(duv_map, uv_l_t, m.TILE)
            if FREEZE_W:
                W = torch.eye(2, device=DEVICE, dtype=duv.dtype).expand(
                    len(uv_p_t), 2, 2)
            else:
                L_raw = m.gather_token(L_map.contiguous(), uv_l_t, m.TILE)
                W = m.build_W_from_L(L_raw)
            delta_hat, _ = solve_kb(
                uv_p_t.unsqueeze(0), duv.unsqueeze(0), W.unsqueeze(0),
                z_t.unsqueeze(0), K_t, dist_t, DOF_ACTIVE,
                n_iter=1, damping=1e-3,
            )
            delta_hat = delta_hat[0]

        delta_h_2 = delta_hat.cpu().numpy().astype(np.float64)
        delta_h_full = np.array([delta_h_2[0], delta_h_2[1], 0, 0, 0, 0],
                                  dtype=np.float64)
        derr = float((delta_hat - delta_target).abs().max().item())
        duv_np = duv.cpu().numpy()
        duv_err_mean = float(np.linalg.norm(duv_np - duv_oracle_np, axis=1).mean())

        J, N = J_from_lin_pt(uv_pert_np, z_pert_np, delta_h_full, K, dist)
        JTJ_inv = np.linalg.inv(J.T @ J)
        proj = lambda x: J @ (JTJ_inv @ (J.T @ x))
        l_flat = duv_np.reshape(-1); o_flat = duv_oracle_np.reshape(-1)
        l_row = proj(l_flat); l_null = l_flat - l_row
        o_row = proj(o_flat); o_null = o_flat - o_row
        rms = lambda x: float(np.sqrt((x**2).sum() / N))
        cos_r = float((l_row @ o_row) /
                      (np.linalg.norm(l_row)*np.linalg.norm(o_row) + 1e-12))
        eval_results.append(dict(
            omx=omx, omy=omy, delta_hat=delta_h_full,
            duv_learned=duv_np, duv_oracle=duv_oracle_np,
            l_row=rms(l_row), l_null=rms(l_null),
            o_row=rms(o_row), o_null=rms(o_null),
            cos_row=cos_r, derr=derr, duv_err_mean=duv_err_mean,
        ))
        tag = f'(ωx={omx:+.1f}, ωy={omy:+.1f})'
        print(f'  {tag:<28}  {derr:>10.4f}  {duv_err_mean:>10.3f}  '
               f'{rms(l_row):>10.3f}  {rms(l_null):>10.3f}  '
               f'{rms(o_null):>10.3f}  {cos_r:>+9.4f}')

    avg_null = np.mean([r['l_null'] for r in eval_results])
    avg_duv_err = np.mean([r['duv_err_mean'] for r in eval_results])
    ref_null_n1 = 15.69
    ref_duv_err_n1 = 14.0
    print(f'\n  ref (N=1 fixed-δ, RGB only): null = {ref_null_n1:.2f} px, '
           f'|Δuv − oracle| ≈ {ref_duv_err_n1:.1f} px')
    print(f'  this run (random-δ, +heatmap+CoordConv): null = {avg_null:.2f} px, '
           f'|Δuv − oracle| = {avg_duv_err:.2f} px')

    # ─── Plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6))

    ax = axes[0]
    ax.semilogy(log_iter, log_loss, '-', lw=2, label='pose loss')
    ax.semilogy(log_iter, log_duv_err, '--', lw=1.8,
                  label='|Δuv − oracle| (px, mean)')
    ax.axhline(ref_duv_err_n1, ls=':', color='tab:purple',
                 label=f'N=1 fixed-δ, RGB-only (~{ref_duv_err_n1:.0f} px)')
    ax.set_xlabel('Adam iteration'); ax.set_ylabel('value (log)')
    ax.set_title('(a) training: random δ, depth-heatmap input')
    ax.grid(which='both', alpha=0.3); ax.legend(loc='best')

    r = eval_results[0]
    ax = axes[1]
    duv_l = r['duv_learned']; duv_o = r['duv_oracle']
    ax.scatter(duv_o[:,0], duv_l[:,0], s=4, alpha=0.35, label='Δu', color='tab:blue')
    ax.scatter(duv_o[:,1], duv_l[:,1], s=4, alpha=0.35, label='Δv', color='tab:orange')
    lim = max(abs(duv_o).max(), abs(duv_l).max()) * 1.05
    ax.plot([-lim,lim], [-lim,lim], 'k--', lw=1, label='y = x')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal')
    ax.set_xlabel('Δuv_oracle  [px]'); ax.set_ylabel('Δuv_learned  [px]')
    ax.set_title(f'(b) RAW Δuv on held-out (ωx={r["omx"]:+.1f}, ωy={r["omy"]:+.1f})\n'
                   f'|Δuv − oracle| = {r["duv_err_mean"]:.2f} px')
    ax.legend(loc='best'); ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(len(eval_results)); w = 0.35
    ax.bar(x - w/2, [r['o_null'] for r in eval_results], w,
             label='oracle null  (KB ε)', color='tab:gray')
    ax.bar(x + w/2, [r['l_null'] for r in eval_results], w,
             label='learned null', color='tab:red')
    ax.axhline(ref_null_n1, ls='--', color='tab:purple',
                 label=f'N=1 fixed-δ ({ref_null_n1:.1f} px)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'(ωx={r["omx"]:+.1f},ωy={r["omy"]:+.1f})'
                          for r in eval_results], fontsize=8)
    ax.set_ylabel('‖Δuv null‖  [px-RMS]')
    ax.set_title(f'(c) null-space drift on held-out δ\n'
                   f'mean = {avg_null:.2f} px')
    ax.legend(loc='best'); ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        'N=1 image, RANDOM δ_pert per step.  '
        'Input = RGB + 1/z heatmap (3ch) + CoordConv at stem.  '
        'BA = closed-form, n_iter=1, 2-DoF.\n'
        'Pass criterion: |Δuv − oracle| → small on held-out δ.',
        y=1.04, fontsize=10)
    fig.tight_layout()
    out_path = OUT / 'overfit_1image_random_delta.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
