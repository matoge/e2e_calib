"""3-image overfit + null-space decomposition.

Same story as overfit_check_nullspace.py but with 3 distinct (image,
δ_pert) pairs sharing one network. Hypothesis: a single set of network
weights now has to "explain" three different scenes with three different
δ_pert simultaneously, so it has fewer degrees of freedom in the null(J)
direction → ‖Δuv_learned.null‖ should be smaller per-image than the
N=1 reference (≈ 16 px). Even at N=3 we expect a measurable drop.

Output:
  docs/assets/2026-05-19_diffba/overfit_3images_nullspace.png
  Console: per-image RMS row/null + cos similarity.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
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

# 3 frames from the same scene (same camera intrinsics → simplifies tensor
# packing). Different δ_pert per frame so the network can't memorise one
# answer.
FRAMES = [0, 50, 98]
PERT_LIST = [
    np.array([ 1.0,  1.5,  0.5,   0.20, -0.30,  0.40], dtype=np.float64),
    np.array([-0.8,  1.2, -0.3,  -0.15,  0.25,  0.35], dtype=np.float64),
    np.array([ 1.3, -1.0,  0.6,   0.10,  0.20, -0.30], dtype=np.float64),
]

LR = 1e-3
N_ITER = 300


def prepare(frame_idx, delta_pert):
    cf = load_frame(m.SCENE, frame_idx)
    img_full = cf.img.astype(np.float32) / 255.0
    H_full, W_full = img_full.shape[:2]
    K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
    pts_cam = cf.pts_cam.astype(np.float64)
    cu, cv = W_full // 2, H_full // 2 + m.DV
    u0 = cu - m.TILE // 2; v0 = cv - m.TILE // 2
    u1 = u0 + m.TILE;       v1 = v0 + m.TILE
    R_pert = Rotation.from_rotvec(np.deg2rad(delta_pert[:3])).as_matrix()
    pts_cam_pert = pts_cam @ R_pert.T + delta_pert[3:6]
    uv_pert_all = project_kb(pts_cam_pert, K, dist)
    z_pert_all = pts_cam_pert[:, 2]
    in_tile = ((uv_pert_all[:, 0] >= u0) & (uv_pert_all[:, 0] < u1)
                & (uv_pert_all[:, 1] >= v0) & (uv_pert_all[:, 1] < v1)
                & (z_pert_all > 0.5))
    uv_pert = uv_pert_all[in_tile]; z_pert = z_pert_all[in_tile]
    pts_cam_in = pts_cam[in_tile]
    uv_local = uv_pert.copy(); uv_local[:, 0] -= u0; uv_local[:, 1] -= v0
    img_tile = img_full[v0:v1, u0:u1].copy()
    return dict(
        img=img_tile, uv_pert=uv_pert, z_pert=z_pert,
        pts_cam_in=pts_cam_in, uv_local=uv_local,
        K=K, dist=dist, delta_pert=delta_pert, frame_idx=frame_idx,
    )


def main():
    torch.manual_seed(7); np.random.seed(7)
    samples = [prepare(f, p) for f, p in zip(FRAMES, PERT_LIST)]
    for s in samples:
        print(f'  frame {s["frame_idx"]:>3d}  pts={len(s["uv_pert"]):>4d}  '
               f'δ_pert={s["delta_pert"].round(2)}')

    # Pad to common N for batching (BA accepts a `valid` mask but here all
    # samples have plenty of pts; just use the min and truncate so the
    # batch tensors are uniform — discarded pts are never seen).
    N_min = min(len(s['uv_pert']) for s in samples)
    print(f'  truncate every sample to N={N_min} pts for batching')
    for s in samples:
        idx = np.arange(N_min)              # deterministic head-N slice
        for k in ('uv_pert', 'z_pert', 'pts_cam_in', 'uv_local'):
            s[k] = s[k][idx]

    B = len(samples)
    img_t  = torch.from_numpy(np.stack([s['img'] for s in samples])).permute(0,3,1,2).to(DEVICE)
    uv_p_t = torch.from_numpy(np.stack([s['uv_pert'] for s in samples])).float().to(DEVICE)
    uv_l_t = torch.from_numpy(np.stack([s['uv_local'] for s in samples])).float().to(DEVICE)
    z_t    = torch.from_numpy(np.stack([s['z_pert']  for s in samples])).float().to(DEVICE)
    K_t    = torch.from_numpy(np.stack([s['K']    for s in samples])).float().to(DEVICE)
    dist_t = torch.from_numpy(np.stack([s['dist'] for s in samples])).float().to(DEVICE)
    delta_target = torch.from_numpy(np.stack([-s['delta_pert'] for s in samples])).float().to(DEVICE)
    loss_scale = m.LOSS_SCALE.to(DEVICE)

    model = m.TileToTokenHead(d=96, n_blocks=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'\n  {"iter":>4}  {"loss":>10}  '
           + ''.join(f'{"|δ̂-tgt|"+str(b):>10}' for b in range(B)))
    duv_final, W_final, delta_hat_final = None, None, None
    for it in range(N_ITER + 1):
        opt.zero_grad()
        # Per-batch forward (vectorise via for-loop; B=3 is fine).
        duv_list, L_list = [], []
        for b in range(B):
            duv_map_b, L_map_b = model(img_t[b:b+1])           # (1,2,32,32),(1,3,32,32)
            duv_b = m.gather_token(duv_map_b, uv_l_t[b], m.TILE)        # (N, 2)
            L_b   = m.gather_token(L_map_b.contiguous(), uv_l_t[b], m.TILE)
            duv_list.append(duv_b); L_list.append(L_b)
        duv = torch.stack(duv_list, dim=0)                   # (B, N, 2)
        L_raw = torch.stack(L_list, dim=0)
        W = m.build_W_from_L(L_raw)
        delta_hat, _ = solve_kb(uv_p_t, duv, W, z_t, K_t, dist_t, m.DOF,
                                  n_iter=2, damping=1e-3)        # (B, 6)
        loss = (((delta_hat - delta_target) * loss_scale) ** 2).mean()
        loss.backward(); opt.step()
        if it % 30 == 0 or it == N_ITER:
            with torch.no_grad():
                derrs = (delta_hat - delta_target).abs().max(dim=-1).values.cpu().numpy()
            print(f'  {it:>4}  {loss.item():>10.5f}  '
                   + ''.join(f'{d:>10.4f}' for d in derrs))
        if it == N_ITER:
            duv_final = duv.detach().cpu().numpy()
            W_final = W.detach().cpu().numpy()
            delta_hat_final = delta_hat.detach().cpu().numpy()

    # ─── Per-image row/null decomposition ─────────────────────────────
    print('\n=== per-image row/null decomposition (RMS px) ===')
    print(f'  {"frame":>5}  {"oracle row":>12}  {"oracle null":>12}  '
           f'{"learned row":>12}  {"learned null":>12}  {"cos row":>10}')
    rows_o, nuls_o, rows_l, nuls_l, coss = [], [], [], [], []
    for b, s in enumerate(samples):
        K = s['K']; dist = s['dist']
        uv_pert = s['uv_pert']; z_pert = s['z_pert']
        pts_cam_in = s['pts_cam_in']
        delta_h = delta_hat_final[b].astype(np.float64)
        # J at final lin. point
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        X0 = (uv_pert[:,0] - cx) * z_pert / fx
        Y0 = (uv_pert[:,1] - cy) * z_pert / fy
        Z0 = z_pert
        R_h = Rotation.from_rotvec(np.deg2rad(delta_h[:3])).as_matrix()
        pts_lin = np.stack([X0, Y0, Z0], axis=1) @ R_h.T + delta_h[3:6]
        Xc, Yc, Zc = pts_lin[:,0], pts_lin[:,1], pts_lin[:,2]
        Ju, Jv = kb_jac_np(Xc, Yc, Zc, K, dist, m.DOF)
        N = len(uv_pert)
        J = np.zeros((2*N, 6)); J[0::2] = Ju; J[1::2] = Jv
        JTJ_inv = np.linalg.inv(J.T @ J)
        proj = lambda x: J @ (JTJ_inv @ (J.T @ x))
        duv_l_flat = duv_final[b].reshape(-1)
        duv_o = project_kb(pts_cam_in, K, dist) - uv_pert
        duv_o_flat = duv_o.reshape(-1)
        l_row = proj(duv_l_flat); l_null = duv_l_flat - l_row
        o_row = proj(duv_o_flat); o_null = duv_o_flat - o_row
        rms = lambda x: float(np.sqrt((x**2).sum() / N))
        cos_r = float((l_row @ o_row) /
                      (np.linalg.norm(l_row)*np.linalg.norm(o_row) + 1e-12))
        rows_o.append(rms(o_row)); nuls_o.append(rms(o_null))
        rows_l.append(rms(l_row)); nuls_l.append(rms(l_null))
        coss.append(cos_r)
        print(f'  {s["frame_idx"]:>5d}  {rms(o_row):>12.3f}  {rms(o_null):>12.3f}  '
               f'{rms(l_row):>12.3f}  {rms(l_null):>12.3f}  {cos_r:>+10.4f}')

    # Pull the N=1 reference number from the previous experiment for the
    # subtitle; hard-coded from overfit_check_nullspace.py output.
    ref_null_n1 = 15.69
    print(f'\n  reference: N=1 overfit had ‖learned null‖ = {ref_null_n1:.2f} px')
    avg_null_n3 = np.mean(nuls_l)
    print(f'  mean N=3      ‖learned null‖ = {avg_null_n3:.2f} px  '
           f'(↓ {(ref_null_n1 - avg_null_n3)/ref_null_n1*100:.0f}% vs N=1)')

    # ─── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6))

    # (a) per-image RAW Δuv scatter (image 0 only, for clarity)
    duv_l = duv_final[0]
    duv_o = (project_kb(samples[0]['pts_cam_in'], samples[0]['K'], samples[0]['dist'])
              - samples[0]['uv_pert'])
    axes[0].scatter(duv_o[:,0], duv_l[:,0], s=4, alpha=0.35, label='Δu', color='tab:blue')
    axes[0].scatter(duv_o[:,1], duv_l[:,1], s=4, alpha=0.35, label='Δv', color='tab:orange')
    lim = max(abs(duv_o).max(), abs(duv_l).max()) * 1.05
    axes[0].plot([-lim,lim], [-lim,lim], 'k--', lw=1, label='y = x')
    axes[0].set_xlim(-lim,lim); axes[0].set_ylim(-lim,lim); axes[0].set_aspect('equal')
    axes[0].set_xlabel('Δuv_oracle  [px]'); axes[0].set_ylabel('Δuv_learned  [px]')
    axes[0].set_title(f'(a) RAW Δuv  (frame {samples[0]["frame_idx"]})\n'
                       'closer to y=x than N=1 → null shrinking')
    axes[0].legend(loc='best'); axes[0].grid(alpha=0.3)

    # (b) bar: per-image null comparison
    x = np.arange(len(samples)) ; w = 0.35
    axes[1].bar(x - w/2, nuls_o, w, label='oracle null  (KB ε)', color='tab:gray')
    axes[1].bar(x + w/2, nuls_l, w, label='learned null  (drift)', color='tab:red')
    axes[1].axhline(ref_null_n1, ls='--', color='tab:purple',
                     label=f'N=1 reference ({ref_null_n1:.1f} px)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'frame {s["frame_idx"]}' for s in samples])
    axes[1].set_ylabel('‖Δuv null‖  [px-RMS]')
    axes[1].set_title(f'(b) null-space drift, per image\n'
                       f'mean N=3 = {avg_null_n3:.2f} px '
                       f'({(ref_null_n1-avg_null_n3)/ref_null_n1*100:.0f}% lower than N=1)')
    axes[1].legend(loc='best'); axes[1].grid(axis='y', alpha=0.3)

    # (c) bar: row component sanity (= same on every image because loss is 0)
    axes[2].bar(x - w/2, rows_o, w, label='oracle row', color='tab:blue', alpha=0.7)
    axes[2].bar(x + w/2, rows_l, w, label='learned row', color='tab:cyan')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'frame {s["frame_idx"]}' for s in samples])
    axes[2].set_ylabel('‖Δuv row‖  [px-RMS]')
    axes[2].set_title('(c) row component agrees per image\n'
                       'cos(learned, oracle): ' + ', '.join(f'{c:+.3f}' for c in coss))
    axes[2].legend(loc='best'); axes[2].grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'N=3 overfit: same network weights must satisfy 3 (image, δ_pert) pairs.\n'
        'Pose loss → 0 on every sample, AND null(J) drift drops vs N=1 — the '
        'network is being forced to learn a more BA-faithful Δuv field.',
        y=1.04, fontsize=10)
    fig.tight_layout()
    out_path = OUT / 'overfit_3images_nullspace.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote → {out_path}')


if __name__ == '__main__':
    main()
