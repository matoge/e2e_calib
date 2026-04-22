"""Multi-frame BA on one PandaSet sequence — share one rig-level extrinsic.

Setup
  • Pick one scene (015) from the mc val cache.
  • Take N=10 of its val frames.
  • Sample ONE cam-frame rig drift (ypr_cam, t_cam) — same for all frames
    (LiDAR↔camera extrinsic is rigid w.r.t. the vehicle).
  • Per frame k apply it as:  R_off_k = R_gt_k @ R_ypr,  cp_off_k = cp_k + R_gt_k @ t_cam
    (convert the cam-frame t to world for the build_sample API).
  • Run detector on every frame's object crops → per-point Gaussians.
  • One pyceres problem with ONE 6-DoF block (θ), residual blocks stacked from
    all frames — θ is the shared rig correction.
  • Iterate 3 times.

Also run single-frame BA on each of the same N frames independently (with the
same per-frame induced drift) for a head-to-head.

Output: experiments/all_v3_mc/ba_multiframe/{batch.json, hero.png}
"""
import json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pyceres
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_pandaset import PandaSetCalibDataset, build_sample
from model_depth import CalibNetDepth
import ba_singleframe as bas  # reuse ypr_to_R, project, run_inference, solve_ba, CalibBACost


CACHE_PATH = '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_cache.pt'
CKPT       = Path('experiments/all_v3_mc/best_model.pt')
OUT_ROOT   = Path('experiments/all_v3_mc/ba_multiframe')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PS_ROOT    = Path('/mnt/mininas/datasets/pandaset')
SCENE      = '015'
N_FRAMES   = 10
SIGMA_YPR  = 0.5
SIGMA_TR   = 0.20
N_ITERS    = 3
SEED       = 42


# ── frame discovery ──────────────────────────────────────────────────────────
def discover_frames(instances, scene, n_frames):
    groups = defaultdict(list)
    for i, inst in enumerate(instances):
        if 'obj_pos' not in inst:
            continue
        key = tuple(inst['cam_pos'].numpy().round(3).tolist())
        groups[key].append(i)

    pose_f = PS_ROOT / scene / 'camera/front_camera/poses.json'
    poses = json.loads(pose_f.read_text())
    fi_of_key = {}
    for fi, pose in enumerate(poses):
        p = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
        for key in groups:
            if np.linalg.norm(np.array(key) - p) < 0.02:
                fi_of_key[key] = fi
                break

    picked = sorted(fi_of_key.items(), key=lambda kv: kv[1])[:n_frames]
    return [(fi, key, groups[key]) for key, fi in picked]


# ── multi-frame BA solve ─────────────────────────────────────────────────────
def solve_multi(all_obs):
    """all_obs: list of (P_cam, uv_ref, d_shift, Sigma, K). One shared θ block."""
    theta = np.zeros(6)
    prob = pyceres.Problem()
    for P, ur, d, Σ, K in all_obs:
        N = P.shape[0]
        L = np.zeros_like(Σ)
        for i in range(N):
            try:
                L[i] = np.linalg.cholesky(np.linalg.inv(Σ[i])).T
            except np.linalg.LinAlgError:
                L[i] = np.eye(2)
        prob.add_residual_block(bas.CalibBACost(P, ur, d, L, K), None, [theta])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.minimizer_progress_to_stdout = False
    summ = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summ)
    return theta, summ


# ── inference per frame, with per-frame cam-frame-converted t ─────────────────
@torch.no_grad()
def frame_observations(model, instances, inst_idxs, ypr_res_cam, t_res_cam,
                        R_gt_c2w, K):
    """Build all obs for ONE frame under the shared cam-frame residual drift."""
    t_res_world = R_gt_c2w @ t_res_cam
    obs_P, obs_uv, obs_d, obs_S = [], [], [], []
    n_crops = 0
    for idx in inst_idxs:
        inst = instances[idx]
        s = build_sample(inst, ypr_res_cam, t_res_world)
        if s is None: continue
        img, _t, dist_uvd, idx_in = s
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if int(is_obj.sum()) < bas.MIN_OBJ_PTS: continue
        pad = torch.zeros(1, dist_uvd.shape[0], dtype=torch.bool, device=DEVICE)
        p = model(img.unsqueeze(0).to(DEVICE),
                  dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                  key_padding_mask=pad)[0].cpu().numpy()
        K_full      = inst['K_full'].numpy()
        R_gt        = inst['R_gt'].numpy()
        cp          = inst['cam_pos'].numpy()
        pts_world   = inst['pts'].numpy()
        crop_size   = float(inst['crop_size'])
        scale       = 64.0 / crop_size

        io = np.arange(len(is_obj)) if bas.USE_BG else np.where(is_obj)[0]
        d_full = p[io, :2] / scale
        sx = np.exp(p[io, 2]); sy = np.exp(p[io, 3]); rho = np.tanh(p[io, 4])
        Σ_full = np.zeros((len(io), 2, 2))
        Σ_full[:, 0, 0] = sx * sx
        Σ_full[:, 1, 1] = sy * sy
        Σ_full[:, 0, 1] = Σ_full[:, 1, 0] = rho * sx * sy
        Σ_full /= (scale * scale)

        R_off_res  = R_gt @ bas.ypr_to_R(ypr_res_cam)
        cp_off_res = cp + t_res_world
        P_world    = pts_world[idx_in[io]]
        P_cam      = (R_off_res.T @ (P_world - cp_off_res).T).T
        uv_ref, _  = bas.project(P_cam, K_full)

        obs_P.append(P_cam); obs_uv.append(uv_ref)
        obs_d.append(d_full); obs_S.append(Σ_full)
        n_crops += 1

    if not obs_P:
        return None
    return (np.concatenate(obs_P), np.concatenate(obs_uv),
            np.concatenate(obs_d), np.concatenate(obs_S), K, n_crops)


def reproj_err_px(frame_pack, ypr_cam, t_cam):
    """Full-frame mean reproj error under (ypr_cam, t_cam) residual drift."""
    inst = frame_pack['first']
    K        = inst['K_full'].numpy()
    R_gt_c2w = inst['R_gt'].numpy()
    cp       = inst['cam_pos'].numpy()
    pts_w    = inst['pts'].numpy()
    t_world = R_gt_c2w @ t_cam
    uv_gt, _ = bas.project((R_gt_c2w.T @ (pts_w - cp).T).T, K)
    R_off  = R_gt_c2w @ bas.ypr_to_R(ypr_cam)
    cp_off = cp + t_world
    uv_off, z = bas.project((R_off.T @ (pts_w - cp_off).T).T, K)
    vis = z > 0.5
    return float(np.linalg.norm(uv_off[vis] - uv_gt[vis], axis=1).mean())


# ── iterate multi-frame ──────────────────────────────────────────────────────
def iterate_multi(model, frames, ypr_gt_cam, t_gt_cam, n_iters=N_ITERS):
    ypr_est_cam = np.zeros(3)
    t_est_cam   = np.zeros(3)
    history = []
    for it in range(n_iters):
        ypr_res = ypr_gt_cam - ypr_est_cam
        t_res   = t_gt_cam   - t_est_cam

        all_obs = []
        total_pts = 0; total_crops = 0
        for fp in frames:
            obs = frame_observations(model, fp['instances'], fp['inst_idxs'],
                                      ypr_res, t_res, fp['R_gt_c2w'], fp['K'])
            if obs is None: continue
            P, ur, d, Σ, K, n_cr = obs
            all_obs.append((P, ur, d, Σ, K))
            total_pts += P.shape[0]; total_crops += n_cr
        if total_pts < 40:
            print(f'  iter {it}: too few obs ({total_pts}), abort'); break

        θ, summ = solve_multi(all_obs)
        Δypr = θ[:3]; Δt = θ[3:6]
        ypr_est_cam = ypr_est_cam + Δypr
        t_est_cam   = t_est_cam   + Δt
        rem_y = ypr_gt_cam - ypr_est_cam
        rem_t = t_gt_cam   - t_est_cam
        history.append(dict(
            iter=it, n_frames=len(all_obs), n_crops=total_crops, n_pts=total_pts,
            dypr=Δypr.round(4).tolist(), dt_cm=(Δt*100).round(3).tolist(),
            rem_ypr=rem_y.round(4).tolist(), rem_t_cm=(rem_t*100).round(3).tolist(),
            initial_cost=float(summ.initial_cost),
            final_cost=float(summ.final_cost),
        ))
        print(f'  iter {it}: frames={len(all_obs)} pts={total_pts}  '
              f'Δypr={Δypr.round(3)} Δt_cm={(Δt*100).round(2)}  '
              f'rem ypr={rem_y.round(3)} t_cm={(rem_t*100).round(2)}  '
              f'cost {summ.initial_cost:.1f}→{summ.final_cost:.1f}')
        if np.linalg.norm(Δypr) < 1e-3 and np.linalg.norm(Δt) < 1e-4:
            print(f'  converged at iter {it}'); break
    return ypr_est_cam, t_est_cam, history


# ── iterate single-frame (reuses existing solve) ─────────────────────────────
def iterate_single(model, fp, ypr_gt_cam, t_gt_cam, n_iters=N_ITERS):
    ypr_est = np.zeros(3); t_est = np.zeros(3)
    for it in range(n_iters):
        ypr_res = ypr_gt_cam - ypr_est
        t_res   = t_gt_cam   - t_est
        obs = frame_observations(model, fp['instances'], fp['inst_idxs'],
                                  ypr_res, t_res, fp['R_gt_c2w'], fp['K'])
        if obs is None or obs[0].shape[0] < 20:
            break
        P, ur, d, Σ, K, _ = obs
        θ, summ = solve_multi([(P, ur, d, Σ, K)])
        ypr_est = ypr_est + θ[:3]; t_est = t_est + θ[3:6]
        if np.linalg.norm(θ[:3]) < 1e-3 and np.linalg.norm(θ[3:6]) < 1e-4:
            break
    return ypr_est, t_est


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print('[1] loading dataset & model')
    ds = PandaSetCalibDataset(CACHE_PATH, split='val')
    model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                          use_convnext=True, use_frustum=True).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f'[2] discovering frames in scene {SCENE}')
    found = discover_frames(ds.instances, SCENE, N_FRAMES)
    print(f'  {len(found)} frames: fi={[f[0] for f in found]}')

    frames = []
    for fi, key, inst_idxs in found:
        first = ds.instances[inst_idxs[0]]
        frames.append(dict(
            fi=fi, key=key, inst_idxs=inst_idxs, instances=ds.instances,
            first=first,
            K        = first['K_full'].numpy(),
            R_gt_c2w = first['R_gt'].numpy(),
            cp       = first['cam_pos'].numpy(),
        ))

    # ONE shared cam-frame drift
    rng = np.random.default_rng(SEED)
    ypr_gt_cam = (rng.random(3)*2 - 1) * SIGMA_YPR
    t_gt_cam   = (rng.random(3)*2 - 1) * SIGMA_TR
    print(f'[3] shared GT cam-frame drift:')
    print(f'    ypr_cam = {ypr_gt_cam.round(3)}° ||={np.linalg.norm(ypr_gt_cam):.3f}°')
    print(f'    t_cam   = {(t_gt_cam*100).round(2)} cm ||={np.linalg.norm(t_gt_cam)*100:.2f}cm')

    # ─── single-frame pass (baseline) ─────────────────────────────────────────
    print(f'\n[4] SINGLE-frame BA on each of {len(frames)} frames (same shared GT)')
    singles = []
    t0 = time.time()
    for i, fp in enumerate(frames):
        ypr_est, t_est_cam = iterate_single(model, fp, ypr_gt_cam, t_gt_cam)
        before = reproj_err_px(fp, ypr_gt_cam, t_gt_cam)
        after  = reproj_err_px(fp, ypr_gt_cam - ypr_est, t_gt_cam - t_est_cam)
        rot_err = float(np.rad2deg(np.arccos(np.clip(
            (np.trace(bas.ypr_to_R(ypr_est).T @ bas.ypr_to_R(ypr_gt_cam)) - 1) / 2, -1, 1))))
        t_err_cm = float(np.linalg.norm(t_est_cam - t_gt_cam) * 100)
        singles.append(dict(
            fi=fp['fi'], ypr_est_deg=ypr_est.tolist(),
            t_est_cm=(t_est_cam*100).tolist(),
            rot_err_deg=rot_err, t_err_cm=t_err_cm,
            reproj_before_px=before, reproj_after_px=after,
        ))
        print(f'  [{i+1}/{len(frames)}] fi={fp["fi"]}  '
              f'rot={rot_err:.3f}° t_err={t_err_cm:.2f}cm  '
              f'reproj {before:.2f}→{after:.2f} px')
    dt_single = time.time() - t0

    # ─── multi-frame pass ─────────────────────────────────────────────────────
    print(f'\n[5] MULTI-frame BA (1 shared 6-DoF over {len(frames)} frames)')
    t0 = time.time()
    ypr_m, t_m_cam, hist_m = iterate_multi(model, frames, ypr_gt_cam, t_gt_cam)
    dt_multi = time.time() - t0
    rot_err_m = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(bas.ypr_to_R(ypr_m).T @ bas.ypr_to_R(ypr_gt_cam)) - 1) / 2, -1, 1))))
    t_err_cm_m = float(np.linalg.norm(t_m_cam - t_gt_cam) * 100)

    # per-frame full-frame reproj under the shared multi-frame estimate
    multi_per_frame = []
    for fp in frames:
        before = reproj_err_px(fp, ypr_gt_cam, t_gt_cam)
        after  = reproj_err_px(fp, ypr_gt_cam - ypr_m, t_gt_cam - t_m_cam)
        multi_per_frame.append(dict(fi=fp['fi'],
                                     reproj_before_px=before, reproj_after_px=after))
        print(f'  fi={fp["fi"]:2d}  reproj {before:.2f}→{after:.2f} px')

    print(f'\n=== SUMMARY ===')
    s_rot = np.array([s['rot_err_deg'] for s in singles])
    s_t   = np.array([s['t_err_cm']    for s in singles])
    s_rep = np.array([s['reproj_after_px'] for s in singles])
    m_rep = np.array([e['reproj_after_px'] for e in multi_per_frame])
    print(f'  SINGLE ({len(singles)} frames, {dt_single:.1f}s):')
    print(f'    rot_err_deg   median={np.median(s_rot):.3f}  max={s_rot.max():.3f}')
    print(f'    t_err_cm      median={np.median(s_t):.2f}   max={s_t.max():.2f}')
    print(f'    reproj_after  median={np.median(s_rep):.2f}  max={s_rep.max():.2f}')
    print(f'    converged <10px: {int((s_rep<10).sum())}/{len(singles)}')
    print(f'  MULTI  (1 shared solve over {len(frames)} frames, {dt_multi:.1f}s):')
    print(f'    rot_err_deg   = {rot_err_m:.3f}   (single median {np.median(s_rot):.3f})')
    print(f'    t_err_cm      = {t_err_cm_m:.2f}    (single median {np.median(s_t):.2f})')
    print(f'    reproj_after  median={np.median(m_rep):.2f}  max={m_rep.max():.2f}')
    print(f'    converged <10px: {int((m_rep<10).sum())}/{len(multi_per_frame)}')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = dict(
        scene=SCENE, n_frames=len(frames),
        ypr_gt_cam_deg=ypr_gt_cam.tolist(),
        t_gt_cam_cm=(t_gt_cam*100).tolist(),
        single=singles,
        multi=dict(
            ypr_est_deg=ypr_m.tolist(),
            t_est_cm=(t_m_cam*100).tolist(),
            rot_err_deg=rot_err_m, t_err_cm=t_err_cm_m,
            per_frame=multi_per_frame,
            history=hist_m,
            wall_s=dt_multi,
        ),
        summary=dict(
            single=dict(rot_med=float(np.median(s_rot)), rot_max=float(s_rot.max()),
                        t_med=float(np.median(s_t)),     t_max=float(s_t.max()),
                        rep_med=float(np.median(s_rep)), rep_max=float(s_rep.max()),
                        conv=int((s_rep<10).sum())),
            multi =dict(rot=rot_err_m, t=t_err_cm_m,
                        rep_med=float(np.median(m_rep)), rep_max=float(m_rep.max()),
                        conv=int((m_rep<10).sum())),
        ),
    )
    (OUT_ROOT / 'batch.json').write_text(json.dumps(out, indent=2))
    print(f'\n  saved -> {OUT_ROOT}/batch.json')

    # ─── hero figure ──────────────────────────────────────────────────────────
    fis = [s['fi'] for s in singles]
    fig, ax = plt.subplots(1, 1, figsize=(11, 4.6), dpi=130)
    fig.patch.set_facecolor('#f6f4ed'); ax.set_facecolor('#ffffff')
    x = np.arange(len(fis))
    w = 0.36
    b1 = ax.bar(x - w/2, [s['reproj_after_px'] for s in singles],
                 width=w, color='#174734', alpha=0.72, label='single-frame BA', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, [e['reproj_after_px'] for e in multi_per_frame],
                 width=w, color='#c13c14', alpha=0.95, label='multi-frame BA (shared)', edgecolor='white', linewidth=0.5)
    for bars, vals in [(b1, [s['reproj_after_px'] for s in singles]),
                       (b2, [e['reproj_after_px'] for e in multi_per_frame])]:
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width()/2, v + 0.08, f'{v:.1f}',
                     ha='center', fontsize=8, family='monospace',
                     color=bb.get_facecolor())
    ax.axhline(1.0, lw=0.7, ls=':', color='#6b6a63', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'fi={f}' for f in fis], fontsize=9)
    ax.set_ylabel('mean full-frame reproj (px)', fontsize=10)
    ax.set_title(f'Scene {SCENE}  ·  {len(fis)} consecutive val frames  ·  same shared 6-DoF drift\n'
                  f'SINGLE rot={np.median(s_rot):.3f}° / t={np.median(s_t):.2f}cm median    '
                  f'MULTI rot={rot_err_m:.3f}° / t={t_err_cm_m:.2f}cm (one solve)',
                  fontsize=11, loc='left', pad=8)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_ROOT / 'hero.png', dpi=130, bbox_inches='tight', facecolor='#f6f4ed')
    plt.close(fig)
    print(f'  saved -> {OUT_ROOT}/hero.png')


if __name__ == '__main__':
    main()
