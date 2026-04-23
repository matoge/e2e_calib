"""Iterative single-frame bundle adjustment using object crops only.

Pipeline (1 frame, all in-distribution object instances from cache):
  for iter in 0..K:
      residual_pert = gt_pert - cumulative_est      # world frame
      for each obj inst: build_sample(residual_pert) -> model -> (shift, Sigma)
      aggregate each crop to 1 obs (inverse-variance weighted mean of obj pts)
      solve 6DoF BA with pyceres -> delta
      cumulative_est += delta                        # composed properly
  iter 0: coarse fit (model sees full perturbation)
  iter 1+: refinement (model sees only residual -> in-distribution, more accurate)
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import pyceres
from scipy.spatial.transform import Rotation

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from dataset_pandaset import PandaSetCalibDataset, build_sample
from model_depth import CalibNetDepth


PS_ROOT     = Path('/mnt/mininas/datasets/pandaset')
CACHE_PATH  = '/tmp/pandaset_cache.pt'
CKPT        = Path('experiments/all_v2/best_model.pt')
MODEL_KW    = dict(img_size=64, in_channels=3, n_layers=4,
                   self_first=False, use_convnext=True, use_frustum=True)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA_YPR   = 0.5   # deg, in-distribution for model
SIGMA_TR    = 0.20  # m
MIN_OBJ_PTS = 3     # min obj pts per crop to use as a BA observation
N_ITERS     = 3
SEED        = 42
USE_BG      = True  # include bg pts in addition to obj pts


# ── helpers ──────────────────────────────────────────────────────────────────
def ypr_to_R(ypr_deg):
    return Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()


def project(pts_cam, K):
    z = pts_cam[..., 2]
    uv = (pts_cam[..., :2] * np.array([K[0,0], K[1,1]]) / z[..., None]
          + np.array([K[0,2], K[1,2]]))
    return uv, z


def find_frame_image(cam_pos, tol=0.01):
    """Reverse-lookup: find (scene, fi) whose pose matches cam_pos, load image."""
    cp = np.asarray(cam_pos)
    for scene_dir in sorted(PS_ROOT.iterdir()):
        cam_dir = scene_dir / 'camera' / 'front_camera'
        pose_f = cam_dir / 'poses.json'
        if not pose_f.exists(): continue
        with open(pose_f) as f: poses = json.load(f)
        for fi, pose in enumerate(poses):
            p = np.array([pose['position']['x'], pose['position']['y'], pose['position']['z']])
            if np.linalg.norm(p - cp) < tol:
                img_path = cam_dir / f'{fi:02d}.jpg'
                if img_path.exists():
                    return scene_dir.name, fi, np.array(Image.open(img_path).convert('RGB'))
    return None, None, None


def group_by_frame(instances):
    """Group obj instances by frame using cam_pos as key."""
    groups = defaultdict(list)
    for i, inst in enumerate(instances):
        if 'obj_pos' not in inst:  # skip random-crop entries
            continue
        key = tuple(inst['cam_pos'].numpy().round(3).tolist())
        groups[key].append(i)
    return groups


# ── inference: per-point observations from obj pts of each crop ─────────────
@torch.no_grad()
def run_inference(model, instances, inst_idxs, ypr_res_world, t_res_world):
    """Apply residual perturbation via build_sample, run model, and emit ONE
    observation PER OBJ POINT. Each obs = (P_cam, uv_ref, d, Sigma) in full-res
    pixels under the current residual-distorted cam frame.
    """
    obs_P, obs_uv, obs_d, obs_S = [], [], [], []
    n_crops_used = 0
    for idx in inst_idxs:
        inst = instances[idx]
        s = build_sample(inst, ypr_res_world, t_res_world)
        if s is None:
            continue
        img, _true, dist_uvd, idx_in = s
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if int(is_obj.sum()) < MIN_OBJ_PTS:
            continue

        pad = torch.zeros(1, dist_uvd.shape[0], dtype=torch.bool, device=DEVICE)
        p = model(img.unsqueeze(0).to(DEVICE),
                  dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                  key_padding_mask=pad)[0].cpu().numpy()

        K         = inst['K_full'].numpy()
        R_gt      = inst['R_gt'].numpy()
        cp        = inst['cam_pos'].numpy()
        pts_world = inst['pts'].numpy()
        crop_size = float(inst['crop_size'])
        scale     = 64.0 / crop_size                              # full-res → 64

        io = np.arange(len(is_obj)) if USE_BG else np.where(is_obj)[0]
        # rescale crop-local predictions → full-res pixels
        d_full = p[io, :2] / scale                                # (M, 2)
        sx = np.exp(p[io, 2]); sy = np.exp(p[io, 3]); rho = np.tanh(p[io, 4])
        Σ_full = np.zeros((len(io), 2, 2))
        Σ_full[:, 0, 0] = sx * sx
        Σ_full[:, 1, 1] = sy * sy
        Σ_full[:, 0, 1] = Σ_full[:, 1, 0] = rho * sx * sy
        Σ_full /= (scale * scale)

        # per-point 3D in current residual cam frame
        R_off_res  = R_gt @ ypr_to_R(ypr_res_world)
        cp_off_res = cp + t_res_world
        P_world    = pts_world[idx_in[io]]                         # (M, 3)
        P_cam      = (R_off_res.T @ (P_world - cp_off_res).T).T    # (M, 3)
        uv_ref, _  = project(P_cam, K)                             # (M, 2)

        obs_P.append(P_cam)
        obs_uv.append(uv_ref)
        obs_d.append(d_full)
        obs_S.append(Σ_full)
        n_crops_used += 1

    if not obs_P:
        return None
    return dict(
        P_cam   = np.concatenate(obs_P),
        uv_ref  = np.concatenate(obs_uv),
        d       = np.concatenate(obs_d),
        Sigma   = np.concatenate(obs_S),
        n_crops = n_crops_used,
    )


# ── BA cost ──────────────────────────────────────────────────────────────────
class CalibBACost(pyceres.CostFunction):
    """residuals = sqrt_info @ (π(R(ypr) P + t) - (uv_ref + d)).  Block=(6,)."""
    def __init__(self, P, uv_ref, d, sqrt_info, K):
        super().__init__()
        self.P, self.uv_ref, self.d, self.L = P, uv_ref, d, sqrt_info
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx, self.cy = K[0,2], K[1,2]
        self.N = P.shape[0]
        self.set_num_residuals(2 * self.N)
        self.set_parameter_block_sizes([6])

    def _residuals(self, θ):
        R = ypr_to_R(θ[:3])
        t = θ[3:6]
        Q = self.P @ R.T + t
        proj = np.stack([self.fx * Q[:,0]/Q[:,2] + self.cx,
                         self.fy * Q[:,1]/Q[:,2] + self.cy], axis=1)
        r_raw = proj - self.uv_ref - self.d
        return np.einsum('nij,nj->ni', self.L, r_raw).ravel()

    def Evaluate(self, params, residuals, jacobians):
        θ = np.asarray(params[0]).copy()
        residuals[:] = self._residuals(θ)
        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-5
            J = np.zeros((2*self.N, 6))
            for k in range(6):
                tp = θ.copy(); tp[k] += eps
                tm = θ.copy(); tm[k] -= eps
                J[:, k] = (self._residuals(tp) - self._residuals(tm)) / (2*eps)
            jacobians[0][:] = J.ravel()
        return True


def solve_ba(obs, K):
    P  = obs['P_cam']
    ur = obs['uv_ref']
    d  = obs['d']
    Σ  = obs['Sigma']
    N  = P.shape[0]
    L  = np.zeros_like(Σ)
    for i in range(N):
        try:
            L[i] = np.linalg.cholesky(np.linalg.inv(Σ[i])).T
        except np.linalg.LinAlgError:
            L[i] = np.eye(2)

    θ = np.zeros(6)
    prob = pyceres.Problem()
    prob.add_residual_block(CalibBACost(P, ur, d, L, K), None, [θ])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.minimizer_progress_to_stdout = False
    summ = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summ)
    return θ, summ


# ── iterative loop ───────────────────────────────────────────────────────────
def iterate(model, instances, inst_idxs, ypr_gt, t_gt_world,
            R_gt_c2w, K, n_iters=N_ITERS):
    ypr_est_world = np.zeros(3)
    t_est_world   = np.zeros(3)
    history = []
    for it in range(n_iters):
        ypr_res = ypr_gt - ypr_est_world
        t_res_w = t_gt_world - t_est_world

        obs = run_inference(model, instances, inst_idxs, ypr_res, t_res_w)
        if obs is None or obs['P_cam'].shape[0] < 20:
            n = 0 if obs is None else obs['P_cam'].shape[0]
            print(f'  iter {it}: too few obs ({n}), abort'); break
        θ, summ = solve_ba(obs, K)
        n_pts   = obs['P_cam'].shape[0]
        n_crops = obs['n_crops']

        Δypr_cam = θ[:3]                               # same axes as world ypr
        Δt_cam   = θ[3:6]                              # cam-frame translation
        # convert Δt_cam back to world using ORIGINAL c2w (t relates via R_gt)
        Δt_world = R_gt_c2w @ Δt_cam
        ypr_est_world = ypr_est_world + Δypr_cam       # small-angle additive ok
        t_est_world   = t_est_world   + Δt_world

        rem_ypr = ypr_gt - ypr_est_world
        rem_tw  = t_gt_world - t_est_world

        history.append(dict(
            iter=it, n_crops=n_crops, n_pts=n_pts,
            Δypr=Δypr_cam.round(4).tolist(),
            Δt_cam_cm=(Δt_cam*100).round(3).tolist(),
            remaining_ypr=rem_ypr.round(4).tolist(),
            remaining_t_world_cm=(rem_tw*100).round(3).tolist(),
            initial_cost=float(summ.initial_cost),
            final_cost=float(summ.final_cost),
        ))
        print(f'  iter {it}: crops={n_crops} pts={n_pts}  '
              f'Δypr={Δypr_cam.round(3)} Δt={(Δt_cam*100).round(2)}cm  '
              f'remaining: ypr={rem_ypr.round(3)} t_w={(rem_tw*100).round(2)}cm  '
              f'cost {summ.initial_cost:.1f}→{summ.final_cost:.1f}')

        if np.linalg.norm(Δypr_cam) < 1e-3 and np.linalg.norm(Δt_cam) < 1e-4:
            print(f'  converged at iter {it}')
            break
    return ypr_est_world, t_est_world, history


# ── main ─────────────────────────────────────────────────────────────────────
def main(frame_choice='mid'):
    print(f'[1] loading dataset & model')
    ds = PandaSetCalibDataset(CACHE_PATH, split='val')
    model = CalibNetDepth(**MODEL_KW).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f'[2] grouping cache by frame')
    groups = group_by_frame(ds.instances)
    sizes  = sorted(((k, len(v)) for k,v in groups.items()), key=lambda x: x[1])
    if   frame_choice == 'min':  key, n = sizes[0]
    elif frame_choice == 'max':  key, n = sizes[-1]
    else:                         key, n = sizes[len(sizes)//2]
    inst_idxs = groups[key]
    print(f'  picked frame: {n} obj instances  (choice={frame_choice})')

    # sample GT perturbation (shared across all obj in the frame)
    rng = np.random.default_rng(SEED)
    ypr_gt     = (rng.random(3)*2-1) * SIGMA_YPR
    t_gt_world = (rng.random(3)*2-1) * SIGMA_TR
    print(f'[3] GT perturbation:')
    print(f'    ypr_gt    = {ypr_gt.round(3)} deg')
    print(f'    t_gt (w)  = {(t_gt_world*100).round(2)} cm')

    # per-frame constants
    first = ds.instances[inst_idxs[0]]
    K         = first['K_full'].numpy()
    R_gt_c2w  = first['R_gt'].numpy()

    print(f'[4] iterative BA (n_iters={N_ITERS}):')
    ypr_est, t_est_w, history = iterate(
        model, ds.instances, inst_idxs,
        ypr_gt, t_gt_world, R_gt_c2w, K, n_iters=N_ITERS)

    # final reporting (full-frame reproj)
    t_gt_cam  = R_gt_c2w.T @ t_gt_world
    t_est_cam = R_gt_c2w.T @ t_est_w
    ypr_err = ypr_est - ypr_gt
    t_err_world = t_est_w - t_gt_world
    R_est = ypr_to_R(ypr_est)
    R_gt_d = ypr_to_R(ypr_gt)
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_est.T @ R_gt_d)-1)/2, -1, 1)))

    # full-frame reproj on ALL LiDAR of the frame
    pts_w = first['pts'].numpy()
    cp    = first['cam_pos'].numpy()
    # GT proj
    uv_gt, _ = project((R_gt_c2w.T @ (pts_w - cp).T).T, K)
    # distorted (initial) proj — full GT perturbation applied
    R_off = R_gt_c2w @ ypr_to_R(ypr_gt)
    cp_off = cp + t_gt_world
    uv_dist, z_dist = project((R_off.T @ (pts_w - cp_off).T).T, K)
    # BA-corrected proj = residual-distorted pose (what's left after subtracting estimate)
    R_res = R_gt_c2w @ ypr_to_R(ypr_gt - ypr_est)
    cp_res = cp + (t_gt_world - t_est_w)
    uv_ba, z_ba = project((R_res.T @ (pts_w - cp_res).T).T, K)

    vis = (z_dist > 0.5)
    err_before = np.linalg.norm(uv_dist[vis] - uv_gt[vis], axis=1).mean()
    err_after  = np.linalg.norm(uv_ba[vis]   - uv_gt[vis], axis=1).mean()

    # per-iter mean obj-point reproj on same frame (directly from model predictions)
    print(f'  iter-wise obj reproj (model prediction on residual):')
    dummy_ypr_res = ypr_gt; dummy_t_res = t_gt_world
    for i, h in enumerate(history):
        rem_y = np.array(h['remaining_ypr'])
        rem_t = np.array(h['remaining_t_world_cm']) / 100
        print(f'    after iter {i}: remaining ||ypr||={np.linalg.norm(rem_y):.3f}° '
              f'||t_w||={np.linalg.norm(rem_t)*100:.2f}cm')

    print(f'\n=== result ===')
    print(f'  ypr_gt  = {ypr_gt.round(3)} deg')
    print(f'  ypr_est = {ypr_est.round(3)} deg   err={ypr_err.round(3)} '
          f'(|| = {np.linalg.norm(ypr_err):.3f}°)')
    print(f'  rot_err (geodesic) = {rot_err_deg:.4f} deg')
    print(f'  t_gt_w  = {(t_gt_world*100).round(2)} cm')
    print(f'  t_est_w = {(t_est_w*100).round(2)} cm   err={(t_err_world*100).round(2)} cm '
          f'(|| = {np.linalg.norm(t_err_world)*100:.2f}cm)')
    print(f'  full-frame reproj: before={err_before:.3f} px  after={err_after:.3f} px')

    out_dir = Path(f'experiments/ba_singleframe/iter_{frame_choice}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(dict(
        frame_choice=frame_choice, n_obj_insts=n,
        ypr_gt_deg=ypr_gt.tolist(), t_gt_world_cm=(t_gt_world*100).tolist(),
        ypr_est_deg=ypr_est.tolist(), t_est_world_cm=(t_est_w*100).tolist(),
        ypr_err_deg=ypr_err.tolist(), rot_err_deg=float(rot_err_deg),
        t_err_world_cm=(t_err_world*100).tolist(),
        t_err_world_norm_cm=float(np.linalg.norm(t_err_world)*100),
        reproj_before_px=float(err_before), reproj_after_px=float(err_after),
        history=history,
    ), indent=2))

    # A/B viz: TOP = distorted, BOTTOM = BA-corrected. color by depth.
    scene, fi, img_full = find_frame_image(cp)
    print(f'  frame: scene={scene} fi={fi}')
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(13, 15), dpi=120)
    for ax in (axT, axB):
        if img_full is not None: ax.imshow(img_full)
        else:                     ax.set_facecolor('#111')
        ax.set_xlim(0, K[0,2]*2); ax.set_ylim(K[1,2]*2, 0)
        ax.set_aspect('equal'); ax.axis('off')

    z_ok = z_dist > 0.5
    depths = np.log10(np.clip(z_dist[z_ok], 1, 80))
    vmin, vmax = depths.min(), depths.max()

    axT.scatter(uv_dist[z_ok, 0], uv_dist[z_ok, 1],
                c=depths, cmap='turbo_r', vmin=vmin, vmax=vmax,
                s=6, alpha=0.8, linewidths=0, zorder=3)
    axT.set_title(f'BEFORE — mis-calibrated   reproj = {err_before:.2f} px',
                  fontsize=16, pad=8, loc='left')

    axB.scatter(uv_ba[z_ok, 0], uv_ba[z_ok, 1],
                c=depths, cmap='turbo_r', vmin=vmin, vmax=vmax,
                s=6, alpha=0.8, linewidths=0, zorder=3)
    axB.set_title(f'AFTER — BA corrected   reproj = {err_after:.2f} px',
                  fontsize=16, pad=8, loc='left')

    plt.suptitle(f'scene {scene} fi={fi}   |   '
                 f'GT pert: ypr={ypr_gt.round(2)}° t={(t_gt_world*100).round(1)}cm  →  '
                 f'recovered: ypr_err={rot_err_deg:.3f}° t_err={np.linalg.norm(t_err_world)*100:.2f}cm   |   '
                 f'{len(inst_idxs)} obj crops   |   '
                 f'{err_before:.2f}→{err_after:.2f} px ({100*(1-err_after/err_before):.0f}% reduction)',
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(out_dir / 'ba_ab.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_dir}/metrics.json, ba_ab.png')


if __name__ == '__main__':
    choice = sys.argv[1] if len(sys.argv) > 1 else 'mid'
    main(choice)
