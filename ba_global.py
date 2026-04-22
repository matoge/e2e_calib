"""Single-frame global 6DOF bundle adjustment on NS / PS / WM.

Per dataset, iterate over N val frames that have >=2 object crops:
  1. Sample GT 6-DOF perturbation (|ypr|<=0.5°, |t|<=0.20m, shared across frame).
  2. Iterative BA: for k iters, run all-v3-mc model on each obj crop under the
     current residual perturbation, aggregate per-point (shift, Σ) observations,
     solve 6DOF update with pyceres.
  3. Record (ypr_err, t_err, reproj_before, reproj_after) per frame.
Aggregate + report median/percentiles per dataset.
"""
import json, sys, time, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import pyceres
from scipy.spatial.transform import Rotation

import dataset_nuscenes as ns
import dataset_pandaset as ps
import dataset_waymo    as wm
from model_depth import CalibNetDepth

DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP   = Path('experiments/all_v3_mc')
CDIR  = '/mnt/nvme6t/e2e_calib_cache'
SIGMA_YPR = 0.5
SIGMA_TR  = 0.20
N_ITERS   = 3
MIN_OBJ_PTS_PER_CROP = 0   # 0 = use any crop w/ build_sample.min_pts points; model is useful on bg too
MIN_CROPS_PER_FRAME  = 2
N_FRAMES_PER_DS      = 60
SEED     = 42


def ypr_to_R(ypr_deg):
    return Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()


def load_model():
    ns_ = {}; exec((EXP / 'config.py').read_text(), ns_); c = ns_['CFG']
    m = CalibNetDepth(img_size=c['img_size'], in_channels=c['in_channels'],
                      n_layers=c['n_layers'], self_first=False,
                      use_convnext=c.get('use_convnext', False),
                      use_frustum=c.get('use_frustum', False)).to(DEV)
    m.load_state_dict(torch.load(EXP / 'best_model.pt',
                                 map_location=DEV, weights_only=True))
    m.eval()
    return m


# ── per-dataset adapter ──────────────────────────────────────────────────────

class DatasetAdapter:
    """Bundles dataset-specific geometry into a single object.

    projection='standard': depth=Z, u=fx*X/Z+cx, v=fy*Y/Z+cy
    projection='waymo':    depth=X, u=-fu*Y/X+cu, v=-fv*Z/X+cv
    """
    def __init__(self, name, module, cache, projection):
        self.name = name
        self.module = module
        self.projection = projection
        if name == 'PS':
            self.ds = ps.PandaSetCalibDataset(cache, split='val')
        elif name == 'NS':
            self.ds = ns.NuScenesCalibDataset(cache, split='val')
        elif name == 'WM':
            self.ds = wm.WaymoCalibDataset(cache, split='val')

    def group_frames(self):
        """Returns {frame_key: [inst_indices]} using only obj (not bg) insts."""
        groups = defaultdict(list)
        for i, inst in enumerate(self.ds.instances):
            if self.name == 'PS':
                if 'obj_pos' not in inst: continue
                key = tuple(inst['cam_pos'].numpy().round(3).tolist())
            elif self.name == 'NS':
                if inst['obj_dims'].norm() == 0: continue
                key = tuple(inst['cam_pos'].numpy().round(3).tolist())
            elif self.name == 'WM':
                if inst.get('obj_pos') is None: continue
                pts = inst['pts_vis']
                # unique-per-frame signature (same pts_vis ⇒ same frame)
                sig = tuple(pts[:3].flatten().round(5).tolist()) + (int(len(pts)),)
                key = (inst['seg'], sig)
            groups[key].append(i)
        return groups

    def K_and_gt(self, inst):
        """Return (K_full, R_gt (cam_from_world/veh as rotmat), t_gt (world pos
        of camera, or veh-frame translation for WM)) — in a form consistent with
        the adapter's build_sample expectations."""
        if self.name == 'PS':
            return (inst['K_full'].numpy(),
                    inst['R_gt'].numpy(),
                    inst['cam_pos'].numpy())
        if self.name == 'NS':
            return (inst['K_full'].numpy(),
                    inst['R_gt'].numpy(),
                    inst['cam_pos'].numpy())
        # WM: K encoded as fu/fv/cu/cv; R_gt as T_veh_from_cam[:3,:3]; t is world-like (veh-frame cam pos)
        fu, fv, cu, cv = inst['fu'], inst['fv'], inst['cu'], inst['cv']
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)
        T_vc = inst['T_veh_from_cam']
        return (K, T_vc[:3, :3], T_vc[:3, 3])

    def get_points_world(self, inst):
        """All LiDAR points in the 'world' frame (world for PS/NS, vehicle for WM)."""
        if self.name == 'WM':
            return np.asarray(inst['pts_vis'])
        return inst['pts'].numpy()

    def build_sample(self, inst, ypr, t_delta):
        return self.module.build_sample(inst, ypr, t_delta, img_size=64,
                                        min_pts=8, sub=None)

    def project(self, pts_cam, K):
        """pts_cam: (N,3) in camera frame → (N,2) uv, (N,) depth."""
        if self.projection == 'standard':
            z = pts_cam[..., 2]
            safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
            uv = np.stack([K[0,0]*pts_cam[...,0]/safe + K[0,2],
                           K[1,1]*pts_cam[...,1]/safe + K[1,2]], axis=-1)
            return uv, z
        else:  # waymo
            z = pts_cam[..., 0]
            safe = np.where(np.abs(z) > 1e-6, z, 1e-6)
            uv = np.stack([K[0,0]*(-pts_cam[...,1])/safe + K[0,2],
                           K[1,1]*(-pts_cam[...,2])/safe + K[1,2]], axis=-1)
            return uv, z


# ── inference: collect per-point (P_cam, uv_ref, d_pred, Σ) observations ─────

@torch.no_grad()
def run_inference(model, adapter, inst_idxs, ypr_res, t_res):
    obs_P, obs_uv, obs_d, obs_S = [], [], [], []
    n_used = 0
    for idx in inst_idxs:
        inst = adapter.ds.instances[idx]
        s = adapter.build_sample(inst, ypr_res, t_res)
        if s is None: continue
        img, _true, dist_uvd, idx_in = s
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if int(is_obj.sum()) < MIN_OBJ_PTS_PER_CROP: continue

        pad = torch.zeros(1, dist_uvd.shape[0], dtype=torch.bool, device=DEV)
        p = model(img.unsqueeze(0).to(DEV),
                  dist_uvd.unsqueeze(0).to(DEV)[..., :3],
                  key_padding_mask=pad)[0].cpu().numpy()

        K, R_gt, t_gt = adapter.K_and_gt(inst)
        pts_w     = adapter.get_points_world(inst)
        crop_size = float(inst['crop_size'])
        scale     = 64.0 / crop_size

        io = np.arange(len(is_obj))                           # use all pts (obj+bg)
        d_full = p[io, :2] / scale
        sx = np.exp(p[io, 2]); sy = np.exp(p[io, 3]); rho = np.tanh(p[io, 4])
        Σ = np.zeros((len(io), 2, 2))
        Σ[:, 0, 0] = sx*sx; Σ[:, 1, 1] = sy*sy
        Σ[:, 0, 1] = Σ[:, 1, 0] = rho*sx*sy
        Σ /= (scale * scale)

        # per-point 3D in RESIDUAL cam frame (whatever residual perturbation this iter sees)
        P_world = pts_w[idx_in[io]]
        R_off  = R_gt @ ypr_to_R(ypr_res)
        t_off  = t_gt + t_res
        P_cam  = (R_off.T @ (P_world - t_off).T).T
        uv_ref, _ = adapter.project(P_cam, K)

        obs_P.append(P_cam); obs_uv.append(uv_ref)
        obs_d.append(d_full); obs_S.append(Σ)
        n_used += 1

    if not obs_P: return None
    return dict(P=np.concatenate(obs_P), uv=np.concatenate(obs_uv),
                d=np.concatenate(obs_d), S=np.concatenate(obs_S), n_crops=n_used)


# ── BA cost ──────────────────────────────────────────────────────────────────

class BACost(pyceres.CostFunction):
    """θ = (ypr, t) in CAMERA frame — residual update to current residual pose.
    residuals = L @ (π(R(ypr) P_cam + t) - (uv_ref + d_pred))  where L L^T = Σ^-1."""
    def __init__(self, P, uv_ref, d, L, K, projection):
        super().__init__()
        self.P, self.uv, self.d, self.L = P, uv_ref, d, L
        self.K = K
        self.projection = projection
        self.N = P.shape[0]
        self.set_num_residuals(2 * self.N)
        self.set_parameter_block_sizes([6])

    def _proj(self, Q):
        K = self.K
        if self.projection == 'standard':
            z = Q[:, 2]
            return np.stack([K[0,0]*Q[:,0]/z + K[0,2],
                             K[1,1]*Q[:,1]/z + K[1,2]], axis=1)
        z = Q[:, 0]
        return np.stack([K[0,0]*(-Q[:,1])/z + K[0,2],
                         K[1,1]*(-Q[:,2])/z + K[1,2]], axis=1)

    def _residuals(self, θ):
        R = ypr_to_R(θ[:3]); t = θ[3:6]
        Q = self.P @ R.T + t
        r = self._proj(Q) - self.uv - self.d
        return np.einsum('nij,nj->ni', self.L, r).ravel()

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


def solve_ba(obs, K, projection):
    P, uv, d, Σ = obs['P'], obs['uv'], obs['d'], obs['S']
    L = np.zeros_like(Σ)
    for i in range(P.shape[0]):
        try:
            L[i] = np.linalg.cholesky(np.linalg.inv(Σ[i])).T
        except np.linalg.LinAlgError:
            L[i] = np.eye(2)
    θ = np.zeros(6)
    prob = pyceres.Problem()
    prob.add_residual_block(BACost(P, uv, d, L, K, projection), None, [θ])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.minimizer_progress_to_stdout = False
    summ = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summ)
    return θ, summ


# ── iterative BA per frame ───────────────────────────────────────────────────

def iterate_frame(model, adapter, inst_idxs, ypr_gt, t_gt_world, n_iters=N_ITERS):
    first = adapter.ds.instances[inst_idxs[0]]
    K, R_gt_c2w, _ = adapter.K_and_gt(first)

    ypr_est = np.zeros(3); t_est = np.zeros(3)
    for it in range(n_iters):
        ypr_res = ypr_gt - ypr_est
        t_res   = t_gt_world - t_est
        obs = run_inference(model, adapter, inst_idxs, ypr_res, t_res)
        if obs is None or obs['P'].shape[0] < 20:
            return None
        θ, _ = solve_ba(obs, K, adapter.projection)
        Δypr = θ[:3]; Δt_cam = θ[3:6]
        Δt_w = R_gt_c2w @ Δt_cam
        ypr_est = ypr_est + Δypr
        t_est   = t_est   + Δt_w
        if np.linalg.norm(Δypr) < 1e-3 and np.linalg.norm(Δt_cam) < 1e-4:
            break
    return dict(ypr_est=ypr_est, t_est=t_est, n_crops=obs['n_crops'], n_pts=obs['P'].shape[0])


# ── full-frame reproj evaluation ─────────────────────────────────────────────

def reproj_error(adapter, inst, ypr_gt, t_gt, ypr_est, t_est):
    K, R_gt_c2w, t_ref = adapter.K_and_gt(inst)
    pts_w = adapter.get_points_world(inst)
    # GT (identity perturbation)
    P_gt = (R_gt_c2w.T @ (pts_w - t_ref).T).T
    uv_gt, z_gt = adapter.project(P_gt, K)
    # fully perturbed (before BA)
    R_off = R_gt_c2w @ ypr_to_R(ypr_gt); t_off = t_ref + t_gt
    P_bf = (R_off.T @ (pts_w - t_off).T).T
    uv_bf, _ = adapter.project(P_bf, K)
    # after BA: residual = gt - est
    R_res = R_gt_c2w @ ypr_to_R(ypr_gt - ypr_est); t_res = t_ref + (t_gt - t_est)
    P_af = (R_res.T @ (pts_w - t_res).T).T
    uv_af, _ = adapter.project(P_af, K)
    # only count points in front of camera (either frame) to avoid NaNs
    ok = (z_gt > 0.5)
    if ok.sum() < 10: return None, None
    return (float(np.linalg.norm(uv_bf[ok] - uv_gt[ok], axis=1).mean()),
            float(np.linalg.norm(uv_af[ok] - uv_gt[ok], axis=1).mean()))


# ── main: run across datasets ────────────────────────────────────────────────

def run_dataset(model, adapter, n_frames=N_FRAMES_PER_DS, seed=SEED):
    print(f"\n── {adapter.name} ──")
    groups = adapter.group_frames()
    keys   = [k for k, idxs in groups.items() if len(idxs) >= MIN_CROPS_PER_FRAME]
    rng    = np.random.default_rng(seed)
    rng.shuffle(keys)
    picked = keys[:n_frames]
    print(f"  frames with ≥{MIN_CROPS_PER_FRAME} obj crops: {len(keys)}  "
          f"(using {len(picked)})")

    results = []
    t0 = time.time()
    for fi, key in enumerate(picked):
        inst_idxs = groups[key]
        ypr_gt = (rng.random(3)*2 - 1) * SIGMA_YPR
        t_gt   = (rng.random(3)*2 - 1) * SIGMA_TR
        res = iterate_frame(model, adapter, inst_idxs, ypr_gt, t_gt)
        if res is None: continue
        ypr_err = res['ypr_est'] - ypr_gt
        t_err   = res['t_est']   - t_gt
        rot_geo = float(np.rad2deg(np.arccos(np.clip(
            (np.trace(ypr_to_R(res['ypr_est']).T @ ypr_to_R(ypr_gt)) - 1)/2, -1, 1))))
        rpj = reproj_error(adapter, adapter.ds.instances[inst_idxs[0]],
                            ypr_gt, t_gt, res['ypr_est'], res['t_est'])
        if rpj is None or rpj[0] is None: continue
        rb, ra = rpj
        results.append(dict(
            n_crops=res['n_crops'], n_pts=res['n_pts'],
            ypr_err_norm_deg=float(np.linalg.norm(ypr_err)),
            rot_err_geo_deg=rot_geo,
            t_err_norm_cm=float(np.linalg.norm(t_err))*100,
            reproj_before_px=rb, reproj_after_px=ra,
        ))
        if (fi+1) % 10 == 0:
            print(f"  {fi+1}/{len(picked)}  ({time.time()-t0:.0f}s)", flush=True)

    if not results:
        print("  no valid frames"); return None

    def q(arr, p): return float(np.percentile(arr, p))
    r_rot_geo = [r['rot_err_geo_deg'] for r in results]
    r_t       = [r['t_err_norm_cm']   for r in results]
    r_rb      = [r['reproj_before_px'] for r in results]
    r_ra      = [r['reproj_after_px']  for r in results]
    r_ncrops  = [r['n_crops']          for r in results]
    summary = dict(
        n=len(results), median_crops=int(np.median(r_ncrops)),
        rot_err_p50=q(r_rot_geo, 50), rot_err_p90=q(r_rot_geo, 90),
        t_err_p50=q(r_t, 50),         t_err_p90=q(r_t, 90),
        reproj_before_p50=q(r_rb, 50), reproj_before_p90=q(r_rb, 90),
        reproj_after_p50=q(r_ra, 50),  reproj_after_p90=q(r_ra, 90),
        reduction_p50=1 - q(r_ra, 50)/q(r_rb, 50),
    )
    print(f"  frames={summary['n']}  median obj crops={summary['median_crops']}")
    print(f"  rot err (geodesic deg): p50={summary['rot_err_p50']:.3f}  p90={summary['rot_err_p90']:.3f}  (init≈{SIGMA_YPR*np.sqrt(3):.2f})")
    print(f"  t   err (cm):           p50={summary['t_err_p50']:.2f}   p90={summary['t_err_p90']:.2f}  (init≈{SIGMA_TR*100*np.sqrt(3):.1f})")
    print(f"  reproj px p50:          {summary['reproj_before_p50']:.2f} → {summary['reproj_after_p50']:.2f}  "
          f"({summary['reduction_p50']*100:+.0f}%)")
    return dict(summary=summary, per_frame=results)


def main():
    model = load_model()
    adapters = [
        DatasetAdapter('NS', ns, f'{CDIR}/nuscenes_mc_cache.pt', 'standard'),
        DatasetAdapter('PS', ps, f'{CDIR}/pandaset_mc_cache.pt', 'standard'),
        DatasetAdapter('WM', wm, f'{CDIR}/waymo_mc_cache.pt',    'waymo'),
    ]
    out = {}
    for a in adapters:
        r = run_dataset(model, a)
        if r is not None: out[a.name] = r

    out_dir = Path('experiments/all_v3_mc/ba')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'ba_global.json').write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_dir}/ba_global.json")


if __name__ == '__main__':
    main()
