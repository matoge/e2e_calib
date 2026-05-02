"""BA evaluation on V3 cache: per-frame sliding-window inference + pyceres BA solve.

Usage:
  python scripts/ba/ba_eval_v3.py \
      --exp experiments/ns_v504r_convnext_deform_l4 \
      --cache /dev/shm/nuscenes_v3_full \
      --sigma-rot 0.5 --sigma-t 0.20 \
      --n-frames 100 --window 384 --stride 192 \
      --clearml

Picks N val frames at random, samples a single (ypr, t) perturbation per frame,
generates an overlapping sliding-window crop grid, runs the model on every
crop, aggregates per-point (Δu, Δv, Σ) observations, and solves a 6-DOF BA via
pyceres for the residual update. Reports rot/t error vs ground truth and
reprojection error before/after BA. ClearML logs scalars and a histogram.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, io, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pyceres
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.pandaset_full import PandaSetCalibDatasetFull, _is_obj_per_point
from models.model_depth import CalibNetDepth

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ypr_to_R(ypr_deg):
    return Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix().astype(np.float32)


def load_model(exp_dir: Path):
    cfg_ns = {}; exec((exp_dir / 'config.py').read_text(), cfg_ns); c = cfg_ns['CFG']
    sd = torch.load(exp_dir / 'best_model.pt', map_location=DEV, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    # detect deform variant from state_dict
    has_deform = any('sampling_offsets' in k for k in sd.keys())
    deform_mode = 'sl' if has_deform else 'none'
    m = CalibNetDepth(img_size=c['img_size'], in_channels=c['in_channels'],
                      n_layers=c['n_layers'], self_first=c.get('self_first', False),
                      use_convnext=c.get('use_convnext', False),
                      use_frustum=c.get('use_frustum', False),
                      deform_mode=deform_mode).to(DEV)
    m.load_state_dict(sd)
    m.eval()
    return m, c


def decode_jpg(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert('RGB'), dtype=np.uint8)


def project(pts_world, R_c2w, t_cam_world, K):
    """R_c2w: 3x3 cam→world rotation (R_gt convention in V3 cache).
    t_cam_world: camera center in world frame.
    Returns (uv (N, 2), z (N,), pts_cam (N, 3))."""
    # row-form: (p - cp) @ R_c2w = R_c2w.T @ (p - cp) = R_w2c @ (p - cp) = pt in cam
    pts_cam = (pts_world - t_cam_world[None, :]) @ R_c2w
    z = pts_cam[:, 2]
    safe = np.maximum(np.abs(z), 1e-6) * np.where(z >= 0, 1.0, -1.0)
    uv = np.stack([K[0, 0] * pts_cam[:, 0] / safe + K[0, 2],
                   K[1, 1] * pts_cam[:, 1] / safe + K[1, 2]], axis=-1)
    return uv.astype(np.float32), z.astype(np.float32), pts_cam.astype(np.float32)


@torch.no_grad()
def infer_crop(model, img_full, K, pts_world, R_off, t_off,
                u0, v0, cs, S, min_pts=8):
    """Run model on one window. Returns dict or None if insufficient points."""
    IH, IW = img_full.shape[:2]
    if u0 < 0 or v0 < 0 or u0 + cs > IW or v0 + cs > IH:
        return None
    # project all points under perturbed pose; keep those landing in the window
    uv_off, z_off, pts_cam_off = project(pts_world, R_off.astype(np.float32),
                                           t_off.astype(np.float32), K)
    in_crop = ((uv_off[:, 0] >= u0) & (uv_off[:, 0] < u0 + cs) &
               (uv_off[:, 1] >= v0) & (uv_off[:, 1] < v0 + cs) &
               (z_off > 0.5))
    if in_crop.sum() < min_pts:
        return None
    sel = np.where(in_crop)[0]
    if len(sel) > 256:
        sel = np.random.default_rng(int(u0 * 1e3 + v0)).choice(sel, 256, replace=False)
    scale = float(S) / float(cs)
    uv_loc = np.stack([(uv_off[sel, 0] - u0) * scale,
                       (uv_off[sel, 1] - v0) * scale], axis=-1)
    z_loc = z_off[sel]
    dist_uvd = np.concatenate([uv_loc, z_loc[:, None]], axis=1).astype(np.float32)

    # crop image, resize
    crop = img_full[v0:v0 + cs, u0:u0 + cs]
    img_t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, cs, cs)
    img_t = F.interpolate(img_t, size=(S, S), mode='bilinear', align_corners=False)
    img_t = (img_t / 255.0).to(DEV)

    Nq = dist_uvd.shape[0]
    pad = torch.zeros(1, Nq, dtype=torch.bool, device=DEV)
    vfp = torch.tensor([float(K[0, 0]) * S / cs], dtype=torch.float32, device=DEV)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(img_t,
                    torch.from_numpy(dist_uvd[None]).to(DEV),
                    key_padding_mask=pad,
                    vfp=vfp)[0].float().cpu().numpy()
    # out: (Nq, 5) [du, dv, log_sx, log_sy, rho]
    du_loc = out[:, 0]; dv_loc = out[:, 1]
    sx_loc = np.exp(out[:, 2]); sy_loc = np.exp(out[:, 3])
    rho    = np.tanh(out[:, 4])
    # convert local-pixel predictions back to full-image pixels
    du_full = du_loc / scale; dv_full = dv_loc / scale
    sx_full = sx_loc / scale; sy_full = sy_loc / scale
    Σ = np.zeros((Nq, 2, 2), dtype=np.float32)
    Σ[:, 0, 0] = sx_full * sx_full
    Σ[:, 1, 1] = sy_full * sy_full
    Σ[:, 0, 1] = Σ[:, 1, 0] = rho * sx_full * sy_full
    return dict(P=pts_cam_off[sel],
                uv_ref=uv_off[sel],
                d=np.stack([du_full, dv_full], axis=1).astype(np.float32),
                Σ=Σ)


class BACost(pyceres.CostFunction):
    """θ = (ypr, t_cam): residual update to current residual perturbation.
    residual = L (π(R(ypr) P + t) - (uv_ref + d_pred))"""
    def __init__(self, P, uv_ref, d, L, K):
        super().__init__()
        self.P, self.uv, self.d, self.L, self.K = P, uv_ref, d, L, K
        self.N = P.shape[0]
        self.set_num_residuals(2 * self.N)
        self.set_parameter_block_sizes([6])

    def _proj(self, Q):
        K = self.K
        z = Q[:, 2]
        return np.stack([K[0, 0] * Q[:, 0] / z + K[0, 2],
                         K[1, 1] * Q[:, 1] / z + K[1, 2]], axis=1)

    def _residuals(self, theta):
        R = ypr_to_R(theta[:3]); t = theta[3:6]
        Q = self.P @ R.T + t
        r = self._proj(Q) - self.uv - self.d
        return np.einsum('nij,nj->ni', self.L, r).ravel()

    def Evaluate(self, params, residuals, jacobians):
        theta = np.asarray(params[0]).copy()
        residuals[:] = self._residuals(theta)
        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-5
            J = np.zeros((2 * self.N, 6))
            for k in range(6):
                tp = theta.copy(); tp[k] += eps
                tm = theta.copy(); tm[k] -= eps
                J[:, k] = (self._residuals(tp) - self._residuals(tm)) / (2 * eps)
            jacobians[0][:] = J.ravel()
        return True


def solve_ba(obs, K):
    P = np.concatenate([o['P']      for o in obs])
    uv= np.concatenate([o['uv_ref'] for o in obs])
    d = np.concatenate([o['d']      for o in obs])
    Σ = np.concatenate([o['Σ']      for o in obs])
    L = np.zeros_like(Σ)
    for i in range(P.shape[0]):
        try:
            L[i] = np.linalg.cholesky(np.linalg.inv(Σ[i])).T
        except np.linalg.LinAlgError:
            L[i] = np.eye(2)
    theta = np.zeros(6)
    prob = pyceres.Problem()
    prob.add_residual_block(BACost(P, uv, d, L, K), None, [theta])
    opts = pyceres.SolverOptions()
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.max_num_iterations = 50
    opts.minimizer_progress_to_stdout = False
    summ = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summ)
    return theta, P.shape[0]


def grid_windows(IW, IH, cs, stride):
    us = list(range(0, max(1, IW - cs + 1), stride))
    vs = list(range(0, max(1, IH - cs + 1), stride))
    if us[-1] != IW - cs: us.append(IW - cs)
    if vs[-1] != IH - cs: vs.append(IH - cs)
    return [(u, v) for v in vs for u in us]


def evaluate_frame(model, inst, ypr_gt, t_gt_world, S, cs, stride, n_iters=3):
    K = inst['K_full'].numpy()
    pts_w = inst['pts'].numpy()
    cam_pos = inst['cam_pos'].numpy()
    R_gt = inst['R_gt'].numpy()
    img = decode_jpg(bytes(inst['jpg_bytes']))
    IH, IW = img.shape[:2]

    windows = grid_windows(IW, IH, cs, stride)

    ypr_est = np.zeros(3, dtype=np.float32)
    t_est = np.zeros(3, dtype=np.float32)
    n_pts_total = 0
    for it in range(n_iters):
        ypr_res = (ypr_gt - ypr_est).astype(np.float32)
        t_res = (t_gt_world - t_est).astype(np.float32)
        R_off = R_gt @ ypr_to_R(ypr_res)
        t_off = cam_pos + t_res
        obs = []
        for (u0, v0) in windows:
            o = infer_crop(model, img, K, pts_w, R_off, t_off, u0, v0, cs, S)
            if o is not None:
                obs.append(o)
        if not obs:
            return None
        n_pts_total = sum(o['P'].shape[0] for o in obs)
        if n_pts_total < 20:
            return None
        theta, n_pts_ba = solve_ba(obs, K)
        d_ypr = theta[:3]
        d_t_cam = theta[3:6]
        d_t_w = R_gt @ d_t_cam
        # debug: should converge to theta ≈ ypr_res / R_gt.T @ t_res_world
        expected_t_cam = R_gt.T @ t_res
        if it == 0:
            print(f'    iter0 obs={len(obs)} pts={n_pts_ba}  '
                  f'expected ypr_res={ypr_res}  got theta_ypr={d_ypr}', flush=True)
            print(f'    expected t_cam={expected_t_cam}  got theta_t_cam={d_t_cam}', flush=True)
        ypr_est += d_ypr
        t_est += d_t_w
        if np.linalg.norm(d_ypr) < 1e-3 and np.linalg.norm(d_t_cam) < 1e-4:
            break

    # reprojection error before/after
    R_off_full = R_gt @ ypr_to_R(ypr_gt)
    t_off_full = cam_pos + t_gt_world
    uv_bf, z_bf, _ = project(pts_w, R_off_full, t_off_full, K)
    uv_gt, z_gt, _ = project(pts_w, R_gt, cam_pos, K)
    R_res_full = R_gt @ ypr_to_R(ypr_gt - ypr_est)
    t_res_full = cam_pos + (t_gt_world - t_est)
    uv_af, _, _ = project(pts_w, R_res_full, t_res_full, K)
    in_img = ((z_gt > 0.5) & (uv_gt[:, 0] >= 0) & (uv_gt[:, 0] < IW) &
                              (uv_gt[:, 1] >= 0) & (uv_gt[:, 1] < IH))
    if in_img.sum() < 10:
        return None

    rb = float(np.linalg.norm(uv_bf[in_img] - uv_gt[in_img], axis=1).mean())
    ra = float(np.linalg.norm(uv_af[in_img] - uv_gt[in_img], axis=1).mean())
    rot_geo = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(ypr_to_R(ypr_est).T @ ypr_to_R(ypr_gt)) - 1) / 2, -1, 1))))
    return dict(
        n_crops=len(obs), n_pts=n_pts_total,
        ypr_gt=ypr_gt.tolist(), t_gt=t_gt_world.tolist(),
        ypr_est=ypr_est.tolist(), t_est=t_est.tolist(),
        rot_err_geo_deg=rot_geo,
        ypr_err_norm_deg=float(np.linalg.norm(ypr_est - ypr_gt)),
        t_err_norm_cm=float(np.linalg.norm(t_est - t_gt_world)) * 100,
        reproj_before_px=rb, reproj_after_px=ra,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--sigma-rot', type=float, default=0.5, help='deg, half-range')
    ap.add_argument('--sigma-t',   type=float, default=0.20, help='m, half-range')
    ap.add_argument('--n-frames', type=int, default=100)
    ap.add_argument('--window', type=int, default=384)
    ap.add_argument('--stride', type=int, default=192)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--clearml', action='store_true')
    args = ap.parse_args()

    cml = None
    if args.clearml:
        from clearml import Task
        cml = Task.init(project_name='e2e_calib/ba',
                        task_name=f'ba_eval_{args.exp.name}',
                        task_type=Task.TaskTypes.qc, reuse_last_task_id=False)
        cml.connect({'sigma_rot_deg': args.sigma_rot, 'sigma_t_m': args.sigma_t,
                     'window': args.window, 'stride': args.stride,
                     'n_frames': args.n_frames})

    model, cfg = load_model(args.exp)
    print(f'model: {args.exp.name}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    ds = PandaSetCalibDatasetFull(args.cache, split='val', img_size=cfg['img_size'],
                                    min_crop_px=128, max_crop_px=args.window, oversample=1)
    print(f'val frames in cache: {len(ds.fnames)}')
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds.fnames), size=min(args.n_frames, len(ds.fnames)),
                      replace=False)

    S = cfg['img_size']
    results = []
    t0 = time.time()
    for fi, idx in enumerate(idxs):
        inst = torch.load(ds.inst_dir / ds.fnames[idx], weights_only=False)
        ypr_gt = (rng.random(3) * 2 - 1) * args.sigma_rot
        t_gt_w = (rng.random(3) * 2 - 1) * args.sigma_t
        try:
            r = evaluate_frame(model, inst, ypr_gt.astype(np.float32),
                                t_gt_w.astype(np.float32), S, args.window, args.stride)
        except Exception as e:
            print(f'  frame {fi}: {type(e).__name__}: {e}')
            continue
        if r is None: continue
        results.append(r)
        if (fi + 1) % 5 == 0:
            print(f'  {fi+1}/{len(idxs)}  ({time.time()-t0:.0f}s)  '
                  f'rot_geo={r["rot_err_geo_deg"]:.3f}°  t_err={r["t_err_norm_cm"]:.1f}cm  '
                  f'reproj {r["reproj_before_px"]:.1f}→{r["reproj_after_px"]:.1f}', flush=True)
            if cml is not None:
                cml.get_logger().report_scalar('progress/rot_err_deg', 'p50_so_far',
                    iteration=fi+1, value=float(np.median([rr['rot_err_geo_deg'] for rr in results])))
                cml.get_logger().report_scalar('progress/t_err_cm', 'p50_so_far',
                    iteration=fi+1, value=float(np.median([rr['t_err_norm_cm'] for rr in results])))

    if not results:
        print('no valid frames'); return

    def q(arr, p): return float(np.percentile(arr, p))
    rg = np.array([r['rot_err_geo_deg'] for r in results])
    te = np.array([r['t_err_norm_cm']   for r in results])
    rb = np.array([r['reproj_before_px'] for r in results])
    ra = np.array([r['reproj_after_px']  for r in results])
    summary = dict(
        n=len(results),
        rot_p50=q(rg, 50), rot_p90=q(rg, 90),
        t_p50=q(te, 50),   t_p90=q(te, 90),
        rb_p50=q(rb, 50),  ra_p50=q(ra, 50),
        sigma_rot=args.sigma_rot, sigma_t=args.sigma_t,
    )
    print()
    print(f'  rot err (geodesic deg): p50={summary["rot_p50"]:.3f}  p90={summary["rot_p90"]:.3f}  '
          f'(init≈{args.sigma_rot*np.sqrt(3):.2f})')
    print(f'  t   err (cm):           p50={summary["t_p50"]:.2f}   p90={summary["t_p90"]:.2f}  '
          f'(init≈{args.sigma_t*100*np.sqrt(3):.1f})')
    print(f'  reproj px p50:          {summary["rb_p50"]:.2f} → {summary["ra_p50"]:.2f}')

    out_dir = args.exp / 'ba'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'ba_eval.json').write_text(json.dumps(dict(summary=summary, per_frame=results), indent=2))
    print(f'saved → {out_dir}/ba_eval.json')

    if cml is not None:
        L = cml.get_logger()
        for k, v in summary.items(): L.report_single_value(name=k, value=float(v))
        L.report_histogram('rot_err_deg_hist', 'all', values=rg.tolist(), iteration=0)
        L.report_histogram('t_err_cm_hist',    'all', values=te.tolist(), iteration=0)
        L.report_histogram('reproj_before_px', 'all', values=rb.tolist(), iteration=0)
        L.report_histogram('reproj_after_px',  'all', values=ra.tolist(), iteration=0)


if __name__ == '__main__':
    main()
