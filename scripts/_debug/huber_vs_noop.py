"""Compare per-frame δ_pred vs δ_gt stats WITH and WITHOUT Huber IRLS.
Loops over seed=0..N on kamikado val @ RANDOM training-range perturbation
(rot=±0.5°, t=±0.2m — same as training val). Prints per-axis stats of |err|.
Zero-pert comparison is noise-floor dominated (|rot|~0.02°, |t|~8mm); Huber
only kicks in when residuals are bigger than info_head's σ estimate."""
import sys
from pathlib import Path
sys.path.insert(0, '/workspace')
import numpy as np, torch
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.calib_lib import build_model
from scripts.ba.gn_pose import solve_pose
from datasets.train_cnd2_ddp import make_info_from_sigma_rho

EXP   = '/workspace/experiments/km_only_s3_gf0_1002_2104'
CACHE = '/raid/home/hfunaya/cache_v5/kamikado_v3_full'
dev   = 'cuda'
N_SEEDS = 20

import importlib.util as _ilu
spec = _ilu.spec_from_file_location('_cfg', f'{EXP}/config.py')
mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); cfg = mod.CFG
IMG_SIZE = int(cfg['img_size'])

ds = PandaSetCalibDatasetFull(
    CACHE, split='val',
    img_size=IMG_SIZE, min_crop_px=512, max_crop_px=512,
    max_rot_deg=cfg['rot_deg'], max_offset_m=cfg['t_m'],
    grid_n=cfg['grid_n'], oversample=40,
    crop_grid=True, share_pert=True, center_band=0.5,
)
model = build_model(EXP, device=dev).eval()


def one_frame(seed, robust=None, pert=None):
    """pert = (t_delta (3,), ypr_deg (3,)). If None, zero (noise floor only)."""
    idx = seed % len(ds)
    inst = ds._load_inst(idx)
    ds._use_grid=True; ds._grid_ju=0; ds._grid_jv=0
    ds._grid_cells = ds._plan_grid(inst); ds._grid_k=0
    samples = []
    t_force, ypr_force = (np.zeros(3), np.zeros(3)) if pert is None else pert
    for k in range(40):
        np.random.seed(seed*40 + k)
        s = ds._build_one_window(idx, _inst_override=inst,
                                 _force_pert=(t_force.copy(), ypr_force.copy()),
                                 _win_i=k)
        if s is None: continue
        samples.append(s)
    if not samples: return None
    batch = collate_full(samples)
    (imgs, trues, dists, pad, vfps, b_uvds, b_valids, pv,
     pts_co, duv_o, K_o, cs_t, _d1, _u0, _v0, _ig) = batch
    img_t  = imgs.float().div(255.0).to(dev)
    dist_t = dists.to(dev); vfp_t = vfps.to(dev)
    bu_t = b_uvds.to(dev); bv_t = b_valids.to(dev)
    pad_t = pad.to(dev)
    point_in = torch.cat([dist_t[..., :3], dist_t[..., 4:5]], dim=-1)
    with torch.no_grad():
        out = model(img_t, point_in, dpose_R=None, vfp=vfp_t,
                     bucket_uvd=bu_t, bucket_valid=bv_t, key_padding_mask=pad_t)
    per_pt       = out[0] if isinstance(out, tuple) else out
    info_head_ba = out[1] if (isinstance(out, tuple) and len(out)>1) else None
    B = per_pt.shape[0]; G = len(samples)
    pts_co = pts_co.to(dev); duv_o = duv_o.to(dev)
    K_o    = K_o.to(dev);    cs_t  = cs_t.to(dev)
    valid  = (~pad_t) & (pts_co[..., 2] > 0.5)
    safe = torch.tensor([0., 0., 10.], device=dev)
    P0   = torch.where(valid.unsqueeze(-1), pts_co, safe)
    s2o = (cs_t / float(IMG_SIZE)).view(B, 1)
    mu_orig = per_pt[..., :2] * s2o.unsqueeze(-1)
    W = info_head_ba / (s2o*s2o).view(B,1,1,1)
    def _grp(x): return x.reshape(B//G, G*x.shape[1], *x.shape[2:])
    P0g, mu_g, duv_g = _grp(P0), _grp(mu_orig), _grp(duv_o)
    W_g, valid_g = _grp(W), _grp(valid)
    K_g = K_o.reshape(B//G, G, 3, 3)[:, 0]
    prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09],
                           device=dev, dtype=torch.float64)
    with torch.no_grad():
        delta_pred, _ = solve_pose(P0g.double(), -mu_g.double(), W_g.double(),
                                     K_g.double(), valid=valid_g, n_iter=10,
                                     damping=1e-3, prior_diag=prior,
                                     robust=robust, huber_k=1.5)
        Wi = make_info_from_sigma_rho(
            torch.ones(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64),
            torch.ones(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64),
            torch.zeros(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64))
        delta_gt, _ = solve_pose(P0g.double(), -duv_g.double(), Wi, K_g.double(),
                                   valid=valid_g, n_iter=10, damping=1e-3,
                                   prior_diag=prior)
    err = (delta_pred[0] - delta_gt[0]).cpu().numpy()
    return err  # (6,) [omega_x_deg, omega_y_deg, omega_z_deg, tx_m, ty_m, tz_m]


def summarize(rows, label):
    rows = np.stack(rows)  # (N,6)
    rot_axes = ['roll(ωx)', 'pitch(ωy)', 'yaw(ωz)']
    t_axes   = ['tx', 'ty', 'tz']
    print(f'=== {label}  (N={len(rows)}) ===')
    for i, name in enumerate(rot_axes):
        a = np.abs(rows[:, i])
        print(f'  {name:>12s}  median={np.median(a):.4f}°  mean={a.mean():.4f}°  p90={np.percentile(a, 90):.4f}°')
    for j, name in enumerate(t_axes):
        a = np.abs(rows[:, 3+j]) * 1000.0
        print(f'  {name:>12s}  median={np.median(a):.1f}mm  mean={a.mean():.1f}mm  p90={np.percentile(a, 90):.1f}mm')
    rr = np.abs(rows[:, :3]).mean(-1)
    tt = np.abs(rows[:, 3:]).mean(-1) * 1000.0
    print(f'  mean|rot|   median={np.median(rr):.4f}°  mean={rr.mean():.4f}°')
    print(f'  mean|t|     median={np.median(tt):.1f}mm  mean={tt.mean():.1f}mm')


rng = np.random.RandomState(42)
rows_noop  = []
rows_huber = []
rows_perts = []
for s in range(N_SEEDS):
    ypr = (rng.rand(3) * 2 - 1) * float(cfg['rot_deg'])          # ±0.5°
    tt  = (rng.rand(3) * 2 - 1) * float(cfg['t_m'])              # ±0.2m
    pert = (tt, ypr)
    e0 = one_frame(s, robust=None,     pert=pert)
    e1 = one_frame(s, robust='huber',  pert=pert)
    if e0 is None or e1 is None:
        print(f'seed={s}: skip', flush=True); continue
    rows_noop.append(e0); rows_huber.append(e1); rows_perts.append(pert)
    print(f'seed={s:2d}  ypr=({ypr[0]:+.2f},{ypr[1]:+.2f},{ypr[2]:+.2f})° '
          f't=({tt[0]*1000:+.0f},{tt[1]*1000:+.0f},{tt[2]*1000:+.0f})mm  '
          f'noop|rot|={np.abs(e0[:3]).mean():.3f}°  '
          f'huber|rot|={np.abs(e1[:3]).mean():.3f}°  '
          f'Δ={np.abs(e1[:3]).mean()-np.abs(e0[:3]).mean():+.3f}°', flush=True)

print()
summarize(rows_noop,  'NO HUBER  (random pert ±0.5°/±0.2m)')
print()
summarize(rows_huber, 'HUBER k=1.5 (random pert ±0.5°/±0.2m)')
