"""Sweep perturbation direction/magnitude on a few val frames to see if
the model recovers consistently (and when it breaks). Fixed frame, varied
pert — pure-axis sweeps + mixed random. Training range: ±0.5°, ±0.2m."""
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

import importlib.util as _ilu
spec = _ilu.spec_from_file_location('_cfg', f'{EXP}/config.py')
mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); cfg = mod.CFG
IMG_SIZE = int(cfg['img_size'])

import os
SPLIT = os.environ.get('SWEEP_SPLIT', 'val')
ds = PandaSetCalibDatasetFull(CACHE, split=SPLIT,
    img_size=IMG_SIZE, min_crop_px=512, max_crop_px=512,
    max_rot_deg=0.5, max_offset_m=0.2,
    grid_n=cfg['grid_n'], oversample=40,
    crop_grid=True, share_pert=True, center_band=0.5)
model = build_model(EXP, device=dev).eval()


def one_pert(idx, t_delta, ypr_deg, seed=0):
    inst = ds._load_inst(idx)
    ds._use_grid=True; ds._grid_ju=0; ds._grid_jv=0
    ds._grid_cells = ds._plan_grid(inst); ds._grid_k=0
    samples = []
    for k in range(40):
        np.random.seed(seed*40 + k)
        s = ds._build_one_window(idx, _inst_override=inst,
                                 _force_pert=(np.asarray(t_delta, dtype=np.float64).copy(),
                                              np.asarray(ypr_deg, dtype=np.float64).copy()),
                                 _win_i=k)
        if s is None: continue
        samples.append(s)
    if not samples: return None, None
    batch = collate_full(samples)
    (imgs, trues, dists, pad, vfps, b_uvds, b_valids, pv,
     pts_co, duv_o, K_o, cs_t, _d1, _u0, _v0, _ig) = batch
    img_t = imgs.float().div(255.0).to(dev); dist_t = dists.to(dev)
    vfp_t = vfps.to(dev); bu_t = b_uvds.to(dev); bv_t = b_valids.to(dev)
    pad_t = pad.to(dev)
    point_in = torch.cat([dist_t[..., :3], dist_t[..., 4:5]], dim=-1)
    with torch.no_grad():
        out = model(img_t, point_in, dpose_R=None, vfp=vfp_t,
                     bucket_uvd=bu_t, bucket_valid=bv_t, key_padding_mask=pad_t)
    per_pt = out[0] if isinstance(out, tuple) else out
    info_head_ba = out[1] if (isinstance(out, tuple) and len(out)>1) else None
    B = per_pt.shape[0]; G = len(samples)
    pts_co = pts_co.to(dev); duv_o = duv_o.to(dev)
    K_o = K_o.to(dev); cs_t = cs_t.to(dev)
    valid = (~pad_t) & (pts_co[..., 2] > 0.5)
    safe = torch.tensor([0., 0., 10.], device=dev)
    P0 = torch.where(valid.unsqueeze(-1), pts_co, safe)
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
                                     damping=1e-3, prior_diag=prior)
        Wi = make_info_from_sigma_rho(
            torch.ones(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64),
            torch.ones(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64),
            torch.zeros(P0g.shape[0], P0g.shape[1], device=dev, dtype=torch.float64))
        delta_gt, _ = solve_pose(P0g.double(), -duv_g.double(), Wi, K_g.double(),
                                   valid=valid_g, n_iter=10, damping=1e-3,
                                   prior_diag=prior)
    return delta_pred[0].cpu().numpy(), delta_gt[0].cpu().numpy()


N = len(ds)
IDXS = [0, N//4, N//2, (3*N)//4]            # 4 frames spread across the split
mags_rot = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
mags_t   = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

def _hdr(title):
    print('\n' + '='*78); print(title); print('='*78)
    print(f'{"frame":5s} {"pert":>22s}  {"err rot (deg)":>14s}  {"err t (mm)":>14s}  {"ok?":5s}')

def _row(idx, lbl, dp, dg):
    if dp is None:
        print(f'{idx:5d} {lbl:>22s}  {"skip":>14s}'); return
    e_rot = np.abs(dp[:3] - dg[:3]).mean()
    e_t   = np.abs(dp[3:] - dg[3:]).mean() * 1000.0
    # "ok" = within 2x training val quality.
    ok = 'ok' if (e_rot < 0.12 and e_t < 25) else 'BAD'
    print(f'{idx:5d} {lbl:>22s}  {e_rot:>14.4f}  {e_t:>14.1f}  {ok:5s}')

for axis_name, axis in [('yaw(z)', 0), ('pitch(y)', 1), ('roll(x)', 2)]:
    _hdr(f'PURE ROTATION: {axis_name}')
    for idx in IDXS:
        for m in mags_rot:
            for sign in (+1, -1) if m > 0 else (+1,):
                ypr = np.zeros(3); ypr[axis] = sign * m
                dp, dg = one_pert(idx, np.zeros(3), ypr)
                _row(idx, f'{axis_name}={sign*m:+.2f}deg', dp, dg)

for axis_name, axis in [('tx', 0), ('ty', 1), ('tz', 2)]:
    _hdr(f'PURE TRANSLATION: {axis_name}')
    for idx in IDXS:
        for m in mags_t:
            for sign in (+1, -1) if m > 0 else (+1,):
                t = np.zeros(3); t[axis] = sign * m
                dp, dg = one_pert(idx, t, np.zeros(3))
                _row(idx, f'{axis_name}={sign*m:+.2f}m', dp, dg)

_hdr('MIXED RANDOM')
rng = np.random.RandomState(7)
for idx in IDXS:
    for trial in range(4):
        ypr = (rng.rand(3)*2-1) * 0.5
        tt  = (rng.rand(3)*2-1) * 0.2
        dp, dg = one_pert(idx, tt, ypr)
        _row(idx, f'ypr={ypr.round(2).tolist()}', dp, dg)
