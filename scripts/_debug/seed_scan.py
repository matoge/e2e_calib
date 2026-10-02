import sys, os
sys.path.insert(0, '/workspace')
import numpy as np, torch
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.calib_lib import build_model
from datasets.train_cnd2_ddp import _ba_pose_loss

EXP = '/workspace/experiments/km_only_s3_gf0_1002_2104'
CACHE = '/raid/home/hfunaya/cache_v5/kamikado_v3_full'
device = 'cuda'
import importlib.util as _ilu
spec = _ilu.spec_from_file_location('_cfg', f'{EXP}/config.py')
mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); cfg = mod.CFG

ds = PandaSetCalibDatasetFull(
    CACHE, split='val',
    img_size=cfg['img_size'], min_crop_px=512, max_crop_px=512,
    max_rot_deg=cfg['rot_deg'], max_offset_m=cfg['t_m'],
    grid_n=cfg['grid_n'], oversample=40,
    crop_grid=True, share_pert=True, center_band=0.5,
)
print(f'val size: {len(ds)}', flush=True)
model = build_model(EXP, device=device).eval()
img_size = cfg['img_size']

for seed in range(0, 8):
    idx = seed % len(ds)
    inst = ds._load_inst(idx)
    ds._use_grid = True; ds._grid_ju=0; ds._grid_jv=0
    ds._grid_cells = ds._plan_grid(inst); ds._grid_k = 0
    samples = []
    for k in range(40):
        np.random.seed(seed*40 + k)
        s = ds._build_one_window(idx, _inst_override=inst, _force_pert=(np.zeros(3), np.zeros(3)), _win_i=k)
        if s is None: continue
        samples.append(s)
    if not samples:
        print(f'seed={seed} idx={idx}: NO samples', flush=True); continue
    batch = collate_full(samples)
    (imgs, trues, dists, pad, vfps, b_uvds, b_valids, pv, pts_co, duv_o, K_o, cs_t, _d1, _u0, _v0, _ig) = batch
    img_t = imgs.float().div(255.0).to(device)
    dist_t = dists.to(device); vfp_t = vfps.to(device)
    bu_t = b_uvds.to(device); bv_t = b_valids.to(device)
    pad_t = pad.to(device)
    point_in = torch.cat([dist_t[..., :3], dist_t[..., 4:5]], dim=-1)
    with torch.no_grad():
        out = model(img_t, point_in, dpose_R=None, vfp=vfp_t, bucket_uvd=bu_t, bucket_valid=bv_t, key_padding_mask=pad_t)
    per_pt = out[0] if isinstance(out, tuple) else out
    info_ba = out[1] if (isinstance(out, tuple) and len(out)>1) else None
    ba_batch = [None]*8 + [pts_co.to(device), duv_o.to(device), K_o.to(device), cs_t.to(device)]
    _loss, diag = _ba_pose_loss(per_pt, dist_t, pad_t, ba_batch, ba_iter=4, damping=1e-3, loss_type='nll', img_size=img_size, group=len(samples), info_head_ba=info_ba, is_grid=None, return_pose=True)
    dp = diag['delta_pred'][0].cpu().numpy()
    dg = diag['delta_gt'][0].cpu().numpy()
    rot_deg = np.degrees(dp[:3])
    print(f'seed={seed} idx={idx} tiles={len(samples)}  δ_pred rot=({rot_deg[0]:+.3f},{rot_deg[1]:+.3f},{rot_deg[2]:+.3f})°  t=({dp[3]*1000:+.1f},{dp[4]*1000:+.1f},{dp[5]*1000:+.1f})mm   |rot|={np.linalg.norm(rot_deg):.3f}° |t|={np.linalg.norm(dp[3:])*1000:.1f}mm   rot_err={diag.get("rot_err",float("nan")):.5f}', flush=True)
