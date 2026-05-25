"""Time the σ-head forward pass alone, varying batch size.

Runs *inside* the calib-api container so it hits the same GPU 15
weights that the live server uses.
"""
import os, time, json
import torch
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full

DEVICE = ess.DEVICE
print("[bench] device:", DEVICE, "  ckpt:", ess.CKPT.name)

cfg = ess._load_cfg()
ds = PandaSetCalibDatasetFull(
    cache_dir=ess.CACHE, split='val',
    img_size=cfg['img_size'],
    min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
    max_offset_m=0.0, max_rot_deg=0.0,
    oversample=1, grid_n=cfg.get('grid_n', 16),
    center_band=0.0, preload=False,
)
ds._cfg = cfg
model = ess._build_model(cfg).to(DEVICE)
sd = torch.load(ess.CKPT, map_location=DEVICE, weights_only=False)
if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
    sd = sd.get("state_dict") or sd.get("model")
model.load_state_dict(sd, strict=False)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

import numpy as np
ypr = np.array([0.3, -0.2, 0.5], dtype=np.float64)
tt  = np.array([0.030, -0.020, 0.040], dtype=np.float64)
inst = ds._load_inst(17)
u0v0 = [(0,0),(256,0),(0,256),(256,256)]

def make_batch(B):
    wins = []
    while len(wins) < B:
        w = ess._build_subwin(ds, inst, tt, ypr,
                              u0=u0v0[len(wins)%4][0],
                              v0=u0v0[len(wins)%4][1], cs=256)
        if w is not None:
            wins.append(w)
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
    (imgs, _u, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _, *_rest) = moved
    use_intensity = getattr(model, "use_intensity", True)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1) if use_intensity else dist_uvd[..., :3]
    return imgs.float().div(255.), point_in, pad_mask, vfp, bucket_uvd, bucket_valid

@torch.no_grad()
def time_fwd(B, n_warm=3, n_iter=20):
    img, pt, pad, vfp, bu, bv = make_batch(B)
    for _ in range(n_warm):
        _ = model(img, pt, key_padding_mask=pad, vfp=vfp,
                  bucket_uvd=bu, bucket_valid=bv)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = model(img, pt, key_padding_mask=pad, vfp=vfp,
                  bucket_uvd=bu, bucket_valid=bv)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter
    return dt

print("=== σ-head forward (cs=256, img_size=%d) ===" % cfg["img_size"])
print(f"{'B':>5}  {'ms_total':>10}  {'ms/sample':>11}")
for B in [1, 4, 16, 32, 64, 128, 256]:
    try:
        dt = time_fwd(B)
        print(f"{B:>5}  {dt*1000:>10.2f}  {dt*1000/B:>11.3f}")
    except RuntimeError as e:
        print(f"{B:>5}  OOM/err: {e}")
        break

# ─── GN solve micro-bench ────────────────────────────────────────────────
from scripts.ba.ba_torch import (
    solve_kb_xyz_shared, make_info_from_sigma_rho,
)

@torch.no_grad()
def time_full_pipeline(B, n_warm=3, n_iter=20):
    """Time forward + GN separately for the same batch."""
    wins = []
    while len(wins) < B:
        w = ess._build_subwin(ds, inst, tt, ypr,
                              u0=u0v0[len(wins)%4][0],
                              v0=u0v0[len(wins)%4][1], cs=256)
        if w is not None:
            wins.append(w)
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
    (imgs, _u, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b) = moved
    valid = ~pad_mask
    pad_full = ~valid
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        P0_orig[pad_full] = torch.tensor([0.,0.,1.], dtype=P0_orig.dtype, device=DEVICE)
    dist_one = inst["distortion"].clone().detach().to(torch.float32).reshape(1,4).to(DEVICE)
    dist_b = dist_one.expand(B, 4).contiguous()
    use_intensity = getattr(model, "use_intensity", True)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1) if use_intensity else dist_uvd[..., :3]
    img_norm = imgs.float().div(255.)

    def fwd():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
        per_pt = out[0] if isinstance(out, tuple) else out
        return per_pt

    def gn(per_pt):
        duv_pred_local = per_pt[..., :2].detach()
        if pad_full.any():
            duv_pred_local = duv_pred_local.clone()
            duv_pred_local[pad_full] = 0.
        sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
        W_local = make_info_from_sigma_rho(sx, sy, rho).detach()
        scale = (cs_b / float(cfg["img_size"])).reshape(-1,1,1)
        inv_l2o = (1.0/scale).reshape(-1,1,1,1)
        duv_orig_pred = duv_pred_local * scale
        W_orig = W_local * inv_l2o.pow(2)
        prior = ess.PRIOR_DIAG.to(DEVICE)
        delta, _H = solve_kb_xyz_shared(
            P0_orig, duv_orig_pred, W_orig, K_orig, dist_b, ess.DOFS,
            valid=valid, n_iter=ess.BA_N_ITER, damping=ess.DAMPING,
            prior_diag=prior,
        )
        return delta

    # warmup
    for _ in range(n_warm):
        per_pt = fwd(); _ = gn(per_pt)
    torch.cuda.synchronize()

    # forward only
    t0 = time.time()
    for _ in range(n_iter):
        per_pt = fwd()
    torch.cuda.synchronize()
    dt_fwd = (time.time() - t0) / n_iter

    # GN only (per_pt cached)
    per_pt_cached = fwd(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        _ = gn(per_pt_cached)
    torch.cuda.synchronize()
    dt_gn = (time.time() - t0) / n_iter

    return dt_fwd, dt_gn

print()
print("=== forward vs GN solve breakdown (cs=256) ===")
print(f"BA_N_ITER={ess.BA_N_ITER}  DOFS={ess.DOFS}  DAMPING={ess.DAMPING}")
print(f"{'B':>5}  {'fwd_ms':>9}  {'gn_ms':>9}  {'sum_ms':>9}")
for B in [4, 16, 32, 64, 128]:
    try:
        df, dg = time_full_pipeline(B)
        print(f"{B:>5}  {df*1000:>9.2f}  {dg*1000:>9.2f}  {(df+dg)*1000:>9.2f}")
    except Exception as e:
        print(f"{B:>5}  err: {e}")
        break

# ─── GN solve vs n_iter ──────────────────────────────────────────────────
print()
print("=== GN solve vs n_iter (B=128) ===")
print(f"{'n_iter':>7}  {'gn_ms':>9}  {'delta_first6 (deg, m)':>30}")
B = 128
wins = []
while len(wins) < B:
    w = ess._build_subwin(ds, inst, tt, ypr,
                          u0=u0v0[len(wins)%4][0],
                          v0=u0v0[len(wins)%4][1], cs=256)
    if w is not None:
        wins.append(w)
moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
(imgs, _u, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, _,
 pts_cam_orig, duv_orig, K_orig, cs_b) = moved
valid = ~pad_mask
pad_full = ~valid
P0_orig = pts_cam_orig.detach().clone()
if pad_full.any():
    P0_orig[pad_full] = torch.tensor([0.,0.,1.], dtype=P0_orig.dtype, device=DEVICE)
dist_one = inst["distortion"].clone().detach().to(torch.float32).reshape(1,4).to(DEVICE)
dist_b = dist_one.expand(B, 4).contiguous()
use_intensity = getattr(model, "use_intensity", True)
point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1) if use_intensity else dist_uvd[..., :3]
img_norm = imgs.float().div(255.)
with torch.no_grad():
    out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
duv_pred_local = per_pt[..., :2].detach()
if pad_full.any():
    duv_pred_local = duv_pred_local.clone(); duv_pred_local[pad_full] = 0.
sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
W_local = make_info_from_sigma_rho(sx, sy, rho).detach()
scale = (cs_b / float(cfg["img_size"])).reshape(-1,1,1)
inv_l2o = (1.0/scale).reshape(-1,1,1,1)
duv_orig_pred = duv_pred_local * scale
W_orig = W_local * inv_l2o.pow(2)
prior = ess.PRIOR_DIAG.to(DEVICE)

@torch.no_grad()
def gn_iter(n_iter, n_warm=3, n_loops=20):
    for _ in range(n_warm):
        delta, _ = solve_kb_xyz_shared(
            P0_orig, duv_orig_pred, W_orig, K_orig, dist_b, ess.DOFS,
            valid=valid, n_iter=n_iter, damping=ess.DAMPING, prior_diag=prior)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_loops):
        delta, _ = solve_kb_xyz_shared(
            P0_orig, duv_orig_pred, W_orig, K_orig, dist_b, ess.DOFS,
            valid=valid, n_iter=n_iter, damping=ess.DAMPING, prior_diag=prior)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_loops, delta

for n_iter in [1, 2, 3, 4, 6, 10]:
    dt, d = gn_iter(n_iter)
    d_np = d.detach().cpu().numpy().tolist()
    s = "[" + ", ".join(f"{x:+.4f}" for x in d_np[:3]) + " | " \
        + ", ".join(f"{x:+.4f}" for x in d_np[3:]) + "]"
    print(f"{n_iter:>7}  {dt*1000:>9.2f}  {s}")
