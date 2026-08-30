"""System test for the nuScenes cam-LiDAR calib data path.

Walks ONE sample through every stage, dumps an image + JSON per stage, and
checks the 6-DOF at the end. Stages S1-S7 need no trained model.

env:
  WDIR   worktree to import from            (default e2e_gn)
  CACHE  v3 LMDB cache (no-tile)            (default ns_v3_notile)
  OUT    debug dir                          (default DEBUG/<tag>)
  IMG    network input side                 (default 256)
  CROP   fixed crop side in original px     (default 256)
  ROT/TM perturbation half-range            (default 0.5 deg / 0.05 m)
  N      samples to dump                    (default 5)
  CKPT   trained model -> enables S8
"""
import os, sys, json, io, time, pathlib, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[1])
WD  = os.environ.get('WDIR', REPO); sys.path.insert(0, WD)
CACHE = os.environ.get('CACHE')
if not CACHE:
    raise SystemExit("set CACHE=<v3 lmdb cache dir>  (e.g. CACHE=.../ns_v3_notile)")
L = os.environ.get('SCRATCH', '/tmp/e2e_calib_systest')
TAG   = os.environ.get('TAG', 'run')
OUT   = os.environ.get('OUT', f'{L}/DEBUG/{TAG}')
IMG   = int(os.environ.get('IMG', '256'))
CROP  = int(os.environ.get('CROP', '256'))
ROT   = float(os.environ.get('ROT', '0.5')); TM = float(os.environ.get('TM', '0.05'))
NDUMP = int(os.environ.get('N', '5'))
CKPT  = os.environ.get('CKPT', '')
os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.ba.gn_pose import solve_pose, _identity_W, project_pinhole, _apply_extrinsic
from scipy.spatial.transform import Rotation

REPORT = {'config': dict(wdir=WD, cache=CACHE, img=IMG, crop=CROP, rot=ROT, tm=TM),
          'samples': [], 'checks': {}}
FAIL = []
def chk(name, cond, msg):
    REPORT['checks'].setdefault(name, []).append(dict(ok=bool(cond), msg=msg))
    if not cond: FAIL.append(f"{name}: {msg}")
    return cond

# a dataset only used for its cache reader + the same crop machinery
ds = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                              min_crop_px=CROP, max_crop_px=CROP, oversample=1,
                              max_rot_deg=ROT, max_offset_m=TM)
print(f"cache {CACHE}: {len(ds)} instances   img {IMG}  crop {CROP}  pert {ROT}deg/{TM}m")

def proj(P, K):
    z = np.maximum(P[:, 2:3], 1e-6)
    return (P[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])

rng = np.random.default_rng(0)
for si in range(NDUMP):
    S = {}
    idx = int(rng.integers(len(ds)))
    inst = ds._load_inst(idx)
    K   = inst['K_full'].numpy().astype(np.float64)
    pts = inst['pts'].numpy().astype(np.float64)          # WORLD
    cp  = inst['cam_pos'].numpy().astype(np.float64)
    Rgt = inst['R_gt'].numpy().astype(np.float64)
    IH, IW = int(inst['IH']), int(inst['IW'])
    full = np.asarray(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    S['idx'] = idx; S['scene'] = str(inst.get('scene')); S['frame'] = int(inst.get('frame', -1))
    S['image_hw'] = [IH, IW]; S['n_pts_world'] = int(len(pts))

    # ---------------- S1 raw frame + GT projection ----------------
    Pc  = (pts - cp) @ Rgt                                  # GT camera frame
    uvg = proj(Pc, K); z = Pc[:, 2]
    inim = (z > 0.5) & (uvg[:, 0] >= 0) & (uvg[:, 0] < IW) & (uvg[:, 1] >= 0) & (uvg[:, 1] < IH)
    S['S1'] = dict(n_in_image=int(inim.sum()),
                   depth=[float(z[inim].min()), float(np.median(z[inim])), float(z[inim].max())],
                   K=[float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])])
    fig, a = plt.subplots(figsize=(13, 7.4), dpi=110); a.imshow(full)
    a.scatter(uvg[inim,0], uvg[inim,1], s=1.5, c=z[inim], cmap='turbo', vmin=2, vmax=60, lw=0)
    a.set_title(f"S1 raw frame {IW}x{IH} + LiDAR at GT extrinsic  ({int(inim.sum())} pts)")
    a.set_xticks([]); a.set_yticks([]); plt.tight_layout()
    plt.savefig(f"{OUT}/s{si}_S1_raw.png", bbox_inches='tight'); plt.close()

    # ---------------- S2 cache consistency ----------------
    uvc = inst['uv_full'].numpy().astype(np.float64)
    d2 = np.linalg.norm(uvc[inim] - uvg[inim], axis=1)
    S['S2'] = dict(max=float(d2.max()), p50=float(np.median(d2)))
    chk('S2_cache_uv', d2.max() < 0.5, f"cached uv vs reprojected: max {d2.max():.4f} px")

    # ---------------- S3 pivot + crop box ----------------
    cand = np.where(inim)[0]
    piv  = int(cand[rng.integers(len(cand))])
    u0 = int(np.clip(uvg[piv,0] - CROP/2, 0, IW - CROP))
    v0 = int(np.clip(uvg[piv,1] - CROP/2, 0, IH - CROP))
    S['S3'] = dict(pivot_uv=[float(uvg[piv,0]), float(uvg[piv,1])], u0=u0, v0=v0, cs=CROP,
                   box_inside=bool(0 <= u0 <= IW-CROP and 0 <= v0 <= IH-CROP))
    chk('S3_box_inside', S['S3']['box_inside'], f"crop box ({u0},{v0},{CROP}) inside {IW}x{IH}")
    fig, a = plt.subplots(figsize=(13, 7.4), dpi=110); a.imshow(full)
    a.add_patch(plt.Rectangle((u0, v0), CROP, CROP, fill=False, ec='red', lw=2.5))
    a.scatter([uvg[piv,0]], [uvg[piv,1]], s=90, marker='x', c='yellow', lw=2.2)
    a.set_title(f"S3 pivot (yellow x) and {CROP}x{CROP} crop at ({u0},{v0})")
    a.set_xticks([]); a.set_yticks([]); plt.tight_layout()
    plt.savefig(f"{OUT}/s{si}_S3_crop.png", bbox_inches='tight'); plt.close()

    # ---------------- S4 perturbation ----------------
    ypr = (rng.random(3)*2 - 1) * ROT
    t   = (rng.random(3)*2 - 1) * TM
    Rd  = Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    R_off, cp_off = Rgt @ Rd, cp + t
    Po  = (pts - cp_off) @ R_off
    uvo = proj(Po, K); zo = Po[:, 2]
    shift = np.linalg.norm(uvo[inim] - uvg[inim], axis=1)
    S['S4'] = dict(inject_ypr_deg=[float(x) for x in ypr], inject_t_m=[float(x) for x in t],
                   shift_px=dict(p50=float(np.median(shift)), p90=float(np.percentile(shift,90)),
                                 max=float(shift.max())))
    sub = full[v0:v0+CROP, u0:u0+CROP]
    ib = (uvo[:,0]>=u0)&(uvo[:,0]<u0+CROP)&(uvo[:,1]>=v0)&(uvo[:,1]<v0+CROP)&(zo>0.5)&(z>0.5)
    fig, a = plt.subplots(figsize=(9, 9), dpi=120); a.imshow(sub, extent=[u0,u0+CROP,v0+CROP,v0])
    a.scatter(uvg[ib,0], uvg[ib,1], s=26, marker='+', c='lime', lw=1.1)
    a.scatter(uvo[ib,0], uvo[ib,1], s=26, facecolors='none', edgecolors='red', lw=0.9)
    a.set_title(f"S4 GT(+green) vs perturbed(o red)  ypr={np.round(ypr,3)} t={np.round(t,3)}\n"
                f"shift p50 {np.median(shift):.2f} px", fontsize=10)
    a.set_xticks([]); a.set_yticks([]); plt.tight_layout()
    plt.savefig(f"{OUT}/s{si}_S4_pert.png", bbox_inches='tight'); plt.close()

    # ---------------- S5 in-crop filter + cell representatives ----------------
    sel = np.where(ib)[0]
    uvl = (uvo[sel] - [u0, v0]) * (IMG / CROP)
    G = 16; cell = IMG / G
    cu = np.clip((uvl[:,0]/cell).astype(int), 0, G-1); cv = np.clip((uvl[:,1]/cell).astype(int), 0, G-1)
    cid = cv*G + cu
    dc = (uvl[:,0]-(cu+0.5)*cell)**2 + (uvl[:,1]-(cv+0.5)*cell)**2
    order = np.lexsort((dc, cid)); _, fp = np.unique(cid[order], return_index=True)
    rep = sel[order[fp]]
    S['S5'] = dict(n_candidates=int(inim.sum()), n_in_crop=int(len(sel)),
                   n_occupied_cells=int(len(np.unique(cid))), n_representatives=int(len(rep)),
                   pts_per_cell=float(len(sel)/max(1,len(np.unique(cid)))))
    chk('S5_reps', len(rep) == len(np.unique(cid)), f"{len(rep)} reps for {len(np.unique(cid))} occupied cells")
    fig, a = plt.subplots(figsize=(9,9), dpi=120); a.imshow(sub, extent=[0,IMG,IMG,0])
    a.scatter(uvl[:,0], uvl[:,1], s=12, c='gray', lw=0, label=f'in-crop {len(sel)}')
    ruv = (uvo[rep]-[u0,v0])*(IMG/CROP)
    a.scatter(ruv[:,0], ruv[:,1], s=34, facecolors='none', edgecolors='red', lw=1.1,
              label=f'representatives {len(rep)}')
    for gg in range(1, G): a.axvline(gg*cell, c='w', lw=0.25, alpha=.4); a.axhline(gg*cell, c='w', lw=0.25, alpha=.4)
    a.legend(fontsize=8); a.set_title(f"S5 in-crop {len(sel)} -> {len(rep)} reps ({G}x{G} cells)")
    a.set_xticks([]); a.set_yticks([]); plt.tight_layout()
    plt.savefig(f"{OUT}/s{si}_S5_reps.png", bbox_inches='tight'); plt.close()

    # ---------------- S6 resize + coordinates ----------------
    scale = IMG / CROP
    gt_loc  = (uvg[rep]-[u0,v0])*scale
    off_loc = (uvo[rep]-[u0,v0])*scale
    duv_win = np.linalg.norm(gt_loc-off_loc, axis=1)
    duv_org = np.linalg.norm(uvg[rep]-uvo[rep], axis=1)
    S['S6'] = dict(scale=float(scale),
                   target_window_px=float(np.median(duv_win)),
                   target_original_px=float(np.median(duv_org)))
    chk('S6_scale', abs(np.median(duv_win)*(CROP/IMG) - np.median(duv_org)) < 1e-3,
        f"scale consistent: window {np.median(duv_win):.3f} x {CROP/IMG:.3f} == orig {np.median(duv_org):.3f}")
    im_in = np.asarray(Image.fromarray(sub).resize((IMG, IMG), Image.BILINEAR))
    fig, a = plt.subplots(figsize=(9,9), dpi=120); a.imshow(im_in, extent=[0,IMG,IMG,0])
    a.scatter(gt_loc[:,0], gt_loc[:,1], s=30, marker='+', c='lime', lw=1.2)
    a.scatter(off_loc[:,0], off_loc[:,1], s=30, facecolors='none', edgecolors='red', lw=1.0)
    a.set_title(f"S6 network input {IMG}x{IMG} (scale {scale:.2f}) target p50 {np.median(duv_win):.2f} win px")
    a.set_xticks([]); a.set_yticks([]); plt.tight_layout()
    plt.savefig(f"{OUT}/s{si}_S6_input.png", bbox_inches='tight'); plt.close()

    # ---------------- S7 6-DOF recovery from the GT residual ----------------
    Pt = torch.from_numpy(Pc[rep])[None].to(dev)
    Dt = torch.from_numpy((uvg[rep]-uvo[rep]))[None].to(dev)
    Kt = torch.from_numpy(K)[None].to(dev)
    vt = (Pt[..., 2] > 0.5)
    prior = torch.tensor([1/9.]*3+[1/0.09]*3, dtype=torch.float64, device=dev)
    d, H = solve_pose(Pt, -Dt, _identity_W(1, Pt.shape[1], torch.float64, dev), Kt,
                      valid=vt, n_iter=15, damping=1e-3, prior_diag=prior)
    uv0 = project_pinhole(Pt, Kt)
    uv1 = project_pinhole(_apply_extrinsic(Pt, d[:, :3], d[:, 3:6]), Kt)
    fit = float(((uv0-uv1)-Dt).norm(dim=-1)[vt].median())
    # the recovered delta expressed the same way as the injection:
    #   camera-frame delta R = R_off^T R_gt = Rd^T  -> compare axis-angle magnitude & the euler
    om = d[0, :3].cpu().numpy(); tv = d[0, 3:6].cpu().numpy()
    R_hat = Rotation.from_rotvec(om*np.pi/180.0).as_matrix()
    # Ground-truth camera-frame delta, taken straight from the point transform
    # (no euler-convention guessing): P_off = R_true P_gt + t_true.
    #   P_gt  = (X - cp)     @ R_gt
    #   P_off = (X - cp_off) @ R_off,  R_off = R_gt @ Rd,  cp_off = cp + t
    # => R_true = Rd^T,  t_true = -Rd^T R_gt^T t
    R_true = Rd.T
    t_true = -(Rd.T @ (Rgt.T @ t))
    dR = R_hat @ R_true.T
    ang_err = float(np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))))
    t_err   = float(np.linalg.norm(tv - t_true))
    # independent numerical check that R_true/t_true really是 the transform
    S_gtalg = float(np.median(np.linalg.norm((Pc[rep] @ R_true.T) + t_true - Po[rep], axis=1)))
    S_algebra = float(np.median(np.linalg.norm((Pc[rep] @ R_hat.T) + tv - Po[rep], axis=1)))
    S['S7'] = dict(fit_residual_px=fit,
                   recovered_omega_deg=[float(x) for x in om], recovered_t_m=[float(x) for x in tv],
                   injected_ypr_deg=[float(x) for x in ypr], injected_t_m=[float(x) for x in t],
                   rot_magnitude_recovered_deg=float(np.linalg.norm(om)),
                   rot_magnitude_injected_deg=float(np.linalg.norm(ypr)),
                   rot_angle_error_deg=ang_err, t_error_m=t_err,
                   gt_transform_check_m=S_gtalg,
                   t_magnitude_recovered_m=float(np.linalg.norm(tv)),
                   t_magnitude_injected_m=float(np.linalg.norm(t)),
                   point_transfer_err_m=S_algebra)
    chk('S7_fit',  fit < 0.05, f"residual after one rigid pose: {fit:.4f} px")
    chk('S7_rot',  ang_err < 0.02, f"rotation error vs injected: {ang_err:.4f} deg")
    chk('S7_xfer', S_algebra < 1e-3, f"recovered delta maps GT->perturbed: {S_algebra:.6f} m")
    chk('S7_gtalg', S_gtalg < 1e-6, f"analytic (R_true,t_true) maps GT->perturbed: {S_gtalg:.9f} m")
    chk('S7_t',   t_err < 1e-3, f"translation error vs injected: {t_err:.6f} m")
    REPORT['samples'].append(S)
    print(f"[{si}] {S['scene']} f{S['frame']}  in-crop {len(sel)} reps {len(rep)}  "
          f"shift p50 {np.median(shift):.2f}px  fit {fit:.4f}px  rot_err {ang_err:.4f}deg")

json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)
print(f"\n--- checks ---")
for k, v in REPORT['checks'].items():
    n_ok = sum(1 for x in v if x['ok'])
    print(f"  {k:16s} {n_ok}/{len(v)} pass   e.g. {v[0]['msg']}")
print(f"\nFAILED: {len(FAIL)}")
for f in FAIL[:10]: print("  -", f)
print(f"\ndebug output -> {OUT}")

# ============================ S8 (needs CKPT) ============================
if CKPT and os.path.exists(CKPT):
    print("\nS8 trained model")
    from models.model_depth import CalibNetDepth
    from datasets.pandaset_full import collate_full
    from torch.utils.data import DataLoader
    m = CalibNetDepth(img_size=IMG, in_channels=3, n_layers=4, use_convnext=True,
                      use_frustum=True, deform_mode=os.environ.get('DEFORM','ml'),
                      use_info_head=os.environ.get('INFO','1')=='1').to(dev)
    sdm = torch.load(CKPT, map_location='cpu'); sdm = sdm.get('model', sdm)
    m.load_state_dict(sdm, strict=False); m.eval()
    dsv = PandaSetCalibDatasetFull(CACHE, split='val', img_size=IMG, min_crop_px=CROP,
                                   max_crop_px=CROP, oversample=1,
                                   max_rot_deg=ROT, max_offset_m=TM)
    dlv = DataLoader(dsv, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_full)
    prior = torch.tensor([1/9.]*3+[1/0.09]*3, dtype=torch.float64, device=dev)
    SAFE  = torch.tensor([0.,0.,10.], dtype=torch.float64, device=dev)
    BA_,ER_,ZM_,RB,RE,TB,TE = [],[],[],[],[],[],[]
    with torch.no_grad():
        for bi,b in enumerate(dlv):
            if bi>=12: break
            imgs,tu,du,pad,vfp,pc,duv,Kb,cs = [x.to(dev) for x in b]
            out = m(imgs.float().div(255.), du[...,:3], key_padding_mask=pad, vfp=vfp)
            p, Wm = out if isinstance(out, tuple) else (out, None)
            p = p.float(); B,N,_ = p.shape
            s2o = (cs/float(IMG)).view(B,1)
            gt = tu[...,:2]-du[...,:2]; v = (~pad)&(pc[...,2]>0.5)
            BA_.append((gt.norm(dim=-1)*s2o)[v]); ER_.append(((p[...,:2]-gt).norm(dim=-1)*s2o)[v])
            sx=p[...,2].exp(); sy=p[...,3].exp(); rho=torch.tanh(p[...,4])*0.99
            e=(p[...,:2]-gt); dx=e[...,0]/sx; dy=e[...,1]/sy
            ZM_.append(((dx*dx+dy*dy-2*rho*dx*dy)/(1-rho*rho))[v])
            mu = (p[...,:2]*s2o.unsqueeze(-1)).double()
            Wq = (Wm.float()/(s2o*s2o).view(B,1,1,1)).double() if Wm is not None \
                 else _identity_W(B,N,torch.float64,dev)
            P = torch.where(v.unsqueeze(-1), pc.double(), SAFE); Kd = Kb.double(); D = duv.double()
            dg,_ = solve_pose(P,-D,_identity_W(B,N,torch.float64,dev),Kd,valid=v,n_iter=12,damping=1e-3,prior_diag=prior)
            dp,_ = solve_pose(P,-mu,Wq,Kd,valid=v,n_iter=12,damping=1e-3,prior_diag=prior)
            ok = torch.isfinite(dp).all(-1)&torch.isfinite(dg).all(-1)&(v.sum(-1)>=20)
            if ok.any():
                RB.append(dg[ok,:3].norm(dim=-1)); RE.append((dp[ok,:3]-dg[ok,:3]).norm(dim=-1))
                TB.append(dg[ok,3:6].norm(dim=-1)); TE.append((dp[ok,3:6]-dg[ok,3:6]).norm(dim=-1))
    BA_,ER_,ZM_,RB,RE,TB,TE = [torch.cat(x).cpu().numpy() for x in (BA_,ER_,ZM_,RB,RE,TB,TE)]
    print(f"  residual : base {np.median(BA_):.3f} -> {np.median(ER_):.3f} px (orig)  ratio {np.median(ER_)/np.median(BA_):.3f}")
    print(f"  pose rot : base {np.median(RB):.4f} -> err {np.median(RE):.4f} deg  ratio {np.median(RE)/np.median(RB):.3f}")
    print(f"  pose t   : base {np.median(TB):.4f} -> err {np.median(TE):.4f} m    ratio {np.median(TE)/np.median(TB):.3f}")
    print(f"  sigma    : Mahalanobis^2 mean {ZM_.mean():.3f} (calibrated 2.0)")
    chk('S8_residual', np.median(ER_) < 0.5*np.median(BA_), f"{np.median(ER_):.3f} vs base {np.median(BA_):.3f} px")
    chk('S8_rot',      np.median(RE)  < 0.5*np.median(RB),  f"{np.median(RE):.4f} vs base {np.median(RB):.4f} deg")
    chk('S8_sigma',    1.0 < ZM_.mean() < 4.0,              f"Mahalanobis^2 {ZM_.mean():.3f}")
    REPORT['S8'] = dict(base_px=float(np.median(BA_)), err_px=float(np.median(ER_)),
                        rot_base_deg=float(np.median(RB)), rot_err_deg=float(np.median(RE)),
                        t_base_m=float(np.median(TB)), t_err_m=float(np.median(TE)),
                        maha2=float(ZM_.mean()))
    json.dump(REPORT, open(f"{OUT}/report.json",'w'), indent=2)
    print(f"\nFAILED: {len(FAIL)}")
    for f in FAIL[:10]: print("  -", f)


# ---------------------------------------------------------------- S9
def S9_shared_delta(NW=6):
    """One frame, NW windows, ONE shared delta -> recover it from the SHIPPED
    solver fields (pts_cam_orig / duv_orig / K_orig) that _ba_pose_loss consumes.

    This is the guard for the BA export. With pts_cam_orig left in WORLD (as it
    shipped until 2026-08-28) 4/6 windows return nan from cholesky_ex and the
    surviving two return the injected magnitude itself, i.e. nothing recovered
    -- and the Z>0.5 guard still passed 54% of points, so it never crashed.
    """
    import datasets.pandaset_full as _PF
    d = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                 min_crop_px=CROP, max_crop_px=CROP, oversample=NW,
                                 max_rot_deg=ROT, max_offset_m=TM)
    rng = np.random.default_rng(0)
    YPR = (rng.random(3)*2-1)*ROT; TD = (rng.random(3)*2-1)*TM
    FIX = [(TD/TM+1)/2, (YPR/ROT+1)/2]          # inverse of (rand(3)*2-1)*max
    _orig = np.random.rand; _k = {'i': 0}
    def _fake(*a):
        if a == (3,):
            v = FIX[_k['i'] % 2]; _k['i'] += 1; return v.copy()
        return _orig(*a)
    np.random.rand = _fake
    try:
        samples = d[0]
    finally:
        np.random.rand = _orig
    inst = d._load_inst(0)
    Rgt = np.asarray(inst['R_gt'].numpy(), dtype=np.float64)
    Rd = Rotation.from_euler('zyx', YPR, degrees=True).as_matrix()
    R_true = Rd.T; t_true = -(Rd.T @ TD)   # camera-frame perturbation       # GT-cam-frame delta
    prior = torch.tensor([1/9.]*3+[1/0.09]*3, dtype=torch.float64, device=dev)
    print(f"\nS9  {inst.get('scene')} f{inst.get('frame')}  {len(samples)} windows, "
          f"shared delta |ypr| {np.linalg.norm(YPR):.4f} deg  |t| {np.linalg.norm(TD):.4f} m")
    print(f"{'win':>3s} {'npts':>5s} {'Z>0.5':>7s} {'fit px':>9s} {'rot err deg':>12s} {'t err m':>10s}")
    Hs = []; bs = []; rows = []
    for i, sm in enumerate(samples):
        P = sm[7].double().to(dev)[None]; D = sm[8].double().to(dev)[None]
        K = sm[9].double().to(dev)[None]
        v = P[..., 2] > 0.5
        dd, H = solve_pose(P, -D, _identity_W(1, P.shape[1], torch.float64, dev), K,
                           valid=v, n_iter=15, damping=1e-3, prior_diag=prior)
        uv0 = project_pinhole(P, K)
        uv1 = project_pinhole(_apply_extrinsic(P, dd[:, :3], dd[:, 3:6]), K)
        fit = float(((uv0-uv1)-D).norm(dim=-1)[v].median())
        om = dd[0, :3].cpu().numpy(); tv = dd[0, 3:6].cpu().numpy()
        Rh = Rotation.from_rotvec(om*np.pi/180.0).as_matrix()
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rh@R_true.T)-1)/2, -1, 1))))
        terr = float(np.linalg.norm(tv - t_true))
        zf = float(v.float().mean())
        print(f"{i:3d} {P.shape[1]:5d} {100*zf:6.1f}% {fit:9.4f} {ang:12.4f} {terr:10.4f}")
        rows.append(dict(npts=int(P.shape[1]), z_frac=zf, fit_px=fit,
                         rot_err_deg=ang, t_err_m=terr))
        Hs.append(H[0]); bs.append(H[0] @ dd[0])
    Hf = torch.stack(Hs).sum(0); bf = torch.stack(bs).sum(0)
    df = torch.linalg.solve(Hf, bf).cpu().numpy()
    Rh = Rotation.from_rotvec(df[:3]*np.pi/180.0).as_matrix()
    fang = float(np.degrees(np.arccos(np.clip((np.trace(Rh@R_true.T)-1)/2, -1, 1))))
    fterr = float(np.linalg.norm(df[3:] - t_true))
    print(f"FUSED (info-form, {len(samples)} windows): rot err {fang:.4f} deg  t err {fterr:.4f} m")
    zmin = min(r['z_frac'] for r in rows)
    chk('S9_zguard', zmin > 0.99, f"min Z>0.5 fraction {100*zmin:.1f}% (world coords give ~54%)")
    chk('S9_rot', max(r['rot_err_deg'] for r in rows) < 0.01,
        f"worst per-window rot err {max(r['rot_err_deg'] for r in rows):.4f} deg")
    chk('S9_t', max(r['t_err_m'] for r in rows) < 0.005,
        f"worst per-window t err {max(r['t_err_m'] for r in rows):.4f} m")
    chk('S9_fused', fang < 0.01 and fterr < 0.005,
        f"fused rot {fang:.4f} deg / t {fterr:.4f} m")
    REPORT['S9'] = dict(injected_rot_deg=float(np.linalg.norm(YPR)),
                        injected_t_m=float(np.linalg.norm(TD)),
                        windows=rows, fused_rot_deg=fang, fused_t_m=fterr)


S9_shared_delta()
json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)
print(f"\nFAILED: {len(FAIL)}")
for f in FAIL[:10]: print("  -", f)

# ---------------------------------------------------------------- S10
def S10_ba_fuses_windows():
    """Does _ba_pose_loss actually fuse the G windows it is told to?

    The BA loss takes `group=oversample` and is supposed to concatenate a
    frame's G windows into ONE pose problem. A branch meant for mixed
    grid/random batches used to fire whenever ANY frame was non-grid, which with
    grid_frac=0.0 sent EVERY frame to the solve-alone path -- so a run
    configured for 28-window fusion silently trained on single windows.

    Check it by counting the points the solver sees: fused must be G x the
    per-window count.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets.pandaset_full import collate_full
    import datasets.train_cnd2_ddp as T
    from scripts.ba import gn_pose
    G = 4
    d = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                 min_crop_px=CROP, max_crop_px=CROP, oversample=G,
                                 max_rot_deg=ROT, max_offset_m=TM, share_pert=True)
    b = next(iter(DataLoader(d, batch_size=2, shuffle=False, num_workers=0,
                             collate_fn=collate_full)))
    B, N = b[2].shape[:2]
    per_pt = torch.zeros(B, N, 5)
    seen = []
    orig = gn_pose.solve_pose
    def spy(pts, duv, W, K, **kw):
        seen.append(int(pts.shape[1]))
        return orig(pts, duv, W, K, **kw)
    gn_pose.solve_pose = spy; T.solve_pose = spy
    try:
        T._ba_pose_loss(per_pt, b[2], b[3], list(b), img_size=IMG, group=G,
                        is_grid=(b[15] if len(b) > 15 else None))
    finally:
        gn_pose.solve_pose = orig; T.solve_pose = orig
    got = max(seen) if seen else 0
    chk('S10_fuse', got == G * N,
        f"solver saw {got} points; {G} windows of {N} should give {G*N}")
    print(f"\nS10  group={G}, {N} pts/window -> solver saw {got} "
          f"({'fused' if got == G*N else 'NOT fused'})")


S10_ba_fuses_windows()
json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)

# ---------------------------------------------------------------- S11
def S11_variable_crop():
    """Scale invariance: crops of DIFFERENT sizes must fuse to the SAME pose.

    With 512 as the standard input, a 256-px crop is upscaled 2x and a 512-px one
    is 1:1. The network works in local crop pixels; the GN works in original
    camera pixels via s2o = cs / img_size. If that conversion is wrong, windows
    of different sizes disagree and the fused pose is garbage -- but each window
    alone still looks fine, so nothing else catches it.

    The truth is read from the sample's own pert_vec, so no monkeypatching of
    np.random is involved (crop-size sampling consumes the RNG too, which made
    an earlier version of this test compare against the wrong delta).
    """
    import torch
    from scripts.ba.gn_pose import solve_pose, _identity_W
    from scipy.spatial.transform import Rotation
    NW = 8
    d = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                 min_crop_px=256, max_crop_px=512, oversample=NW,
                                 max_rot_deg=ROT, max_offset_m=TM, share_pert=True)
    samples = d[0]
    pv = samples[0][6].numpy().astype(np.float64)      # tx,ty,tz + euler zyx
    same = all(np.allclose(w[6].numpy().astype(np.float64), pv) for w in samples)
    chk('S11_shared', same, 'every window of the frame carries the same delta')
    Rd = Rotation.from_euler('zyx', pv[3:6], degrees=True).as_matrix()
    R_true = Rd.T; t_true = -(Rd.T @ pv[:3])           # camera-frame perturbation
    prior = torch.tensor([1/9.]*3 + [1/0.09]*3, dtype=torch.float64, device=dev)
    print(f"\nS11  crops 256..512 -> input {IMG}px, {len(samples)} windows, "
          f"shared delta |ypr| {np.linalg.norm(pv[3:6]):.4f} deg "
          f"|t| {np.linalg.norm(pv[:3]):.4f} m")
    print(f"{'win':>3s} {'cs':>4s} {'scale':>6s} {'npts':>5s} {'rot err':>9s} {'t err':>9s}")
    Hs = []; bs = []; sizes = []
    for i, sm in enumerate(samples):
        P = sm[7].double().to(dev)[None]; D = sm[8].double().to(dev)[None]
        K = sm[9].double().to(dev)[None]; cs = float(sm[10])
        v = P[..., 2] > 0.5
        dd, H = solve_pose(P, -D, _identity_W(1, P.shape[1], torch.float64, dev), K,
                           valid=v, n_iter=15, damping=1e-3, prior_diag=prior)
        om = dd[0, :3].cpu().numpy(); tv = dd[0, 3:6].cpu().numpy()
        Rh = Rotation.from_rotvec(om * np.pi / 180.0).as_matrix()
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rh @ R_true.T) - 1) / 2, -1, 1))))
        terr = float(np.linalg.norm(tv - t_true))
        sizes.append(cs)
        print(f"{i:3d} {int(cs):4d} {IMG/cs:6.3f} {P.shape[1]:5d} {ang:9.4f} {terr:9.4f}")
        Hs.append(H[0]); bs.append(H[0] @ dd[0])
    Hf = torch.stack(Hs).sum(0); bf = torch.stack(bs).sum(0)
    df = torch.linalg.solve(Hf, bf).cpu().numpy()
    Rh = Rotation.from_rotvec(df[:3] * np.pi / 180.0).as_matrix()
    fang = float(np.degrees(np.arccos(np.clip((np.trace(Rh @ R_true.T) - 1) / 2, -1, 1))))
    fterr = float(np.linalg.norm(df[3:] - t_true))
    print(f"FUSED across sizes {int(min(sizes))}..{int(max(sizes))} px: "
          f"rot {fang:.4f} deg  t {fterr:.4f} m")
    chk('S11_sizes', max(sizes) - min(sizes) > 32,
        f"crop sizes actually varied: {int(min(sizes))}..{int(max(sizes))} px")
    chk('S11_cs_range', max(sizes) <= 512 + 1,
        f"crop size stayed within max_crop_px=512 (saw {int(max(sizes))})")
    chk('S11_fused', fang < 0.01 and fterr < 0.005,
        f"fused across crop sizes: rot {fang:.4f} deg / t {fterr:.4f} m")
    REPORT['S11'] = dict(sizes=sizes, fused_rot_deg=fang, fused_t_m=fterr)


S11_variable_crop()
json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)

# ---------------------------------------------------------------- S12
def S12_ba_needs_warmstart():
    """The BA pose loss needs a warm start; the test records how much.

    -0.5*logdet H rewards shrinking sigma, and with a random mu the GN solution
    is far from the target, so the quadratic term is enormous. Training
    512/scratch with the BA loss on blew up (tr_nll 3.1e14 at ep18, NaN at ep19)
    and nothing caught it.

    Measured on one batch (256 px, group=4):

        untrained                     loss 7.2e+04   max|grad| 1.8e+07
        head_ns200 (BA-free warm)     loss 1.3e+04   max|grad| 1.0e+06
        cam_fuse28 (trained with BA)  loss   -20.5   max|grad| 1.1e+03

    So: **never switch the BA loss on from scratch.** Warm-start without it
    first. This stage only asserts the loss/grad are finite (a NaN here means
    something is broken outright) and prints the magnitude so the warm-start
    requirement stays visible.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets.pandaset_full import collate_full
    import datasets.train_cnd2_ddp as T
    from models.calibnet2 import CalibNet2
    G = 4
    d = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                 min_crop_px=CROP, max_crop_px=CROP, oversample=G,
                                 max_rot_deg=ROT, max_offset_m=TM, share_pert=True)
    b = next(iter(DataLoader(d, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_full)))
    b = [x.to(dev) if torch.is_tensor(x) else x for x in b]
    m = CalibNet2(d=128, img_size=IMG, in_channels=3, use_intensity=True,
                  frustum_grid_n=int(round(IMG / 16)), n_iter=4, n_heads=4,
                  d_scalar=8, n_type1=40, kv_schedule=None,
                  fourier_head_n_freq=0, fourier_head_scale=10.0,
                  point_mlp_fourier_n_freq=0, use_info_head=True).to(dev)
    imgs, true_uvd, dist_uvd, pad, vfp, bu, bv = b[:7]
    out = m(imgs.float().div(255.0),
            torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1),
            dpose_R=None, vfp=vfp, bucket_uvd=bu, bucket_valid=bv,
            key_padding_mask=pad)
    per_pt = out[0]; W_head = out[1] if isinstance(out, tuple) and len(out) > 1 else None
    loss, diag = T._ba_pose_loss(per_pt, dist_uvd, pad, list(b), img_size=IMG,
                                 group=G, W_head=W_head,
                                 is_grid=(b[15] if len(b) > 15 else None))
    lv = float('nan') if loss is None else float(loss)
    fin = loss is not None and np.isfinite(lv)
    gmax = float('nan')
    if fin:
        loss.backward()
        gs = [p.grad.abs().max().item() for p in m.parameters() if p.grad is not None]
        gmax = max(gs) if gs else 0.0
    print(f"\nS12  UNTRAINED model: BA loss {lv:.4g}   max|grad| {gmax:.4g}")
    print(f"     (a trained one gives ~-20 and ~1e3 -- warm-start before enabling --ba-loss)")
    chk('S12_finite', fin and np.isfinite(gmax),
        f"BA loss and its gradient are finite (loss {lv:.4g}, grad {gmax:.4g})")
    chk('S12_frac_pd', diag.get('frac_pd', 0.0) > 0.5,
        f"most tiles stay positive-definite even untrained "
        f"(frac_pd {diag.get('frac_pd', float('nan')):.2f})")
    REPORT['S12'] = dict(loss=lv, grad_max=gmax, frac_pd=diag.get('frac_pd'))


S12_ba_needs_warmstart()
json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)

# ---------------------------------------------------------------- S13
def S13_training_perturbation_is_random():
    """Training must draw a FRESH delta every time a frame is used.

    The eval path needs a reproducible delta (so two checkpoints see identical
    data), so a seeded per-frame RNG was added. It also fired during training,
    which froze the perturbation at ONE delta per frame -- 363 for this cache,
    no matter how many epochs. cam_rnd400b was 2x worse than cam_rnd400 for
    exactly that reason (tr_mse 19.9 vs 9.9 at ep10) and nothing caught it.
    """
    kw = dict(split='train', img_size=IMG, min_crop_px=CROP, max_crop_px=CROP,
              oversample=4, max_rot_deg=ROT, max_offset_m=TM, share_pert=True)
    d = PandaSetCalibDatasetFull(CACHE, **kw)
    seen = []
    for t in range(4):
        np.random.seed(9100 + t)
        seen.append(tuple(np.round(d[0][0][6].numpy()[:6], 6)))
    n_uniq = len(set(seen))
    d2 = PandaSetCalibDatasetFull(CACHE, pert_seed=1234, **kw)
    fixed = []
    for t in range(3):
        np.random.seed(9200 + t)
        fixed.append(tuple(np.round(d2[0][0][6].numpy()[:6], 6)))
    n_fixed = len(set(fixed))
    print(f"\nS13  training: {n_uniq}/4 distinct deltas for the same frame   "
          f"eval(pert_seed=1234): {n_fixed}/3 distinct")
    chk('S13_train_random', n_uniq == 4,
        f"training draws a new delta each time ({n_uniq}/4 distinct)")
    chk('S13_eval_fixed', n_fixed == 1,
        f"pert_seed makes eval reproducible ({n_fixed}/3 distinct, want 1)")
    REPORT['S13'] = dict(train_distinct=n_uniq, eval_distinct=n_fixed)


S13_training_perturbation_is_random()


# ---------------------------------------------------------------- S14
def S14_ba_w_source():
    """The GN's per-point 2x2 information matrix has two possible sources and
    they are NOT interchangeable.

    InfoHead2x2's last layer is zero-weight / constant-bias at init, so its W
    is the SAME matrix for every point -- uniform weighting, no robustness.
    A checkpoint whose info_head never received gradient (anything trained
    before the head was routed into the GN, e.g. head_ns200) still sits at
    that init, so switching BA on with --ba-w-source infohead weights every
    point equally and the squared term is set by the worst mu. Measured:
    tr_nll 14651 at ep1 resuming head_ns200, tr_mse rising 10.69 -> 13.53,
    against cam_rnd400's 10.103 -> 9.169 -> 8.809 on the sigma path.

    This stage pins the init (so a silent change to it is caught) and pins
    that --ba-w-source exists with both values.
    """
    import argparse, inspect
    from models.model_depth import InfoHead2x2
    d = 64
    ih = InfoHead2x2(d)
    with torch.no_grad():
        W = ih(torch.randn(2, 50, d))                      # (B, N, 2, 2)
    Wf = W.reshape(-1, 4)
    spread = float((Wf - Wf[:1]).abs().max())
    diag0  = float(W[0, 0, 0, 0])
    offd   = float(W[0, 0, 0, 1].abs())
    print(f"\nS14  InfoHead2x2 at init: W = {diag0:.4f}*I  "
          f"(off-diag {offd:.2e}, spread over points {spread:.2e})")

    chk('S14_init_constant', spread < 1e-6,
        f'InfoHead W varies by {spread:.2e} across points at init; it must be '
        f'constant, otherwise this stage no longer describes the head')
    chk('S14_init_scale', 0.3 < diag0 < 0.8,
        f'InfoHead init W diag {diag0:.4f} outside [0.3, 0.8]; softplus(0)^2 '
        f'= 0.48 is what the BA-blowup analysis assumed')

    src = open('datasets/train_cnd2_ddp.py').read()
    has_flag = "--ba-w-source" in src
    has_sigma = "_ba_w_source" in src and "W_head = None" in src
    print(f"     trainer --ba-w-source present: {has_flag}   "
          f"sigma path wired: {has_sigma}")
    chk('S14_flag', has_flag,
        'train_cnd2_ddp.py has no --ba-w-source; the GN would be locked to '
        'one W source and cam_rnd400 (sigma) could not be reproduced')
    chk('S14_sigma_path', has_sigma,
        '--ba-w-source sigma does not actually drop W_head')

    REPORT['S14'] = dict(init_diag=diag0, spread=spread,
                         flag=has_flag, sigma_path=has_sigma)


S14_ba_w_source()


# ---------------------------------------------------------------- S15
def S15_pose_metrics_reported():
    """The training curve must carry the POSE error, not only the point error.

    tr_mse / va_mse are ||mu - gt|| in local crop px -- the per-point error.
    Every curve in ClearML up to 2026-08-29 was that, so a run could look fine
    (or broken) with nobody ever seeing the rot/t the calibration is for. On
    the ep100 BA-free checkpoint the point error was 4.96 px while the pose was
    0.093 deg / 0.022 m and chi2_reduced = 224 -- three numbers that move
    independently.

    This stage pins that _ba_pose_loss actually returns all of them, and that
    the trainer evaluates them on val regardless of --ba-loss.
    """
    import torch
    from torch.utils.data import DataLoader
    from datasets.pandaset_full import collate_full
    import datasets.train_cnd2_ddp as T
    from models.calibnet2 import CalibNet2
    G = 4
    d = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                 min_crop_px=CROP, max_crop_px=CROP, oversample=G,
                                 max_rot_deg=ROT, max_offset_m=TM, share_pert=True)
    b = next(iter(DataLoader(d, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_full)))
    b = [x.to(dev) if torch.is_tensor(x) else x for x in b]
    m = CalibNet2(d=128, img_size=IMG, in_channels=3, use_intensity=True,
                  frustum_grid_n=int(round(IMG / 16)), n_iter=4, n_heads=4,
                  d_scalar=8, n_type1=40, kv_schedule=None,
                  fourier_head_n_freq=0, fourier_head_scale=10.0,
                  point_mlp_fourier_n_freq=0, use_info_head=True).to(dev)
    imgs, true_uvd, dist_uvd, pad, vfp, bu, bv = b[:7]
    with torch.no_grad():
        out = m(imgs.float().div(255.0),
                torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1),
                dpose_R=None, vfp=vfp, bucket_uvd=bu, bucket_valid=bv,
                key_padding_mask=pad)
        _, diag = T._ba_pose_loss(out[0], dist_uvd, pad, list(b), img_size=IMG,
                                  group=G, W_head=None, loss_type='nll',
                                  is_grid=(b[15] if len(b) > 15 else None))
    need = ('rot_err', 't_err', 'quad', 'logdet', 'frac_pd', 'sigma_px')
    have = [k for k in need if diag.get(k) is not None]
    print(f"\nS15  pose diag keys present: {have}")
    chk('S15_diag_keys', len(have) == len(need),
        f'_ba_pose_loss returned {have}, needs all of {list(need)}; a missing '
        f'key silently drops that curve from ClearML')

    src = open('datasets/train_cnd2_ddp.py').read()
    ev = "(not train) and getattr(accel, '_pose_eval', None) is not None" in src
    cm = ("pose/rot_err_deg" in src and "pose/chi2_reduced" in src
          and "pose/sigma_px" in src)
    print(f"     eval hook: {ev}   ClearML pose scalars: {cm}")
    chk('S15_eval_hook', ev,
        'the trainer does not evaluate pose on val independently of --ba-loss')
    chk('S15_clearml', cm,
        'pose/rot_err_deg and pose/chi2_reduced are not reported to ClearML')
    REPORT['S15'] = dict(keys=have, eval_hook=ev, clearml=cm)


S15_pose_metrics_reported()


# ---------------------------------------------------------------- S16
def S16_scene_split_is_disjoint():
    """--scene-split must give train and val that share NO scene.

    build_nuscenes_v3 splits by scene already (rng.shuffle(scenes_sorted);
    n_val = len(scenes_sorted) * val_frac) and stores it in meta.pt. The
    trainer's default path concatenated both sides and reshuffled at FRAME
    level, so a val frame's temporal neighbours sat in train -- consecutive
    frames of the same drive, seconds apart. Every pose number measured
    before 2026-08-30 (mini rot 0.0300 deg, blob01 rot 0.0285 deg) was on
    that split and is NOT a generalisation figure.

    This stage reads the scene of every frame on both sides and asserts the
    two sets do not intersect.
    """
    tr = PandaSetCalibDatasetFull(CACHE, split='train', img_size=IMG,
                                  min_crop_px=CROP, max_crop_px=CROP, oversample=1)
    va = PandaSetCalibDatasetFull(CACHE, split='val', img_size=IMG,
                                  min_crop_px=CROP, max_crop_px=CROP, oversample=1)

    # build_nuscenes_v3 hands scene i the gid block [i*4000, (i+1)*4000)
    # (gid_stride = 4000), and every instance file is named f'{gid:08d}.pt'.
    # So gid // 4000 IS the scene index -- no need to open the records.
    GID_STRIDE = 4000

    def scenes_of(ds):
        out = set()
        for fn in getattr(ds, 'fnames', []):
            m = re.match(r'(\d{8})\.pt$', str(fn))
            if m:
                out.add(int(m.group(1)) // GID_STRIDE)
        return out

    str_s, va_s = scenes_of(tr), scenes_of(va)
    both = str_s & va_s
    print(f"\nS16  cache split: train {len(tr)} frames / {len(str_s)} scenes, "
          f"val {len(va)} frames / {len(va_s)} scenes")
    print(f"     val scene indices: {sorted(va_s)}")
    print(f"     shared scenes: {len(both)}  {sorted(both)[:5]}")

    chk('S16_scenes_readable', bool(str_s) and bool(va_s),
        f'could not read scene ids off the cache (train {len(str_s)}, '
        f'val {len(va_s)}); S16 cannot verify disjointness')
    if str_s and va_s:
        chk('S16_disjoint', not both,
            f'{len(both)} scene(s) appear in BOTH the cache train and val '
            f'split: {sorted(both)[:5]} — --scene-split would leak')

    src = open('datasets/train_cnd2_ddp.py').read()
    has = "--scene-split" in src and "args.scene_split" in src
    warns = "NOT scene-disjoint" in src
    print(f"     trainer --scene-split: {has}   default path warns: {warns}")
    chk('S16_flag', has,
        'train_cnd2_ddp.py has no --scene-split, so the cache scene split '
        'cannot be used and every val number stays frame-shuffled')
    chk('S16_warns', warns,
        'the frame-level path does not say it is NOT scene-disjoint; that '
        'silence is how the leak went unnoticed')
    REPORT['S16'] = dict(train_scenes=len(str_s), val_scenes=len(va_s),
                         shared=len(both), flag=has)


S16_scene_split_is_disjoint()
json.dump(REPORT, open(f"{OUT}/report.json", 'w'), indent=2)

# ---------------------------------------------------------------- exit code
_n = len(FAIL)
print(f"\n{'='*62}")
print(f"SYSTEM TEST: {'PASS' if _n == 0 else 'FAIL'}   ({_n} failing checks)")
print(f"{'='*62}")
sys.exit(1 if _n else 0)
