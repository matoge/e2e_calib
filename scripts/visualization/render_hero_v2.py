"""Hero image v2 — per-point non-uniform shifts from REAL __getitem__ drift.

Uses the dataset's native 6-DOF perturbation range (≤0.2 m translation,
≤0.5° rotation) — identical to training distribution. Each panel is a
different val sample with its own random drift; arrow magnitudes and
directions vary per point because the underlying perturbation is 3-D
(rotation spreads shifts angularly, depth-varying translation spreads
them radially).

Selects 4 diverse samples where per-point shift variance is visually
obvious (so the "not uniform" story reads at a glance).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch, numpy as np, random as _r
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

from datasets.pandaset import PandaSetCalibDataset
from models.model_depth import CalibNetDepth

BG      = "#f6f4ed"
CARD    = "#ffffff"
ACCENT  = "#c13c14"
ACCENT2 = "#174734"
LINE    = "#d9d6cd"

CACHE   = '/tmp/pandaset_cache.pt'
EXP_DIR = Path('experiments/ps_v9_objsplit')
OUT     = EXP_DIR / 'vis' / 'hero.png'

MAX_OFFSET_M = 0.20
MAX_ROT_DEG  = 0.5


def reproject(inst, ypr_deg, t_delta, S):
    """Same pipeline as PandaSetCalibDataset.__getitem__, but with explicit
    ypr/t instead of random ones."""
    pts       = inst['pts'].numpy()
    cp        = inst['cam_pos'].numpy()
    R_gt      = inst['R_gt'].numpy()
    T_gt      = inst['T_gt'].numpy()
    K         = inst['K_full'].numpy()
    u0        = inst['u0']
    v0        = inst['v0']
    crop_size = inst['crop_size']

    R_off = R_gt @ Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()
    cp_off = cp + np.array(t_delta, dtype=np.float32)
    T_off = np.eye(4, dtype=np.float32)
    T_off[:3, :3] = R_off.T
    T_off[:3,  3] = -(R_off.T @ cp_off)

    pts_cam_off = T_off[:3, :3] @ pts.T + T_off[:3, 3:]
    z_off = pts_cam_off[2]
    uv_off = ((K @ pts_cam_off)[:2] / z_off).T

    in_crop = ((uv_off[:,0] >= u0) & (uv_off[:,0] < u0+crop_size) &
               (uv_off[:,1] >= v0) & (uv_off[:,1] < v0+crop_size) &
               (z_off > 0.5))
    if in_crop.sum() < 40:
        return None

    scale = S / crop_size
    uv_d_crop = np.stack([(uv_off[in_crop,0]-u0) * scale,
                           (uv_off[in_crop,1]-v0) * scale], axis=1)

    grid, cell = 16, float(S) / 16
    sel = []
    for gi in range(grid):
        for gj in range(grid):
            d2 = ((uv_d_crop[:,0]-(gj+.5)*cell)**2 +
                  (uv_d_crop[:,1]-(gi+.5)*cell)**2)
            sel.append(int(d2.argmin()))
    sel = sorted(set(sel))

    idx_in  = np.where(in_crop)[0][sel]
    pts_sel = pts[idx_in]

    pts_cam_gt = T_gt[:3, :3] @ pts_sel.T + T_gt[:3, 3:]
    uv_gt = ((K @ pts_cam_gt)[:2] / pts_cam_gt[2]).T
    uv_gt_c  = np.stack([(uv_gt[:,0]-u0) * scale,
                          (uv_gt[:,1]-v0) * scale], axis=1)
    uv_off_c = uv_d_crop[sel]

    dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)

    if 'obj_pos' not in inst:
        return None
    obj_pos = inst['obj_pos'].numpy()
    obj_dims = inst['obj_dims'].numpy()
    obj_yaw = inst['obj_yaw']
    c_y, s_y = np.cos(obj_yaw), np.sin(obj_yaw)
    R_obj = np.array([[c_y, s_y, 0],
                       [-s_y, c_y, 0],
                       [0,    0,   1]], dtype=np.float32)
    pts_local = (R_obj @ (pts_sel - obj_pos).T).T
    half = obj_dims / 2.0
    is_obj = ((np.abs(pts_local[:,0]) <= half[0]) &
              (np.abs(pts_local[:,1]) <= half[1]) &
              (np.abs(pts_local[:,2]) <= half[2])).astype(np.float32)
    return dict(uv_gt=uv_gt_c.astype(np.float32),
                uv_dist=uv_off_c.astype(np.float32),
                dist_m=dist_m, is_obj=is_obj)


def draw_perturbation(rng):
    ypr = ((rng.random(3) * 2 - 1) * MAX_ROT_DEG).tolist()
    t   = ((rng.random(3) * 2 - 1) * MAX_OFFSET_M).tolist()
    return ypr, t


def scan_candidates(ds, n_candidates=400, seed=1,
                    min_mean_shift=6.0, max_obj_depth=35.0,
                    min_obj_span_frac=0.28):
    """Find val samples where (a) the object is CLOSE (mean obj depth <
    max_obj_depth m) so it fills the crop and reads clearly in the image,
    (b) the object occupies at least min_obj_span_frac of the crop side,
    (c) the __getitem__-style perturbation yields a per-point mean shift
    >= min_mean_shift px so the correction is obvious."""
    S = 64
    rng = np.random.default_rng(seed)
    idxs = list(range(len(ds))); _r.Random(seed).shuffle(idxs)
    found = []
    for idx in idxs:
        inst = ds.instances[idx]
        if 'obj_pos' not in inst: continue
        # quick pre-filter: reject far objects by instance-level obj_pos
        obj_dist = float(np.linalg.norm(inst['obj_pos'].numpy() -
                                        inst['cam_pos'].numpy()))
        if obj_dist > max_obj_depth: continue
        # allow several re-draws per instance to hit the ≥ min_mean_shift tail
        for _k in range(8):
            ypr, t = draw_perturbation(rng)
            out = reproject(inst, ypr, t, S)
            if out is None: continue
            is_obj = out['is_obj'].astype(bool)
            if is_obj.sum() < 14: continue
            tu = out['uv_gt']; obj = tu[is_obj]
            w = obj[:,0].max() - obj[:,0].min()
            h = obj[:,1].max() - obj[:,1].min()
            cx, cy = obj[:,0].mean(), obj[:,1].mean()
            if max(w, h) < min_obj_span_frac * S: continue
            if max(w, h) > 0.80 * S: continue
            if min(w, h) < 0.18 * S: continue
            if abs(cx - S/2) > 0.22*S or abs(cy - S/2) > 0.22*S: continue
            shift = out['uv_dist'] - out['uv_gt']
            mag = np.linalg.norm(shift, axis=1)
            if mag.mean() < min_mean_shift: continue
            cv = float(mag.std() / max(mag.mean(), 1e-6))
            # sort key: bigger obj × closer × reasonable cv
            size_score = max(w, h) / S
            key = size_score * 10 + cv + (max_obj_depth - obj_dist) * 0.05
            found.append((key, idx, ypr, t, float(mag.mean()), obj_dist, size_score))
            break
        if len(found) >= n_candidates: break
    found.sort(key=lambda x: -x[0])
    return found


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = EXP_DIR / 'config.py'
    ns = {}; exec(cfg_path.read_text(), ns); cfg = ns['CFG']
    model = CalibNetDepth(img_size=cfg['img_size'], in_channels=cfg['in_channels'],
                          n_layers=cfg['n_layers'], self_first=False,
                          use_convnext=cfg.get('use_convnext', False),
                          use_frustum=cfg.get('use_frustum', False)).to(dev)
    model.load_state_dict(torch.load(EXP_DIR / 'best_model.pt',
                                     map_location=dev, weights_only=True))
    model.eval()

    ds = PandaSetCalibDataset(CACHE, split='val',
                              max_offset_m=MAX_OFFSET_M,
                              max_rot_deg=MAX_ROT_DEG)
    candidates = scan_candidates(ds, n_candidates=800, seed=7)
    print(f"scanned → {len(candidates)} candidates (close + large obj + ≥8px shift)")

    # Diversify: for each candidate compute dominant shift direction + depth.
    # Pick 4 whose (angle_bucket, depth_bucket) tuples are all distinct.
    def _aug(c):
        _, idx, ypr, t, mmag, obj_dist, size = c
        inst = ds.instances[idx]
        out = reproject(inst, ypr, t, 64)
        shift = out['uv_dist'] - out['uv_gt']
        dx, dy = shift.mean(0)
        ang = float(np.degrees(np.arctan2(dy, dx)))  # −180..+180
        # 8 directional buckets (45 deg each)
        ang_bucket = int(((ang + 180 + 22.5) % 360) // 45)
        # 4 depth buckets
        d_bucket = min(3, int(obj_dist // 10))
        return ang_bucket, d_bucket, c
    aug = [_aug(c) for c in candidates]

    picks = []
    used_idx  = set()
    used_ang  = set()
    used_dep  = set()
    # Pass 1: strict — distinct direction AND distinct depth
    for ab, db, c in aug:
        _, idx, *_ = c
        if idx in used_idx: continue
        if ab in used_ang or db in used_dep: continue
        picks.append(c); used_idx.add(idx); used_ang.add(ab); used_dep.add(db)
        if len(picks) == 4: break
    # Pass 2: relax — only distinct direction
    if len(picks) < 4:
        for ab, db, c in aug:
            _, idx, *_ = c
            if idx in used_idx: continue
            if ab in used_ang: continue
            picks.append(c); used_idx.add(idx); used_ang.add(ab)
            if len(picks) == 4: break
    # Pass 3: fill remainder from best-scoring untouched
    for ab, db, c in aug:
        if len(picks) == 4: break
        _, idx, *_ = c
        if idx in used_idx: continue
        picks.append(c); used_idx.add(idx)

    picks = [(idx, ypr, t, key, mmag, obj_dist, size_score)
             for (key, idx, ypr, t, mmag, obj_dist, size_score) in picks]
    assert len(picks) == 4, f"only found {len(picks)} suitable samples"

    S = 64
    panels = []
    for idx, ypr, t, key, mmag, obj_dist, size_score in picks:
        inst = ds.instances[idx]
        out = reproject(inst, ypr, t, S)
        is_obj = out['is_obj'].astype(bool)
        dist_uv = out['uv_dist']; true_uv = out['uv_gt']
        synth = np.concatenate([dist_uv, out['dist_m'][:, None],
                                out['is_obj'][:, None]], axis=1).astype(np.float32)
        img_cache = inst.get('img_cache', inst.get('img_64'))
        if img_cache.shape[-1] != S:
            img = torch.nn.functional.interpolate(
                img_cache.float().unsqueeze(0), size=(S, S),
                mode='bilinear', align_corners=False).squeeze(0) / 255.0
        else:
            img = img_cache.float() / 255.0
        with torch.no_grad():
            pad = torch.zeros(1, synth.shape[0], dtype=torch.bool, device=dev)
            params = model(img.unsqueeze(0).to(dev),
                           torch.from_numpy(synth).unsqueeze(0).to(dev)[..., :3],
                           key_padding_mask=pad)[0].cpu().float().numpy()
        pred_uv = dist_uv + params[:, :2]

        eb = float(np.linalg.norm(dist_uv[is_obj] - true_uv[is_obj], axis=1).mean())
        ea = float(np.linalg.norm(pred_uv[is_obj] - true_uv[is_obj], axis=1).mean())
        print(f" idx={idx:5d}  dist={obj_dist:5.1f}m  size={size_score:.2f}  "
              f"shift μ={mmag:.2f}  obj_err {eb:.2f} → {ea:.2f} px")

        panels.append(dict(idx=idx, ypr=ypr, t=t,
                           dist_uv=dist_uv, true_uv=true_uv, pred_uv=pred_uv,
                           is_obj=is_obj, err_before=eb, err_after=ea,
                           img=img.permute(1,2,0).numpy()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), dpi=120)
    fig.patch.set_facecolor(BG)
    for ax, p in zip(axes.flatten(), panels):
        ax.set_facecolor(CARD)
        ax.imshow(p['img'], extent=[0, S, S, 0])

        u_d = p['dist_uv'][:, 0]; v_d = p['dist_uv'][:, 1]
        u = p['pred_uv'][:, 0] - u_d
        v = p['pred_uv'][:, 1] - v_d
        is_obj = p['is_obj']
        if (~is_obj).any():
            ax.quiver(u_d[~is_obj], v_d[~is_obj], u[~is_obj], v[~is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='#79c4ff', width=0.0045,
                      headwidth=3.5, headlength=4.5, alpha=0.9, zorder=3)
        if is_obj.any():
            ax.quiver(u_d[is_obj], v_d[is_obj], u[is_obj], v[is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='#ffb133', width=0.007,
                      headwidth=3.5, headlength=4.5, alpha=1.0, zorder=5)

        d_obj = p['dist_uv'][is_obj]; t_obj = p['true_uv'][is_obj]
        x0 = float(d_obj[:,0].min()); y0 = float(d_obj[:,1].min())
        x1 = float(d_obj[:,0].max()); y1 = float(d_obj[:,1].max())
        tx0 = float(t_obj[:,0].min()); ty0 = float(t_obj[:,1].min())
        tx1 = float(t_obj[:,0].max()); ty1 = float(t_obj[:,1].max())
        pad = 2.5
        ax.add_patch(plt.Rectangle((x0-pad, y0-pad),
                                   (x1-x0)+2*pad, (y1-y0)+2*pad,
                                   fill=False, ec=ACCENT, lw=2.2, zorder=6))
        ax.add_patch(plt.Rectangle((tx0-pad, ty0-pad),
                                   (tx1-tx0)+2*pad, (ty1-ty0)+2*pad,
                                   fill=False, ec='#2fcf7a', lw=2.2,
                                   linestyle='--', zorder=6))

        ypr = p['ypr']; t = p['t']
        desc = (f"ypr=({ypr[0]:+.2f},{ypr[1]:+.2f},{ypr[2]:+.2f})°  "
                f"t=({t[0]:+.2f},{t[1]:+.2f},{t[2]:+.2f})m")
        ax.text(S*0.023, S*0.96,
                f" {p['err_before']:.1f} px  →  {p['err_after']:.2f} px ",
                color='white', fontsize=12, family='monospace', fontweight='700',
                bbox=dict(facecolor=ACCENT, edgecolor='none', pad=5), zorder=8)
        ax.text(S*0.977, S*0.047,
                f" {desc} ",
                color='white', fontsize=9.5, family='monospace', fontweight='600',
                ha='right', va='top',
                bbox=dict(facecolor=ACCENT2, edgecolor='none', pad=5), zorder=8)

        ax.set_xlim(0, S); ax.set_ylim(S, 0)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=1.2)
    plt.savefig(OUT, facecolor=BG, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
