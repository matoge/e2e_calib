"""Inference results on val samples for all_v3_mc: show both obj and bg samples with predicted mean + sigma ellipses."""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dataset_nuscenes  import NuScenesCalibDataset
from dataset_pandaset  import PandaSetCalibDataset
from dataset_waymo     import WaymoCalibDataset
from model_depth import CalibNetDepth

BG      = "#f6f4ed"
CARD    = "#ffffff"
INK     = "#0f0f0e"
INK_DIM = "#6b6a63"
LINE    = "#d9d6cd"
ACCENT  = "#c13c14"
ACCENT2 = "#174734"
GT      = "#22aa44"
PRED    = "#2266ee"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = 'experiments/all_v3_mc/best_model.pt'

model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                       use_convnext=True, use_frustum=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()

ns = NuScenesCalibDataset('/mnt/nvme6t/e2e_calib_cache/nuscenes_mc_cache.pt', split='val')
ps = PandaSetCalibDataset('/mnt/nvme6t/e2e_calib_cache/pandaset_mc_cache.pt', split='val')
wm = WaymoCalibDataset  ('/mnt/nvme6t/e2e_calib_cache/waymo_mc_cache.pt',     split='val')

def sample(ds, name, want_obj, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(ds))[:5000]:
        idx = int(i)
        try:
            img, true_uvd, dist_uvd = ds[idx]
        except Exception:
            continue
        is_obj = (dist_uvd[:, 3] > 0.5).numpy()
        if want_obj and is_obj.sum() < 8: continue
        if (not want_obj) and is_obj.sum() > 0: continue
        out.append((name, idx, img, true_uvd, dist_uvd))
        if len(out) == n: break
    return out

picks = []
picks += sample(ns, 'NS obj', True, 2, 11)
picks += sample(ps, 'PS obj', True, 2, 22)
picks += sample(wm, 'WM obj', True, 2, 33)
picks += sample(ns, 'NS bg',  False, 1, 41)
picks += sample(ps, 'PS bg',  False, 1, 52)
picks += sample(wm, 'WM bg',  False, 1, 63)

rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(cols*4.6, rows*4.7), dpi=130)
fig.patch.set_facecolor(BG)

for ax, (name, idx, img, true_uvd, dist_uvd) in zip(axes.ravel(), picks):
    im = img.permute(1,2,0).numpy()
    ax.set_facecolor(CARD)
    ax.imshow(im)

    with torch.no_grad():
        im_b   = img.unsqueeze(0).to(DEVICE)
        d_uvd  = dist_uvd.unsqueeze(0).to(DEVICE)
        params = model(im_b, d_uvd[..., :3]).squeeze(0).cpu().numpy()

    du = dist_uvd[:, :2].numpy()
    tu = true_uvd[:, :2].numpy()
    is_obj = (dist_uvd[:, 3] > 0.5).numpy()

    pred_offset = params[:, :2]
    log_sx = params[:, 2]; log_sy = params[:, 3]; rho = params[:, 4]
    sx = np.exp(log_sx); sy = np.exp(log_sy)

    pred_uv = du + pred_offset
    gt_offset = tu - du

    for i in range(len(du)):
        ax.plot([du[i,0], pred_uv[i,0]], [du[i,1], pred_uv[i,1]],
                c=PRED, lw=0.5, alpha=0.55)

    for i in range(len(du)):
        cov = np.array([[sx[i]**2,         sx[i]*sy[i]*rho[i]],
                        [sx[i]*sy[i]*rho[i], sy[i]**2]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
        w, h = 2 * np.sqrt(eigvals)
        ell = Ellipse(xy=pred_uv[i], width=w, height=h, angle=angle,
                      facecolor='none', edgecolor=PRED, lw=0.5, alpha=0.45)
        ax.add_patch(ell)

    ax.scatter(du[:,0], du[:,1],     c=ACCENT, s=14, alpha=0.9, edgecolors='white', lw=0.4)
    ax.scatter(tu[:,0], tu[:,1],     c=GT,     s=14, alpha=0.9, edgecolors='white', lw=0.4)
    ax.scatter(pred_uv[:,0], pred_uv[:,1], c=PRED, s=10, alpha=0.95, marker='x', lw=1.2)

    if is_obj.any():
        ax.scatter(tu[is_obj,0], tu[is_obj,1], facecolors='none',
                    edgecolors='yellow', s=90, lw=1.3)

    d_px = np.linalg.norm(pred_uv - tu, axis=1).mean()
    sig_mean = 0.5 * (sx.mean() + sy.mean())
    ax.set_title(f'{name}  ·  N={len(du)}  ·  mean err {d_px:.1f} px  ·  ⟨σ⟩ {sig_mean:.1f}',
                  color=INK, fontsize=10, fontweight='600', loc='left', pad=6)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(LINE)

import matplotlib.lines as mlines
handles = [
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor=ACCENT, markersize=9, label='input (perturbed)'),
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor=GT,     markersize=9, label='ground truth'),
    mlines.Line2D([], [], marker='x', color=PRED, markersize=9, lw=0,    label='predicted μ'),
    mlines.Line2D([], [], color=PRED, lw=1, alpha=0.6, label='1σ ellipse'),
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='yellow', markersize=12, markeredgewidth=1.6,
                   label='object point (is_obj=1)'),
]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
            fontsize=10, bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Inference on val samples — all_v3_mc  (NS·PS·WM, obj + bg mix)',
              color=INK, fontsize=14, fontweight='700', x=0.04, y=0.995, ha='left')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = 'experiments/all_v3_mc/infer_grid.png'
plt.savefig(out, facecolor=BG, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
