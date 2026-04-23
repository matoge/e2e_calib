"""
vis_depth.py  –  Visualise depth-aware uncertainty: obj1/obj2/background.

Expected:  obj points   → small ellipses (confident)
           bg   points  → large ellipses (uncertain)
"""
import math, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

from datasets.synthetic import make_image_and_points_depth, DEPTH_OBJ1, DEPTH_OBJ2, DEPTH_BG, DEPTH_NORM
from models.model_depth import CalibNetDepth

CKPT      = "best_model_depth.pt"
OUT       = "result_depth.png"
N_SAMPLES = 8
BG_RATIO  = 1   # change to 3, 5, etc. to increase BG density
N_POINTS  = 255
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cov_ellipse(ax, cx, cy, sx, sy, rho, n_std=1.5, **kw):
    cov = np.array([[sx**2, rho*sx*sy],[rho*sx*sy, sy**2]])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    w, h = 2*n_std*np.sqrt(vals)
    angle = math.degrees(math.atan2(vecs[1,1], vecs[0,1]))
    ax.add_patch(Ellipse((cx,cy), width=w, height=h, angle=angle, **kw))


def main():
    model = CalibNetDepth().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    n_obj = N_POINTS // (2 + BG_RATIO)
    n_bg  = N_POINTS - 2 * n_obj

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
    axes = axes.flatten()

    # Per-group sigma stats
    all_sx = {"obj1":[], "obj2":[], "bg":[]}
    all_sy = {"obj1":[], "obj2":[], "bg":[]}

    for i in range(N_SAMPLES):
        ax = axes[i]
        img, true_uvd, dist_uvd = make_image_and_points_depth(
            seed=400_100 + i*7, bg_ratio=BG_RATIO, n_points=N_POINTS)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()

        tx  = params[:,0].numpy()
        ty  = params[:,1].numpy()
        sx  = params[:,2].exp().numpy()
        sy  = params[:,3].exp().numpy()
        rho = params[:,4].numpy()

        pred_uv = dist_uvd[:,:2].numpy().copy()
        pred_uv[:,0] += tx; pred_uv[:,1] += ty
        pred_uv = pred_uv.clip(0,127)

        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0,128,128,0])

        groups = [
            (slice(0,         n_obj),       "obj1", "#00ff88", "#ff4444"),
            (slice(n_obj,     2*n_obj),     "obj2", "#00ccff", "#ff9900"),
            (slice(2*n_obj,   2*n_obj+n_bg),"bg",   "#ffffff", "#ff00ff"),
        ]
        for sl, name, gt_c, el_c in groups:
            # GT points
            ax.scatter(true_uvd[sl,0].numpy(), true_uvd[sl,1].numpy(),
                       c=gt_c, s=4, alpha=0.5, linewidths=0)
            # Ellipses (subsample)
            idx = list(range(sl.start, sl.stop, max(1,(sl.stop-sl.start)//20)))
            for j in idx:
                fc = tuple(int(el_c.lstrip('#')[k:k+2], 16)/255 for k in (0,2,4)) + (0.12,)
                cov_ellipse(ax, pred_uv[j,0], pred_uv[j,1], sx[j], sy[j], rho[j],
                            n_std=1.5, linewidth=0.8,
                            edgecolor=el_c, facecolor=fc)
            all_sx[name].append(sx[sl].mean())
            all_sy[name].append(sy[sl].mean())

        ax.set_xlim(0,127); ax.set_ylim(127,0)
        ax.set_xticks([]); ax.set_yticks([])
        s1 = sx[:n_obj].mean(); s2 = sx[n_obj:2*n_obj].mean(); sb = sx[2*n_obj:].mean()
        ax.set_title(f"s{i}  σx: obj1={s1:.1f} obj2={s2:.1f} bg={sb:.1f}", fontsize=7)

    handles = [
        mpatches.Patch(color="#00ff88", label="GT obj1 (depth=10)"),
        mpatches.Patch(color="#00ccff", label="GT obj2 (depth=20)"),
        mpatches.Patch(color="white",   label="GT bg   (depth=40)"),
        mpatches.Patch(color="#ff4444", label="ellipse obj1"),
        mpatches.Patch(color="#ff9900", label="ellipse obj2"),
        mpatches.Patch(color="#ff00ff", label="ellipse bg"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Depth-aware uncertainty:  obj1/obj2=small ellipse,  bg=large ellipse",
                 fontsize=11)
    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(OUT, bbox_inches="tight", dpi=120)
    print(f"Saved → {OUT}")

    # Print sigma summary
    print(f"\n=== Per-group σ summary (avg over {N_SAMPLES} samples) ===")
    for name, vals in all_sx.items():
        if vals:
            print(f"  {name:4s}  σx={np.mean(vals):.2f}  σy={np.mean(all_sy[name]):.2f}")


if __name__ == "__main__":
    main()
