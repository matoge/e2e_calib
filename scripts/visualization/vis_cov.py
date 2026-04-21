"""
vis_cov.py  –  Visualise predicted uncertainty ellipses.

Each point is drawn as an ellipse proportional to (σx, σy, ρ).
Color = confidence: blue=high, red=low.
"""
import math, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

from dataset import make_image_and_points
from model_cov import CalibNetCov

CKPT   = "best_model_cov.pt"
OUT    = "result_cov.png"
N_SAMPLES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cov_ellipse(ax, cx, cy, sx, sy, rho, n_std=1.5, **kw):
    """Draw 1.5σ confidence ellipse for 2D Gaussian."""
    cov = np.array([[sx**2, rho*sx*sy],
                    [rho*sx*sy, sy**2]])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-8)
    w, h = 2 * n_std * np.sqrt(vals)
    angle = math.degrees(math.atan2(vecs[1,1], vecs[0,1]))
    e = Ellipse((cx, cy), width=w, height=h, angle=angle, **kw)
    ax.add_patch(e)


def main():
    model = CalibNetCov().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    cols = 4
    rows = math.ceil(N_SAMPLES / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=100)
    axes = axes.flatten()

    for i in range(N_SAMPLES):
        ax = axes[i]
        img, true_uv, dist_uv = make_image_and_points(seed=200_000 + i*7)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uv.unsqueeze(0).to(DEVICE))[0].cpu().float()  # (N,5)

        tx   = params[:, 0].numpy()
        ty   = params[:, 1].numpy()
        sx   = params[:, 2].exp().numpy()
        sy   = params[:, 3].exp().numpy()
        rho  = params[:, 4].numpy()

        pred_uv = dist_uv.numpy().copy()
        pred_uv[:, 0] += tx
        pred_uv[:, 1] += ty
        pred_uv = pred_uv.clip(0, 127)

        conf = 1.0 / (sx * sy + 1e-6)          # inverse area = confidence
        conf_n = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0, 128, 128, 0])

        # draw ellipses (subsample for clarity)
        step = max(1, len(pred_uv) // 64)
        cmap = plt.cm.RdYlBu
        for j in range(0, len(pred_uv), step):
            c = cmap(conf_n[j])
            cov_ellipse(ax, pred_uv[j,0], pred_uv[j,1],
                        sx[j], sy[j], rho[j],
                        n_std=1.5, linewidth=0.6,
                        edgecolor=c, facecolor=(*c[:3], 0.15))

        # GT and distorted
        ax.scatter(dist_uv[:,0], dist_uv[:,1], c="red",  s=3, alpha=0.5, linewidths=0)
        ax.scatter(true_uv[:,0], true_uv[:,1], c="lime", s=3, alpha=0.5, linewidths=0)
        ax.scatter(pred_uv[:,0], pred_uv[:,1], c="deepskyblue", s=3, alpha=0.7, linewidths=0)

        # stats
        gt = (true_uv - dist_uv)
        err_b = float((dist_uv - true_uv).norm(dim=1).mean())
        err_a = float(torch.tensor(pred_uv).float().sub(true_uv).norm(dim=1).mean())
        mean_sx = sx.mean(); mean_sy = sy.mean()
        ax.set_title(f"s{i}  err: {err_b:.1f}→{err_a:.1f}px  σ=({mean_sx:.1f},{mean_sy:.1f})",
                     fontsize=7)
        ax.set_xlim(0,127); ax.set_ylim(127,0)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(N_SAMPLES, len(axes)):
        axes[j].set_visible(False)

    handles = [
        mpatches.Patch(color="red",   label="distorted (input)"),
        mpatches.Patch(color="lime",  label="GT"),
        mpatches.Patch(color="deepskyblue", label="corrected (pred)"),
        mpatches.Patch(color="blue",  label="high confidence (small ellipse)"),
        mpatches.Patch(color="red",   label="low confidence  (large ellipse)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Uncertainty ellipses (1.5σ)  —  blue=confident, red=uncertain",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT, bbox_inches="tight", dpi=120)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
