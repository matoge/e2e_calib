"""
app.py  –  Flask backend for the interactive UV correction demo
Supports single-object (?mode=single), multi-object (?mode=multi),
and depth-aware covariance (?mode=depth) modes.
"""
import io, base64, os, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from flask import Flask, jsonify, request, send_from_directory, redirect

from dataset import make_image_and_points, make_image_and_points_multi, make_image_and_points_depth
from sim3d import make_sample as make_sample_sim3d
from model import CalibNet
from model_depth import CalibNetDepth

app = Flask(__name__, static_folder="static", static_url_path="")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
_models  = {}   # cache per ckpt path


def get_model(ckpt, model_cls=None):
    if ckpt in _models:
        return _models[ckpt]
    if not os.path.exists(ckpt):
        return None
    cls = model_cls if model_cls is not None else CalibNet
    m = cls().to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.eval()
    _models[ckpt] = m
    return m


def draw_arrows(ax, src, dst, color, n_arrows=20, step=None):
    """Draw subsampled arrows from src → dst."""
    if step is None:
        step = max(1, len(src) // n_arrows)
    idx = range(0, len(src), step)
    for i in idx:
        dx, dy = dst[i,0]-src[i,0], dst[i,1]-src[i,1]
        if abs(dx)+abs(dy) < 0.5:
            continue
        ax.annotate("", xy=(dst[i,0], dst[i,1]), xytext=(src[i,0], src[i,1]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.8, mutation_scale=6),
                    annotation_clip=True)


def _draw_arrows(ax, src, dst, color, n_max=15):
    """Subsample and draw dist→pred arrows."""
    n = len(src)
    step = max(1, n // n_max)
    for i in range(0, n, step):
        dx = dst[i, 0] - src[i, 0]
        dy = dst[i, 1] - src[i, 1]
        if abs(dx) + abs(dy) < 0.8:
            continue
        ax.annotate("", xy=(dst[i,0], dst[i,1]), xytext=(src[i,0], src[i,1]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=0.9, mutation_scale=7, alpha=0.75),
                    annotation_clip=True)


def render_png(img_np, true_uv, dist_uv, pred_uv=None, multi=False, depth=False, gt_only=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=128)
    if img_np.ndim == 3:   # RGB (H, W, 3)
        ax.imshow(img_np, origin="upper", extent=[0, IMG_SIZE, IMG_SIZE, 0])
    else:                  # grayscale (H, W)
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0, IMG_SIZE, IMG_SIZE, 0])

    if depth:
        n = len(true_uv) // 3
        ax.scatter(true_uv[:n,0],   true_uv[:n,1],   c="#00ff88", s=30, alpha=0.9, marker='x', linewidths=1.2, label="GT obj1")
        ax.scatter(true_uv[n:2*n,0],true_uv[n:2*n,1],c="#00ccff", s=30, alpha=0.9, marker='x', linewidths=1.2, label="GT obj2")
        ax.scatter(true_uv[2*n:,0], true_uv[2*n:,1], c="white",   s=12, alpha=0.5, marker='x', linewidths=0.7, label="GT bg")
        if not gt_only:
            ax.scatter(dist_uv[:n,0],   dist_uv[:n,1],   c="red",     s=8,  alpha=0.7, linewidths=0, label="dist obj1")
            ax.scatter(dist_uv[n:2*n,0],dist_uv[n:2*n,1],c="orange",  s=8,  alpha=0.7, linewidths=0, label="dist obj2")
            ax.scatter(dist_uv[2*n:,0], dist_uv[2*n:,1], c="#ff88ff", s=4,  alpha=0.4, linewidths=0, label="dist bg")
            if pred_uv is not None:
                ax.scatter(pred_uv[:n,0],   pred_uv[:n,1],   c="#ff4444", s=30, alpha=0.9, marker='+', linewidths=1.5, label="pred obj1")
                ax.scatter(pred_uv[n:2*n,0],pred_uv[n:2*n,1],c="#ff9900", s=30, alpha=0.9, marker='+', linewidths=1.5, label="pred obj2")
                ax.scatter(pred_uv[2*n:,0], pred_uv[2*n:,1], c="#ff00ff", s=12, alpha=0.6, marker='+', linewidths=0.8, label="pred bg")
                _draw_arrows(ax, dist_uv[:n],    pred_uv[:n],    "#ff4444", n_max=12)
                _draw_arrows(ax, dist_uv[n:2*n], pred_uv[n:2*n],"#ff9900", n_max=12)
    elif multi:
        n = len(true_uv) // 2
        ax.scatter(true_uv[:n,0], true_uv[:n,1], c="lime", s=30, alpha=0.9, marker='x', linewidths=1.2, label="GT obj1")
        ax.scatter(true_uv[n:,0], true_uv[n:,1], c="cyan", s=30, alpha=0.9, marker='x', linewidths=1.2, label="GT obj2")
        if not gt_only:
            ax.scatter(dist_uv[:n,0], dist_uv[:n,1], c="red",    s=8, alpha=0.7, linewidths=0, label="dist obj1")
            ax.scatter(dist_uv[n:,0], dist_uv[n:,1], c="orange", s=8, alpha=0.7, linewidths=0, label="dist obj2")
            if pred_uv is not None:
                ax.scatter(pred_uv[:n,0], pred_uv[:n,1], c="deepskyblue", s=30, alpha=0.9, marker='+', linewidths=1.5, label="pred obj1")
                ax.scatter(pred_uv[n:,0], pred_uv[n:,1], c="violet",      s=30, alpha=0.9, marker='+', linewidths=1.5, label="pred obj2")
                _draw_arrows(ax, dist_uv[:n], pred_uv[:n], "deepskyblue", n_max=15)
                _draw_arrows(ax, dist_uv[n:], pred_uv[n:], "violet",      n_max=15)
    else:
        ax.scatter(true_uv[:,0], true_uv[:,1], c="lime", s=30, alpha=0.9, marker='x', linewidths=1.2, label="GT")
        if not gt_only:
            ax.scatter(dist_uv[:,0], dist_uv[:,1], c="red", s=8, alpha=0.7, linewidths=0, label="distorted")
            if pred_uv is not None:
                ax.scatter(pred_uv[:,0], pred_uv[:,1], c="deepskyblue", s=30, alpha=0.9, marker='+', linewidths=1.5, label="corrected")
                _draw_arrows(ax, dist_uv, pred_uv, "deepskyblue", n_max=20)

    ax.set_xlim(0, IMG_SIZE); ax.set_ylim(IMG_SIZE, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=5, loc="upper right", framealpha=0.6)
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=128)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/sample")
def api_sample():
    seed     = int(request.args.get("seed", 0))
    mode     = request.args.get("mode", "single")
    dataset  = request.args.get("dataset", "synthetic")
    bg_ratio = int(request.args.get("bg_ratio", 1))
    gt_only  = request.args.get("gt_only", "0") == "1"
    multi    = (mode == "multi")
    depth    = (mode == "depth") or (dataset == "sim3d")

    if dataset == "sim3d":
        ckpt = "best_model_sim3d.pt"
        img, true_uvd, dist_uvd = make_sample_sim3d(
            seed=seed + 500_100, bg_ratio=bg_ratio)
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
        img_gray = img.mean(dim=0, keepdim=True)   # (1,H,W) for model
    elif depth:
        ckpt = "best_model_depth.pt"
        img, true_uvd, dist_uvd = make_image_and_points_depth(
            seed=seed + 400_100, bg_ratio=bg_ratio)
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
    elif multi:
        ckpt = "best_model_multi.pt"
        img, true_uv, dist_uv = make_image_and_points_multi(img_size=IMG_SIZE, seed=seed + 300_100)
    else:
        ckpt = "best_model.pt"
        img, true_uv, dist_uv = make_image_and_points(img_size=IMG_SIZE, seed=seed + 200_000)

    model_cls  = CalibNetDepth if depth else CalibNet
    model      = get_model(ckpt, model_cls)
    pred_uv    = None
    err_after  = None
    shifts_gt  = []
    shifts_pred = []
    sigma_stats = None

    if depth:  # covers both mode=depth and dataset=sim3d
        n_total = len(true_uv)
        n_obj = n_total // (2 + bg_ratio)
        n_bg  = n_total - 2 * n_obj
        group_names = (["car", "pole", "bg"] if dataset == "sim3d"
                       else ["obj1", "obj2", "bg"])
        for name, sl in zip(group_names, [slice(0, n_obj), slice(n_obj, 2*n_obj),
                                          slice(2*n_obj, 2*n_obj+n_bg)]):
            off = (true_uv[sl] - dist_uv[sl]).mean(0)
            shifts_gt.append({"label": name, "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
    elif multi:
        n = len(true_uv) // 2
        for i in range(2):
            off = (true_uv[i*n:(i+1)*n] - dist_uv[i*n:(i+1)*n]).mean(0)
            shifts_gt.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
    else:
        off = (true_uv - dist_uv).mean(0)
        shifts_gt.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})

    err_before = float((dist_uv - true_uv).norm(dim=1).mean())

    if model is not None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if depth:
                # sim3d returns 3-ch RGB; use grayscale for model inference
                img_in = img_gray if dataset == "sim3d" else img
                params = model(img_in.unsqueeze(0).to(DEVICE),
                               dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()
                tx_pred = params[:, 0]; ty_pred = params[:, 1]
                offset_pred = torch.stack([tx_pred, ty_pred], dim=1)
                sx = params[:, 2].exp().numpy()
                sy = params[:, 3].exp().numpy()
                sigma_stats = {
                    "obj1": {"sx": round(float(sx[:n_obj].mean()),        3), "sy": round(float(sy[:n_obj].mean()),        3)},
                    "obj2": {"sx": round(float(sx[n_obj:2*n_obj].mean()), 3), "sy": round(float(sy[n_obj:2*n_obj].mean()), 3)},
                    "bg":   {"sx": round(float(sx[2*n_obj:].mean()),      3), "sy": round(float(sy[2*n_obj:].mean()),      3)},
                }
            else:
                offset_pred = model(img.unsqueeze(0).to(DEVICE),
                                    dist_uv.unsqueeze(0).to(DEVICE))[0].cpu()

        pred_uv_t = (dist_uv + offset_pred).clamp(0, IMG_SIZE - 1)
        err_after = float((pred_uv_t - true_uv).norm(dim=1).mean())
        pred_uv   = pred_uv_t.numpy()

        if depth:
            for j, (name, sz) in enumerate(zip(group_names, [n_obj, n_obj, n_bg])):
                start = sum([n_obj, n_obj, n_bg][:j])
                sl = slice(start, start + sz)
                off = offset_pred[sl].mean(0)
                shifts_pred.append({"label": name, "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
        elif multi:
            n = len(true_uv) // 2
            for i in range(2):
                off = offset_pred[i*n:(i+1)*n].mean(0)
                shifts_pred.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
        else:
            off = offset_pred.mean(0)
            shifts_pred.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})

    # Build display image: sim3d returns (3,H,W) RGB, others (1,H,W) grayscale
    if dataset == "sim3d":
        img_disp = img.numpy().transpose(1, 2, 0)  # (H, W, 3)
    else:
        img_disp = img[0].numpy()                  # (H, W)

    if depth:
        png = render_png(img_disp, true_uv.numpy(), dist_uv.numpy(), pred_uv, depth=True,  gt_only=gt_only)
    else:
        png = render_png(img_disp, true_uv.numpy(), dist_uv.numpy(), pred_uv, multi=multi, gt_only=gt_only)

    return jsonify({
        "png":          png,
        "mode":         mode,
        "shifts_gt":    shifts_gt,
        "shifts_pred":  shifts_pred if shifts_pred else None,
        "err_before":   round(err_before, 3),
        "err_after":    round(err_after, 3) if err_after is not None else None,
        "model_loaded": model is not None,
        "sigma_stats":  sigma_stats,
        "dataset":      dataset,
    })


@app.route("/api/model_status")
def api_model_status():
    return jsonify({
        "single": os.path.exists("best_model.pt"),
        "multi":  os.path.exists("best_model_multi.pt"),
        "depth":  os.path.exists("best_model_depth.pt"),
        "sim3d":  os.path.exists("best_model_sim3d.pt"),
    })


@app.route("/model-graph")
def model_graph():
    import threading, netron
    onnx_path = os.path.abspath("model.onnx")
    def _start():
        netron.start(onnx_path, address=("0.0.0.0", 5002), browse=False)
    threading.Thread(target=_start, daemon=True).start()
    import time; time.sleep(1)
    return redirect("http://localhost:5002", code=302)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
