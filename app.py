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


def render_png(img_np, true_uv, dist_uv, pred_uv=None, multi=False, depth=False):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=128)
    ax.imshow(img_np, cmap="gray", vmin=0, vmax=1,
              origin="upper", extent=[0, IMG_SIZE, IMG_SIZE, 0])

    if depth:
        n = len(true_uv) // 3
        ax.scatter(dist_uv[:n,0],   dist_uv[:n,1],   c="red",     s=5, alpha=0.7, linewidths=0, label="dist obj1")
        ax.scatter(dist_uv[n:2*n,0],dist_uv[n:2*n,1],c="orange",  s=5, alpha=0.7, linewidths=0, label="dist obj2")
        ax.scatter(dist_uv[2*n:,0], dist_uv[2*n:,1], c="#ff88ff", s=3, alpha=0.5, linewidths=0, label="dist bg")
        ax.scatter(true_uv[:n,0],   true_uv[:n,1],   c="#00ff88", s=5, alpha=0.7, linewidths=0, label="GT obj1")
        ax.scatter(true_uv[n:2*n,0],true_uv[n:2*n,1],c="#00ccff", s=5, alpha=0.7, linewidths=0, label="GT obj2")
        ax.scatter(true_uv[2*n:,0], true_uv[2*n:,1], c="white",   s=3, alpha=0.4, linewidths=0, label="GT bg")
        if pred_uv is not None:
            ax.scatter(pred_uv[:n,0],   pred_uv[:n,1],   c="#ff4444",   s=5, alpha=0.9, linewidths=0, label="pred obj1")
            ax.scatter(pred_uv[n:2*n,0],pred_uv[n:2*n,1],c="#ff9900",   s=5, alpha=0.9, linewidths=0, label="pred obj2")
            ax.scatter(pred_uv[2*n:,0], pred_uv[2*n:,1], c="#ff00ff",   s=3, alpha=0.6, linewidths=0, label="pred bg")
    elif multi:
        n = len(true_uv) // 2
        # obj1: red/lime/blue, obj2: orange/cyan/violet
        ax.scatter(dist_uv[:n,0], dist_uv[:n,1], c="red",    s=7, alpha=0.8, linewidths=0, label="dist obj1")
        ax.scatter(dist_uv[n:,0], dist_uv[n:,1], c="orange", s=7, alpha=0.8, linewidths=0, label="dist obj2")
        ax.scatter(true_uv[:n,0], true_uv[:n,1], c="lime",   s=7, alpha=0.7, linewidths=0, label="GT obj1")
        ax.scatter(true_uv[n:,0], true_uv[n:,1], c="cyan",   s=7, alpha=0.7, linewidths=0, label="GT obj2")
        if pred_uv is not None:
            ax.scatter(pred_uv[:n,0], pred_uv[:n,1], c="deepskyblue", s=7, alpha=0.9, linewidths=0, label="pred obj1")
            ax.scatter(pred_uv[n:,0], pred_uv[n:,1], c="violet",      s=7, alpha=0.9, linewidths=0, label="pred obj2")
    else:
        ax.scatter(dist_uv[:,0], dist_uv[:,1], c="red",          s=8, alpha=0.8, linewidths=0, label="distorted")
        ax.scatter(true_uv[:,0], true_uv[:,1], c="lime",          s=8, alpha=0.7, linewidths=0, label="GT")
        if pred_uv is not None:
            ax.scatter(pred_uv[:,0], pred_uv[:,1], c="deepskyblue", s=8, alpha=0.9, linewidths=0, label="corrected")

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
    seed  = int(request.args.get("seed", 0))
    mode  = request.args.get("mode", "single")   # "single" | "multi" | "depth"
    multi = (mode == "multi")
    depth = (mode == "depth")

    if depth:
        ckpt = "best_model_depth.pt"
        img, true_uvd, dist_uvd = make_image_and_points_depth(seed=seed + 400_100)
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

    if depth:
        n = len(true_uv) // 3
        for name, sl in [("obj1", slice(0, n)), ("obj2", slice(n, 2*n)), ("bg", slice(2*n, 3*n))]:
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
                params = model(img.unsqueeze(0).to(DEVICE),
                               dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()
                tx_pred = params[:, 0]; ty_pred = params[:, 1]
                offset_pred = torch.stack([tx_pred, ty_pred], dim=1)
                sx = params[:, 2].exp().numpy()
                sy = params[:, 3].exp().numpy()
                n  = len(true_uv) // 3
                sigma_stats = {
                    "obj1": {"sx": round(float(sx[:n].mean()),   3), "sy": round(float(sy[:n].mean()),   3)},
                    "obj2": {"sx": round(float(sx[n:2*n].mean()),3), "sy": round(float(sy[n:2*n].mean()),3)},
                    "bg":   {"sx": round(float(sx[2*n:].mean()), 3), "sy": round(float(sy[2*n:].mean()), 3)},
                }
            else:
                offset_pred = model(img.unsqueeze(0).to(DEVICE),
                                    dist_uv.unsqueeze(0).to(DEVICE))[0].cpu()

        pred_uv_t = (dist_uv + offset_pred).clamp(0, IMG_SIZE - 1)
        err_after = float((pred_uv_t - true_uv).norm(dim=1).mean())
        pred_uv   = pred_uv_t.numpy()

        if depth:
            n = len(true_uv) // 3
            for j, name in enumerate(["obj1", "obj2", "bg"]):
                sl = slice(j*n, (j+1)*n)
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

    if depth:
        png = render_png(img[0].numpy(), true_uv.numpy(), dist_uv.numpy(), pred_uv, depth=True)
    else:
        png = render_png(img[0].numpy(), true_uv.numpy(), dist_uv.numpy(), pred_uv, multi=multi)

    return jsonify({
        "png":          png,
        "mode":         mode,
        "shifts_gt":    shifts_gt,
        "shifts_pred":  shifts_pred if shifts_pred else None,
        "err_before":   round(err_before, 3),
        "err_after":    round(err_after, 3) if err_after is not None else None,
        "model_loaded": model is not None,
        "sigma_stats":  sigma_stats,
    })


@app.route("/api/model_status")
def api_model_status():
    return jsonify({
        "single": os.path.exists("best_model.pt"),
        "multi":  os.path.exists("best_model_multi.pt"),
        "depth":  os.path.exists("best_model_depth.pt"),
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
