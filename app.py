"""
app.py  –  Flask backend for the interactive UV correction demo
Supports single-object (?mode=single), multi-object (?mode=multi),
and depth-aware covariance (?mode=depth) modes.
"""
import io, base64, os, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from flask import Flask, jsonify, request, send_from_directory, redirect

from dataset import (make_image_and_points, make_image_and_points_multi,
                     make_image_and_points_depth, make_image_and_points_grid,
                     make_image_and_points_grid_depth)
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


def _detect_groups(uvd):
    """Return list of (start, end, depth_val) for contiguous depth groups."""
    depths = uvd[:, 2]
    groups, start, cur = [], 0, depths[0].item()
    for i in range(1, len(depths)):
        d = depths[i].item()
        if abs(d - cur) > 1e-4:
            groups.append((start, i, cur)); start = i; cur = d
    groups.append((start, len(depths), cur))
    return groups


def render_png(img_np, true_uv, dist_uv, pred_uv=None, multi=False, depth=False,
               gt_only=False, img_size=None, grid_depth_uvd=None):
    """
    grid_depth_uvd: pass true_uvd (N,3) to enable depth-coloured grid_depth rendering.
    """
    sz = img_size or IMG_SIZE
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=128)
    if img_np.ndim == 3:
        ax.imshow(img_np, origin="upper", extent=[0, sz, sz, 0])
    else:
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0, sz, sz, 0])

    if grid_depth_uvd is not None:
        groups = _detect_groups(grid_depth_uvd)
        cmap   = plt.cm.tab10
        for gi, (s, e, d) in enumerate(groups):
            c     = cmap(gi % 10)
            label = "bg" if gi == len(groups) - 1 else f"obj{gi+1}"
            ax.scatter(true_uv[s:e, 0], true_uv[s:e, 1],
                       color=c, s=30, alpha=0.9, marker='x', linewidths=1.2, label=f"GT {label}")
            if not gt_only:
                ax.scatter(dist_uv[s:e, 0], dist_uv[s:e, 1],
                           color=c, s=8, alpha=0.45, linewidths=0)
                if pred_uv is not None:
                    ax.scatter(pred_uv[s:e, 0], pred_uv[s:e, 1],
                               color=c, s=30, alpha=0.9, marker='+', linewidths=1.5, label=f"pred {label}")
                    _draw_arrows(ax, dist_uv[s:e], pred_uv[s:e], c, n_max=8)
    elif depth:
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

    ax.set_xlim(0, sz); ax.set_ylim(sz, 0)
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
    res      = int(request.args.get("res", 128))   # 64 or 128
    sampling = request.args.get("sampling", "dense")  # dense / grid4 / grid8
    multi      = (mode == "multi")
    depth      = (mode == "depth") or (dataset == "sim3d")
    gd_mode    = (mode == "grid_depth")   # grid_depth handled separately

    # Determine resolution: grid_depth and sim3d are always 64/128 respectively
    if mode == "grid_depth":
        cur_size = 64
    elif dataset == "sim3d" or (mode == "depth"):
        cur_size = IMG_SIZE
    else:
        cur_size = res

    img_gray = None  # only used for sim3d

    if mode == "grid_depth":
        from config_grid_depth import CFG as _GD_CFG
        from pathlib import Path as _Path
        ckpt = str(_Path("experiments") / _GD_CFG["name"] / "best_model.pt")
        img, true_uvd, dist_uvd = make_image_and_points_grid_depth(
            img_size=64, seed=seed + 700_000,
            random_depths=_GD_CFG.get("random_depths", False))
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
        bg_ratio = 1   # 3 equal groups
    elif dataset == "sim3d":
        ckpt = "best_model_sim3d.pt"
        img, true_uvd, dist_uvd = make_sample_sim3d(
            seed=seed + 500_100, bg_ratio=bg_ratio)
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
        img_gray = img.mean(dim=0, keepdim=True)
    elif mode == "depth":
        ckpt = "best_model_depth.pt"
        img, true_uvd, dist_uvd = make_image_and_points_depth(
            seed=seed + 400_100, bg_ratio=bg_ratio)
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
    elif multi:
        ckpt = "best_model_multi_64.pt" if cur_size == 64 else "best_model_multi.pt"
        img, true_uv, dist_uv = make_image_and_points_multi(img_size=cur_size, seed=seed + 300_100)
    elif cur_size == 64:
        ckpt = "best_model_64.pt"
        gs = {"grid4": 4, "grid8": 8}.get(sampling)
        if gs:
            img, true_uv, dist_uv = make_image_and_points_grid(
                img_size=64, grid_size=gs, seed=seed + 200_000)
        else:
            img, true_uv, dist_uv = make_image_and_points(img_size=64, seed=seed + 200_000)
    else:
        ckpt = "best_model.pt"
        gs = {"grid4": 4, "grid8": 8}.get(sampling)
        if gs:
            img, true_uv, dist_uv = make_image_and_points_grid(
                img_size=IMG_SIZE, grid_size=gs, seed=seed + 200_000)
        else:
            img, true_uv, dist_uv = make_image_and_points(img_size=IMG_SIZE, seed=seed + 200_000)

    if gd_mode:
        from config_grid_depth import CFG as _GD
        model = get_model(ckpt, lambda: CalibNetDepth(
            img_size=_GD["img_size"], in_channels=_GD["in_channels"],
            n_layers=_GD["n_layers"], self_first=_GD["self_first"]))
    elif cur_size == 64 and not depth:
        model = get_model(ckpt, lambda: CalibNet(img_size=64))
    elif depth:
        model = get_model(ckpt, CalibNetDepth)
    else:
        model = get_model(ckpt, CalibNet)
    pred_uv      = None
    err_after    = None
    shifts_gt    = []
    shifts_pred  = []
    sigma_stats  = None
    group_errors = None
    gd_groups    = None   # for render_png grid_depth path

    if gd_mode:
        gd_groups = _detect_groups(true_uvd)
        for gi, (s, e, d) in enumerate(gd_groups):
            label = "bg" if gi == len(gd_groups) - 1 else f"obj{gi+1}"
            off = (true_uv[s:e] - dist_uv[s:e]).mean(0)
            shifts_gt.append({"label": label, "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
    elif depth:  # mode=depth or sim3d
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
            if gd_mode:
                params = model(img.unsqueeze(0).to(DEVICE),
                               dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()
                offset_pred = params[:, :2]
                sx = params[:, 2].exp().numpy()
                sy = params[:, 3].exp().numpy()
                gd_groups_s = _detect_groups(true_uvd)
                sigma_stats = {}
                for gi, (s, e, d) in enumerate(gd_groups_s):
                    label = "bg" if gi == len(gd_groups) - 1 else f"obj{gi+1}"
                    sigma_stats[label] = {
                        "sx": round(float(sx[s:e].mean()), 3),
                        "sy": round(float(sy[s:e].mean()), 3),
                    }
            elif depth:
                img_in = img_gray if dataset == "sim3d" else img
                params = model(img_in.unsqueeze(0).to(DEVICE),
                               dist_uvd.unsqueeze(0).to(DEVICE))[0].cpu().float()
                offset_pred = params[:, :2]
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

        pred_uv_t = (dist_uv + offset_pred).clamp(0, cur_size - 1)
        err_after = float((pred_uv_t - true_uv).norm(dim=1).mean())
        pred_uv   = pred_uv_t.numpy()

        if gd_mode:
            group_errors = []
            for gi, (s, e, d) in enumerate(gd_groups):
                lbl = "bg" if gi == len(gd_groups) - 1 else f"obj{gi+1}"
                eb  = float((dist_uv[s:e] - true_uv[s:e]).norm(dim=1).mean())
                ea  = float((pred_uv_t[s:e] - true_uv[s:e]).norm(dim=1).mean())
                group_errors.append({"label": lbl, "before": round(eb,3), "after": round(ea,3)})

        if gd_mode:
            for gi, (s, e, d) in enumerate(gd_groups):
                label = "bg" if gi == len(gd_groups) - 1 else f"obj{gi+1}"
                off = offset_pred[s:e].mean(0)
                shifts_pred.append({"label": label, "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
        elif depth:
            for j, (name, sz) in enumerate(zip(group_names, [n_obj, n_obj, n_bg])):
                start = sum([n_obj, n_obj, n_bg][:j])
                off = offset_pred[start:start+sz].mean(0)
                shifts_pred.append({"label": name, "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
        elif multi:
            n = len(true_uv) // 2
            for i in range(2):
                off = offset_pred[i*n:(i+1)*n].mean(0)
                shifts_pred.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
        else:
            off = offset_pred.mean(0)
            shifts_pred.append({"tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})

    if dataset == "sim3d" or (gd_mode and img.shape[0] == 3):
        img_disp = img.numpy().transpose(1, 2, 0)  # (H, W, 3) RGB
    else:
        img_disp = img[0].numpy()                  # (H, W) grayscale

    if gd_mode:
        png = render_png(img_disp, true_uv.numpy(), dist_uv.numpy(), pred_uv,
                         gt_only=gt_only, img_size=cur_size,
                         grid_depth_uvd=true_uvd.numpy())
    elif depth:
        png = render_png(img_disp, true_uv.numpy(), dist_uv.numpy(), pred_uv,
                         depth=True, gt_only=gt_only, img_size=cur_size)
    else:
        png = render_png(img_disp, true_uv.numpy(), dist_uv.numpy(), pred_uv,
                         multi=multi, gt_only=gt_only, img_size=cur_size)

    return jsonify({
        "png":          png,
        "mode":         mode,
        "shifts_gt":    shifts_gt,
        "shifts_pred":  shifts_pred if shifts_pred else None,
        "err_before":   round(err_before, 3),
        "err_after":    round(err_after, 3) if err_after is not None else None,
        "model_loaded": model is not None,
        "sigma_stats":  sigma_stats,
        "group_errors": group_errors if gd_mode and pred_uv is not None else None,
        "dataset":      dataset,
    })


@app.route("/api/model_status")
def api_model_status():
    return jsonify({
        "single":      os.path.exists("best_model.pt"),
        "multi":       os.path.exists("best_model_multi.pt"),
        "depth":       os.path.exists("best_model_depth.pt"),
        "sim3d":       os.path.exists("best_model_sim3d.pt"),
        "64":          os.path.exists("best_model_64.pt"),
        "grid_depth":  os.path.exists(
            str(__import__("pathlib").Path("experiments") /
                __import__("config_grid_depth").CFG["name"] / "best_model.pt")),
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
