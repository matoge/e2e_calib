"""
app.py  –  Flask backend for the interactive UV correction demo
Supports single-object (?mode=single), multi-object (?mode=multi),
and depth-aware covariance (?mode=depth) modes.
"""
import io, base64, os, re, torch, numpy as np, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, matplotlib.patheffects as pe
from flask import Flask, jsonify, request, send_from_directory, redirect
from pathlib import Path

from dataset import (make_image_and_points, make_image_and_points_multi,
                     make_image_and_points_depth, make_image_and_points_grid,
                     make_image_and_points_grid_depth)
from sim3d import make_sample as make_sample_sim3d
from model import CalibNet
from model_depth import CalibNetDepth

_ps_dataset = None
def _get_ps_dataset(cache='/tmp/pandaset_full_cache.pt'):
    global _ps_dataset
    if _ps_dataset is None:
        try:
            from dataset_pandaset import PandaSetCalibDataset
            _ps_dataset = PandaSetCalibDataset(cache, split='val')
        except Exception:
            pass
    return _ps_dataset

_ns_dataset = None
def _get_ns_dataset(cache='/tmp/nuscenes_static_cache.pt'):
    global _ns_dataset
    if _ns_dataset is None:
        try:
            from dataset_nuscenes import NuScenesCalibDataset
            _ns_dataset = NuScenesCalibDataset(cache, split='val')
        except Exception as e:
            print(f"NuScenes load error: {e}")
    return _ns_dataset

_ns_ps_dataset = None
def _get_ns_ps_dataset():
    global _ns_ps_dataset
    if _ns_ps_dataset is None:
        try:
            from torch.utils.data import ConcatDataset
            ns = _get_ns_dataset()
            ps = _get_ps_dataset('/tmp/pandaset_cache.pt')
            if ns is not None and ps is not None:
                _ns_ps_dataset = ConcatDataset([ns, ps])
        except Exception as e:
            print(f"NS+PS load error: {e}")
    return _ns_ps_dataset

app = Flask(__name__, static_folder="static", static_url_path="")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
_models  = {}   # cache per ckpt path
_exp_cfg_cache = {}  # cache parsed experiment configs


def _load_exp_cfg(exp_name: str) -> dict | None:
    """Load config from experiments/{exp_name}/config.py."""
    if exp_name in _exp_cfg_cache:
        return _exp_cfg_cache[exp_name]
    cfg_path = Path("experiments") / exp_name / "config.py"
    if not cfg_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_exp_cfg", cfg_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.CFG
    _exp_cfg_cache[exp_name] = cfg
    return cfg


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
    elif dataset in ("sim3d", "pandaset", "nuscenes", "ns_ps") or (mode == "depth"):
        cur_size = IMG_SIZE if dataset == "sim3d" else 64
    else:
        cur_size = res

    img_gray = None  # only used for sim3d

    if dataset in ("pandaset", "nuscenes", "ns_ps"):
        if dataset == "pandaset":
            ds = _get_ps_dataset()
            default_exp = "ps_v7_frustum"
        elif dataset == "nuscenes":
            ds = _get_ns_dataset()
            default_exp = "ns_ps_v2"
        else:  # ns_ps
            ds = _get_ns_ps_dataset()
            default_exp = "ns_ps_v2"
        if ds is None:
            return jsonify({"error": f"{dataset} cache not found"}), 404
        img, true_uvd, dist_uvd = ds[seed % len(ds)]
        true_uv = true_uvd[:, :2]
        dist_uv = dist_uvd[:, :2]
        is_obj  = dist_uvd[:, 3].numpy() > 0.5
        exp_name = request.args.get("experiment", default_exp)
        ckpt     = str(Path("experiments") / exp_name / "best_model.pt")
        cfg_ps   = _load_exp_cfg(exp_name) or {}
        cur_size = 64
    elif mode == "grid_depth":
        exp_name = request.args.get("experiment", None)
        if exp_name:
            _GD_CFG = _load_exp_cfg(exp_name) or {}
            if not _GD_CFG:
                from config_grid_depth import CFG as _GD_CFG
        else:
            from config_grid_depth import CFG as _GD_CFG
            exp_name = _GD_CFG["name"]
        ckpt = str(Path("experiments") / exp_name / "best_model.pt")
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

    if dataset in ("pandaset", "nuscenes", "ns_ps"):
        _PS = cfg_ps
        model = get_model(ckpt, lambda: CalibNetDepth(
            img_size=64, in_channels=_PS.get("in_channels", 3),
            n_layers=_PS.get("n_layers", 3), self_first=False,
            use_convnext=_PS.get("use_convnext", False),
            use_frustum=_PS.get("use_frustum", False)))
    elif gd_mode:
        _GD = _GD_CFG
        model = get_model(ckpt, lambda: CalibNetDepth(
            img_size=_GD["img_size"], in_channels=_GD["in_channels"],
            n_layers=_GD["n_layers"], self_first=_GD["self_first"],
            kv_self_attn=_GD.get("kv_self_attn", False),
            cross_temp=_GD.get("cross_temp", 1.0)))
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

    if dataset in ("pandaset", "nuscenes", "ns_ps"):
        off = (true_uv - dist_uv).mean(0)
        shifts_gt.append({"label": "obj", "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})

    err_before = float((dist_uv - true_uv).norm(dim=1).mean())

    if model is not None:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if dataset in ("pandaset", "nuscenes", "ns_ps"):
                params = model(img.unsqueeze(0).to(DEVICE),
                               dist_uvd[:, :3].unsqueeze(0).to(DEVICE))[0].cpu().float()
                offset_pred = params[:, :2]
                sx = params[:, 2].exp().numpy()
                sy = params[:, 3].exp().numpy()
                sigma_stats = {
                    "obj": {"sx": round(float(sx[is_obj].mean()) if is_obj.any() else 0, 3),
                            "sy": round(float(sy[is_obj].mean()) if is_obj.any() else 0, 3)},
                    "bg":  {"sx": round(float(sx[~is_obj].mean()) if (~is_obj).any() else 0, 3),
                            "sy": round(float(sy[~is_obj].mean()) if (~is_obj).any() else 0, 3)},
                }
                off = offset_pred.mean(0)
                shifts_pred.append({"label": "obj", "tx": round(float(off[0]),2), "ty": round(float(off[1]),2)})
            elif gd_mode:
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

    if dataset in ("sim3d", "pandaset", "nuscenes", "ns_ps") or (gd_mode and img.shape[0] == 3):
        img_disp = img.numpy().transpose(1, 2, 0)  # (H, W, 3) RGB
    else:
        img_disp = img[0].numpy()                  # (H, W) grayscale

    if dataset in ("pandaset", "nuscenes", "ns_ps"):
        # render with obj/bg coloring
        pred_uv_ps = (dist_uv + offset_pred).clamp(0, 63).numpy() if model is not None else None
        err_after  = float(((torch.from_numpy(pred_uv_ps) if pred_uv_ps is not None
                             else dist_uv) - true_uv).norm(dim=1).mean()) if pred_uv_ps is not None else None
        fig, ax = plt.subplots(figsize=(4,4), dpi=128)
        ax.imshow(img_disp, extent=[0,64,64,0], origin='upper')
        t_uv = true_uv.numpy(); d_uv = dist_uv.numpy()
        for mask, cgt, cd, cp, lbl in [
            (is_obj,   'lime',  'red',    'cyan',        'obj'),
            (~is_obj,  'yellow','orange', 'deepskyblue', 'bg'),
        ]:
            if not mask.any(): continue
            ax.scatter(t_uv[mask,0], t_uv[mask,1], c=cgt, s=18, marker='x', linewidths=1.1, label=f'GT {lbl}', zorder=3)
            if not gt_only:
                ax.scatter(d_uv[mask,0], d_uv[mask,1], c=cd, s=7, alpha=0.5, zorder=2)
                if pred_uv_ps is not None:
                    ax.scatter(pred_uv_ps[mask,0], pred_uv_ps[mask,1], c=cp, s=18, marker='+', linewidths=1.3, zorder=4)
                    _draw_arrows(ax, d_uv[mask], pred_uv_ps[mask], cp, n_max=10)
        # obj BB: red around dist, cyan dashed shifted by mean predicted offset
        if is_obj.any() and not gt_only and pred_uv_ps is not None:
            d_obj = d_uv[is_obj]; p_obj = pred_uv_ps[is_obj]
            x0, y0 = float(d_obj[:,0].min()), float(d_obj[:,1].min())
            x1, y1 = float(d_obj[:,0].max()), float(d_obj[:,1].max())
            dx = float((p_obj - d_obj)[:,0].mean()); dy = float((p_obj - d_obj)[:,1].mean())
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec='red',  lw=1.2, zorder=5))
            ax.add_patch(plt.Rectangle((x0+dx, y0+dy), x1-x0, y1-y0, fill=False, ec='cyan', lw=1.2, linestyle='--', zorder=5))
            cx, cy = (x0+x1)/2, (y0+y1)/2
            arr = ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx, cy),
                              arrowprops=dict(arrowstyle='-|>,head_width=0.45,head_length=0.6',
                                              color='white', lw=2.2, mutation_scale=14),
                              zorder=7)
            arr.arrow_patch.set_path_effects([
                pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
        ax.set_xlim(0,64); ax.set_ylim(64,0); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=5, loc='upper right', framealpha=0.6, ncol=2)
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=128)
        plt.close(fig); buf.seek(0)
        png = base64.b64encode(buf.read()).decode()
        pred_uv = pred_uv_ps
    elif gd_mode:
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


def _parse_best_epoch(log_path: Path) -> dict:
    """Parse train.log to get metrics at the best val NLL epoch."""
    result = {"best_nll": None, "best_obj_nll": None, "best_mse": None}
    if not log_path.exists():
        return result
    best_nll, best_line = None, ""
    for line in log_path.read_text().splitlines():
        m = re.search(r"Best val NLL: ([\d.]+)", line)
        if m:
            best_nll = float(m.group(1))
        # keep track of the saved-epoch line just before each "↳ saved"
        if "↳ saved" in line and best_line:
            pass  # best_line already set
        if re.search(r"\[\s*\d+/\d+\]", line):
            best_line = line
    result["best_nll"] = best_nll
    # find the epoch line matching best_nll
    if best_nll is not None:
        for line in log_path.read_text().splitlines():
            m = re.search(r"val nll=([+-]?[\d.]+)\(obj=([+-]?[\d.]+).*?mse=([\d.]+)", line)
            if m and abs(float(m.group(1)) - best_nll) < 1e-3:
                result["best_obj_nll"] = float(m.group(2))
                result["best_mse"]     = float(m.group(3))
    return result


@app.route("/api/experiments")
def api_experiments():
    exps = []
    for cfg_path in sorted(Path("experiments").glob("*/config.py")):
        name = cfg_path.parent.name
        try:
            cfg     = _load_exp_cfg(name)
            metrics = _parse_best_epoch(cfg_path.parent / "train.log")
            exps.append({
                "name":          name,
                "n_layers":      cfg.get("n_layers"),
                "random_depths": cfg.get("random_depths", False),
                "cross_temp":    cfg.get("cross_temp", 1.0),
                "best_nll":      metrics["best_nll"],
                "best_obj_nll":  metrics["best_obj_nll"],
                "best_mse":      metrics["best_mse"],
                "ckpt_exists":   (cfg_path.parent / "best_model.pt").exists(),
            })
        except Exception:
            pass
    return jsonify(exps)


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


@app.route("/experiments/<path:filename>")
def serve_experiment_file(filename):
    return send_from_directory("experiments", filename)


@app.route("/api/report_list")
def api_report_list():
    reports = []
    for p in sorted(Path("experiments").glob("*/report.html")):
        name = p.parent.name
        log  = p.parent / "train.log"
        best_nll = None
        if log.exists():
            for line in log.read_text().splitlines():
                m = re.search(r"Best val NLL: ([\d.]+)", line)
                if m:
                    best_nll = float(m.group(1))
        reports.append({"name": name, "url": f"/experiments/{name}/report.html",
                         "best_nll": best_nll})
    reports.sort(key=lambda x: (x["best_nll"] or 99))
    return jsonify(reports)


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
