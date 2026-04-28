"""Regenerate val vis for a PS experiment: 48 samples with BB overlay.

Red BB = around distorted obj points.
Cyan BB (dashed) = same size, shifted by the mean of per-point predicted offsets.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys, torch, importlib.util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from datasets.pandaset import PandaSetCalibDataset
from datasets.pandaset_lazy import PandaSetCalibDatasetLazy
from models.model_depth import CalibNetDepth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cfg(exp_dir: Path) -> dict:
    spec = importlib.util.spec_from_file_location("_cfg", exp_dir / "config.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.CFG


def main(exp: str, n_vis: int = 48, cache: str = '/tmp/pandaset_cache.pt',
         split: str = 'val', idx_range: tuple | None = None,
         min_shift_px: float = 0.0, min_obj_bb_px: float = 22.0,
         max_scan: int = 5000):
    exp_dir = Path("experiments") / exp
    c = load_cfg(exp_dir)
    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"], self_first=False,
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False)).to(DEVICE)
    model.load_state_dict(torch.load(exp_dir / "best_model.pt",
                                     map_location=DEVICE, weights_only=True))
    model.eval()

    # Same perturbation as training; shift filter naturally picks close/large objects.
    # Auto-detect: directory → lazy disk-backed; file → legacy in-memory.
    if Path(cache).is_dir():
        ds = PandaSetCalibDatasetLazy(cache, split=split)
    else:
        ds = PandaSetCalibDataset(cache, split=split)
    lo, hi = idx_range if idx_range else (0, len(ds))
    hi = min(hi, len(ds))
    print(f"[{exp}] split={split} range=[{lo},{hi})  scanning for {n_vis} large-obj samples"
          f"  (min shift={min_shift_px}px, min bb={min_obj_bb_px}px)")

    # Half the tiles are object-centered (large obj BB), half are random crops
    # (sampled uniformly regardless of whether an obj is present).
    n_obj_tiles = n_vis // 2
    n_rand_tiles = n_vis - n_obj_tiles
    import random as _r
    scan_idxs = list(range(lo, hi))
    _r.Random(0).shuffle(scan_idxs)

    picked_obj, picked_rand = [], []
    used = set()
    for scan_i, idx in enumerate(scan_idxs[:max_scan]):
        if len(picked_obj) >= n_obj_tiles: break
        img, true_uvd, dist_uvd = ds[idx]
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if not is_obj.any(): continue
        d_obj = dist_uvd[is_obj, :2].numpy()
        t_obj = true_uvd[is_obj, :2].numpy()
        bb_w = float(d_obj[:,0].max() - d_obj[:,0].min())
        bb_h = float(d_obj[:,1].max() - d_obj[:,1].min())
        shift_mag = float(((t_obj - d_obj)**2).sum(1).mean()**0.5)
        if max(bb_w, bb_h) < min_obj_bb_px: continue
        if shift_mag < min_shift_px: continue
        picked_obj.append((idx, img, true_uvd, dist_uvd)); used.add(idx)

    rand_pool = _r.Random(1).sample(scan_idxs, min(len(scan_idxs), n_rand_tiles * 4))
    for idx in rand_pool:
        if len(picked_rand) >= n_rand_tiles: break
        if idx in used: continue
        img, true_uvd, dist_uvd = ds[idx]
        picked_rand.append((idx, img, true_uvd, dist_uvd)); used.add(idx)

    picked = picked_obj + picked_rand
    print(f"  picked {len(picked_obj)} obj + {len(picked_rand)} random (total {len(picked)})")
    val_ds = ds  # kept for legacy naming
    idxs = [p[0] for p in picked]
    preload = {p[0]: p[1:] for p in picked}

    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    for old in vis_dir.glob("val_*.png"): old.unlink()

    for vi, idx in enumerate(idxs):
        img, true_uvd, dist_uvd = preload[idx]
        with torch.no_grad():
            pad = torch.zeros(1, true_uvd.shape[0], dtype=torch.bool, device=DEVICE)
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                           key_padding_mask=pad)[0].cpu().float()
        pred_uv = (dist_uvd[:, :2] + params[:, :2]).numpy()
        true_uv = true_uvd[:, :2].numpy()
        dist_uv = dist_uvd[:, :2].numpy()
        is_obj  = dist_uvd[:, 3].numpy() > 0.5

        err_b_obj = float(abs(dist_uv[is_obj]  - true_uv[is_obj]).sum(1).mean())  if is_obj.any()  else float('nan')
        err_a_obj = float(abs(pred_uv[is_obj]  - true_uv[is_obj]).sum(1).mean())  if is_obj.any()  else float('nan')
        err_b_bg  = float(abs(dist_uv[~is_obj] - true_uv[~is_obj]).sum(1).mean()) if (~is_obj).any() else float('nan')
        err_a_bg  = float(abs(pred_uv[~is_obj] - true_uv[~is_obj]).sum(1).mean()) if (~is_obj).any() else float('nan')

        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        ax.imshow(img.permute(1,2,0).numpy())

        # per-point correction field (dist -> pred) + GT × marker
        u = pred_uv[:,0] - dist_uv[:,0]
        v = pred_uv[:,1] - dist_uv[:,1]
        if (~is_obj).any():
            ax.quiver(dist_uv[~is_obj,0], dist_uv[~is_obj,1], u[~is_obj], v[~is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='deepskyblue', width=0.004, headwidth=3.5, headlength=4,
                      alpha=0.55, zorder=2)
        if is_obj.any():
            ax.quiver(dist_uv[is_obj,0], dist_uv[is_obj,1], u[is_obj], v[is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='orange', width=0.006, headwidth=3.5, headlength=4,
                      alpha=0.95, zorder=4)
        ax.scatter(true_uv[is_obj,0],  true_uv[is_obj,1],  c='lime',   s=14, marker='x', linewidths=0.9, label='GT obj', zorder=5)
        ax.scatter(true_uv[~is_obj,0], true_uv[~is_obj,1], c='yellow', s=6,  marker='x', linewidths=0.5, alpha=0.5, label='GT bg', zorder=3)

        # faint obj BB + mean-shift summary arrow (secondary info)
        if is_obj.any():
            d_obj = dist_uv[is_obj]; p_obj = pred_uv[is_obj]
            x0, y0 = float(d_obj[:,0].min()), float(d_obj[:,1].min())
            x1, y1 = float(d_obj[:,0].max()), float(d_obj[:,1].max())
            dx, dy = float((p_obj - d_obj)[:,0].mean()), float((p_obj - d_obj)[:,1].mean())
            pad = 2.0
            ax.add_patch(plt.Rectangle((x0-pad, y0-pad), (x1-x0)+2*pad, (y1-y0)+2*pad,
                                       fill=False, ec='red', lw=1.4, alpha=1.0, zorder=6))
            ax.add_patch(plt.Rectangle((x0-pad+dx, y0-pad+dy), (x1-x0)+2*pad, (y1-y0)+2*pad,
                                       fill=False, ec='cyan', lw=1.4, linestyle='--', zorder=6))
            cx, cy = (x0+x1)/2, (y0+y1)/2
            arr = ax.annotate("", xy=(cx+dx, cy+dy), xytext=(cx, cy),
                              arrowprops=dict(arrowstyle='-|>,head_width=0.45,head_length=0.6',
                                              color='white', lw=2.2, mutation_scale=14),
                              zorder=7)
            arr.arrow_patch.set_path_effects([
                pe.Stroke(linewidth=4.5, foreground='black'),
                pe.Normal()])
        else:
            dx = dy = 0.0

        ax.set_title(f"obj:{err_b_obj:.2f}→{err_a_obj:.2f}  bg:{err_b_bg:.2f}→{err_a_bg:.2f}px  mean obj shift:({dx:+.2f},{dy:+.2f})"
                     if is_obj.any() else
                     f"bg:{err_b_bg:.2f}→{err_a_bg:.2f}px", fontsize=6)
        ax.axis('off')
        ax.legend(fontsize=4, loc='upper right', framealpha=0.5, ncol=2)
        plt.tight_layout(pad=0.2)
        plt.savefig(vis_dir / f"val_{vi:02d}.png", dpi=96, bbox_inches='tight')
        plt.close(fig)

    # Regenerate report.html referencing all images
    log = exp_dir / "train.log"
    best_nll = None
    if log.exists():
        import re
        for line in log.read_text().splitlines():
            m = re.search(r"Best val NLL: ([\d.]+)", line)
            if m: best_nll = float(m.group(1))
    vis_imgs = "\n".join(
        f'      <img src="vis/val_{i:02d}.png" style="width:200px;margin:4px;border:1px solid #333;">'
        for i in range(len(idxs)))
    report = f"""<!doctype html><meta charset="utf-8">
<title>{exp} — report</title>
<style>body{{background:#0d1117;color:#eee;font-family:sans-serif;padding:20px;}} pre{{background:#161b22;padding:10px;border-radius:6px;}} h2{{border-bottom:1px solid #333;padding-bottom:4px;}}</style>
<h1>{exp}</h1>
<p>best val NLL: <b>{best_nll}</b></p>
<pre>{chr(10).join(f'{k:<15}= {v!r}' for k,v in c.items())}</pre>
<h2>Curves</h2>
<img src="vis/curves.png" style="max-width:800px;">
<h2>Val samples (BB: red=dist, cyan=shifted by mean obj offset)</h2>
<div style="display:flex;flex-wrap:wrap;">
{vis_imgs}
</div>
"""
    (exp_dir / "report.html").write_text(report)
    print(f"Saved {len(idxs)} vis → {vis_dir}  |  report → {exp_dir}/report.html")


if __name__ == "__main__":
    exp   = sys.argv[1] if len(sys.argv) > 1 else "ps_v9_overfit500_1k"
    n     = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    split = sys.argv[3] if len(sys.argv) > 3 else 'val'
    rng   = None
    if len(sys.argv) > 5:
        rng = (int(sys.argv[4]), int(sys.argv[5]))
    main(exp, n, split=split, idx_range=rng)
