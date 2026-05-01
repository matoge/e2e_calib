"""V3 cache visualization — v9 style minus rectangles (multi-obj crops).

  GT obj : lime ×
  GT bg  : yellow ×
  dist→pred quiver: deepskyblue (bg), orange (obj)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import importlib.util, pickle
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.pandaset_full import PandaSetCalibDatasetFull
from datasets.pandaset import USEFUL_LABELS
from models.model_depth import CalibNetDepth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PANDASET_SRC = '/mnt/nvme6t/pandaset'


class PandaSetCalibDatasetFullRepaired(PandaSetCalibDatasetFull):
    """Override _load_inst: replace cache's frame-shifted lidar+cuboids with the
    source .pkl for the correct (scene, frame). The cache was built with a
    sorted(.pkl + .pkl.gz) bug that off-by-2x the lidar/cuboid frame index.
    Image and pose were correct → swap pts/cuboids with the matching frame's
    source data and the projection lines up."""

    def _load_inst(self, idx: int) -> dict:
        inst = super()._load_inst(idx)
        scene = inst['scene']; frame = int(inst['frame'])
        ld = Path(PANDASET_SRC) / scene / 'lidar' / f'{frame:02d}.pkl'
        cb = Path(PANDASET_SRC) / scene / 'annotations' / 'cuboids' / f'{frame:02d}.pkl'
        if ld.exists():
            df = pickle.load(open(ld, 'rb'))
            if 'd' in df.columns:
                df = df[df['d'] == 0]
            pts_world = df[['x', 'y', 'z']].values.astype(np.float32)
            # Re-filter to in-image points using inst's (correct-frame) pose
            T_gt = inst['T_gt'].numpy(); K = inst['K_full'].numpy()
            IH = int(inst['img'].shape[-2]); IW = int(inst['img'].shape[-1])
            homo = np.column_stack([pts_world, np.ones(len(pts_world))])
            pcam = (T_gt @ homo.T)[:3].T
            z = pcam[:, 2]
            uv = ((K @ pcam.T)[:2] / np.maximum(pcam[:, 2:].T, 1e-6)).T
            vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
            inst['pts'] = torch.from_numpy(pts_world[vis])
            # invalidate stale cached projections so __getitem__ recomputes
            for k in ('uv_full', 'z_cam', 'is_obj'):
                inst.pop(k, None)
        if cb.exists():
            df = pickle.load(open(cb, 'rb'))
            if 'label' in df.columns:
                df = df[df['label'].isin(USEFUL_LABELS)]
            cubs = []
            for _, obj in df.iterrows():
                cubs.append(dict(
                    pos=np.array([obj['position.x'], obj['position.y'], obj['position.z']], dtype=np.float32),
                    dims=np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']], dtype=np.float32),
                    yaw=float(obj['yaw']),
                    label=str(obj['label']) if 'label' in obj else '',
                ))
            inst['cuboids'] = cubs
        return inst


def load_cfg(exp_dir: Path) -> dict:
    cfg = {"img_size": 64, "in_channels": 3, "n_layers": 2,
           "self_first": False, "use_convnext": False, "use_frustum": True,
           "deform_mode": "none"}
    p = exp_dir / "config.py"
    if p.exists():
        spec = importlib.util.spec_from_file_location("_cfg", p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        cfg.update(m.CFG)
    return cfg


def main(exp_name="ps_v3_full_e100", n_vis=48,
         cache="/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full",
         seed=0, min_obj_bb_px=22.0, max_scan=4000):
    exp_dir = Path("experiments") / exp_name
    vis_dir = exp_dir / "vis"; vis_dir.mkdir(exist_ok=True)
    for old in vis_dir.glob("val_*.png"): old.unlink()

    c = load_cfg(exp_dir)
    model = CalibNetDepth(img_size=c["img_size"], in_channels=c["in_channels"],
                          n_layers=c["n_layers"],
                          self_first=c.get("self_first", False),
                          use_convnext=c.get("use_convnext", False),
                          use_frustum=c.get("use_frustum", False),
                          deform_mode=c.get("deform_mode", "none")).to(DEVICE)
    sd = torch.load(exp_dir / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd if 'state_dict' not in sd else sd['state_dict'], strict=False)
    model.eval()

    ds = PandaSetCalibDatasetFull(cache, split='val', img_size=c["img_size"],
                                   min_crop_px=128, max_crop_px=512, oversample=1)

    n_obj_tiles  = n_vis // 2
    n_rand_tiles = n_vis - n_obj_tiles
    import random as _r
    scan_idxs = list(range(len(ds)))
    _r.Random(seed).shuffle(scan_idxs)

    def _inst_name(i):
        return ds.fnames[i % len(ds.fnames)]

    picked_obj, picked_rand = [], []
    used = set()
    for idx in scan_idxs[:max_scan]:
        if len(picked_obj) >= n_obj_tiles: break
        try:
            sample = ds[idx]
        except Exception:
            continue
        img, true_uvd, dist_uvd, vfp = sample
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if not is_obj.any(): continue
        d_obj = dist_uvd[is_obj, :2].numpy()
        bb_w = float(d_obj[:, 0].max() - d_obj[:, 0].min())
        bb_h = float(d_obj[:, 1].max() - d_obj[:, 1].min())
        if max(bb_w, bb_h) < min_obj_bb_px: continue
        picked_obj.append((idx, sample)); used.add(idx)

    rand_pool = _r.Random(seed + 1).sample(scan_idxs, min(len(scan_idxs), n_rand_tiles * 4))
    for idx in rand_pool:
        if len(picked_rand) >= n_rand_tiles: break
        if idx in used: continue
        try:
            picked_rand.append((idx, ds[idx])); used.add(idx)
        except Exception:
            continue

    picked = picked_obj + picked_rand
    print(f"  picked {len(picked_obj)} obj + {len(picked_rand)} random (total {len(picked)})")

    import json
    manifest = {}
    S = int(c["img_size"])
    saved = 0
    for vi, (idx, (img, true_uvd, dist_uvd, vfp)) in enumerate(picked):
        manifest[f"val_{vi:02d}"] = dict(ds_idx=int(idx), inst_file=_inst_name(idx))
        Nmax = true_uvd.shape[0]
        pad = torch.zeros(1, Nmax, dtype=torch.bool, device=DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            params = model(img.unsqueeze(0).to(DEVICE),
                           dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                           key_padding_mask=pad, vfp=vfp.view(1).to(DEVICE))
        p = params[0].float().cpu().numpy()
        true_uv = true_uvd[:, :2].numpy()
        dist_uv = dist_uvd[:, :2].numpy()
        pred_uv = dist_uv + p[:, :2]
        is_obj  = dist_uvd[:, 3].numpy() > 0.5

        err_b_obj = float(abs(dist_uv[is_obj]  - true_uv[is_obj]).sum(1).mean())  if is_obj.any()  else float('nan')
        err_a_obj = float(abs(pred_uv[is_obj]  - true_uv[is_obj]).sum(1).mean())  if is_obj.any()  else float('nan')
        err_b_bg  = float(abs(dist_uv[~is_obj] - true_uv[~is_obj]).sum(1).mean()) if (~is_obj).any() else float('nan')
        err_a_bg  = float(abs(pred_uv[~is_obj] - true_uv[~is_obj]).sum(1).mean()) if (~is_obj).any() else float('nan')

        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        ax.imshow(np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1))

        u_corr = pred_uv[:, 0] - dist_uv[:, 0]
        v_corr = pred_uv[:, 1] - dist_uv[:, 1]
        if (~is_obj).any():
            ax.quiver(dist_uv[~is_obj, 0], dist_uv[~is_obj, 1],
                      u_corr[~is_obj], v_corr[~is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='deepskyblue', width=0.004, headwidth=3.5, headlength=4,
                      alpha=0.55, zorder=2)
        if is_obj.any():
            ax.quiver(dist_uv[is_obj, 0], dist_uv[is_obj, 1],
                      u_corr[is_obj], v_corr[is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='orange', width=0.006, headwidth=3.5, headlength=4,
                      alpha=0.95, zorder=4)
        ax.scatter(true_uv[is_obj, 0],  true_uv[is_obj, 1],
                   c='lime',   s=14, marker='x', linewidths=0.9, label='GT obj', zorder=5)
        ax.scatter(true_uv[~is_obj, 0], true_uv[~is_obj, 1],
                   c='yellow', s=6,  marker='x', linewidths=0.5, alpha=0.5, label='GT bg', zorder=3)

        if is_obj.any():
            d_obj = dist_uv[is_obj]; p_obj = pred_uv[is_obj]
            dx = float((p_obj - d_obj)[:, 0].mean())
            dy = float((p_obj - d_obj)[:, 1].mean())
        else:
            dx = dy = 0.0

        ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.axis('off')
        title = (f"obj:{err_b_obj:.2f}→{err_a_obj:.2f}  bg:{err_b_bg:.2f}→{err_a_bg:.2f}px  mean obj shift:({dx:+.2f},{dy:+.2f})"
                 if is_obj.any() else
                 f"bg:{err_b_bg:.2f}→{err_a_bg:.2f}px")
        ax.set_title(title, fontsize=6)
        ax.legend(fontsize=4, loc='upper right', framealpha=0.5, ncol=2)
        plt.tight_layout(pad=0.2)
        plt.savefig(vis_dir / f"val_{vi:02d}.png", dpi=96, bbox_inches='tight')
        plt.close(fig)
        saved += 1

    (vis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log = exp_dir / "train.log"
    best_nll = None
    if log.exists():
        import re
        for line in log.read_text().splitlines():
            m = re.search(r"Best val NLL: ([\d.]+)", line)
            if m: best_nll = float(m.group(1))
    vis_imgs = "\n".join(
        f'      <img src="vis/val_{i:02d}.png" style="width:200px;margin:4px;border:1px solid #333;">'
        for i in range(saved))
    report = f"""<!doctype html><meta charset="utf-8">
<title>{exp_name} — report</title>
<style>body{{background:#0d1117;color:#eee;font-family:sans-serif;padding:20px;}} pre{{background:#161b22;padding:10px;border-radius:6px;}} h2{{border-bottom:1px solid #333;padding-bottom:4px;}}</style>
<h1>{exp_name}</h1>
<p>best val NLL: <b>{best_nll}</b></p>
<pre>{chr(10).join(f'{k:<15}= {v!r}' for k,v in c.items())}</pre>
<h2>Val samples — GT obj=lime ×, GT bg=yellow ×, quiver=dist→pred (orange obj / deepskyblue bg)</h2>
<div style="display:flex;flex-wrap:wrap;">
{vis_imgs}
</div>
"""
    (exp_dir / "report.html").write_text(report)
    print(f"saved {saved} vis → {vis_dir}  |  report → {exp_dir}/report.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_name', nargs='?', default='ps_v3_full_e100')
    ap.add_argument('--n-vis', type=int, default=48)
    ap.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full')
    args = ap.parse_args()
    main(args.exp_name, args.n_vis, args.cache)
