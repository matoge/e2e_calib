"""Evaluate experiments/all_v2 (joint NS+PS+WM) on each dataset's held-out val
split, dump per-dataset metrics + 6 sample tiles per dataset.

Reconstructs the exact split from train_all_v1.py:
  seed=42, shuffle, val = first 500, train = next 20 000.
"""
import json, random as _r
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from torch.utils.data import ConcatDataset, Subset, DataLoader

from dataset_nuscenes import NuScenesCalibDataset
from dataset_pandaset import PandaSetCalibDataset
from dataset_waymo     import WaymoCalibDataset
from model_depth import CalibNetDepth
from model_cov import gaussian2d_nll

BG      = "#f6f4ed"
CARD    = "#ffffff"
INK     = "#0f0f0e"
INK_DIM = "#6b6a63"
LINE    = "#d9d6cd"
ACCENT  = "#c13c14"
ACCENT2 = "#174734"
OBJ_COL = "#ffb133"
BG_COL  = "#79c4ff"

EXP_DIR  = Path("experiments/all_v2")
VIS_DIR  = EXP_DIR / "vis"; VIS_DIR.mkdir(parents=True, exist_ok=True)
CKPT     = EXP_DIR / "best_model.pt"
NS_CACHE = '/tmp/nuscenes_static_cache.pt'
PS_CACHE = '/tmp/pandaset_cache.pt'
WM_CACHE = '/tmp/waymo_v2_cache.pt'
N_TRAIN_PER = 20000
N_VAL_PER   = 500
SEED        = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_mixed(batch):
    imgs, true_uvds, dist_uvds = zip(*batch)
    max_n = max(t.shape[0] for t in true_uvds)
    out_t = torch.zeros(len(batch), max_n, true_uvds[0].shape[1])
    out_d = torch.zeros(len(batch), max_n, dist_uvds[0].shape[1])
    mask  = torch.ones(len(batch), max_n, dtype=torch.bool)
    for i, (t, d) in enumerate(zip(true_uvds, dist_uvds)):
        n = t.shape[0]
        out_t[i, :n] = t; out_d[i, :n] = d; mask[i, :n] = False
    return torch.stack(imgs), out_t, out_d, mask


@torch.no_grad()
def eval_ds(model, val_sub):
    loader = DataLoader(val_sub, batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True,
                        collate_fn=collate_mixed)
    tot = dict(nll=0., mse=0., n=0,
               obj_nll=0., obj_mse=0., obj_n=0,
               bg_nll=0.,  bg_mse=0.,  bg_n=0)
    for imgs, true_uvd, dist_uvd, pad in loader:
        imgs     = imgs.to(DEVICE)
        true_uvd = true_uvd.to(DEVICE)
        dist_uvd = dist_uvd.to(DEVICE)
        pad      = pad.to(DEVICE)
        gt       = true_uvd[..., :2] - dist_uvd[..., :2]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            params = model(imgs, dist_uvd[..., :3], key_padding_mask=pad)
            valid  = ~pad
            loss   = gaussian2d_nll(params[valid], gt[valid])
        mse    = (params[valid][..., :2] - gt[valid]).norm(dim=-1).mean().item()
        is_obj = valid & (dist_uvd[..., 3] > 0.5)
        is_bg  = valid & (dist_uvd[..., 3] < 0.5)
        tot['nll'] += loss.item(); tot['mse'] += mse; tot['n'] += 1
        if is_obj.any():
            tot['obj_nll'] += gaussian2d_nll(params[is_obj], gt[is_obj]).item()
            tot['obj_mse'] += (params[is_obj][..., :2] - gt[is_obj]).norm(dim=-1).mean().item()
            tot['obj_n']  += 1
        if is_bg.any():
            tot['bg_nll']  += gaussian2d_nll(params[is_bg],  gt[is_bg]).item()
            tot['bg_mse']  += (params[is_bg][..., :2] - gt[is_bg]).norm(dim=-1).mean().item()
            tot['bg_n']   += 1
    return dict(
        nll      = tot['nll']    / max(tot['n'], 1),
        mse      = tot['mse']    / max(tot['n'], 1),
        obj_nll  = tot['obj_nll']/ max(tot['obj_n'], 1),
        obj_mse  = tot['obj_mse']/ max(tot['obj_n'], 1),
        bg_nll   = tot['bg_nll'] / max(tot['bg_n'],  1),
        bg_mse   = tot['bg_mse'] / max(tot['bg_n'],  1),
    )


@torch.no_grad()
def render_tile(model, ds, idx, out_path, title_prefix):
    img, true_uvd, dist_uvd = ds[idx]
    pad = torch.zeros(1, true_uvd.shape[0], dtype=torch.bool, device=DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        params = model(img.unsqueeze(0).to(DEVICE),
                       dist_uvd.unsqueeze(0).to(DEVICE)[..., :3],
                       key_padding_mask=pad)[0].cpu().float()
    pred_uv = (dist_uvd[:, :2] + params[:, :2]).numpy()
    dist_uv = dist_uvd[:, :2].numpy()
    true_uv = true_uvd[:, :2].numpy()
    is_obj  = dist_uvd[:, 3].numpy() > 0.5

    obj_mask = is_obj
    bg_mask  = ~is_obj
    if obj_mask.sum() > 0:
        eb_o = float(np.linalg.norm(dist_uv[obj_mask] - true_uv[obj_mask], axis=1).mean())
        ea_o = float(np.linalg.norm(pred_uv[obj_mask] - true_uv[obj_mask], axis=1).mean())
        mean_shift = (pred_uv[obj_mask] - dist_uv[obj_mask]).mean(0)
    else:
        eb_o = ea_o = float('nan'); mean_shift = np.zeros(2)
    eb_b = float(np.linalg.norm(dist_uv[bg_mask] - true_uv[bg_mask], axis=1).mean()) if bg_mask.sum() else float('nan')
    ea_b = float(np.linalg.norm(pred_uv[bg_mask] - true_uv[bg_mask], axis=1).mean()) if bg_mask.sum() else float('nan')

    S = img.shape[-1]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    ax.imshow(img.permute(1, 2, 0).numpy(), extent=[0, S, S, 0])

    u = pred_uv[:, 0] - dist_uv[:, 0]
    v = pred_uv[:, 1] - dist_uv[:, 1]
    if bg_mask.any():
        ax.quiver(dist_uv[bg_mask, 0], dist_uv[bg_mask, 1],
                  u[bg_mask], v[bg_mask],
                  angles='xy', scale_units='xy', scale=1,
                  color=BG_COL, width=0.004,
                  headwidth=3.5, headlength=4.5, alpha=0.45, zorder=3)
    if obj_mask.any():
        ax.quiver(dist_uv[obj_mask, 0], dist_uv[obj_mask, 1],
                  u[obj_mask], v[obj_mask],
                  angles='xy', scale_units='xy', scale=1,
                  color=OBJ_COL, width=0.007,
                  headwidth=3.5, headlength=4.5, alpha=0.7, zorder=5)
    if obj_mask.any():
        d_obj = dist_uv[obj_mask]
        x0, y0 = float(d_obj[:, 0].min()), float(d_obj[:, 1].min())
        x1, y1 = float(d_obj[:, 0].max()), float(d_obj[:, 1].max())
        padbb = 2.0
        ax.add_patch(plt.Rectangle((x0 - padbb, y0 - padbb),
                                   (x1 - x0) + 2 * padbb, (y1 - y0) + 2 * padbb,
                                   fill=False, ec=ACCENT, lw=1.8, zorder=6))
        ax.add_patch(plt.Rectangle((x0 - padbb + mean_shift[0], y0 - padbb + mean_shift[1]),
                                   (x1 - x0) + 2 * padbb, (y1 - y0) + 2 * padbb,
                                   fill=False, ec='#2fcf7a', lw=1.8,
                                   linestyle='--', zorder=6))

    txt = f"{title_prefix}  obj {eb_o:.1f}→{ea_o:.2f}  bg {eb_b:.1f}→{ea_b:.2f} px"
    ax.text(S*0.022, S*0.955, txt, color='white',
            fontsize=10, family='monospace', fontweight='700',
            bbox=dict(facecolor=ACCENT, edgecolor='none', pad=4), zorder=8)

    ax.set_xlim(0, S); ax.set_ylim(S, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, facecolor=BG, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return dict(eb_o=eb_o, ea_o=ea_o, eb_b=eb_b, ea_b=ea_b,
                n_obj=int(obj_mask.sum()), n_bg=int(bg_mask.sum()))


def pick_diverse(val_sub, seed=0, n=12, min_obj=12):
    """Return indices into val_sub that have decent-sized obj masks."""
    rng = _r.Random(seed)
    order = list(range(len(val_sub))); rng.shuffle(order)
    picks = []
    for i in order:
        try:
            _, true_uvd, dist_uvd = val_sub[i]
        except Exception:
            continue
        n_obj = int((dist_uvd[:, 3] > 0.5).sum())
        if n_obj < min_obj: continue
        picks.append(i)
        if len(picks) >= n: break
    return picks


def main():
    print(f"loading {CKPT} ...")
    model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                          self_first=False, use_convnext=True,
                          use_frustum=True).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    datasets = {}
    for name, cls, cache in [
        ('NS', NuScenesCalibDataset, NS_CACHE),
        ('PS', PandaSetCalibDataset, PS_CACHE),
        ('WM', WaymoCalibDataset,    WM_CACHE),
    ]:
        tr = cls(cache, split='train')
        va = cls(cache, split='val')
        datasets[name] = ConcatDataset([tr, va])
        print(f"{name}: {len(datasets[name])} instances")

    rng = _r.Random(SEED)
    val_subs = {}
    for name, ds in datasets.items():
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        val_part = idxs[:N_VAL_PER]
        val_subs[name] = Subset(ds, val_part)

    metrics = {}
    for name, sub in val_subs.items():
        print(f"\n── {name} val ({len(sub)} samples) ──")
        m = eval_ds(model, sub)
        metrics[name] = m
        print(f"  nll={m['nll']:+.3f} (obj={m['obj_nll']:+.3f} bg={m['bg_nll']:+.3f})  "
              f"mse={m['mse']:.3f} (obj={m['obj_mse']:.3f} bg={m['bg_mse']:.3f})")

    N_TILES = 6
    tile_info = {}
    for name, sub in val_subs.items():
        picks = pick_diverse(sub, seed=11 + ord(name[0]), n=N_TILES, min_obj=12)
        print(f"\n{name} tile picks: {picks}")
        infos = []
        for k, i in enumerate(picks):
            out = VIS_DIR / f"{name.lower()}_{k:02d}.png"
            info = render_tile(model, sub, i, out, title_prefix=name)
            infos.append(info)
            print(f"  saved {out.name}  obj {info['eb_o']:.2f}→{info['ea_o']:.2f}  bg {info['eb_b']:.2f}→{info['ea_b']:.2f}")
        tile_info[name] = infos

    out_json = EXP_DIR / "eval_metrics.json"
    out_json.write_text(json.dumps(dict(
        generated_at = datetime.now().isoformat(timespec='seconds'),
        ckpt         = str(CKPT),
        seed         = SEED,
        n_val_per    = N_VAL_PER,
        metrics      = metrics,
        tiles        = tile_info,
    ), indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
