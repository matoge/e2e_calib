"""Render a grid of zoomed-in tiles for the v0.2 blog.

For each tile, draw the parent-tile crop with three point sets overlaid:
  yellow = GT LiDAR projection
  red    = perturbed projection (the input to the σ-head)
  lime   = BA-corrected projection (after δ̂)

Output: docs/assets/2026-05-22_v02_raw_frame/tile_zoom_grid.png
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.data.adapters.kamikado import TILE_LAYOUT
from scripts.data.tile_cutter import frame_to_tiles
from scripts.util.projection import project_kannala
from services.calib_api.raw_pipeline import (
    load_kamikado_frame_from_disk,
    solve_from_calib_frame,
)


SCENE = Path("/raw/kamikado/scenes/points_ip664_D_20260226_224648_d005_3000_3020")
FRAME = 0
YPR = np.array([0.30, -0.20, 0.50], dtype=np.float64)
T = np.array([0.030, -0.020, 0.040], dtype=np.float64)
OUT = REPO / "docs/assets/2026-05-22_v02_raw_frame/tile_zoom_grid.png"
N_PICK = 6  # how many tiles to show


def main():
    cfg = ess._load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=ess.CACHE,
        split="val",
        img_size=cfg["img_size"],
        min_crop_px=cfg["min_crop_px"],
        max_crop_px=cfg["max_crop_px"],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1,
        grid_n=cfg.get("grid_n", 16),
        center_band=0.0, preload=False,
    )
    model = ess._build_model(cfg).to(ess.DEVICE)
    sd = torch.load(ess.CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict") or sd.get("model")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cf = load_kamikado_frame_from_disk(SCENE, FRAME)
    delta, B, n_tiles, n_subcrops = solve_from_calib_frame(
        model, ds, cf, ypr_target=YPR, t_target=T, cs=256, n_per_inst=4)
    print(f"[solve] tiles={n_tiles} sub-crops={n_subcrops} B={B}")
    delta_np = delta.detach().cpu().numpy().astype(np.float64)
    print(f"  δ̂ ω={delta_np[:3]} t={delta_np[3:]}")

    tiles = frame_to_tiles(cf, **TILE_LAYOUT, min_pts=8)
    # Pick well-spread tiles by point count: top-N descending.
    tiles_sorted = sorted(tiles, key=lambda t: -int(t["pts"].shape[0]))
    pick = tiles_sorted[:N_PICK]

    R_pert = Rotation.from_euler("zyx", YPR, degrees=True).as_matrix()
    R_d = Rotation.from_rotvec(np.deg2rad(delta_np[:3])).as_matrix()

    K_full = cf.K.astype(np.float64)
    dist = cf.dist.astype(np.float64)

    cols = 3
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6), dpi=140)
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, inst in enumerate(pick):
        ax = axes[i // cols, i % cols]
        pts_cam = inst["pts"].numpy().astype(np.float64)
        u0 = float(inst["tile_u0"])
        v0 = float(inst["tile_v0"])
        img = np.array(Image.open(io.BytesIO(inst["jpg_bytes"])).convert("RGB"))
        H, W = img.shape[:2]

        pts_pert = (pts_cam - T) @ R_pert
        pts_corr = pts_pert @ R_d.T + delta_np[3:]

        def proj(P):
            uv = project_kannala(P, K_full, dist)
            return uv - np.array([u0, v0])

        uv_gt = proj(pts_cam)
        uv_pert = proj(pts_pert)
        uv_corr = proj(pts_corr)

        def in_view(uv, z):
            return ((z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H))

        m_g = in_view(uv_gt, pts_cam[:, 2])
        m_p = in_view(uv_pert, pts_pert[:, 2])
        m_c = in_view(uv_corr, pts_corr[:, 2])

        e_p = float(np.linalg.norm(uv_pert[m_g & m_p] - uv_gt[m_g & m_p], axis=1).mean()) if (m_g & m_p).any() else float("nan")
        e_c = float(np.linalg.norm(uv_corr[m_g & m_c] - uv_gt[m_g & m_c], axis=1).mean()) if (m_g & m_c).any() else float("nan")

        ax.imshow(img)
        ax.scatter(uv_gt[m_g, 0], uv_gt[m_g, 1], s=3, c="yellow", alpha=0.85, marker=".", linewidths=0)
        ax.scatter(uv_pert[m_p, 0], uv_pert[m_p, 1], s=3, c="red", alpha=0.6, marker=".", linewidths=0)
        ax.scatter(uv_corr[m_c, 0], uv_corr[m_c, 1], s=3, c="lime", alpha=0.85, marker=".", linewidths=0)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        ax.set_title(
            f"tile {inst['tile_id']}  @ ({int(u0)},{int(v0)})\n"
            f"{int(m_g.sum())} pts  pert {e_p:.1f}px → corr {e_c:.2f}px",
            fontsize=8.5, linespacing=1.25,
        )

    fig.suptitle(
        f"v0.2 raw-frame · {SCENE.name}@{FRAME} · "
        f"δ_target ypr={YPR.tolist()} t={T.tolist()}  ·  "
        f"{N_PICK} tiles (of {n_tiles})  ·  yellow=GT  red=perturbed  lime=BA-corrected",
        fontsize=10, y=1.0,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {OUT}  ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
