"""Smoke the v0.2 raw-frame pipeline against a kamikado scene/frame.

Run inside calib-api container so it sees the cuda model.

    docker exec calib-api python scripts/_debug/_smoke_raw_pipeline.py

Compares against v0.1 idx=17 whole_frame (≈ 0.82 px corr).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import PandaSetCalibDatasetFull
from services.calib_api.raw_pipeline import (
    load_kamikado_frame_from_disk,
    solve_from_calib_frame,
)


def main():
    SCENE = Path("/raw/kamikado/scenes/points_ip664_D_20260226_224648_d005_3000_3020")
    FRAME = 0  # default scene's first frame — the v0.1 default tile (idx=17)
                # belongs to *this* scene/frame too, since `key=('points_ip664...', '', 0)`.

    cfg = ess._load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=ess.CACHE,
        split="val",
        img_size=cfg["img_size"],
        min_crop_px=cfg["min_crop_px"],
        max_crop_px=cfg["max_crop_px"],
        max_offset_m=0.0,
        max_rot_deg=0.0,
        oversample=1,
        grid_n=cfg.get("grid_n", 16),
        center_band=0.0,
        preload=False,
    )
    print(f"[ds] min_pts={ds.min_pts} n_full={ds.n_full}")

    model = ess._build_model(cfg).to(ess.DEVICE)
    sd = torch.load(ess.CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict") or sd.get("model")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[load] {SCENE.name} frame={FRAME}")
    t0 = time.time()
    cf = load_kamikado_frame_from_disk(SCENE, FRAME)
    print(f"  loaded in {time.time()-t0:.2f}s; img={cf.hw}, pts={len(cf.pts_cam)}")

    ypr = np.array([0.30, -0.20, 0.50], dtype=np.float64)
    t = np.array([0.030, -0.020, 0.040], dtype=np.float64)

    print(f"[solve] cs=256 n_per_inst=4")
    t0 = time.time()
    delta, B, n_tiles, n_subcrops = solve_from_calib_frame(
        model, ds, cf,
        ypr_target=ypr, t_target=t,
        cs=256, n_per_inst=4,
    )
    dt = time.time() - t0
    print(f"  tiles={n_tiles}  sub-crops batched={n_subcrops}  B={B}  in {dt:.2f}s")
    delta_np = delta.detach().cpu().numpy().tolist()
    print(f"  δ̂ ω={delta_np[:3]}  t={delta_np[3:]}")
    print(f"  expected ω≈[0.467, -0.198, 0.277]  t≈[0.032, -0.025, 0.045]  (v0.1 idx=17)")


if __name__ == "__main__":
    main()
