"""Calibration API — FastAPI server, GPU-15 resident σ-head + closed-form GN.

Endpoints
  GET  /calibrate                   → drag-drop UI (static/index.html)
  GET  /api/default                 → metadata for the built-in idx=17 sample
  GET  /api/default/parent.png      → the original camera tile (cached PNG)
  POST /api/calibrate               → JSON body
        { ypr: [r,p,y], t: [x,y,z], n_tiles: int, cs: 256|512 }
        runs sub-tile sweep, shared-GN, returns { delta, pert_px, corr_px,
        overlay_url, took_s, B }

Run
  CUDA_VISIBLE_DEVICES=15 \
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python -m uvicorn \
      services.calib_api.server:app --host 0.0.0.0 --port 5002
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.eval_shared_256x800 as ess
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.ba.ba_torch import solve_kb_xyz_shared, make_info_from_sigma_rho
from services.calib_api import __version__ as API_VERSION
from services.calib_api.raw_pipeline import (
    build_calib_frame,
    load_calib_bytes,
    load_kamikado_frame_from_disk,
    load_points_text,
    solve_from_calib_frame,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STATIC = HERE / "static"
HISTORY = HERE / "history.jsonl"
RAW_KAMIKADO_DIR = Path("/raw/kamikado/scenes")
RESULTS.mkdir(exist_ok=True)
DEFAULT_IDX = 17


def _git_rev() -> str:
    """Read short SHA without invoking git — the container has no git binary."""
    try:
        head = (REPO / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            sha = (REPO / ".git" / ref).read_text().strip()
        else:
            sha = head
        return sha[:7]
    except Exception:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            return out.decode().strip()
        except Exception:
            return "unknown"


GIT_REV = _git_rev()

app = FastAPI(title="e2e_calib calibration API", version=API_VERSION)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/calibrate/result", StaticFiles(directory=RESULTS), name="results")


def _solve_from_idx_list(model, ds, idx_list, *, cs, n_per_inst,
                         ypr_target, t_target, dist_one, cfg):
    """Mirror of ess._solve_one but the batch is built from a fixed list of
    instance idxs (sibling tiles of the same frame) rather than random val.
    """
    if cs == 256:
        u0v0_list = [(0, 0), (256, 0), (0, 256), (256, 256)][:n_per_inst]
    elif cs == 512:
        u0v0_list = [(0, 0)]
    else:
        raise ValueError(f"unsupported cs={cs}")

    wins = []
    for idx in idx_list:
        try:
            inst_i = ds._load_inst(int(idx))
        except Exception:
            continue
        for (u0, v0) in u0v0_list:
            w = ess._build_subwin(ds, inst_i, t_target, ypr_target,
                                  u0=u0, v0=v0, cs=cs)
            if w is not None:
                wins.append(w)
    if not wins:
        raise RuntimeError("no usable sub-tiles in idx_list")

    moved = [t.to(ess.DEVICE) if torch.is_tensor(t) else t
             for t in collate_full(wins)]
    # collate_full は 12 → 13 tuple 化済み (末尾 delta1_se3)。calib API は
    # split_pert OFF (δ1=0) で回すので末尾は捨てる。
    (imgs, _true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b) = moved[:12]
    valid = ~pad_mask
    pad_full = ~valid
    B, _N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        P0_orig[pad_full] = torch.tensor([0.0, 0.0, 1.0],
                                         dtype=P0_orig.dtype,
                                         device=P0_orig.device)
    dist = dist_one.to(ess.DEVICE).expand(B, 4).contiguous()

    use_intensity = getattr(model, "use_intensity", True)
    if use_intensity:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    sx = per_pt[..., 2].exp()
    sy = per_pt[..., 3].exp()
    rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    scale_l2o = (cs_b / float(cfg["img_size"])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)

    prior = ess.PRIOR_DIAG.to(ess.DEVICE)
    with torch.no_grad():
        delta_shared, _H = solve_kb_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, dist, ess.DOFS,
            valid=valid, n_iter=ess.BA_N_ITER, damping=ess.DAMPING,
            prior_diag=prior,
        )
    return delta_shared, B


# ─── model + dataset, loaded once at startup ───────────────────────────────
_state = {}


def _bootstrap():
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
    model = ess._build_model(cfg).to(ess.DEVICE)
    sd = torch.load(ess.CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict") or sd.get("model")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Pre-cache default tile so the UI shows it instantly.
    inst = ds._load_inst(DEFAULT_IDX)
    parent = np.array(Image.open(io.BytesIO(inst["jpg_bytes"])).convert("RGB"))
    parent_path = RESULTS / "default_idx17_parent.png"
    Image.fromarray(parent).save(parent_path)

    dist_one = inst["distortion"].clone().detach().to(torch.float32).reshape(1, 4)

    # Whole-frame siblings: every tile sharing (scene, cam, frame) with idx=17.
    def _sig(d):
        return (str(d.get("scene", "")),
                str(d.get("cam", "")),
                int(d.get("frame", -1)))

    # Group every val instance by (scene, cam, frame). Each group is a "frame"
    # the user can pick from the UI; sibling tiles within a group share the
    # original-camera image, so a per-frame shared-GN solve is well-posed.
    from collections import defaultdict as _dd
    groups: dict[tuple, list[int]] = _dd(list)
    for i in range(len(ds.fnames)):
        try:
            ii = ds._load_inst(i)
        except Exception:
            continue
        groups[_sig(ii)].append(i)
    key = _sig(inst)
    siblings = groups.get(key, [DEFAULT_IDX])
    print(f"[boot] frame groups: {len(groups)}  default key={key}  siblings={len(siblings)}")
    # Frame catalog: only fisheye groups with ≥ 4 sibling tiles (so the BA has
    # something to chew on). Sorted by sibling count desc so big frames bubble
    # up. Cap at 200 entries to keep /api/default JSON small.
    catalog = []
    for k, v in groups.items():
        if len(v) < 4:
            continue
        try:
            ii0 = ds._load_inst(v[0])
        except Exception:
            continue
        if not bool(ii0.get("is_fisheye", False)):
            continue
        catalog.append({
            "scene": k[0],
            "cam": k[1],
            "frame": k[2],
            "n_tiles": len(v),
            "first_idx": v[0],
            "idxs": v,
        })
    catalog.sort(key=lambda d: -d["n_tiles"])
    catalog = catalog[:200]
    print(f"[boot] fisheye frame catalog: {len(catalog)} (top {catalog[0]['n_tiles']} tiles)")

    # Index catalog by first_idx so /api/calibrate target_idx lookup is O(1).
    catalog_by_id: dict[int, dict] = {c["first_idx"]: c for c in catalog}

    _state["cfg"] = cfg
    _state["ds"] = ds
    _state["model"] = model
    _state["dist_one"] = dist_one
    _state["default_inst"] = inst
    _state["default_parent_size"] = (int(parent.shape[1]), int(parent.shape[0]))
    _state["sibling_idxs"] = siblings
    _state["catalog"] = catalog
    _state["catalog_by_id"] = catalog_by_id
    _state["parent_cache"] = {DEFAULT_IDX: (int(parent.shape[1]), int(parent.shape[0]))}
    print(f"[boot] model on {ess.DEVICE}, ckpt={ess.CKPT.name}, default idx={DEFAULT_IDX}")
    print(f"[boot] parent tile {_state['default_parent_size']} cached at {parent_path}")


@app.on_event("startup")
def _startup():
    _bootstrap()


# ─── routes ────────────────────────────────────────────────────────────────
@app.get("/calibrate", response_class=HTMLResponse)
def page():
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _ensure_parent_png(idx: int) -> tuple[str, tuple[int, int]]:
    """Cache `parent_<idx>.png` on disk, return (url, (W, H))."""
    cache = _state["parent_cache"]
    if idx in cache:
        W, H = cache[idx]
        if idx == DEFAULT_IDX:
            return "/calibrate/result/default_idx17_parent.png", (W, H)
        return f"/calibrate/result/parent_{idx}.png", (W, H)
    ds = _state["ds"]
    inst = ds._load_inst(int(idx))
    arr = np.array(Image.open(io.BytesIO(inst["jpg_bytes"])).convert("RGB"))
    out = RESULTS / f"parent_{idx}.png"
    Image.fromarray(arr).save(out)
    W, H = int(arr.shape[1]), int(arr.shape[0])
    cache[idx] = (W, H)
    return f"/calibrate/result/{out.name}", (W, H)


@app.get("/api/default")
def default_meta():
    pW, pH = _state["default_parent_size"]
    n_sib = len(_state["sibling_idxs"])
    return {
        "api_version": API_VERSION,
        "git_rev": GIT_REV,
        "idx": DEFAULT_IDX,
        "parent_url": "/calibrate/result/default_idx17_parent.png",
        "parent_size": [pW, pH],
        "ckpt": str(ess.CKPT.relative_to(REPO)),
        "img_size": _state["cfg"]["img_size"],
        "n_sibling_tiles": n_sib,
        "n_frames": len(_state["catalog"]),
        "presets": [
            {"label": "1 × 512 size (single tile)",
             "mode": "tile_only", "n_tiles": 1,   "cs": 512},
            {"label": "4 × 256 size (single tile, all 4 sub-crops)",
             "mode": "tile_only", "n_tiles": 1,   "cs": 256},
            {"label": f"whole frame, 256² ({n_sib} tiles × 4 sub-crops ≈ {n_sib*4})",
             "mode": "whole_frame", "n_tiles": n_sib, "cs": 256},
            {"label": "800 × 256 size (paper headline, random val mix)",
             "mode": "random_val", "n_tiles": 200, "cs": 256},
        ],
    }


@app.get("/api/frames")
def frames_list():
    """Catalog of fisheye frames with ≥4 sibling tiles. Pick one to switch
    the source frame for /api/calibrate via `target_idx` (= first_idx)."""
    out = []
    for c in _state["catalog"]:
        out.append({
            "scene": c["scene"],
            "cam": c["cam"],
            "frame": c["frame"],
            "n_tiles": c["n_tiles"],
            "first_idx": c["first_idx"],
        })
    return {"frames": out, "default_idx": DEFAULT_IDX}


@app.get("/api/frame/{first_idx}")
def frame_meta(first_idx: int):
    cat = _state["catalog_by_id"].get(int(first_idx))
    if cat is None and int(first_idx) != DEFAULT_IDX:
        raise HTTPException(404, f"first_idx={first_idx} not in catalog")
    url, (W, H) = _ensure_parent_png(int(first_idx))
    n_sib = (cat["n_tiles"] if cat is not None
             else len(_state["sibling_idxs"]))
    scene = cat["scene"] if cat is not None else None
    cam = cat["cam"] if cat is not None else None
    frame = cat["frame"] if cat is not None else None
    return {
        "first_idx": int(first_idx),
        "scene": scene, "cam": cam, "frame": frame,
        "n_sibling_tiles": int(n_sib),
        "parent_url": url,
        "parent_size": [W, H],
    }


class CalibReq(BaseModel):
    ypr: List[float] = Field(..., min_items=3, max_items=3,
                              description="ZYX-Euler array: [arr0=roll(Z), arr1=yaw(Y), arr2=pitch(X)] in deg")
    t:   List[float] = Field(..., min_items=3, max_items=3,
                              description="[tx, ty, tz] in m")
    mode: str = Field("whole_frame",
                      description="tile_only | whole_frame | random_val")
    n_tiles: int = Field(32, ge=1, le=400)
    cs: int = Field(256, description="256 or 512")
    seed: int = Field(1008, description="RNG for random_val sampling")
    target_idx: int = Field(
        DEFAULT_IDX,
        description=("Which val instance to anchor on. Must be the `first_idx` "
                     "of a frame in /api/frames, or DEFAULT_IDX (17). The frame's "
                     "sibling tiles are used for whole_frame mode."),
    )
    fast: bool = Field(True,
        description="if True: skip 3-panel matplotlib PNG, drop occluded/oob pts")


def _resolve_target(target_idx: int):
    """Return (inst, sibling_idxs) for the requested anchor."""
    if int(target_idx) == DEFAULT_IDX:
        return _state["default_inst"], _state["sibling_idxs"]
    cat = _state["catalog_by_id"].get(int(target_idx))
    if cat is None:
        raise HTTPException(
            404,
            f"target_idx={target_idx} not in catalog (use /api/frames)",
        )
    inst = _state["ds"]._load_inst(int(target_idx))
    return inst, list(cat["idxs"])


@app.post("/api/calibrate")
def calibrate(req: CalibReq):
    if req.cs not in (256, 512):
        raise HTTPException(400, "cs must be 256 or 512")
    n_per_inst = 4 if req.cs == 256 else 1
    cfg = _state["cfg"]
    ds = _state["ds"]
    model = _state["model"]
    dist_one = _state["dist_one"]
    inst, siblings = _resolve_target(req.target_idx)

    ypr = np.asarray(req.ypr, dtype=np.float64)
    t = np.asarray(req.t, dtype=np.float64)

    t0 = time.time()
    if req.mode == "tile_only":
        idxs = [int(req.target_idx)]
        delta, B = _solve_from_idx_list(
            model, ds, idxs, cs=req.cs, n_per_inst=n_per_inst,
            ypr_target=ypr, t_target=t, dist_one=dist_one, cfg=cfg)
    elif req.mode == "whole_frame":
        idxs = siblings[:req.n_tiles]
        delta, B = _solve_from_idx_list(
            model, ds, idxs, cs=req.cs, n_per_inst=n_per_inst,
            ypr_target=ypr, t_target=t, dist_one=dist_one, cfg=cfg)
    elif req.mode == "random_val":
        rng = np.random.RandomState(req.seed)
        delta, B, _H = ess._solve_one(
            model, ds, target_idx=int(req.target_idx), n_inst=req.n_tiles,
            cs=req.cs, n_per_inst=n_per_inst, rng=rng,
            ypr_target=ypr, t_target=t, dist_one=dist_one,
            cfg=cfg, label="api")
    else:
        raise HTTPException(400, f"unknown mode {req.mode!r}")
    solve_s = time.time() - t0

    rid = uuid.uuid4().hex[:12]
    out_png = RESULTS / f"overlay_{rid}.png"
    t1 = time.time()
    if req.fast:
        info = None
    else:
        info = ess.render_3panel_overlay(
            inst, ypr, t, delta,
            out_path=out_png,
            suptitle=f"idx={int(req.target_idx)}  mode={req.mode}  B={B}  cs={req.cs}  "
                     f"δ_target ypr={ypr.tolist()} t={t.tolist()}",
            panel_label=f"BA-corrected ({req.mode}, B={B})",
        )
    render_s = time.time() - t1
    t2 = time.time()
    geom = ess.compute_overlay_geom(inst, ypr, t, delta, drop_invisible=req.fast)
    geom_s = time.time() - t2
    t3 = time.time()
    parent_url, _ = _ensure_parent_png(int(req.target_idx))
    parent_s = time.time() - t3
    print(f"[timing /api/calibrate fast={req.fast}] solve={solve_s:.2f}s "
          f"3panel={render_s:.2f}s geom={geom_s:.2f}s parent={parent_s:.2f}s "
          f"B={B} N_uv={len(geom['uv_gt'])}")

    delta_np = delta.detach().cpu().numpy().tolist()
    geom_url, geom_n, geom_bytes = _write_geom_blob(rid, geom)
    out = {
        "delta": {"omega_deg": delta_np[:3], "t_m": delta_np[3:]},
        "mode": req.mode,
        "B": int(B),
        "cs": req.cs,
        "target_idx": int(req.target_idx),
        "pert_px_mean": float(geom["stats"]["pert_mean_px"]),
        "corr_px_mean": float(geom["stats"]["corr_mean_px"]),
        "took_s": round(solve_s, 2),
        "overlay_url": (f"/calibrate/result/{out_png.name}" if info is not None
                         else None),
        "parent_url": parent_url,
        "parent_size": geom["parent_size"],
        "geom_url": geom_url,
        "geom_n": geom_n,
        "geom_bytes": geom_bytes,
    }
    _history_append({
        "ts": time.time(),
        "endpoint": "/api/calibrate",
        "source": f"tile target_idx={req.target_idx} mode={req.mode}",
        "ypr": req.ypr, "t": req.t, "cs": req.cs, "B": int(B),
        "pert_px_mean": float(geom["stats"]["pert_mean_px"]),
        "corr_px_mean": float(geom["stats"]["corr_mean_px"]),
        "took_s": round(solve_s, 2),
        "overlay_url": out["overlay_url"],
        "parent_url": out.get("parent_url"),
        "parent_size": out.get("parent_size"),
        "geom_url": out.get("geom_url"),
        "geom_n": out.get("geom_n"),
        "delta": out["delta"],
    })
    return out


# ─── geom binary blob ───────────────────────────────────────────────────────
def _write_geom_blob(rid: str, geom: dict) -> tuple[str, int, int]:
    """Pack uv_gt/pert/corr (Nx2 f32) + z_gt/pert/corr (N f32) into one
    little-endian Float32Array blob. Saves the JSON tax for ~55k points.

    Layout (all float32 LE):
        [N×2 uv_gt | N×2 uv_pert | N×2 uv_corr | N z_gt | N z_pert | N z_corr]
    Total = 36 N bytes. Browser reads via fetch().arrayBuffer() + Float32Array.
    """
    N = int(geom["uv_gt"].shape[0])
    parts = [
        geom["uv_gt"].astype(np.float32, copy=False).reshape(-1),
        geom["uv_pert"].astype(np.float32, copy=False).reshape(-1),
        geom["uv_corr"].astype(np.float32, copy=False).reshape(-1),
        geom["z_gt"].astype(np.float32, copy=False).reshape(-1),
        geom["z_pert"].astype(np.float32, copy=False).reshape(-1),
        geom["z_corr"].astype(np.float32, copy=False).reshape(-1),
    ]
    buf = np.concatenate(parts).tobytes()  # native little-endian on x86
    out_path = RESULTS / f"geom_{rid}.bin"
    out_path.write_bytes(buf)
    return f"/calibrate/result/{out_path.name}", N, len(buf)


# ─── v0.2 raw-frame endpoints ───────────────────────────────────────────────
def _history_append(entry: dict):
    """Append one JSON line to history.jsonl. Best-effort; never raise."""
    try:
        with HISTORY.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[history] append failed: {e}")


def _solve_raw_frame(cf, *, ypr, t, cs, n_per_inst, source: str, fast: bool = True):
    """Run σ-head + shared GN on a CalibFrame, render overlay, log history.

    Returns the response dict (same shape as /api/calibrate).
    """
    ds = _state["ds"]
    model = _state["model"]
    t0 = time.time()
    delta, B, n_tiles, n_subcrops = solve_from_calib_frame(
        model, ds, cf,
        ypr_target=np.asarray(ypr, dtype=np.float64),
        t_target=np.asarray(t, dtype=np.float64),
        cs=cs, n_per_inst=n_per_inst,
    )
    solve_s = time.time() - t0

    # Render against the FULL frame (whole 3840×2160 image + all projected
    # LiDAR points), not just the anchor tile window.
    rid = uuid.uuid4().hex[:12]
    out_png = RESULTS / f"overlay_{rid}.png"
    ypr_np = np.asarray(ypr, dtype=np.float64)
    t_np = np.asarray(t, dtype=np.float64)
    full_inst = {
        "img": cf.img,
        "K_full": torch.from_numpy(cf.K.astype(np.float64)),
        "distortion": torch.from_numpy(cf.dist.astype(np.float64)),
        "pts": torch.from_numpy(cf.pts_cam.astype(np.float64)),
        "tile_u0": 0,
        "tile_v0": 0,
    }
    t1 = time.time()
    if fast:
        info = None
    else:
        info = ess.render_3panel_overlay(
            full_inst, ypr_np, t_np, delta,
            out_path=out_png,
            suptitle=f"raw {cf.scene_id}@{cf.frame_id}  full-frame overlay  "
                     f"B={B}  tiles={n_tiles}  cs={cs}  "
                     f"δ_target ypr={ypr_np.tolist()} t={t_np.tolist()}",
            panel_label=f"BA-corrected (whole frame, B={B})",
        )
    render_s = time.time() - t1

    t2 = time.time()
    geom = ess.compute_overlay_geom(full_inst, ypr_np, t_np, delta,
                                     drop_invisible=fast)
    geom_s = time.time() - t2

    t3 = time.time()
    # Cache parent PNG by content signature so repeated requests re-use it.
    sig = f"{cf.scene_id}_{cf.frame_id}"
    parent_png = RESULTS / f"parent_{sig}.png"
    if not parent_png.exists():
        Image.fromarray(cf.img.astype(np.uint8)).save(parent_png)
    parent_url = f"/calibrate/result/{parent_png.name}"
    parent_s = time.time() - t3
    print(f"[timing _solve_raw_frame fast={fast}] solve={solve_s:.2f}s "
          f"3panel={render_s:.2f}s geom={geom_s:.2f}s parent={parent_s:.2f}s "
          f"B={B} tiles={n_tiles} N_uv={len(geom['uv_gt'])}")

    delta_np = delta.detach().cpu().numpy().tolist()
    geom_url, geom_n, geom_bytes = _write_geom_blob(rid, geom)
    resp = {
        "delta": {"omega_deg": delta_np[:3], "t_m": delta_np[3:]},
        "B": int(B),
        "cs": int(cs),
        "n_tiles": int(n_tiles),
        "n_subcrops": int(n_subcrops),
        "scene": cf.scene_id,
        "frame": int(cf.frame_id),
        "pert_px_mean": float(geom["stats"]["pert_mean_px"]),
        "corr_px_mean": float(geom["stats"]["corr_mean_px"]),
        "took_s": round(solve_s, 2),
        "overlay_url": (f"/calibrate/result/{out_png.name}" if info is not None
                         else None),
        "parent_url": parent_url,
        "parent_size": geom["parent_size"],
        "geom_url": geom_url,
        "geom_n": geom_n,
        "geom_bytes": geom_bytes,
    }
    _history_append({
        "ts": time.time(),
        "endpoint": "/api/calibrate_scene" if source.startswith("scene:") else "/api/calibrate_frame",
        "source": source,
        "ypr": list(map(float, ypr)),
        "t": list(map(float, t)),
        "cs": int(cs), "B": int(B),
        "n_tiles": int(n_tiles), "n_subcrops": int(n_subcrops),
        "pert_px_mean": float(geom["stats"]["pert_mean_px"]),
        "corr_px_mean": float(geom["stats"]["corr_mean_px"]),
        "took_s": round(solve_s, 2),
        "overlay_url": resp["overlay_url"],
        "parent_url": resp["parent_url"],
        "parent_size": resp["parent_size"],
        "geom_url": resp["geom_url"],
        "geom_n": resp["geom_n"],
        "delta": resp["delta"],
    })
    return resp


@app.get("/api/scenes")
def list_scenes():
    """Enumerate kamikado scenes available on the server (under /raw/kamikado/scenes)."""
    if not RAW_KAMIKADO_DIR.exists():
        return {"scenes": [], "raw_dir": str(RAW_KAMIKADO_DIR), "available": False}
    out = []
    for d in sorted(RAW_KAMIKADO_DIR.iterdir()):
        if not d.is_dir():
            continue
        n_frames = len(list(d.glob("image_*.png")))
        has_calib = (d / "calib.calib").exists()
        if n_frames > 0 and has_calib:
            out.append({"name": d.name, "n_frames": n_frames})
    return {"scenes": out, "raw_dir": str(RAW_KAMIKADO_DIR), "available": True}


class CalibSceneReq(BaseModel):
    scene: str = Field(..., description="scene directory name under /raw/kamikado/scenes")
    frame: int = Field(0, ge=0, description="frame index (image_<N>.png)")
    ypr: List[float] = Field(..., min_items=3, max_items=3)
    t: List[float] = Field(..., min_items=3, max_items=3)
    cs: int = Field(256)
    fast: bool = Field(True,
        description="if True: skip 3-panel matplotlib PNG, drop occluded/oob pts. "
                    "Drops latency from ~3s to ~0.3s for slider-driven UI.")


@app.post("/api/calibrate_scene")
def calibrate_scene(req: CalibSceneReq):
    """GT-demo: load a server-side raw frame, perturb, solve, return δ̂."""
    if req.cs not in (256, 512):
        raise HTTPException(400, "cs must be 256 or 512")
    scene_dir = RAW_KAMIKADO_DIR / req.scene
    if not scene_dir.is_dir():
        raise HTTPException(404, f"scene {req.scene!r} not found under {RAW_KAMIKADO_DIR}")
    if not (scene_dir / f"image_{req.frame}.png").exists():
        raise HTTPException(404, f"frame {req.frame} not in scene {req.scene}")
    try:
        cf = load_kamikado_frame_from_disk(scene_dir, req.frame)
    except Exception as e:
        raise HTTPException(500, f"failed to load frame: {e}")
    n_per_inst = 4 if req.cs == 256 else 1
    return _solve_raw_frame(
        cf, ypr=req.ypr, t=req.t, cs=req.cs, n_per_inst=n_per_inst,
        source=f"scene:{req.scene}@{req.frame}", fast=bool(req.fast),
    )


@app.post("/api/calibrate_frame")
async def calibrate_frame(
    image: UploadFile = File(..., description="PNG/JPG camera image"),
    points: UploadFile = File(..., description="kamikado points_V_*.txt (x y z intensity)"),
    calib: UploadFile = File(..., description="kamikado calib.calib JSON"),
    ypr: str = Form("[0,0,0]", description="JSON [roll,pitch,yaw] deg"),
    t: str = Form("[0,0,0]", description="JSON [tx,ty,tz] m"),
    cs: int = Form(256),
    scene_id: str = Form("upload"),
    frame_id: int = Form(0),
):
    """User-supplied raw frame upload: image + points_V + calib.calib → δ̂."""
    if cs not in (256, 512):
        raise HTTPException(400, "cs must be 256 or 512")
    try:
        ypr_v = json.loads(ypr)
        t_v = json.loads(t)
        if not (isinstance(ypr_v, list) and len(ypr_v) == 3):
            raise ValueError("ypr must be 3-list")
        if not (isinstance(t_v, list) and len(t_v) == 3):
            raise ValueError("t must be 3-list")
    except Exception as e:
        raise HTTPException(400, f"invalid ypr/t: {e}")

    img_bytes = await image.read()
    pts_bytes = await points.read()
    calib_bytes = await calib.read()
    try:
        img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        pts_V = load_points_text(pts_bytes.decode("utf-8"))
        K, dist, T_SV = load_calib_bytes(calib_bytes)
        cf = build_calib_frame(
            img_arr=img_arr, pts_V=pts_V, K=K, dist=dist, T_SV=T_SV,
            scene_id=scene_id, frame_id=int(frame_id),
        )
    except Exception as e:
        raise HTTPException(400, f"failed to parse uploaded frame: {e}")

    n_per_inst = 4 if cs == 256 else 1
    return _solve_raw_frame(
        cf, ypr=ypr_v, t=t_v, cs=cs, n_per_inst=n_per_inst,
        source=f"upload:{scene_id}@{frame_id}",
    )


@app.get("/api/history")
def history(limit: int = 20):
    """Return the most recent N history entries (newest first)."""
    if not HISTORY.exists():
        return {"entries": []}
    try:
        with HISTORY.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(500, f"history read error: {e}")
    out = []
    for ln in lines[-max(1, int(limit)):][::-1]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return {"entries": out}


@app.get("/calibrate/frame", response_class=HTMLResponse)
def page_frame():
    return (STATIC / "frame.html").read_text(encoding="utf-8")
