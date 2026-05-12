"""3-DoF [ωx, ωy, Δfx] BA correction for PandaSet side cameras.

Use a front-cam-only converged model (zero-shot on side cams) to discover
left/right calibration bias. Pool per-pt residuals across all frames + scenes,
solve 1-step linearized GN with the model's per-pt 2x2 inv-cov as weight.

Output:
  - configs/ba/pandaset_3cam_corrections.json — per-cam (ωx, ωy, Δfx) + provenance
  - experiments/_cache_check/ps_3cam_ba_check/ — BEFORE/AFTER projection PNGs

Reproducible via config file. See configs/ba/pandaset_3cam_corr.json for schema.
"""
import argparse, json, hashlib, pickle, gzip, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_depth import CalibNetDepth


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return cfg


def build_model(model_cfg: dict, device: torch.device) -> CalibNetDepth:
    m = CalibNetDepth(
        img_size=model_cfg["img_size"],
        in_channels=model_cfg["in_channels"],
        n_layers=model_cfg["n_layers"],
        use_convnext=model_cfg["use_convnext"],
        use_frustum=model_cfg["use_frustum"],
        deform_mode=model_cfg["deform_mode"],
        use_frame_pose=model_cfg.get("use_frame_pose", False),
        frame_pose_dof=model_cfg.get("frame_pose_dof", 6),
    ).to(device)
    ckpt = Path(model_cfg["ckpt"])
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # BA only needs per-pt head; drop CLS frame_pose_head entirely so cov-shape
    # differences between ckpt (e.g. full Cholesky) and the model we build here
    # (default diagonal) don't fail load.
    state = {k: v for k, v in state.items() if not k.startswith("frame_pose_head")}
    miss, unexp = m.load_state_dict(state, strict=False)
    print(f"  model: miss={len(miss)} unexp={len(unexp)} ckpt={ckpt}")
    m.eval()
    return m


def load_frame(root: Path, scene: str, cam: str, frame: int):
    sc = root / scene
    cd = sc / "camera" / cam
    if not (cd / "intrinsics.json").exists():
        return None
    intr = json.loads((cd / "intrinsics.json").read_text())
    poses = json.loads((cd / "poses.json").read_text())
    if frame >= len(poses):
        return None
    p = poses[frame]
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    q = p["heading"]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
    cpw = np.array([p["position"]["x"], p["position"]["y"], p["position"]["z"]], dtype=np.float32)
    T_wc = np.eye(4); T_wc[:3, :3] = R; T_wc[:3, 3] = cpw
    T_cw = np.linalg.inv(T_wc).astype(np.float32)
    ld = sc / "lidar" / f"{frame:02d}.pkl"
    if ld.exists():
        df = pickle.load(open(ld, "rb"))
    else:
        ld_gz = sc / "lidar" / f"{frame:02d}.pkl.gz"
        if not ld_gz.exists():
            return None
        df = pickle.load(gzip.open(ld_gz, "rb"))
    df = df[df["d"] == 0]
    pts_w = df[["x", "y", "z"]].values.astype(np.float32)
    homo = np.column_stack([pts_w, np.ones(len(pts_w), dtype=np.float32)])
    pts_c = (T_cw @ homo.T).T[:, :3]
    z = pts_c[:, 2]
    uv = (K @ pts_c.T).T[:, :2] / np.maximum(z[:, None], 1e-6)
    jpg = cd / f"{frame:02d}.jpg"
    if not jpg.exists():
        return None
    img = np.asarray(Image.open(jpg).convert("RGB"))
    return img, K, T_cw, pts_w, pts_c, uv, z


def build_bucket(uvd_local: np.ndarray, S: int, grid_n: int, K_per_cell: int):
    """Replicate datasets/pandaset_full.py bucket layout. uvd_local is (n,3)
    with uv in MODEL_S-scale pixels and d normalized /100."""
    G = grid_n
    cell_S = float(S) / G
    cu = np.clip((uvd_local[:, 0] / cell_S).astype(np.int32), 0, G - 1)
    cv = np.clip((uvd_local[:, 1] / cell_S).astype(np.int32), 0, G - 1)
    cell_id = cv * G + cu
    n = uvd_local.shape[0]
    if n == 0:
        return (np.zeros((G * G, K_per_cell, 3), dtype=np.float32),
                np.zeros((G * G, K_per_cell), dtype=bool))
    shuf = np.random.permutation(n)
    sorted_idx = shuf[np.argsort(cell_id[shuf], kind="stable")]
    sorted_uvd = uvd_local[sorted_idx]
    sorted_cid = cell_id[sorted_idx]
    counts = np.bincount(sorted_cid, minlength=G * G)
    cell_starts = np.zeros(G * G + 1, dtype=np.int64)
    cell_starts[1:] = counts.cumsum()
    intra = np.arange(n, dtype=np.int64) - cell_starts[sorted_cid]
    keep = intra < K_per_cell
    slots = intra[keep]
    cells = sorted_cid[keep]
    bucket_uvd = np.zeros((G * G, K_per_cell, 3), dtype=np.float32)
    bucket_valid = np.zeros((G * G, K_per_cell), dtype=bool)
    bucket_uvd[cells, slots] = sorted_uvd[keep]
    bucket_valid[cells, slots] = True
    return bucket_uvd, bucket_valid


@torch.no_grad()
def infer_tiles(model, img: np.ndarray, uv: np.ndarray, z: np.ndarray, K: np.ndarray,
                 ba_cfg: dict, device: torch.device):
    """Sliding tile grid over image → per-tile model forward → per-pt (Δu,Δv,σ_u,σ_v,ρ).
    Returns (uv_full[N,2], par[N,5], z_cam[N]). All in FULL-image px units."""
    H, W = img.shape[:2]
    TILE = ba_cfg["tile_size"]
    S = ba_cfg["model_input_size"]
    max_pts = ba_cfg["max_pts_per_tile"]
    min_pts = ba_cfg["min_pts_per_tile"]
    stride = ba_cfg.get("tile_stride", 128)
    scale = TILE / S
    # Sliding window with stride (matches /tmp/redo_flow_left.py and canonical
    # proj_3dof_*.png coverage style). Includes a final tile flush to right/bottom.
    us = list(range(0, max(W - TILE, 0) + 1, stride))
    vs = list(range(0, max(H - TILE, 0) + 1, stride))
    if us and us[-1] != W - TILE:
        us.append(W - TILE)
    if vs and vs[-1] != H - TILE:
        vs.append(H - TILE)
    out_uv, out_par, out_z = [], [], []
    for v0 in vs:
        for u0 in us:
            crop = img[v0:v0 + TILE, u0:u0 + TILE]
            cH, cW = crop.shape[:2]
            if cH < TILE // 2 or cW < TILE // 2:
                continue
            if cH < TILE or cW < TILE:
                pad = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                pad[:cH, :cW] = crop
                crop = pad
            in_tile = ((uv[:, 0] >= u0) & (uv[:, 0] < u0 + TILE) &
                       (uv[:, 1] >= v0) & (uv[:, 1] < v0 + TILE) & (z > 0.5))
            if int(in_tile.sum()) < min_pts:
                continue
            idx = np.where(in_tile)[0]
            if len(idx) > max_pts:
                idx = np.random.choice(idx, max_pts, replace=False)
            # Local UVD: model expects uv in [0, S) and d normalized to /100.
            # Depth d = Euclidean distance ‖p_cam‖ (matches canonical pipeline
            # in /tmp/ba_mc_corr_s001f30.py and /tmp/redo_flow_left.py).
            u_local = (uv[idx, 0] - u0) * (S / TILE)
            v_local = (uv[idx, 1] - v0) * (S / TILE)
            z_local = z[idx]
            # Reconstruct cam-frame 3D pts → Euclidean norm
            X = (uv[idx, 0] - K[0, 2]) * z_local / K[0, 0]
            Y = (uv[idx, 1] - K[1, 2]) * z_local / K[1, 1]
            d_eucl = np.sqrt(X * X + Y * Y + z_local * z_local)
            uvd = np.stack([u_local, v_local, (d_eucl / 100.0).astype(np.float32)], axis=1).astype(np.float32)
            # vfp = fx * S / cs (matches dataset)
            vfp = float(K[0, 0]) * S / TILE
            # Bucket from same uvd
            buc, buc_v = build_bucket(uvd, S, grid_n=16, K_per_cell=8)
            # GPU tensors
            crop_r = np.asarray(Image.fromarray(crop).resize((S, S)))
            img_t = torch.from_numpy(crop_r).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
            dist_uvd = torch.from_numpy(uvd).unsqueeze(0).to(device)
            pad_mask = torch.zeros(1, len(idx), dtype=torch.bool, device=device)
            vfp_t = torch.tensor([vfp], dtype=torch.float32, device=device)
            buc_t = torch.from_numpy(buc).unsqueeze(0).to(device)
            buc_v_t = torch.from_numpy(buc_v).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img_t, dist_uvd, key_padding_mask=pad_mask,
                            vfp=vfp_t, bucket_uvd=buc_t, bucket_valid=buc_v_t)
            params = out[0] if isinstance(out, tuple) else out
            params = params.float().cpu().numpy()[0]  # (M, 5)
            out_uv.append(uv[idx])
            out_par.append(np.column_stack([
                params[:, 0] * scale,             # Δu in full-image px
                params[:, 1] * scale,
                np.exp(params[:, 2]) * scale,     # σ_u
                np.exp(params[:, 3]) * scale,
                np.tanh(params[:, 4]) * 0.99,     # ρ
            ]))
            out_z.append(z_local)
    if not out_uv:
        return None
    return np.concatenate(out_uv), np.concatenate(out_par), np.concatenate(out_z)


def solve_3dof(uv: np.ndarray, par: np.ndarray, z: np.ndarray, K: np.ndarray,
                damping: float = 1e-3):
    """1-step linearized GN for [ωx_deg, ωy_deg, Δfx_px]. Matches canonical."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (uv[:, 0] - cx) * z / fx
    Y = (uv[:, 1] - cy) * z / fy
    Z = z
    du_ox = -(fx * X * Y) / (Z * Z)
    dv_ox = -fy - (fy * Y * Y) / (Z * Z)
    du_oy =  fx + (fx * X * X) / (Z * Z)
    dv_oy =  (fy * X * Y) / (Z * Z)
    du_dfx = (uv[:, 0] - cx) / fx
    dv_dfx = (uv[:, 1] - cy) / fy
    d2r = np.pi / 180
    J_u = np.column_stack([du_ox * d2r, du_oy * d2r, du_dfx])
    J_v = np.column_stack([dv_ox * d2r, dv_oy * d2r, dv_dfx])
    r_u, r_v = par[:, 0], par[:, 1]
    su, sv, rho = par[:, 2], par[:, 3], par[:, 4]
    det = su * su * sv * sv * (1 - rho * rho)
    Wuu = (sv * sv) / det
    Wvv = (su * su) / det
    Wuv = -(rho * su * sv) / det
    H = np.zeros((3, 3)); b = np.zeros(3)
    for i in range(3):
        for j in range(3):
            H[i, j] = ((J_u[:, i] * Wuu * J_u[:, j]).sum()
                       + (J_v[:, i] * Wvv * J_v[:, j]).sum()
                       + (J_u[:, i] * Wuv * J_v[:, j]).sum()
                       + (J_v[:, i] * Wuv * J_u[:, j]).sum())
        b[i] = ((J_u[:, i] * Wuu * r_u).sum()
                + (J_v[:, i] * Wvv * r_v).sum()
                + (J_u[:, i] * Wuv * r_v).sum()
                + (J_v[:, i] * Wuv * r_u).sum())
    H += damping * np.eye(3)
    return np.linalg.solve(H, b)


def make_T_corr(ox_deg: float, oy_deg: float) -> np.ndarray:
    R = Rotation.from_euler("xy", [ox_deg, oy_deg], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R
    return T


def render_before_after(out_dir: Path, scene: str, frame: int, cam: str,
                         frame_data, delta: np.ndarray):
    img, K, T_cw, pts_w, _, uv, z = frame_data
    K_new = K.copy()  # fx/fy fixed — 2-DoF BA only solves ωx, ωy
    T_corr = make_T_corr(delta[0], delta[1])
    T_cw_new = (T_corr @ T_cw).astype(np.float32)
    homo = np.column_stack([pts_w, np.ones(len(pts_w), dtype=np.float32)])
    pts_c2 = (T_cw_new @ homo.T).T[:, :3]
    z2 = pts_c2[:, 2]
    uv2 = (K_new @ pts_c2.T).T[:, :2] / np.maximum(z2[:, None], 1e-6)
    H_, W_ = img.shape[:2]
    v1 = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
    v2 = (z2 > 0.5) & (uv2[:, 0] >= 0) & (uv2[:, 0] < W_) & (uv2[:, 1] >= 0) & (uv2[:, 1] < H_)
    # Keep native image resolution so 1-px shifts are visible. figsize × dpi
    # must equal (W_, 2*H_) exactly — don't shrink, don't crop with bbox_inches.
    fig, axes = plt.subplots(2, 1, figsize=(W_ / 100.0, H_ / 100.0 * 2), dpi=100)
    axes[0].imshow(img)
    axes[0].scatter(uv[v1, 0], uv[v1, 1], s=0.3, c=z[v1],
                     cmap="turbo_r", alpha=0.7, vmin=1, vmax=80, marker=".")
    axes[0].set_title(f"{cam} sc={scene} f={frame} — BEFORE")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(img)
    axes[1].scatter(uv2[v2, 0], uv2[v2, 1], s=0.3, c=z2[v2],
                     cmap="turbo_r", alpha=0.7, vmin=1, vmax=80, marker=".")
    axes[1].set_title(f"{cam} — AFTER  ωx={delta[0]:+.3f}° ωy={delta[1]:+.3f}°")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout()
    fp = out_dir / f"{scene}_f{frame:02d}_{cam}.png"
    plt.savefig(fp, dpi=100)
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config} (sha256={cfg['_sha256']})")
    print(f"  description: {cfg['description']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], device)

    root = Path(cfg["data"]["root"])
    scenes = cfg["data"]["scenes"]
    if scenes == "all":
        scenes = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    frames_cfg = cfg["data"]["frames"]
    cams = cfg["data"]["cams"]
    ba_cfg = cfg["ba"]

    vis_dir = Path(cfg["output"]["vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old in vis_dir.glob("*.png"):
        old.unlink()
    vis_scenes = set(cfg["output"]["vis_scenes"])
    vis_frames = set(cfg["output"]["vis_frames"])

    # Cache one frame_data per (scene, cam, frame) for vis (lazy)
    vis_cache = {}

    # Pool residuals across all (scene, frame) per cam
    corrections = {}
    t0 = time.time()
    for cam in cams:
        print(f"\n[BA] cam={cam}")
        pooled_uv, pooled_par, pooled_z = [], [], []
        K_ref = None
        per_scene = 0
        per_frame = 0
        for scene in scenes:
            cam_dir = root / scene / "camera" / cam
            if not cam_dir.exists():
                continue
            poses_p = cam_dir / "poses.json"
            if not poses_p.exists():
                continue
            n_frames = len(json.loads(poses_p.read_text()))
            if frames_cfg == "all":
                fr_iter = range(n_frames)
            else:
                fr_iter = [f for f in frames_cfg if f < n_frames]
            scene_used = False
            for frame in fr_iter:
                fd = load_frame(root, scene, cam, frame)
                if fd is None:
                    continue
                if K_ref is None:
                    K_ref = fd[1]
                if scene in vis_scenes and frame in vis_frames:
                    vis_cache[(scene, frame, cam)] = fd
                img, K, T_cw, pts_w, pts_c, uv, z = fd
                res = infer_tiles(model, img, uv, z, K, ba_cfg, device)
                if res is None:
                    continue
                u, p, zc = res
                pooled_uv.append(u); pooled_par.append(p); pooled_z.append(zc)
                per_frame += 1
                scene_used = True
            if scene_used:
                per_scene += 1
            if per_scene % 10 == 0 and scene_used:
                el = time.time() - t0
                n_pt_so_far = sum(len(x) for x in pooled_uv)
                print(f"  {cam}: scenes={per_scene} frames={per_frame} pts={n_pt_so_far} t={el:.0f}s")
        if not pooled_uv:
            print(f"  {cam}: NO DATA")
            continue
        UV = np.concatenate(pooled_uv)
        PAR = np.concatenate(pooled_par)
        Z = np.concatenate(pooled_z)
        delta = solve_2dof(UV, PAR, Z, K_ref, damping=ba_cfg["damping"])
        corrections[cam] = {
            "omega_x_deg": float(delta[0]),
            "omega_y_deg": float(delta[1]),
            "fx_ref": float(K_ref[0, 0]),
            "n_scenes": int(per_scene),
            "n_frames": int(per_frame),
            "n_pts": int(len(UV)),
        }
        print(f"  → ωx={delta[0]:+.4f}°  ωy={delta[1]:+.4f}°  "
              f"scenes={per_scene} frames={per_frame} pts={len(UV)}")

    # BEFORE/AFTER vis
    print("\n[VIS] rendering BEFORE/AFTER")
    for (scene, frame, cam), fd in sorted(vis_cache.items()):
        if cam not in corrections:
            continue
        delta = np.array([corrections[cam]["omega_x_deg"],
                          corrections[cam]["omega_y_deg"]])
        fp = render_before_after(vis_dir, scene, frame, cam, fd, delta)
        print(f"  {fp}")

    # Provenance
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:10]
    except Exception:
        git_sha = "unknown"
    ckpt_p = Path(cfg["model"]["ckpt"])
    out_obj = {
        "provenance": {
            "config_path": cfg["_path"],
            "config_sha256_prefix": cfg["_sha256"],
            "git_sha": git_sha,
            "run_started": datetime.now().isoformat(timespec="seconds"),
            "model_ckpt": str(ckpt_p),
            "model_ckpt_mtime": datetime.fromtimestamp(ckpt_p.stat().st_mtime).isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "ba_param": "2dof_omega_x_omega_y_fx_fy_fixed",
        "corrections": corrections,
    }
    out_p = Path(cfg["output"]["corrections_json"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out_obj, indent=2))
    print(f"\nwrote → {out_p}")
    print(f"vis  → {vis_dir}")


if __name__ == "__main__":
    main()
