"""Quick-look visualization of ZOD frames_mini: lidar → camera projection.

ZOD uses a Kannala-Brandt fisheye model (camera_type='kannala') with:
  - intrinsics  K (3x4)  : fx, fy, cx, cy on a 3848x2168 front camera
  - extrinsics  T_vc     : camera pose in vehicle frame  (4x4)
  - lidar_extrinsics T_vl: lidar  pose in vehicle frame  (4x4)

Projection:
  P_cam = inv(T_vc) @ T_vl @ P_lidar
  r     = sqrt(x^2 + y^2), theta = atan2(r, z)
  theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
  u = fx (theta_d * x/r) + cx
  v = fy (theta_d * y/r) + cy

Usage:
    python scripts/visualization/vis_zod_mini.py \
        --root /mnt/fsx/tmp/hfunaya/zod/frames_mini \
        --out  /mnt/fsx/tmp/hfunaya/zod/vis \
        --n 6

Outputs N PNGs (one per frame) with lidar points color-coded by depth.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def kannala_project(xyz_cam: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """xyz_cam: (N,3) in camera frame (Z forward for pinhole convention).
    Returns (uv (N,2), z (N,)) — uv only valid where z>0 and theta<pi/2.

    NOTE: ZOD stores the cam frame with its own convention (see extrinsics).
    Caller is expected to pass points already in the camera frame where:
      - x: right, y: down, z: forward (OpenCV convention)
    After the vehicle→camera transform using T_cv = inv(T_vc), this holds
    automatically because T_vc was authored in that convention.
    """
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(r, z)
    k1, k2, k3, k4 = dist
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    # guard r==0 (on-axis points): set to 1 since (x/r, y/r) → 0 anyway
    r_safe = np.where(r > 1e-9, r, 1.0)
    u = K[0, 0] * (theta_d * x / r_safe) + K[0, 2]
    v = K[1, 1] * (theta_d * y / r_safe) + K[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, z


def load_frame(frame_dir: Path):
    """Read calibration, image, and nearest-timestamp lidar sweep."""
    calib = json.loads((frame_dir / "calibration.json").read_text())["FC"]
    K = np.asarray(calib["intrinsics"], dtype=np.float64)          # (3,4)
    K3 = K[:3, :3]
    dist = np.asarray(calib["distortion"], dtype=np.float64)
    T_vc = np.asarray(calib["extrinsics"], dtype=np.float64)        # cam-in-vehicle
    T_vl = np.asarray(calib["lidar_extrinsics"], dtype=np.float64)  # lidar-in-vehicle
    IW, IH = int(calib["image_dimensions"][0]), int(calib["image_dimensions"][1])

    img_path = sorted((frame_dir / "camera_front_dnat").glob("*.jpg"))[0]
    img = np.asarray(Image.open(img_path).convert("RGB"))

    # Pick the lidar sweep whose timestamp is closest to the image timestamp
    # (image filename contains ISO-8601; we just pick the middle sweep as a
    # good-enough approximation — ZOD tools do proper motion compensation)
    lidars = sorted((frame_dir / "lidar_velodyne").glob("*.npy"))
    sweep = np.load(lidars[len(lidars) // 2], allow_pickle=False)

    pts_lidar = np.stack([sweep["x"], sweep["y"], sweep["z"]], axis=1).astype(np.float64)

    # Transform lidar → vehicle → camera
    N = pts_lidar.shape[0]
    homo = np.concatenate([pts_lidar, np.ones((N, 1))], axis=1)   # (N,4)
    pts_veh = (T_vl @ homo.T).T[:, :3]                             # (N,3) vehicle frame
    pts_veh_homo = np.concatenate([pts_veh, np.ones((N, 1))], axis=1)
    T_cv = np.linalg.inv(T_vc)                                     # vehicle → camera
    pts_cam = (T_cv @ pts_veh_homo.T).T[:, :3]

    return dict(img=img, K=K3, dist=dist, T_vc=T_vc, T_vl=T_vl,
                IW=IW, IH=IH, pts_cam=pts_cam,
                frame_id=frame_dir.name)


def render(frame_dir: Path, out_path: Path):
    f = load_frame(frame_dir)
    uv, z = kannala_project(f["pts_cam"], f["K"], f["dist"])
    in_front = z > 0.5
    in_img = ((uv[:, 0] >= 0) & (uv[:, 0] < f["IW"])
              & (uv[:, 1] >= 0) & (uv[:, 1] < f["IH"]))
    m = in_front & in_img
    z_m, uv_m = z[m], uv[m]

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    ax.imshow(f["img"])
    sc = ax.scatter(uv_m[:, 0], uv_m[:, 1], c=z_m, s=0.4, cmap="turbo",
                    vmin=1.0, vmax=60.0, alpha=0.7)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="depth (m)")
    ax.set_title(
        f"ZOD frame {f['frame_id']}   {f['IW']}×{f['IH']}   "
        f"fx={f['K'][0,0]:.1f}  N_proj={m.sum()}/{len(z)}",
        fontsize=10,
    )
    ax.set_xlim(0, f["IW"])
    ax.set_ylim(f["IH"], 0)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return m.sum(), len(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="extracted frames_mini dir (contains single_frames/)")
    ap.add_argument("--out", required=True, help="output dir for PNGs")
    ap.add_argument("--n", type=int, default=6, help="number of frames to render")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frames = sorted((root / "single_frames").iterdir())[: args.n]
    print(f"Rendering {len(frames)} frames → {out}")
    for fd in frames:
        png = out / f"{fd.name}.png"
        n_proj, n_total = render(fd, png)
        print(f"  {fd.name}:  {n_proj}/{n_total} points projected  →  {png}")


if __name__ == "__main__":
    main()
