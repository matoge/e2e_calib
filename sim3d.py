"""
sim3d.py  —  Minimal 3D LiDAR/Camera simulator

Scene: ground plane + axis-aligned buildings (boxes) + vertical poles (cylinders)
Sensor: pinhole camera + spinning LiDAR, both at (0,0,CAM_HEIGHT)

Coordinate system
  X = forward  Y = left  Z = up
  Camera looks along +X

Output: same format as make_image_and_points_depth()
  image        (1, IMG_SIZE, IMG_SIZE) float32
  true_uvd     (N, 3)  [u, v, d_norm]  true projections
  distorted_uvd (N, 3)  with per-group UV offset added
"""

import math
import warnings
import numpy as np
import torch
np.seterr(divide="ignore", invalid="ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
CAM_FOV_H    = math.radians(70)
CAM_FOV_V    = math.radians(70)
CAM_HEIGHT   = 1.5          # m above ground

LIDAR_CHANNELS   = 16
LIDAR_V_UP       = math.radians(15)
LIDAR_V_DOWN     = math.radians(-15)
LIDAR_H_RES      = math.radians(0.5)   # 720 rays / ring
LIDAR_MAX_RANGE  = 50.0

DEPTH_NORM = 50.0   # normalisation  → d_norm ∈ [0,1]


# ── Primitives ─────────────────────────────────────────────────────────────────

class Box:
    """Axis-aligned box (building)."""
    def __init__(self, center, half_size, label="building"):
        self.c     = np.asarray(center,    float)
        self.h     = np.asarray(half_size, float)
        self.label = label

    def intersect(self, origins, dirs):
        """Vectorised AABB slab method.  returns t (N,), inf = no hit."""
        inv = np.where(np.abs(dirs) < 1e-12, np.sign(dirs) * np.inf, 1.0 / dirs)
        lo  = (self.c - self.h - origins) * inv
        hi  = (self.c + self.h - origins) * inv
        tmin = np.minimum(lo, hi).max(axis=1)
        tmax = np.maximum(lo, hi).min(axis=1)
        hit  = (tmax >= tmin) & (tmax > 1e-4)
        t    = np.where(hit, np.where(tmin > 1e-4, tmin, tmax), np.inf)
        return t


class Cylinder:
    """Vertical cylinder (pole), axis along Z."""
    def __init__(self, xy, radius, z_base, height, label="pole"):
        self.xy    = np.asarray(xy, float)
        self.r     = radius
        self.zb    = z_base
        self.zt    = z_base + height
        self.label = label

    def intersect(self, origins, dirs):
        ox = origins[:, 0] - self.xy[0]
        oy = origins[:, 1] - self.xy[1]
        dx, dy = dirs[:, 0], dirs[:, 1]
        a   = dx * dx + dy * dy
        b   = 2 * (ox * dx + oy * dy)
        c   = ox * ox + oy * oy - self.r ** 2
        disc = b * b - 4 * a * c
        ok   = (disc >= 0) & (a > 1e-12)
        sq   = np.sqrt(np.where(ok, disc, 0.0))
        a2   = 2 * a + 1e-30
        t1   = np.where(ok, (-b - sq) / a2, np.inf)
        t2   = np.where(ok, (-b + sq) / a2, np.inf)
        t    = np.full(len(origins), np.inf)
        for ti in (t1, t2):
            finite = np.isfinite(ti)
            z    = np.where(finite, origins[:, 2] + ti * dirs[:, 2], np.inf)
            good = (ti > 1e-4) & (z >= self.zb) & (z <= self.zt) & (ti < t)
            t    = np.where(good, ti, t)
        return t


class Ground:
    """Flat ground plane."""
    def __init__(self, z=0.0, label="ground"):
        self.z     = z
        self.label = label

    def intersect(self, origins, dirs):
        dz = dirs[:, 2]
        t  = np.where(np.abs(dz) > 1e-12,
                      (self.z - origins[:, 2]) / (dz + 1e-30),
                      np.inf)
        return np.where(t > 1e-4, t, np.inf)


# ── Scene ──────────────────────────────────────────────────────────────────────

class Scene:
    def __init__(self):
        self.objects: list = []

    def add(self, obj):
        self.objects.append(obj)

    def cast(self, origins, dirs):
        """Return (t, obj_idx) for each ray, shape (N,)."""
        best_t   = np.full(len(origins), np.inf)
        best_idx = np.full(len(origins), -1, int)
        for i, obj in enumerate(self.objects):
            t = obj.intersect(origins, dirs)
            better = t < best_t
            best_t   = np.where(better, t,  best_t)
            best_idx = np.where(better, i,  best_idx)
        return best_t, best_idx


def make_scene(rng: np.random.Generator) -> Scene:
    """Random scene: road + N buildings + M poles."""
    scene = Scene()
    scene.add(Ground(z=0.0, label="ground"))

    # Buildings on both sides
    for _ in range(rng.integers(3, 8)):
        x  = rng.uniform(4, 25)
        y  = rng.choice([-1, 1]) * rng.uniform(3, 9)
        hw = rng.uniform(1, 4)
        hd = rng.uniform(1, 4)
        hh = rng.uniform(2, 7)
        scene.add(Box([x, y, hh], [hw, hd, hh], "building"))

    # Poles along road edges
    for _ in range(rng.integers(2, 5)):
        x = rng.uniform(2, 28)
        y = rng.choice([-1, 1]) * rng.uniform(1.5, 4)
        scene.add(Cylinder([x, y], 0.08, 0.0, rng.uniform(2, 5), "pole"))

    # Cars on road (boxes ~4m×2m×1.5m)
    for _ in range(rng.integers(1, 4)):
        x  = rng.uniform(5, 25)
        y  = rng.uniform(-1.5, 1.5)   # near road center
        hx = rng.uniform(1.8, 2.5)    # half-length
        hy = rng.uniform(0.8, 1.0)    # half-width
        hz = 0.75                      # half-height (1.5m tall)
        scene.add(Box([x, y, hz], [hx, hy, hz], "car"))

    return scene


# ── Camera renderer ────────────────────────────────────────────────────────────

def render_camera(scene: Scene, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Ray-trace a grayscale image (H, W) float32 ∈ [0, 1].
    Camera at (0, 0, CAM_HEIGHT) looking along +X.
    """
    H = W = img_size
    fx = (W / 2) / math.tan(CAM_FOV_H / 2)
    fy = (H / 2) / math.tan(CAM_FOV_V / 2)
    cx = cy = img_size / 2.0

    u_px, v_px = np.meshgrid(np.arange(W), np.arange(H))
    rx = np.ones((H, W)) * fx
    ry = cx - u_px           # +Y = left
    rz = cy - v_px           # +Z = up
    dirs = np.stack([rx, ry, rz], axis=-1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    origins = np.broadcast_to([0.0, 0.0, CAM_HEIGHT], (H * W, 3)).copy()
    t, idx  = scene.cast(origins, dirs)

    img  = np.zeros(H * W, np.float32)
    hit  = t < LIDAR_MAX_RANGE

    for i, obj in enumerate(scene.objects):
        mask = (idx == i) & hit
        if not mask.any():
            continue
        if obj.label == "ground":
            img[mask] = 0.12
        elif obj.label == "building":
            img[mask] = 0.45 + 0.55 * (1 - t[mask] / LIDAR_MAX_RANGE)
        elif obj.label == "pole":
            img[mask] = 1.0
        elif obj.label == "car":
            img[mask] = 0.72 + 0.28 * (1 - t[mask] / LIDAR_MAX_RANGE)

    return img.reshape(H, W)


# ── LiDAR scanner ──────────────────────────────────────────────────────────────

def scan_lidar(scene: Scene):
    """
    Spinning LiDAR at (0,0,CAM_HEIGHT).
    Returns pts (M,3), label_name (M,) as strings.
    """
    v_angs = np.linspace(LIDAR_V_DOWN, LIDAR_V_UP, LIDAR_CHANNELS)
    h_angs = np.arange(0, 2 * math.pi, LIDAR_H_RES)
    vv, hh = np.meshgrid(v_angs, h_angs)
    vv, hh = vv.ravel(), hh.ravel()
    cv = np.cos(vv)
    dirs = np.stack([cv * np.cos(hh), cv * np.sin(hh), np.sin(vv)], axis=1)

    n = len(dirs)
    origins = np.broadcast_to([0.0, 0.0, CAM_HEIGHT], (n, 3)).copy()
    t, idx  = scene.cast(origins, dirs)

    hit  = (t < LIDAR_MAX_RANGE) & (idx >= 0)
    pts  = origins[hit] + t[hit, None] * dirs[hit]
    lbls = np.array([scene.objects[i].label for i in idx[hit]])
    return pts, lbls


# ── UV projection ──────────────────────────────────────────────────────────────

def project(pts_world: np.ndarray, img_size: int = IMG_SIZE):
    """
    Project world points to camera UV.
    Returns u, v, depth — only in-FOV points.
    """
    p = pts_world.copy()
    p[:, 2] -= CAM_HEIGHT          # camera frame: z = world_z - cam_h
    fwd = p[:, 0] > 0.3
    p   = p[fwd]

    H = W = img_size
    fx = (W / 2) / math.tan(CAM_FOV_H / 2)
    fy = (H / 2) / math.tan(CAM_FOV_V / 2)
    cx = cy = img_size / 2.0

    d = p[:, 0]
    u = cx - fx * p[:, 1] / d
    v = cy - fy * p[:, 2] / d

    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[ok], v[ok], d[ok], fwd.nonzero()[0][ok]


# ── Full sample ────────────────────────────────────────────────────────────────

def make_sample(seed=None, n_points=255, max_offset=8.0, bg_ratio=1,
                img_size=IMG_SIZE):
    """
    Same interface as make_image_and_points_depth().
    Returns:
        image        (1, img_size, img_size) float32
        true_uvd     (N, 3)  [u, v, d_norm]
        distorted_uvd (N, 3)
    Groups:
        [0 : n_obj]       obj1  — left-side objects  (Y > 0)
        [n_obj : 2*n_obj] obj2  — right-side objects (Y < 0)
        [2*n_obj : N]     bg    — ground points
    """
    rng   = np.random.default_rng(seed)
    scene = make_scene(rng)

    img_np = render_camera(scene, img_size)
    image  = torch.from_numpy(img_np).unsqueeze(0)

    pts, lbls = scan_lidar(scene)
    u, v, d, src = project(pts, img_size)
    lbls_proj = lbls[src]

    # Classify
    is_bg  = lbls_proj == "ground"
    is_fg  = ~is_bg
    # Further split FG: left (Y>0) vs right (Y<0)
    fy_val = pts[src, 1]
    is_left  = is_fg & (fy_val >= 0)
    is_right = is_fg & (fy_val <  0)

    n_obj   = n_points // (2 + bg_ratio)
    n_bg_pt = n_points - 2 * n_obj

    def _sample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            idx = np.where(is_fg)[0]  # fallback: any FG
        if len(idx) == 0:
            idx = np.arange(len(u))
        return rng.choice(idx, n, replace=len(idx) < n)

    s1 = _sample(is_left,  n_obj)
    s2 = _sample(is_right, n_obj)
    sb = _sample(is_bg,    n_bg_pt)
    sel = np.concatenate([s1, s2, sb])

    d_norm = np.clip(d[sel] / DEPTH_NORM, 0.0, 1.0)
    true_uvd = torch.tensor(
        np.stack([u[sel], v[sel], d_norm], axis=1), dtype=torch.float32)

    # Per-group UV offset (calibration error)
    sizes = [n_obj, n_obj, n_bg_pt]
    dist_parts = []
    off = 0
    for sz in sizes:
        tx = float(rng.uniform(-1, 1) * max_offset)
        ty = float(rng.uniform(-1, 1) * max_offset)
        uv  = true_uvd[off:off+sz, :2].clone()
        uvd = (uv + torch.tensor([tx, ty])).clamp(0, img_size - 1)
        dist_parts.append(torch.cat([uvd, true_uvd[off:off+sz, 2:3]], dim=1))
        off += sz

    distorted_uvd = torch.cat(dist_parts, dim=0)
    return image, true_uvd, distorted_uvd


# ── Dataset wrappers ───────────────────────────────────────────────────────────

from torch.utils.data import Dataset, DataLoader


class Sim3DDataset(Dataset):
    def __init__(self, length=4000, n_points=255, max_offset=15.0,
                 bg_ratio=1, base_seed=0, random_each_epoch=False):
        self.length = length
        self.n_points = n_points
        self.max_offset = max_offset
        self.bg_ratio = bg_ratio
        self.base_seed = base_seed
        self.random_each_epoch = random_each_epoch

    def __len__(self): return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2**30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        return make_sample(seed, self.n_points, self.max_offset, self.bg_ratio)


def build_loaders_sim3d(train_size=8000, val_size=800, batch_size=32,
                        num_workers=4, bg_ratio=1):
    train_ds = Sim3DDataset(length=train_size, random_each_epoch=True,  bg_ratio=bg_ratio)
    val_ds   = Sim3DDataset(length=val_size,   base_seed=500_000,       bg_ratio=bg_ratio)
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw))


# ── Quick visualisation ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=100)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img, true_uvd, dist_uvd = make_sample(seed=i * 17, bg_ratio=1)
        n = img.shape[-1]
        n_obj = true_uvd.shape[0] // 3

        ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1,
                  origin="upper", extent=[0, n, n, 0])
        ax.scatter(true_uvd[:n_obj, 0],       true_uvd[:n_obj, 1],       c="#00ff88", s=4, alpha=0.7, linewidths=0)
        ax.scatter(true_uvd[n_obj:2*n_obj, 0],true_uvd[n_obj:2*n_obj, 1],c="#00ccff", s=4, alpha=0.7, linewidths=0)
        ax.scatter(true_uvd[2*n_obj:, 0],      true_uvd[2*n_obj:, 1],     c="white",   s=2, alpha=0.3, linewidths=0)
        ax.scatter(dist_uvd[:, 0], dist_uvd[:, 1], c="red", s=2, alpha=0.4, linewidths=0)

        d_obj = true_uvd[:2*n_obj, 2].mean().item() * DEPTH_NORM
        d_bg  = true_uvd[2*n_obj:, 2].mean().item() * DEPTH_NORM
        ax.set_title(f"s{i}  d_obj={d_obj:.1f}m  d_bg={d_bg:.1f}m", fontsize=7)
        ax.set_xlim(0, n); ax.set_ylim(n, 0)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("sim3d: road+buildings+poles  green=obj1  cyan=obj2  white=bg  red=distorted",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig("result_sim3d.png", dpi=120, bbox_inches="tight")
    print("Saved → result_sim3d.png")
