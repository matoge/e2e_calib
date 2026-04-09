"""
dataset.py  –  128×128 synthetic calibration dataset
Images: (B, 1, 128, 128) black background, white shapes.
true_uv: (B, N, 2) N=256 points sampled inside / on shape edges.
distorted_uv: true_uv + smooth random offset (max ±15 px).
"""
import math
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _randint(lo: int, hi: int, rng: torch.Generator) -> int:
    """Uniform integer in [lo, hi)."""
    return lo + int(torch.randint(hi - lo, (1,), generator=rng).item())


def _rand_rect(rng: torch.Generator, img_size: int = 128, small: bool = False):
    """Returns (cx, cy, hw, hh) for a random rectangle."""
    lo, hi = (8, 22) if small else (14, 40)
    hw = _randint(lo, hi, rng)
    hh = _randint(lo, hi, rng)
    cx = _randint(hw + 2, img_size - hw - 2, rng)
    cy = _randint(hh + 2, img_size - hh - 2, rng)
    return cx, cy, hw, hh


def _rand_circle(rng: torch.Generator, img_size: int = 128, small: bool = False):
    """Returns (cx, cy, r) for a random circle."""
    lo, hi = (7, 20) if small else (12, 38)
    r  = _randint(lo, hi, rng)
    cx = _randint(r + 2, img_size - r - 2, rng)
    cy = _randint(r + 2, img_size - r - 2, rng)
    return cx, cy, r


def make_image_and_points(
    n_points: int = 256,
    img_size: int = 128,
    max_offset: float = 15.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        image        (1, H, W) float32 in [0, 1]
        true_uv      (N, 2) float32  in pixel coords  [0, img_size)
        distorted_uv (N, 2) float32
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    H = W = img_size
    image = torch.zeros(1, H, W, dtype=torch.float32)

    # --- choose shape type ---
    shape_type = int(torch.randint(0, 2, (1,), generator=rng).item())  # 0=rect 1=circle

    if shape_type == 0:
        cx, cy, hw, hh = _rand_rect(rng, img_size)
        image[0, cy - hh : cy + hh, cx - hw : cx + hw] = 1.0

        # uniform interior sampling
        xs = torch.randint(cx - hw, cx + hw, (n_points,), generator=rng).float()
        ys = torch.randint(cy - hh, cy + hh, (n_points,), generator=rng).float()

    else:  # circle
        cx, cy, r = _rand_circle(rng, img_size)
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        image[0][mask] = 1.0

        # uniform interior sampling via rejection
        collected_x, collected_y = [], []
        while len(collected_x) == 0 or torch.cat(collected_x).shape[0] < n_points:
            bx = torch.randint(cx - r, cx + r + 1, (n_points * 4,), generator=rng).float()
            by = torch.randint(cy - r, cy + r + 1, (n_points * 4,), generator=rng).float()
            inside = ((bx - cx) ** 2 + (by - cy) ** 2) <= r ** 2
            bx, by = bx[inside], by[inside]
            if bx.numel() > 0:
                collected_x.append(bx)
                collected_y.append(by)
        xs = torch.cat(collected_x)[:n_points]
        ys = torch.cat(collected_y)[:n_points]

    true_uv = torch.stack([xs, ys], dim=1)  # (N, 2)  [x, y]

    # --- uniform translation (same shift for all points) ---
    tx = (torch.rand(1, generator=rng) * 2 - 1) * max_offset  # scalar in [-max, +max]
    ty = (torch.rand(1, generator=rng) * 2 - 1) * max_offset

    offset = torch.stack([tx.expand(n_points), ty.expand(n_points)], dim=1)  # (N,2)

    distorted_uv = (true_uv + offset).clamp(0.0, img_size - 1.0)

    return image, true_uv, distorted_uv


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalibDataset(Dataset):
    def __init__(
        self,
        length: int = 4000,
        n_points: int = 256,
        img_size: int = 128,
        max_offset: float = 15.0,
        base_seed: int = 0,
        random_each_epoch: bool = False,
    ):
        self.length = length
        self.n_points = n_points
        self.img_size = img_size
        self.max_offset = max_offset
        self.base_seed = base_seed
        self.random_each_epoch = random_each_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if self.random_each_epoch:
            # Different sample every epoch — prevents memorisation
            seed = int(torch.randint(0, 2**30, (1,)).item())
        else:
            seed = self.base_seed + idx
        img, true_uv, dist_uv = make_image_and_points(
            n_points=self.n_points,
            img_size=self.img_size,
            max_offset=self.max_offset,
            seed=seed,
        )
        return img, true_uv, dist_uv


def build_loaders(
    train_size: int = 4000,
    val_size: int = 400,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_ds = CalibDataset(length=train_size, base_seed=0, random_each_epoch=True)
    val_ds   = CalibDataset(length=val_size,   base_seed=100_000, random_each_epoch=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Multi-object (2 objects, independent shifts)
# ---------------------------------------------------------------------------

def _sample_shape_points(shape_type, cx, cy, hw_or_r, hh_or_r, n, rng, img_size):
    """Sample n points uniformly inside a shape. shape_type 0=rect, 1=circle."""
    if shape_type == 0:
        hw, hh = hw_or_r, hh_or_r
        xs = torch.randint(cx - hw, cx + hw, (n,), generator=rng).float()
        ys = torch.randint(cy - hh, cy + hh, (n,), generator=rng).float()
    else:
        r = hw_or_r
        collected_x, collected_y = [], []
        while len(collected_x) == 0 or torch.cat(collected_x).shape[0] < n:
            bx = torch.randint(cx - r, cx + r + 1, (n * 4,), generator=rng).float()
            by = torch.randint(cy - r, cy + r + 1, (n * 4,), generator=rng).float()
            inside = ((bx - cx)**2 + (by - cy)**2) <= r**2
            bx, by = bx[inside], by[inside]
            if bx.numel() > 0:
                collected_x.append(bx); collected_y.append(by)
        xs = torch.cat(collected_x)[:n]
        ys = torch.cat(collected_y)[:n]
    return xs.clamp(0, img_size-1), ys.clamp(0, img_size-1)


def _shapes_overlap(s1, s2, margin=4):
    """Returns True if two bounding boxes overlap (with margin)."""
    t1, t2 = s1[0], s2[0]  # shape type
    if t1 == 0:  cx1,cy1,hw1,hh1 = s1[1],s1[2],s1[3],s1[4]
    else:        cx1,cy1,hw1,hh1 = s1[1],s1[2],s1[3],s1[3]
    if t2 == 0:  cx2,cy2,hw2,hh2 = s2[1],s2[2],s2[3],s2[4]
    else:        cx2,cy2,hw2,hh2 = s2[1],s2[2],s2[3],s2[3]
    return (abs(cx1-cx2) < hw1+hw2+margin and abs(cy1-cy2) < hh1+hh2+margin)


def make_image_and_points_multi(
    n_points: int = 256,       # total; split equally between 2 objects
    img_size: int = 128,
    max_offset: float = 15.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two objects with independent uniform shifts.
    Returns:
        image        (1, H, W)
        true_uv      (N, 2)   first N//2 = obj1, last N//2 = obj2
        distorted_uv (N, 2)
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    H = W = img_size
    image = torch.zeros(1, H, W, dtype=torch.float32)
    n_per = n_points // 2

    # Generate 2 non-overlapping shapes
    shapes = []
    attempts = 0
    while len(shapes) < 2:
        st = int(torch.randint(0, 2, (1,), generator=rng).item())
        if st == 0:
            cx, cy, hw, hh = _rand_rect(rng, img_size, small=True)
            s = (st, cx, cy, hw, hh)
        else:
            cx, cy, r = _rand_circle(rng, img_size, small=True)
            s = (st, cx, cy, r, r)
        if all(not _shapes_overlap(s, prev) for prev in shapes):
            shapes.append(s)
        attempts += 1
        if attempts > 200:  # fallback: accept overlap
            shapes.append(s)
            break

    all_xs, all_ys = [], []
    for st, cx, cy, a, b in shapes:
        if st == 0:  # rect
            image[0, cy-b:cy+b, cx-a:cx+a] = 1.0
        else:        # circle
            yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                    torch.arange(W, dtype=torch.float32), indexing="ij")
            image[0][(xx-cx)**2 + (yy-cy)**2 <= a**2] = 1.0
        xs, ys = _sample_shape_points(st, cx, cy, a, b, n_per, rng, img_size)
        all_xs.append(xs); all_ys.append(ys)

    true_uv = torch.stack([torch.cat(all_xs), torch.cat(all_ys)], dim=1)  # (N,2)

    # Independent shift per object
    dist_parts = []
    for i in range(2):
        tx = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        ty = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        uv_i = true_uv[i*n_per:(i+1)*n_per]
        dist_parts.append((uv_i + torch.stack([tx.expand(n_per),
                                                ty.expand(n_per)], dim=1)).clamp(0, img_size-1))

    distorted_uv = torch.cat(dist_parts, dim=0)
    return image, true_uv, distorted_uv


class MultiObjDataset(Dataset):
    def __init__(self, length=4000, n_points=256, img_size=128,
                 max_offset=15.0, base_seed=0, random_each_epoch=False):
        self.length = length; self.n_points = n_points
        self.img_size = img_size; self.max_offset = max_offset
        self.base_seed = base_seed; self.random_each_epoch = random_each_epoch

    def __len__(self): return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2**30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        return make_image_and_points_multi(self.n_points, self.img_size,
                                           self.max_offset, seed)


# ---------------------------------------------------------------------------
# Depth-aware dataset: 2 objects + background plane
# Input points: (U, V, D_norm)  D_norm = depth / 50.0
#   obj1   depth=10  → n_pts//3 points on shape1
#   obj2   depth=20  → n_pts//3 points on shape2
#   bg     depth=40  → remaining points, uniformly scattered in background
# Each group has its own independent (tx, ty) shift.
# ---------------------------------------------------------------------------

DEPTH_OBJ1 = 10.0
DEPTH_OBJ2 = 20.0
DEPTH_BG   = 40.0
DEPTH_NORM = 50.0   # normalisation factor → [0,1] range


def make_image_and_points_depth(
    n_points: int = 255,   # divisible by 3
    img_size: int = 128,
    max_offset: float = 15.0,
    seed: int | None = None,
):
    """
    Returns:
        image        (1, H, W)
        true_uvd     (N, 3)  [U, V, D_norm]   true positions
        distorted_uvd (N, 3) [U, V, D_norm]   distorted (only U,V shifted; D unchanged)

    Groups (contiguous):
        [0   : n//3]       obj1  depth=DEPTH_OBJ1
        [n//3: 2*n//3]     obj2  depth=DEPTH_OBJ2
        [2*n//3 : n]       bg    depth=DEPTH_BG
    """
    n_points = (n_points // 3) * 3   # ensure divisible by 3
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    H = W = img_size
    image = torch.zeros(1, H, W, dtype=torch.float32)
    n_per = n_points // 3

    # --- 2 non-overlapping shapes (smaller) ---
    shapes = []
    for _ in range(200):
        if len(shapes) >= 2:
            break
        st = int(torch.randint(0, 2, (1,), generator=rng).item())
        if st == 0:
            cx, cy, hw, hh = _rand_rect(rng, img_size, small=True)
            s = (st, cx, cy, hw, hh)
        else:
            cx, cy, r = _rand_circle(rng, img_size, small=True)
            s = (st, cx, cy, r, r)
        if all(not _shapes_overlap(s, prev, margin=6) for prev in shapes):
            shapes.append(s)

    while len(shapes) < 2:   # fallback
        st = int(torch.randint(0, 2, (1,), generator=rng).item())
        if st == 0:
            cx, cy, hw, hh = _rand_rect(rng, img_size, small=True)
            shapes.append((st, cx, cy, hw, hh))
        else:
            cx, cy, r = _rand_circle(rng, img_size, small=True)
            shapes.append((st, cx, cy, r, r))

    # render shapes and sample object points
    obj_xs, obj_ys = [], []
    for st, cx, cy, a, b in shapes:
        if st == 0:
            image[0, cy-b:cy+b, cx-a:cx+a] = 1.0
        else:
            yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                    torch.arange(W, dtype=torch.float32), indexing="ij")
            image[0][(xx-cx)**2 + (yy-cy)**2 <= a**2] = 1.0
        xs, ys = _sample_shape_points(st, cx, cy, a, b, n_per, rng, img_size)
        obj_xs.append(xs); obj_ys.append(ys)

    # background points: uniform over pixels NOT on any shape
    bg_xs, bg_ys = [], []
    while len(bg_xs) == 0 or torch.cat(bg_xs).shape[0] < n_per:
        bx = torch.randint(0, img_size, (n_per * 4,), generator=rng).float()
        by = torch.randint(0, img_size, (n_per * 4,), generator=rng).float()
        # keep only background pixels (image value == 0)
        ix = bx.long().clamp(0, img_size-1)
        iy = by.long().clamp(0, img_size-1)
        on_bg = image[0, iy, ix] < 0.5
        bx, by = bx[on_bg], by[on_bg]
        if bx.numel() > 0:
            bg_xs.append(bx); bg_ys.append(by)
    bg_x = torch.cat(bg_xs)[:n_per]
    bg_y = torch.cat(bg_ys)[:n_per]

    # stack all U,V coords
    all_u = torch.cat([obj_xs[0], obj_xs[1], bg_x])   # (N,)
    all_v = torch.cat([obj_ys[0], obj_ys[1], bg_y])   # (N,)

    # depth labels (normalised)
    depths = torch.cat([
        torch.full((n_per,), DEPTH_OBJ1 / DEPTH_NORM),
        torch.full((n_per,), DEPTH_OBJ2 / DEPTH_NORM),
        torch.full((n_per,), DEPTH_BG   / DEPTH_NORM),
    ])

    true_uvd = torch.stack([all_u, all_v, depths], dim=1)   # (N,3)

    # independent shifts per group
    dist_parts = []
    for i in range(3):
        tx = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        ty = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        uv_i = true_uvd[i*n_per:(i+1)*n_per, :2]
        uv_d  = (uv_i + torch.stack([tx.expand(n_per),
                                      ty.expand(n_per)], dim=1)).clamp(0, img_size-1)
        dist_parts.append(torch.cat([uv_d, true_uvd[i*n_per:(i+1)*n_per, 2:3]], dim=1))

    distorted_uvd = torch.cat(dist_parts, dim=0)   # (N,3)
    return image, true_uvd, distorted_uvd


class DepthDataset(Dataset):
    def __init__(self, length=4000, n_points=255, img_size=128,
                 max_offset=15.0, base_seed=0, random_each_epoch=False):
        self.length = length; self.n_points = n_points
        self.img_size = img_size; self.max_offset = max_offset
        self.base_seed = base_seed; self.random_each_epoch = random_each_epoch

    def __len__(self): return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2**30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        return make_image_and_points_depth(self.n_points, self.img_size,
                                            self.max_offset, seed)


def build_loaders_depth(train_size=8000, val_size=800, batch_size=32, num_workers=4):
    train_ds = DepthDataset(length=train_size, random_each_epoch=True)
    val_ds   = DepthDataset(length=val_size,   base_seed=400_000)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),
    )


def build_loaders_multi(train_size=8000, val_size=800, batch_size=32, num_workers=4):
    train_ds = MultiObjDataset(length=train_size, random_each_epoch=True)
    val_ds   = MultiObjDataset(length=val_size,   base_seed=300_000)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True, persistent_workers=True),
    )
