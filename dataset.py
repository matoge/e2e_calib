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


def _rand_pole(rng: torch.Generator, img_size: int = 128):
    """Returns (cx, cy, hw, hh) for a thin pole — thin width, long length.
    50% chance of spanning the full image (hh = img_size//2 - 1).
    """
    vertical  = int(torch.randint(0, 2, (1,), generator=rng).item())
    spanning  = int(torch.randint(0, 2, (1,), generator=rng).item())  # 50% full-span
    thin = max(3, _randint(3, 6, rng))                                # 3-5 px half-width → 6-10px visible
    if spanning:
        long = img_size // 2 - 1                                      # edge-to-edge
    else:
        s    = img_size / 128
        long = max(thin + 4, _randint(max(8, int(15 * s)), max(9, int(45 * s)) + 1, rng))
    if vertical:
        hw, hh = thin, long
        cx = _randint(hw + 1, img_size - hw - 1, rng)
        cy = img_size // 2 if spanning else _randint(hh + 1, img_size - hh - 1, rng)
    else:
        hw, hh = long, thin
        cy = _randint(hh + 1, img_size - hh - 1, rng)
        cx = img_size // 2 if spanning else _randint(hw + 1, img_size - hw - 1, rng)
    return cx, cy, hw, hh


def _rand_rect(rng: torch.Generator, img_size: int = 128, small: bool = False):
    """Returns (cx, cy, hw, hh) for a random rectangle."""
    s = img_size / 128
    lo = max(4, int((8  if small else 14) * s))
    hi = max(lo + 2, int((22 if small else 40) * s))
    hw = _randint(lo, hi, rng)
    hh = _randint(lo, hi, rng)
    cx = _randint(hw + 2, img_size - hw - 2, rng)
    cy = _randint(hh + 2, img_size - hh - 2, rng)
    return cx, cy, hw, hh


def _rand_circle(rng: torch.Generator, img_size: int = 128, small: bool = False):
    """Returns (cx, cy, r) for a random circle."""
    s = img_size / 128
    lo = max(3, int((7  if small else 12) * s))
    hi = max(lo + 2, int((20 if small else 38) * s))
    r  = _randint(lo, hi, rng)
    cx = _randint(r + 2, img_size - r - 2, rng)
    cy = _randint(r + 2, img_size - r - 2, rng)
    return cx, cy, r


def _rand_ellipse(rng: torch.Generator, img_size: int = 128, small: bool = False):
    """Returns (cx, cy, rx, ry) for a random ellipse."""
    s = img_size / 128
    lo = max(3, int((6  if small else 10) * s))
    hi = max(lo + 2, int((20 if small else 36) * s))
    rx = _randint(lo, hi, rng)
    ry = _randint(lo, hi, rng)
    # ensure aspect ratio is at least 1.3 (otherwise it's just a circle)
    if rx < ry:
        rx, ry = ry, rx
    ry = min(ry, max(3, int(rx * 0.65)))
    cx = _randint(rx + 2, img_size - rx - 2, rng)
    cy = _randint(ry + 2, img_size - ry - 2, rng)
    return cx, cy, rx, ry


def make_image_and_points_grid(
    img_size: int = 128,
    max_offset: float = 15.0,
    grid_size: float | None = None,  # None = random [4.0, 8.0]
    jitter: float | None = None,     # None = random [0.5, 2.0]
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Grid-sampled points (LiDAR-like): one point every grid_size pixels with jitter.
    Points are sampled over the whole image regardless of shape content.
    grid_size is a float, so the grid is truly irregular.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Randomise grid_size (float) and jitter if not specified
    if grid_size is None:
        grid_size = float(torch.rand(1, generator=rng).item() * 4 + 4)  # [4.0, 8.0]
    if jitter is None:
        jitter = float(torch.rand(1, generator=rng).item() * 1.5 + 0.5)  # [0.5, 2.0]

    H = W = img_size
    image = torch.zeros(1, H, W, dtype=torch.float32)

    # Draw shape (same as make_image_and_points)
    shape_type = int(torch.randint(0, 2, (1,), generator=rng).item())
    if shape_type == 0:
        cx, cy, hw, hh = _rand_rect(rng, img_size)
        image[0, cy - hh : cy + hh, cx - hw : cx + hw] = 1.0
    else:
        cx, cy, r = _rand_circle(rng, img_size)
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        image[0][(((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2)] = 1.0

    # Grid sampling with random origin — prevents memorising grid positions
    ox = float(torch.rand(1, generator=rng).item() * grid_size)
    oy = float(torch.rand(1, generator=rng).item() * grid_size)
    gx = torch.arange(ox, W, grid_size, dtype=torch.float32)
    gy = torch.arange(oy, H, grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    xs = grid_x.reshape(-1)
    ys = grid_y.reshape(-1)

    # Jitter
    n = len(xs)
    xs = (xs + (torch.rand(n, generator=rng) * 2 - 1) * jitter).clamp(0, W - 1)
    ys = (ys + (torch.rand(n, generator=rng) * 2 - 1) * jitter).clamp(0, H - 1)

    true_uv = torch.stack([xs, ys], dim=1)

    tx = (torch.rand(1, generator=rng) * 2 - 1) * max_offset
    ty = (torch.rand(1, generator=rng) * 2 - 1) * max_offset
    distorted_uv = (true_uv + torch.tensor([tx, ty])).clamp(0, img_size - 1)

    return image, true_uv, distorted_uv


def make_image_and_points_grid_depth(
    img_size: int = 64,
    max_offset: float = 16.0,
    grid_size: float | None = None,  # None → random [4.0, 8.0]
    jitter: float | None = None,     # None → random [0.5, 2.0]
    seed: int | None = None,
    random_depths: bool = False,     # True → all 3 depths random; False → legacy fixed ranges
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Grayscale image, 2 white shapes on black background.
    Grid-sampled points (float spacing, random origin).
    Each point gets depth: d1 (obj1), d2 (obj2), or 1.0 (bg).
    Output sorted by depth so groups are contiguous.

    Returns:
        image    (1, H, W)
        true_uvd (N, 3)  [u, v, d_norm]  sorted: obj1 | obj2 | bg
        dist_uvd (N, 3)  u,v shifted per group; d unchanged
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    if grid_size is None:
        grid_size = float(torch.rand(1, generator=rng).item() * 4 + 4)
    if jitter is None:
        jitter = float(torch.rand(1, generator=rng).item() * 1.5 + 0.5)

    H = W = img_size

    # random RGB colors: bg and each object get distinct random colors
    def rand_color():
        return torch.rand(3, generator=rng)

    bg_color = rand_color()
    obj_colors = []
    for _ in range(2):
        c = rand_color()
        # ensure contrast with bg (L2 distance >= 0.35)
        while (c - bg_color).norm().item() < 0.35:
            c = rand_color()
        obj_colors.append(c)

    image = bg_color[:, None, None].expand(3, H, W).clone()

    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                             torch.arange(W, dtype=torch.float32), indexing="ij")

    # Generate 2 non-overlapping shapes
    shapes = []
    for _attempt in range(300):
        if len(shapes) >= 2:
            break
        st    = _randint(0, 4, rng)   # 0=rect 1=circle 2=ellipse 3=pole
        small = bool(_randint(0, 2, rng))
        if st == 0:
            cx, cy, hw, hh = _rand_rect(rng, img_size, small=small)
            s = (st, cx, cy, hw, hh)
        elif st == 1:
            cx, cy, r = _rand_circle(rng, img_size, small=small)
            s = (st, cx, cy, r, r)
        elif st == 2:
            cx, cy, rx, ry = _rand_ellipse(rng, img_size, small=small)
            s = (st, cx, cy, rx, ry)
        else:
            cx, cy, hw, hh = _rand_pole(rng, img_size)
            s = (st, cx, cy, hw, hh)
        if all(not _shapes_overlap(s, prev, margin=4) for prev in shapes):
            shapes.append(s)
    while len(shapes) < 2:  # fallback
        st    = _randint(0, 4, rng)
        small = bool(_randint(0, 2, rng))
        if st == 0:
            cx, cy, hw, hh = _rand_rect(rng, img_size, small=small)
            shapes.append((st, cx, cy, hw, hh))
        elif st == 1:
            cx, cy, r = _rand_circle(rng, img_size, small=small)
            shapes.append((st, cx, cy, r, r))
        elif st == 2:
            cx, cy, rx, ry = _rand_ellipse(rng, img_size, small=small)
            shapes.append((st, cx, cy, rx, ry))
        else:
            cx, cy, hw, hh = _rand_pole(rng, img_size)
            shapes.append((st, cx, cy, hw, hh))

    masks = []
    for i, (st, cx, cy, a, b) in enumerate(shapes):
        if st in (0, 3):  # rect or pole
            mask = torch.zeros(H, W, dtype=torch.bool)
            mask[cy - b : cy + b, cx - a : cx + a] = True
        elif st == 1:     # circle
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= a ** 2
        else:             # ellipse
            mask = ((xx - cx).pow(2) / a ** 2 + (yy - cy).pow(2) / b ** 2) <= 1.0
        image[:, mask] = obj_colors[i][:, None]
        masks.append(mask)

    if random_depths:
        for _ in range(30):
            _d = torch.rand(3, generator=rng)
            _ds, _ = _d.sort()
            if (_ds[1:] - _ds[:-1]).min().item() >= 0.10:
                break
        else:
            _ds = torch.tensor([0.10, 0.40, 0.80])
        d1, d2, d_bg = _ds[0].item(), _ds[1].item(), _ds[2].item()
    else:
        d1   = float(torch.rand(1, generator=rng).item() * 0.35 + 0.10)  # [0.10, 0.45]
        d2   = float(torch.rand(1, generator=rng).item() * 0.35 + 0.50)  # [0.50, 0.85]
        d_bg = 1.0

    # grid over extended area (image + max_offset margin on all sides)
    # After shifting, only keep points whose DISTORTED position falls inside [0, img_size).
    # BG points that originate outside the frame create genuine uncertainty.
    margin = max_offset
    ox = float(torch.rand(1, generator=rng).item() * grid_size)
    oy = float(torch.rand(1, generator=rng).item() * grid_size)
    gx = torch.arange(-margin + ox, W + margin, grid_size, dtype=torch.float32)
    gy = torch.arange(-margin + oy, H + margin, grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    xs = grid_x.reshape(-1)
    ys = grid_y.reshape(-1)
    n  = len(xs)
    xs = xs + (torch.rand(n, generator=rng) * 2 - 1) * jitter
    ys = ys + (torch.rand(n, generator=rng) * 2 - 1) * jitter

    # assign depth using image mask (clamp to valid pixel coords for lookup)
    xi = xs.long().clamp(0, W - 1)
    yi = ys.long().clamp(0, H - 1)
    in_frame = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    depths = torch.full((n,), d_bg)
    depths[in_frame & masks[0][yi, xi]] = d1
    depths[in_frame & masks[1][yi, xi]] = d2

    # independent shift per depth group; keep only points visible AFTER shift
    # BG uses ALL grid points (including those on objects) so there is no
    # object-shaped hole in the BG dist cloud that would reveal the BG shift.
    true_parts, dist_parts = [], []
    for d_val in [d1, d2, d_bg]:
        if abs(d_val - d_bg) < 1e-5:  # BG: include every grid point
            sel = torch.ones(n, dtype=torch.bool)
        else:
            sel = (torch.abs(depths - d_val) < 1e-5)
        if not sel.any():
            continue
        tx = (torch.rand(1, generator=rng) * 2 - 1) * max_offset
        ty = (torch.rand(1, generator=rng) * 2 - 1) * max_offset
        uv_true = torch.stack([xs[sel], ys[sel]], dim=1)
        uv_dist = uv_true + torch.tensor([[tx.item(), ty.item()]])
        # keep only points whose distorted position is inside the image
        visible = ((uv_dist[:, 0] >= 0) & (uv_dist[:, 0] < W) &
                   (uv_dist[:, 1] >= 0) & (uv_dist[:, 1] < H))
        if not visible.any():
            continue
        d_col = torch.full((visible.sum(), 1), d_val)
        true_parts.append(torch.cat([uv_true[visible], d_col], dim=1))
        dist_parts.append(torch.cat([uv_dist[visible], d_col], dim=1))

    true_uvd = torch.cat(true_parts, dim=0)
    dist_uvd = torch.cat(dist_parts, dim=0)
    return image, true_uvd, dist_uvd


def collate_grid_depth(batch):
    """Collate variable-length point clouds by zero-padding to max N in batch."""
    imgs, true_list, dist_list = zip(*batch)
    imgs    = torch.stack(imgs)
    max_n   = max(t.shape[0] for t in true_list)
    B       = len(true_list)
    true_p  = torch.zeros(B, max_n, 3)
    dist_p  = torch.zeros(B, max_n, 3)
    pad_mask = torch.ones(B, max_n, dtype=torch.bool)   # True = padding (ignored in attn)
    for i, (t, d) in enumerate(zip(true_list, dist_list)):
        n = t.shape[0]
        true_p[i, :n] = t
        dist_p[i, :n] = d
        pad_mask[i, :n] = False
    return imgs, true_p, dist_p, pad_mask


class GridDepthDataset(Dataset):
    def __init__(self, length=4000, img_size=64,
                 max_offset=8.0, base_seed=0, random_each_epoch=False,
                 random_depths=False):
        self.length = length
        self.img_size = img_size
        self.max_offset = max_offset
        self.base_seed = base_seed
        self.random_each_epoch = random_each_epoch
        self.random_depths = random_depths

    def __len__(self): return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2 ** 30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        return make_image_and_points_grid_depth(
            img_size=self.img_size,
            max_offset=self.max_offset,
            seed=seed,
            random_depths=self.random_depths,
        )


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
    if t1 in (0, 3): cx1,cy1,hw1,hh1 = s1[1],s1[2],s1[3],s1[4]
    else:            cx1,cy1,hw1,hh1 = s1[1],s1[2],s1[3],s1[3]
    if t2 in (0, 3): cx2,cy2,hw2,hh2 = s2[1],s2[2],s2[3],s2[4]
    else:            cx2,cy2,hw2,hh2 = s2[1],s2[2],s2[3],s2[3]
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
    n_points: int = 255,   # total points (obj1 + obj2 + bg)
    img_size: int = 128,
    max_offset: float = 15.0,
    seed: int | None = None,
    bg_ratio: int = 1,     # bg points = n_obj_per * bg_ratio
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
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    H = W = img_size
    image = torch.zeros(1, H, W, dtype=torch.float32)
    n_obj = n_points // (2 + bg_ratio)   # points per object
    n_bg  = n_points - 2 * n_obj         # background points
    n_per = n_obj                         # alias for legacy code below

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
    while len(bg_xs) == 0 or torch.cat(bg_xs).shape[0] < n_bg:
        bx = torch.randint(0, img_size, (n_bg * 4,), generator=rng).float()
        by = torch.randint(0, img_size, (n_bg * 4,), generator=rng).float()
        ix = bx.long().clamp(0, img_size-1)
        iy = by.long().clamp(0, img_size-1)
        on_bg = image[0, iy, ix] < 0.5
        bx, by = bx[on_bg], by[on_bg]
        if bx.numel() > 0:
            bg_xs.append(bx); bg_ys.append(by)
    bg_x = torch.cat(bg_xs)[:n_bg]
    bg_y = torch.cat(bg_ys)[:n_bg]

    # stack all U,V coords
    all_u = torch.cat([obj_xs[0], obj_xs[1], bg_x])   # (N,)
    all_v = torch.cat([obj_ys[0], obj_ys[1], bg_y])   # (N,)

    # depth labels (normalised)
    depths = torch.cat([
        torch.full((n_obj,), DEPTH_OBJ1 / DEPTH_NORM),
        torch.full((n_obj,), DEPTH_OBJ2 / DEPTH_NORM),
        torch.full((n_bg,),  DEPTH_BG   / DEPTH_NORM),
    ])

    true_uvd = torch.stack([all_u, all_v, depths], dim=1)   # (N,3)

    # independent shifts per group
    group_sizes = [n_obj, n_obj, n_bg]
    dist_parts = []
    offset = 0
    for sz in group_sizes:
        tx = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        ty = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        uv_i = true_uvd[offset:offset+sz, :2]
        uv_d  = (uv_i + torch.stack([tx.expand(sz), ty.expand(sz)], dim=1)).clamp(0, img_size-1)
        dist_parts.append(torch.cat([uv_d, true_uvd[offset:offset+sz, 2:3]], dim=1))
        offset += sz

    distorted_uvd = torch.cat(dist_parts, dim=0)   # (N,3)
    return image, true_uvd, distorted_uvd


class DepthDataset(Dataset):
    def __init__(self, length=4000, n_points=255, img_size=128,
                 max_offset=15.0, base_seed=0, random_each_epoch=False, bg_ratio=1):
        self.length = length; self.n_points = n_points
        self.img_size = img_size; self.max_offset = max_offset
        self.base_seed = base_seed; self.random_each_epoch = random_each_epoch
        self.bg_ratio = bg_ratio

    def __len__(self): return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2**30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        return make_image_and_points_depth(self.n_points, self.img_size,
                                            self.max_offset, seed, self.bg_ratio)


def build_loaders_depth(train_size=8000, val_size=800, batch_size=32, num_workers=4,
                        bg_ratio=1):
    train_ds = DepthDataset(length=train_size, random_each_epoch=True, bg_ratio=bg_ratio)
    val_ds   = DepthDataset(length=val_size,   base_seed=400_000,      bg_ratio=bg_ratio)
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
