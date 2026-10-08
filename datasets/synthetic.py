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

    # depth uses a separate RNG so main rng sequence (image/grid/offsets)
    # is identical regardless of random_depths flag
    rng_d = torch.Generator()
    rng_d.manual_seed((seed if seed is not None else 0) + 1_000_000)
    if random_depths:
        for _ in range(30):
            _d = torch.rand(3, generator=rng_d)
            _ds, _ = _d.sort()
            if (_ds[1:] - _ds[:-1]).min().item() >= 0.10:
                break
        else:
            _ds = torch.tensor([0.10, 0.40, 0.80])
        d1, d2, d_bg = _ds[0].item(), _ds[1].item(), _ds[2].item()
    else:
        d1   = float(torch.rand(1, generator=rng_d).item() * 0.35 + 0.10)
        d2   = float(torch.rand(1, generator=rng_d).item() * 0.35 + 0.50)
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


def make_image_and_points_lidar(
    img_size: int = 64,
    max_offset: float = 16.0,
    seed: int | None = None,
    n_obj: tuple[int, int] = (1, 4),
    spacing: tuple[float, float] = (2.0, 12.0),
    jitter_frac: float = 0.25,
    drop: tuple[float, float] = (0.0, 0.3),
    spacing_v: tuple[float, float] | None = None,
    return_meta: bool = False,
    shift_bg: bool = True,     # False: 背景はずらさない (ずれ 0)。物体だけずらす
    keep_bg: bool = True,      # False: 背景の点を出さない (物体の点だけ)
):
    """ベンチの最小セット: 物体 n_obj 個 + 背景、深度はランダム、LiDAR の密度もランダム。

    物体:   既存の形 (矩形 / 円 / 楕円 / ポール) を n_obj 個。重なりを許し、
            近いものが手前に描かれる (z-buffer)。深度 U(0.05, 0.85)、背景は
            U(物体の最大 + 0.05, 1.0)。
    LiDAR:  横間隔 su と縦間隔 sv を spacing から別々に対数一様で引く (2 px = 64px 画像で
            32×32 の密、12 px ≈ 5×5 の疎)。原点はランダム、各点を ±jitter_frac·s
            だけ揺らし、drop の割合を抜く。各点の深度は真の位置の z-buffer の値
            (物体の裏の背景点は出ない)。
    ずれ:   深度グループ (物体ごと + 背景) ごとに独立に ±max_offset px。
            ずらした後に画像内に入る点だけ残す。

    Returns: image (3,H,W), true_uvd (N,3), dist_uvd (N,3)  d は [0,1]、背景が最大。
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    H = W = img_size
    U = lambda lo, hi: float(torch.rand(1, generator=rng).item() * (hi - lo) + lo)
    # シーンはクロップより P px ずつ大きいキャンバスに作り、最後に 64×64 を切り出す。
    # 64×64 の中だけで作ると、物体が画像の端で切れ、真の位置が外にある点は物体の上でも
    # 背景の深度になる → ずらすと「ずらした方向の端に物体の点が無い帯」ができ、
    # それがずれの手がかりになっていた。実際のクロップでは物体はクロップの外へ続く。
    # 座標はクロップ基準 (キャンバスでは +P)。
    P = int(math.ceil(max_offset)) + 4
    HB = H + 2 * P

    bg_color = torch.rand(3, generator=rng)
    image = bg_color[:, None, None].expand(3, HB, HB).clone()
    yy, xx = torch.meshgrid(torch.arange(HB, dtype=torch.float32) - P,
                            torch.arange(HB, dtype=torch.float32) - P, indexing="ij")
    in_crop = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)

    n = _randint(n_obj[0], n_obj[1] + 1, rng)
    objs = []
    for _ in range(n):
        for _try in range(50):
            st, small = _randint(0, 4, rng), bool(_randint(0, 2, rng))
            if st == 0:
                cx, cy, a, b = _rand_rect(rng, img_size, small=small)
            elif st == 1:
                cx, cy, a = _rand_circle(rng, img_size, small=small); b = a
            elif st == 2:
                cx, cy, a, b = _rand_ellipse(rng, img_size, small=small)
            else:
                cx, cy, a, b = _rand_pole(rng, img_size)
                # 端から端まで伸びるポール (半長 = img_size//2 - 1) はキャンバスの端まで伸ばす
                if max(a, b) == img_size // 2 - 1:
                    if b > a: b = HB // 2
                    else:     a = HB // 2
            # 大きさはそのまま、中心だけをキャンバス全体 (クロップ ±P) で動かす
            cx = cx + U(-P, P); cy = cy + U(-P, P)
            if st in (0, 3):
                m = (xx - cx).abs().lt(a) & (yy - cy).abs().lt(b)
            elif st == 1:
                m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= a ** 2
            else:
                m = ((xx - cx).pow(2) / a ** 2 + (yy - cy).pow(2) / b ** 2) <= 1.0
            if (m & in_crop).sum() >= 16:      # クロップに 16 px 以上入る物体だけ採用
                break
        c = torch.rand(3, generator=rng)
        while (c - bg_color).norm().item() < 0.35:
            c = torch.rand(3, generator=rng)
        objs.append((U(0.05, 0.85), m, c))
    d_bg = U(max(o[0] for o in objs) + 0.05, 1.0) if objs else 1.0

    # 遠い順に描く → 近いものが上書き (z-buffer)
    depth_map = torch.full((HB, HB), d_bg)
    for d, m, c in sorted(objs, key=lambda o: -o[0]):
        image[:, m] = c[:, None]
        depth_map[m] = d

    # 対数一様: 一様だと密 (両方 2 px 付近) がほとんど出ない (中央値 78 点)
    LU = lambda lo, hi: math.exp(U(math.log(lo), math.log(hi)))
    su, sv = LU(*spacing), LU(*(spacing_v or spacing))
    margin = max_offset
    gx = torch.arange(-margin + U(0, su), W + margin, su)
    gy = torch.arange(-margin + U(0, sv), H + margin, sv)
    gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
    xs, ys = gxx.reshape(-1), gyy.reshape(-1)
    xs = xs + (torch.rand(len(xs), generator=rng) * 2 - 1) * jitter_frac * su
    ys = ys + (torch.rand(len(ys), generator=rng) * 2 - 1) * jitter_frac * sv
    drop_frac = U(*drop)
    keep = torch.rand(len(xs), generator=rng) >= drop_frac
    xs, ys = xs[keep], ys[keep]
    # 全点の深度をキャンバスの z-buffer から取る (クロップの外の点も物体に当たりうる)
    xi = (xs + P).floor().long().clamp(0, HB - 1)
    yi = (ys + P).floor().long().clamp(0, HB - 1)
    depths = depth_map[yi, xi].clone()

    # ずれは別の乱数で引く。同じ seed なら点の密度に関係なく同じずれになる
    # (点の生成で消費した乱数の数に引きずられない)。物体を手前から順に
    # 0, 1, ... 番、背景を最後として、その順番でずれを割り当てる。
    rng_t = torch.Generator()
    rng_t.manual_seed((seed if seed is not None else 0) + 2_000_000)
    order = sorted([o[0] for o in objs]) + [d_bg]
    f32 = lambda x: float(torch.tensor(x, dtype=torch.float32))   # depths は float32
    tab = {f32(d): ((torch.rand(2, generator=rng_t) * 2 - 1) * max_offset) for d in order}
    if not shift_bg:
        tab[f32(d_bg)] = torch.zeros(2)
    true_parts, dist_parts, shifts = [], [], {}
    for d_val in sorted(set(depths.tolist())):
        if not keep_bg and d_val == f32(d_bg):
            continue
        sel = depths == d_val
        t = tab[d_val]
        shifts[d_val] = t.tolist()
        uv_true = torch.stack([xs[sel], ys[sel]], 1)
        uv_dist = uv_true + t[None]
        vis = ((uv_dist[:, 0] >= 0) & (uv_dist[:, 0] < W) &
               (uv_dist[:, 1] >= 0) & (uv_dist[:, 1] < H))
        if not vis.any():
            continue
        dc = torch.full((int(vis.sum()), 1), d_val)
        true_parts.append(torch.cat([uv_true[vis], dc], 1))
        dist_parts.append(torch.cat([uv_dist[vis], dc], 1))
    image = image[:, P:P + H, P:P + W].contiguous()
    if not true_parts:      # 物体に点が 1 つも当たらなかった (keep_bg=False のとき)
        true_parts = [torch.zeros(0, 3)]; dist_parts = [torch.zeros(0, 3)]
    out = (image, torch.cat(true_parts), torch.cat(dist_parts))
    if return_meta:
        return out + (dict(su=su, sv=sv, drop=drop_frac, d_bg=d_bg, shifts=shifts),)
    return out


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
    color: bool = False,   # True: 背景・物体をランダムな色の RGB (3ch) にする
    random_depth: bool = False,  # True: 深度をランダム (物体 U(0.05,0.85)、背景 U(物体の最大+0.05, 1))
    grid_spacing: float | None = None,  # 数値: 点をランダムでなく LiDAR 格子 (この間隔 px、原点ランダム、±0.25 間隔の揺れ) に置く
    area_uniform: bool = False,  # True: 画像全体に n_points 点をランダムに撒き、所属をその位置の形で決める (面積どおり)
    lidar_spacing: tuple[float, float] | None = None,  # (lo, hi): LiDAR のように横・縦の間隔を別々に対数一様で引き、0〜30% 間引く
    canvas: bool = False,  # True: 格子を画像の外 (各辺 max_offset) まで撒き、ずらした後に画像内の点だけ残す (端に寄せない)
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
    image = torch.zeros(1, H, W, dtype=torch.float32)   # 物体マスク (1 = 物体)。color のときも点の配置はこれで決める
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
    if grid_spacing is not None:
        # LiDAR 格子: 画像全体に間隔 grid_spacing の格子を置き、各点の所属 (物体 1 / 物体 2 /
        # 背景) をその位置の形で決める。点の数とグループの大きさは形と間隔で決まる。
        rng_g = torch.Generator()
        rng_g.manual_seed((seed if seed is not None else 0) + 5_000_000)
        if lidar_spacing is not None:
            lo, hi = math.log(lidar_spacing[0]), math.log(lidar_spacing[1])
            su = math.exp(float(torch.rand(1, generator=rng_g)) * (hi - lo) + lo)
            sv = math.exp(float(torch.rand(1, generator=rng_g)) * (hi - lo) + lo)
        else:
            su = sv = float(grid_spacing)
        sp = su
        mg = float(max_offset) if canvas else 0.0
        gx = torch.arange(-mg + float(torch.rand(1, generator=rng_g)) * su, W + mg, su)
        gy = torch.arange(-mg + float(torch.rand(1, generator=rng_g)) * sv, H + mg, sv)
        gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
        px = gxx.reshape(-1) + (torch.rand(gxx.numel(), generator=rng_g) * 2 - 1) * 0.25 * su
        py = gyy.reshape(-1) + (torch.rand(gyy.numel(), generator=rng_g) * 2 - 1) * 0.25 * sv
        if not canvas:
            px, py = px.clamp(0, W - 1), py.clamp(0, H - 1)
        if lidar_spacing is not None:
            keep = torch.rand(px.numel(), generator=rng_g) >= float(torch.rand(1, generator=rng_g)) * 0.3
            px, py = px[keep], py[keep]
        if area_uniform:
            # 格子でなく一様ランダム (点の数は格子と同じ)。格子かどうかと、物体の点の割合を切り分ける用
            px = torch.rand(px.numel(), generator=rng_g) * (W - 1)
            py = torch.rand(py.numel(), generator=rng_g) * (H - 1)
        yy_, xx_ = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                  torch.arange(W, dtype=torch.float32), indexing="ij")
        def _mask(st, cx, cy, a, b):
            if st == 0:
                m = torch.zeros(H, W, dtype=torch.bool); m[cy-b:cy+b, cx-a:cx+a] = True; return m
            return (xx_-cx)**2 + (yy_-cy)**2 <= a**2
        lab = torch.full((px.numel(),), 2, dtype=torch.long)      # 2 = 背景 (画像の外の点も背景)
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        ix, iy = px.floor().long().clamp(0, W - 1), py.floor().long().clamp(0, H - 1)
        for k, sh in enumerate(shapes):
            lab[_mask(*sh)[iy, ix] & inside] = k
        sel = [lab == 0, lab == 1, lab == 2]
        obj_xs, obj_ys = [px[sel[0]], px[sel[1]]], [py[sel[0]], py[sel[1]]]
        bg_x, bg_y = px[sel[2]], py[sel[2]]
        all_u = torch.cat([obj_xs[0], obj_xs[1], bg_x])
        all_v = torch.cat([obj_ys[0], obj_ys[1], bg_y])
        n_obj1, n_obj2, n_bg = int(sel[0].sum()), int(sel[1].sum()), int(sel[2].sum())
    else:
        n_obj1 = n_obj2 = n_obj

    # depth labels (normalised)
    if random_depth:
        # 深度は別の乱数で引く (点の配置・ずれは random_depth=False と同じになる)
        rng_d = torch.Generator()
        rng_d.manual_seed((seed if seed is not None else 0) + 4_000_000)
        d1 = float(torch.rand(1, generator=rng_d) * 0.80 + 0.05)
        d2 = float(torch.rand(1, generator=rng_d) * 0.80 + 0.05)
        lo = max(d1, d2) + 0.05
        dbg = float(torch.rand(1, generator=rng_d) * (1.0 - lo) + lo)
    else:
        d1, d2, dbg = DEPTH_OBJ1 / DEPTH_NORM, DEPTH_OBJ2 / DEPTH_NORM, DEPTH_BG / DEPTH_NORM
    depths = torch.cat([
        torch.full((n_obj1,), d1),
        torch.full((n_obj2,), d2),
        torch.full((n_bg,),  dbg),
    ])

    true_uvd = torch.stack([all_u, all_v, depths], dim=1)   # (N,3)

    # independent shifts per group
    group_sizes = [n_obj1, n_obj2, n_bg]
    dist_parts = []
    offset = 0
    for sz in group_sizes:
        tx = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        ty = (torch.rand(1, generator=rng)*2 - 1) * max_offset
        uv_i = true_uvd[offset:offset+sz, :2]
        uv_d  = uv_i + torch.stack([tx.expand(sz), ty.expand(sz)], dim=1)
        if not canvas:
            uv_d = uv_d.clamp(0, img_size-1)
        dist_parts.append(torch.cat([uv_d, true_uvd[offset:offset+sz, 2:3]], dim=1))
        offset += sz

    distorted_uvd = torch.cat(dist_parts, dim=0)   # (N,3)
    if canvas:
        # ずらした後に画像内に入る点だけ残す
        vis = ((distorted_uvd[:, 0] >= 0) & (distorted_uvd[:, 0] < W) &
               (distorted_uvd[:, 1] >= 0) & (distorted_uvd[:, 1] < H))
        true_uvd, distorted_uvd = true_uvd[vis], distorted_uvd[vis]
    if color:
        # 色は別の乱数で引く (点の配置・ずれは color=False と同じになる)
        rng_c = torch.Generator()
        rng_c.manual_seed((seed if seed is not None else 0) + 3_000_000)
        bgc = torch.rand(3, generator=rng_c)
        rgb = bgc[:, None, None].expand(3, H, W).clone()
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        for st, cx, cy, a, b in shapes:
            c = torch.rand(3, generator=rng_c)
            while (c - bgc).norm().item() < 0.35:
                c = torch.rand(3, generator=rng_c)
            m = torch.zeros(H, W, dtype=torch.bool)
            if st == 0:
                m[cy-b:cy+b, cx-a:cx+a] = True
            else:
                m = (xx-cx)**2 + (yy-cy)**2 <= a**2
            rgb[:, m] = c[:, None]
        image = rgb
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
