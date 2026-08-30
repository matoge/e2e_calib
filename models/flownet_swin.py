"""FlowNetSwin — where does this point go in the other frame, and how sure are we.

The problem is the same one CalibNet2 solves, with the transform relabelled: calibration asks how a
fixed camera->lidar offset shifts a projection, pose asks how a per-frame rigid motion does. Both
want a function that takes a hypothesis transform and returns the residual with a covariance. This
net is the pose-across-frames form, sized for full images rather than object crops.

What is different from CalibNetDepth / CalibNet2:

  * ONE token stream. Image patches and lidar returns land in the same 16x16 grid and are fused
    there, instead of the lidar living in a separate Q built by PointMLP3. A patch that has lidar
    carries it; one that does not still exists and still answers - which matters because lidar
    covers 31% of the 16x16 patches inside the front camera and 0.21% of the pixels, and the far
    points that pin rotation have none at all.

  * Swin, not global self-attention. At 1600x900 a /16 grid is 100x56 = 5600 tokens and global
    attention is 31 M pairs; at 4K it is 32400 tokens and 1.05 G. Windowed attention with shifts is
    linear in the token count and its receptive field is a design parameter (window x depth) rather
    than "everything". Cost volumes are local for the same reason.

  * Depth is a FEATURE, not a position. Position encoding is (u, v) only. Depth enters as a channel
    of the token, because it is an observation about what is there, not where the token is. It is
    carried as inverse depth: d/50 clipped to [0,1] (what the crop-based nets used) puts every point
    past 50 m at exactly 1.0, and the far band here runs to 140 m. Inverse depth is also what the
    disparity is proportional to, so it lines up with the residual the head predicts.

  * The query carries a SUB-PIXEL uv. A patch is 16 px; two points inside one would otherwise be the
    same query. The continuous (u, v) is Fourier-lifted and added to the patch token, so the
    position the head can resolve is not limited by the patch grid.

  * The hypothesis transform rotates the query (RoPE, as in CalibNet2's RoPEPoseEmb) rather than
    switching the KV. KV stays the target frame's own tokens.

  * Distortion is left alone. Residuals live in distorted pixel coordinates, which is where the
    image is; undistorting resamples and stretches the periphery. The distortion model is applied
    forward when a pose step turns a 3D point into a predicted pixel, so no inverse is ever needed.

Output per query: (du, dv, log_su, log_sv, rho, logit_vis)
  du, dv     correction to the hypothesis position, in target-frame pixels
  log_su/sv  anisotropic sigma - a point on an edge is uncertain ALONG the edge and sure across it,
             and a scalar confidence cannot say that
  rho        correlation, so the covariance can tilt with the edge
  logit_vis  visibility in [0,1] after sigmoid, supervised by the Gaussian transmittance
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fourier lift for continuous coordinates
# ---------------------------------------------------------------------------
class FourierUV(nn.Module):
    """(u, v) in [0,1] -> [u, v, sin(pi 2^k u), cos(...), ...] -> d dims."""

    def __init__(self, d: int, n_freq: int = 6):
        super().__init__()
        self.n_freq = n_freq
        self.proj = nn.Linear(2 * (1 + 2 * n_freq), d)
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:      # (..., 2) -> (..., d)
        a = uv[..., None] * self.freqs                        # (..., 2, F)
        lift = torch.cat([uv, torch.sin(a).flatten(-2), torch.cos(a).flatten(-2)], -1)
        return self.proj(lift)


# ---------------------------------------------------------------------------
# Windowed (Swin) self-attention over the token grid
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, w: int):
    """(B, H, W, C) -> (B*nH*nW, w*w, C)"""
    B, H, Wd, C = x.shape
    x = x.view(B, H // w, w, Wd // w, w, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, w * w, C)


def window_reverse(win: torch.Tensor, w: int, B: int, H: int, Wd: int):
    C = win.shape[-1]
    x = win.view(B, H // w, Wd // w, w, w, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, Wd, C)


class SwinBlock(nn.Module):
    """Pre-norm windowed self-attention + FFN, with an optional half-window shift.

    The shift is what stops the windows from being independent images: alternating shifted and
    unshifted blocks lets information cross a window edge, so a feature that straddles a boundary is
    not cut in two. Receptive field after L blocks is about window * L, which is the number to set
    from how far apart the things that need to see each other are.
    """

    def __init__(self, d: int, n_heads: int = 4, window: int = 8, shift: bool = False):
        super().__init__()
        self.window, self.shift = window, (window // 2 if shift else 0)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d))

    def forward(self, x: torch.Tensor, H: int, Wd: int) -> torch.Tensor:  # (B, H*W, C)
        B, L, C = x.shape
        w = self.window
        # pad up to a whole number of windows; the pad is masked out by being zero-valued and
        # dropped again, so it cannot leak into a real token's answer
        ph, pw = (w - H % w) % w, (w - Wd % w) % w
        h = x.view(B, H, Wd, C)
        if ph or pw:
            h = F.pad(h, (0, 0, 0, pw, 0, ph))
        Hp, Wp = H + ph, Wd + pw
        if self.shift:
            h = torch.roll(h, (-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(self.norm1(h), w)
        a, _ = self.attn(win, win, win, need_weights=False)
        h = h + window_reverse(a, w, B, Hp, Wp)
        if self.shift:
            h = torch.roll(h, (self.shift, self.shift), dims=(1, 2))
        h = h[:, :H, :Wd].reshape(B, L, C)
        return h + self.ffn(self.norm2(h))


# ---------------------------------------------------------------------------
# Tokens: image patches and lidar in one grid
# ---------------------------------------------------------------------------
class PatchTokens(nn.Module):
    """image -> 16x16 patch tokens, with lidar scattered into the same grid.

    Lidar arrives as (u, v, inv_depth) per return. Each is projected and added into the patch it
    falls in, averaged over the returns that share a patch, so a patch that has lidar is the image
    patch PLUS what the lidar says is there, and one that does not is unchanged. No separate stream,
    no gating flag - absence is just a zero addition.
    """

    def __init__(self, d: int, patch: int = 16, in_ch: int = 3, n_freq: int = 6):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, d, patch, patch)
        self.pe = FourierUV(d, n_freq)
        self.lidar_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.lidar_mlp[-1].weight)      # start as image-only
        nn.init.zeros_(self.lidar_mlp[-1].bias)
        self.norm = nn.LayerNorm(d)

    def forward(self, img: torch.Tensor, lidar_uv: torch.Tensor = None,
                lidar_invd: torch.Tensor = None):
        B, _, H, Wd = img.shape
        f = self.proj(img)                                   # (B, d, h, w)
        h, w = f.shape[-2:]
        tok = f.flatten(2).transpose(1, 2)                   # (B, h*w, d)
        gy, gx = torch.meshgrid(torch.arange(h, device=img.device),
                                torch.arange(w, device=img.device), indexing="ij")
        uv = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], -1).reshape(1, -1, 2)
        tok = tok + self.pe(uv)
        if lidar_uv is not None and lidar_uv.numel():
            idx = ((lidar_uv[..., 1] * h).long().clamp(0, h - 1) * w
                   + (lidar_uv[..., 0] * w).long().clamp(0, w - 1))          # (B, M)
            val = self.lidar_mlp(lidar_invd[..., None])                       # (B, M, d)
            acc = torch.zeros_like(tok).scatter_add_(1, idx[..., None].expand_as(val), val)
            cnt = torch.zeros(B, h * w, 1, device=img.device).scatter_add_(
                1, idx[..., None], torch.ones_like(lidar_invd[..., None]))
            tok = tok + acc / cnt.clamp(min=1)
        return self.norm(tok), h, w


# ---------------------------------------------------------------------------
# RoPE on the query: SO(3) acts on the type-1 chunk (CalibNet2's construction)
# ---------------------------------------------------------------------------
class RoPEQ(nn.Module):
    def __init__(self, d: int, d_scalar: int = 8):
        super().__init__()
        assert (d - d_scalar) % 3 == 0
        self.ds, self.k = d_scalar, (d - d_scalar) // 3
        self.t_mlp = nn.Sequential(nn.Linear(3, max(d_scalar, 16)), nn.GELU(),
                                   nn.Linear(max(d_scalar, 16), d_scalar))
        nn.init.zeros_(self.t_mlp[-1].weight); nn.init.zeros_(self.t_mlp[-1].bias)

    def forward(self, q, R=None, t=None):
        B, N, D = q.shape
        qs, qv = q[..., :self.ds], q[..., self.ds:].reshape(B, N, self.k, 3)
        if R is not None:
            qv = torch.einsum("bij,bnkj->bnki", R, qv)
        if t is not None:
            qs = qs + self.t_mlp(t)[:, None]
        return torch.cat([qs, qv.reshape(B, N, self.k * 3)], -1)


# ---------------------------------------------------------------------------
class FlowNetSwin(nn.Module):
    def __init__(self, d: int = 128, patch: int = 16, in_ch: int = 3,
                 window: int = 8, n_swin: int = 4, n_iter: int = 3,
                 n_heads: int = 4, d_scalar: int = 8, n_freq: int = 6):
        super().__init__()
        self.tokens = PatchTokens(d, patch, in_ch, n_freq)
        self.swin = nn.ModuleList([SwinBlock(d, n_heads, window, shift=(i % 2 == 1))
                                   for i in range(n_swin)])
        self.q_uv = FourierUV(d, n_freq)          # sub-pixel position of the query
        self.q_d = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.q_d[-1].weight); nn.init.zeros_(self.q_d[-1].bias)
        self.rope = RoPEQ(d, d_scalar)
        self.n_iter = n_iter
        self.ca = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.sa = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        # read out from the ACCUMULATED update, never from q: q carries the query's absolute
        # position and the head must not be able to memorise it (CalibNet2's abs-PE leak gate)
        self.head = nn.Linear(d, 6)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = self.head.bias[3] = math.log(2.0)   # sigma ~ 2 px at init
            self.head.bias[5] = 2.0                                 # visible by default

    def encode(self, img, lidar_uv=None, lidar_invd=None):
        tok, h, w = self.tokens(img, lidar_uv, lidar_invd)
        for blk in self.swin:
            tok = blk(tok, h, w)
        return tok

    def forward(self, img_tgt, q_uv, q_invd, *, R=None, t=None,
                kv=None, lidar_uv=None, lidar_invd=None):
        """q_uv (B,N,2) in [0,1] on the SOURCE frame, q_invd (B,N) inverse depth.
        Returns (duv (B,N,2), log_sig (B,N,2), rho (B,N), vis (B,N))."""
        if kv is None:
            kv = self.encode(img_tgt, lidar_uv, lidar_invd)
        q = self.q_uv(q_uv) + self.q_d(q_invd[..., None])
        q = self.rope(q, R, t)
        acc = torch.zeros_like(q)
        for _ in range(self.n_iter):
            a, _ = self.ca(self.n1(q), kv, kv, need_weights=False)
            q = q + a
            s, _ = self.sa(self.n2(q), self.n2(q), self.n2(q), need_weights=False)
            q = q + s
            f = self.ffn(self.n3(q))
            q = q + f
            acc = acc + a + s + f
        raw = self.head(acc)
        duv = raw[..., :2]
        log_sig = raw[..., 2:4].clamp(-4.0, 6.0)
        rho = torch.tanh(raw[..., 4]) * 0.95
        vis = torch.sigmoid(raw[..., 5])
        return duv, log_sig, rho, vis


def gaussian2d_nll(duv, log_sig, rho, target_duv, vis_w=None):
    """Anisotropic 2D Gaussian NLL. vis_w (0..1) downweights points the geometry says are hidden -
    a hidden point still has a position, but it should not be asked to be sharp about it."""
    e = target_duv - duv
    su, sv = log_sig[..., 0].exp(), log_sig[..., 1].exp()
    zu, zv = e[..., 0] / su, e[..., 1] / sv
    om = (1 - rho ** 2).clamp(min=1e-4)
    nll = (0.5 / om * (zu ** 2 - 2 * rho * zu * zv + zv ** 2)
           + log_sig[..., 0] + log_sig[..., 1] + 0.5 * om.log())
    return (nll * vis_w).sum() / vis_w.sum().clamp(min=1) if vis_w is not None else nll.mean()
