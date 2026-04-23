"""
model_cov.py  –  CalibNet with per-point covariance output

Output per point: (tx, ty, log_sx, log_sy, rho_raw)
  tx, ty    : predicted offset (pixel space)
  log_sx    : log(sigma_x), ensures sigma_x > 0
  log_sy    : log(sigma_y)
  rho_raw   : pre-tanh correlation, ensures |rho| < 1

Parameterisation prevents:
  - variance collapse: clamp log_s >= log(MIN_SIGMA)
  - variance explosion: clamp log_s <= log(MAX_SIGMA)
  - invalid correlation: rho = tanh(rho_raw) * 0.99
"""
import torch
import torch.nn as nn
from models.model import CNNBackbone, PointMLP, D

MIN_SIGMA = 0.3   # px  — floor prevents NLL collapsing to -inf
MAX_SIGMA = 30.0  # px  — ceiling prevents explosion


class CrossAttentionBlockCov(nn.Module):
    """Cross-attn block. Order: [kv_self_attn →] cross → [self_first? sa :] sa → FFN.
    kv_self_attn=True enriches image tokens with self-attn before cross-attn.
    """
    def __init__(self, d: int = D, n_heads: int = 4, kv_self_attn: bool = False,
                 cross_temp: float = 1.0):
        super().__init__()
        self.cross_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q      = nn.LayerNorm(d)
        self.norm_kv     = nn.LayerNorm(d)
        self.self_attn   = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self   = nn.LayerNorm(d)
        self.drop        = nn.Dropout(0.1)
        self.ffn         = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d),
        )
        self.norm_ffn    = nn.LayerNorm(d)
        self.proj        = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69
        # optional image encoder self-attn
        if kv_self_attn:
            self.kv_sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            self.norm_kv_sa = nn.LayerNorm(d)
        self._kv_self_attn = kv_self_attn
        self._cross_temp   = cross_temp

    def forward(self, q, feat, uv_01, key_padding_mask=None, self_first=False):
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)   # (B, H*W, D)

        # image self-attn (optional)
        if self._kv_self_attn:
            img_sa, _ = self.kv_sa(self.norm_kv_sa(kv), self.norm_kv_sa(kv), self.norm_kv_sa(kv))
            kv = kv + self.drop(img_sa)

        # temperature scaling: divide Q by T → softmax(QK^T / (T·√d_k))
        nq = self.norm_q(q) / self._cross_temp

        if self_first:
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                   key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
            ca, _ = self.cross_attn(nq, self.norm_kv(kv), self.norm_kv(kv))
            q = q + self.drop(ca)
        else:
            ca, _ = self.cross_attn(nq, self.norm_kv(kv), self.norm_kv(kv))
            q = q + self.drop(ca)
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                   key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))   # (B,N,5)
        return q, raw


class TransformerDecoderBlock(nn.Module):
    """Standard Transformer Decoder block:
    image self-attn → point self-attn → cross-attn → FFN
    """
    def __init__(self, d: int = D, n_heads: int = 4):
        super().__init__()
        # encoder (image) self-attn
        self.img_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_img      = nn.LayerNorm(d)
        # decoder (point) self-attn
        self.self_attn     = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self     = nn.LayerNorm(d)
        # cross-attn
        self.cross_attn    = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q        = nn.LayerNorm(d)
        self.norm_kv       = nn.LayerNorm(d)
        self.drop          = nn.Dropout(0.1)
        self.ffn           = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d),
        )
        self.norm_ffn      = nn.LayerNorm(d)
        self.proj          = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69

    def forward(self, q, feat, uv_01, key_padding_mask=None):
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)          # (B, H*W, D)

        # 1. image self-attn
        img_sa, _ = self.img_self_attn(self.norm_img(kv), self.norm_img(kv), self.norm_img(kv))
        kv = kv + self.drop(img_sa)

        # 2. point self-attn
        pt_sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                  key_padding_mask=key_padding_mask)
        q = q + self.drop(pt_sa)

        # 3. cross-attn
        ca, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(ca)

        # 4. FFN
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))  # (B,N,5)
        return q, raw


def clamp_params(raw: torch.Tensor, img_size: int = 128):
    """
    raw: (B, N, 5)  →  (B, N, 5) with valid parameterisation
    Returns tensor with columns: [tx, ty, log_sx, log_sy, rho]
    """
    tx, ty      = raw[..., 0], raw[..., 1]
    import math
    log_sx      = raw[..., 2].clamp(min=math.log(MIN_SIGMA), max=math.log(MAX_SIGMA))
    log_sy      = raw[..., 3].clamp(min=math.log(MIN_SIGMA), max=math.log(MAX_SIGMA))
    rho         = torch.tanh(raw[..., 4]) * 0.99   # |rho| < 0.99
    # scale mean to pixel space
    tx = tx * img_size
    ty = ty * img_size
    return torch.stack([tx, ty, log_sx, log_sy, rho], dim=-1)


class CalibNetCov(nn.Module):
    def __init__(self, d: int = D, img_size: int = 128):
        super().__init__()
        self.img_size     = img_size
        self.cnn          = CNNBackbone(d)
        self.point_mlp    = PointMLP(d)
        self.cross_coarse = CrossAttentionBlockCov(d)
        self.cross_fine   = CrossAttentionBlockCov(d)

    def forward(self, image, distorted_uv):
        """
        Returns params (B, N, 5): [tx, ty, log_sx, log_sy, rho]
        All in pixel space.
        """
        coarse_feat, fine_feat = self.cnn(image)
        uv_01 = distorted_uv / self.img_size

        q = self.point_mlp(uv_01)
        q, raw_c = self.cross_coarse(q, coarse_feat, uv_01)

        # warp using coarse mean (columns 0,1 are in [0,1] before scaling → use raw)
        offset_c_01 = raw_c[..., :2]   # still un-scaled at this point
        uv_w = (uv_01 + offset_c_01).clamp(0, 1)
        q_w  = self.point_mlp(uv_w) + q
        _, raw_f = self.cross_fine(q_w, fine_feat, uv_w)

        # sum coarse + fine raw, then clamp/scale
        raw = raw_c + raw_f
        return clamp_params(raw, self.img_size)   # (B,N,5)


# ---------------------------------------------------------------------------
# 2-D Gaussian NLL loss
# ---------------------------------------------------------------------------

def gaussian2d_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    params : (B, N, 5)  [tx, ty, log_sx, log_sy, rho]
    target : (B, N, 2)  [gt_tx, gt_ty]
    Returns scalar mean NLL.
    """
    tx, ty       = params[..., 0], params[..., 1]
    log_sx, log_sy = params[..., 2], params[..., 3]
    rho          = params[..., 4]

    dx = target[..., 0] - tx
    dy = target[..., 1] - ty
    sx = log_sx.exp()
    sy = log_sy.exp()

    # normalised residuals
    zx = dx / sx
    zy = dy / sy
    r2 = 1.0 - rho**2

    # log det of covariance = log(sx²·sy²·(1-ρ²)) = 2log_sx + 2log_sy + log(1-ρ²)
    log_det = 2*log_sx + 2*log_sy + torch.log(r2.clamp(min=1e-6))

    # Mahalanobis
    maha = (zx**2 - 2*rho*zx*zy + zy**2) / r2.clamp(min=1e-6)

    nll = 0.5 * (log_det + maha)  # + const (log 2π, ignored)
    return nll.mean()
