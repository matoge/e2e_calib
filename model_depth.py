"""
model_depth.py  –  Depth-aware CalibNet with covariance output

Input points: (U, V, D_norm)  — 3-channel instead of 2
Output per point: (tx, ty, log_sx, log_sy, rho_raw)

The depth channel lets the model learn:
  - obj points (D~0.2,0.4) → small confident covariance
  - bg  points (D~0.8)     → large/degenerate covariance
"""
import math, torch, torch.nn as nn
from model import CNNBackbone, ConvNeXtBackbone, D as D_DIM
from model_cov import CrossAttentionBlockCov, TransformerDecoderBlock, clamp_params, gaussian2d_nll


class PointMLP3(nn.Module):
    """Point MLP for 3-channel input (U, V, D_norm)."""
    def __init__(self, d: int = D_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  nn.GELU(),
            nn.Linear(64, d),  nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
        )

    def forward(self, uvd: torch.Tensor) -> torch.Tensor:
        return self.net(uvd)


class FrustumLocalEncoder(nn.Module):
    """Local neighborhood feature via box-filter + top-k + MaxPool (PointNet2 style).

    For each point i, finds neighbors j where |Δu| < r_uv AND |Δv| < r_uv AND |Δd| < r_d,
    takes the k UV-nearest, applies shared MLP on relative (Δu, Δv, Δd), MaxPool.
    """
    def __init__(self, d_out: int = D_DIM, r_uv: float = 8.0, r_d: float = 0.004, k: int = 8):
        super().__init__()
        self.r_uv = r_uv
        self.r_d  = r_d
        self.k    = k
        self.mlp = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(),
            nn.Linear(32, d_out), nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, uvd: torch.Tensor, pad_mask=None) -> torch.Tensor:
        # uvd: (B, N, 3)  U,V in [0,img_size], D in [0,1]
        B, N, _ = uvd.shape

        # relative coords: rel[b,i,j] = uvd[b,j] - uvd[b,i]
        rel = uvd.unsqueeze(2) - uvd.unsqueeze(1)   # (B, N, N, 3)

        # box filter
        in_box = ((rel[..., 0].abs() < self.r_uv) &
                  (rel[..., 1].abs() < self.r_uv) &
                  (rel[..., 2].abs() < self.r_d))
        # exclude self
        in_box[:, range(N), range(N)] = False
        # exclude padded neighbors
        if pad_mask is not None:
            in_box = in_box & ~pad_mask.unsqueeze(1)   # (B, 1, N)

        # UV-squared distance; mask out non-box with large value
        uv_d2 = rel[..., 0] ** 2 + rel[..., 1] ** 2  # (B, N, N)
        uv_d2 = uv_d2.masked_fill(~in_box, 1e9)

        # top-k nearest
        k = min(self.k, N)
        _, topk_idx = uv_d2.topk(k, dim=-1, largest=False)   # (B, N, k)

        # gather relative features
        idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        topk_rel = rel.gather(2, idx_exp)              # (B, N, k, 3)
        valid    = uv_d2.gather(2, topk_idx) < 1e8    # (B, N, k)

        # shared MLP + mask invalid
        feat = self.mlp(topk_rel)                      # (B, N, k, d_out)
        feat = feat.masked_fill(~valid.unsqueeze(-1), -1e9)

        # MaxPool over k neighbors
        feat, _ = feat.max(dim=2)                      # (B, N, d_out)

        # zero out points with no valid neighbor
        feat = feat.masked_fill(~valid.any(dim=-1, keepdim=True), 0.0)
        return feat


class CalibNetDepth(nn.Module):
    def __init__(self, d: int = D_DIM, img_size: int = 128, in_channels: int = 1,
                 n_layers: int = 3, self_first: bool = False, kv_self_attn: bool = False,
                 cross_temp: float = 1.0, use_convnext: bool = False,
                 use_frustum: bool = False, r_uv: float = 8.0, r_d: float = 0.004, k_nb: int = 8):
        super().__init__()
        self.img_size    = img_size
        self.n_layers    = n_layers
        self.cnn         = (ConvNeXtBackbone(d, in_channels=in_channels)
                            if use_convnext else CNNBackbone(d, in_channels=in_channels))
        self.point_mlp   = PointMLP3(d)
        self.frustum_enc = FrustumLocalEncoder(d, r_uv=r_uv, r_d=r_d, k=k_nb) if use_frustum else None

        kw = dict(kv_self_attn=kv_self_attn, cross_temp=cross_temp) if not self_first else {}
        Block = TransformerDecoderBlock if self_first else CrossAttentionBlockCov
        self.cross_coarse  = Block(d, **kw)
        self.cross_fine    = Block(d, **kw)
        if n_layers >= 3:
            self.cross_refine  = Block(d, **kw)
        if n_layers >= 4:
            self.cross_fine2   = Block(d, **kw)
        self._self_first = self_first

    def set_cross_temp(self, t: float):
        for m in self.modules():
            if hasattr(m, '_cross_temp'):
                m._cross_temp = t

    def _block(self, block, q, feat, uv_01, mask):
        if self._self_first:
            return block(q, feat, uv_01, key_padding_mask=mask)
        return block(q, feat, uv_01, key_padding_mask=mask, self_first=False)

    def forward(self, image: torch.Tensor, distorted_uvd: torch.Tensor,
                key_padding_mask=None):
        """
        image           : (B, C, H, W)
        distorted_uvd   : (B, N, 3)  [U, V, D_norm]
        key_padding_mask: (B, N) bool  True = padding position
        Returns params  : (B, N, 5)  [tx, ty, log_sx, log_sy, rho]
        """
        coarse_feat, fine_feat = self.cnn(image)

        uv_01    = distorted_uvd[..., :2] / self.img_size
        uvd_norm = torch.cat([uv_01, distorted_uvd[..., 2:3]], dim=-1)
        d3       = distorted_uvd[..., 2:3]

        q = self.point_mlp(uvd_norm)
        if self.frustum_enc is not None:
            q = q + self.frustum_enc(distorted_uvd, pad_mask=key_padding_mask)
        q, raw_c = self._block(self.cross_coarse, q, coarse_feat, uv_01, key_padding_mask)
        raw = raw_c

        if self.n_layers == 2:
            uv_w = (uv_01 + raw_c[..., :2]).clamp(0, 1)
            q_w  = self.point_mlp(torch.cat([uv_w, d3], dim=-1)) + q
            _, raw_f = self._block(self.cross_fine, q_w, fine_feat, uv_w, key_padding_mask)
            raw = raw_c + raw_f

        elif self.n_layers == 3:  # coarse → coarse → fine
            uv_w = (uv_01 + raw_c[..., :2]).clamp(0, 1)
            q_w  = self.point_mlp(torch.cat([uv_w, d3], dim=-1)) + q
            q_w, raw_c2 = self._block(self.cross_refine, q_w, coarse_feat, uv_w, key_padding_mask)

            uv_w2 = (uv_01 + (raw_c + raw_c2)[..., :2]).clamp(0, 1)
            q_w2  = self.point_mlp(torch.cat([uv_w2, d3], dim=-1)) + q_w
            _, raw_f = self._block(self.cross_fine, q_w2, fine_feat, uv_w2, key_padding_mask)
            raw = raw_c + raw_c2 + raw_f

        else:  # n_layers == 4: coarse → coarse → fine → fine
            uv_w = (uv_01 + raw_c[..., :2]).clamp(0, 1)
            q_w  = self.point_mlp(torch.cat([uv_w, d3], dim=-1)) + q
            q_w, raw_c2 = self._block(self.cross_refine, q_w, coarse_feat, uv_w, key_padding_mask)

            uv_w2 = (uv_01 + (raw_c + raw_c2)[..., :2]).clamp(0, 1)
            q_w2  = self.point_mlp(torch.cat([uv_w2, d3], dim=-1)) + q_w
            q_w2, raw_f = self._block(self.cross_fine, q_w2, fine_feat, uv_w2, key_padding_mask)

            uv_w3 = (uv_01 + (raw_c + raw_c2 + raw_f)[..., :2]).clamp(0, 1)
            q_w3  = self.point_mlp(torch.cat([uv_w3, d3], dim=-1)) + q_w2
            _, raw_f2 = self._block(self.cross_fine2, q_w3, fine_feat, uv_w3, key_padding_mask)
            raw = raw_c + raw_c2 + raw_f + raw_f2

        return clamp_params(raw, self.img_size)   # (B,N,5)
