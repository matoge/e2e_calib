"""
model_depth.py  –  Depth-aware CalibNet with covariance output

Input points: (U, V, D_norm)  — 3-channel instead of 2
Output per point: (tx, ty, log_sx, log_sy, rho_raw)

The depth channel lets the model learn:
  - obj points (D~0.2,0.4) → small confident covariance
  - bg  points (D~0.8)     → large/degenerate covariance
"""
import math, torch, torch.nn as nn
from model import CNNBackbone, D as D_DIM
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


class CalibNetDepth(nn.Module):
    def __init__(self, d: int = D_DIM, img_size: int = 128, in_channels: int = 1,
                 n_layers: int = 3, self_first: bool = False, kv_self_attn: bool = False):
        super().__init__()
        self.img_size  = img_size
        self.n_layers  = n_layers
        self.cnn       = CNNBackbone(d, in_channels=in_channels)
        self.point_mlp = PointMLP3(d)

        kw = dict(kv_self_attn=kv_self_attn) if not self_first else {}
        Block = TransformerDecoderBlock if self_first else CrossAttentionBlockCov
        self.cross_coarse  = Block(d, **kw)
        self.cross_fine    = Block(d, **kw)
        if n_layers >= 3:
            self.cross_refine  = Block(d, **kw)
        if n_layers >= 4:
            self.cross_fine2   = Block(d, **kw)
        self._self_first = self_first

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
