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
from model_cov import CrossAttentionBlockCov, clamp_params, gaussian2d_nll  # reuse


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
    def __init__(self, d: int = D_DIM, img_size: int = 128):
        super().__init__()
        self.img_size     = img_size
        self.cnn          = CNNBackbone(d)
        self.point_mlp    = PointMLP3(d)
        self.cross_coarse = CrossAttentionBlockCov(d)
        self.cross_fine   = CrossAttentionBlockCov(d)

    def forward(self, image: torch.Tensor, distorted_uvd: torch.Tensor):
        """
        image        : (B, 1, 128, 128)
        distorted_uvd: (B, N, 3)  [U, V, D_norm]
        Returns params (B, N, 5): [tx, ty, log_sx, log_sy, rho]
        """
        coarse_feat, fine_feat = self.cnn(image)

        # UV for positional encoding (only first 2 dims, normalised to [0,1])
        uv_01 = distorted_uvd[..., :2] / self.img_size   # (B,N,2)

        # full (U,V,D) for the point MLP
        uvd_norm = torch.cat([uv_01, distorted_uvd[..., 2:3]], dim=-1)   # (B,N,3)

        q = self.point_mlp(uvd_norm)
        q, raw_c = self.cross_coarse(q, coarse_feat, uv_01)

        # warp UV by coarse offset estimate
        offset_c_01 = raw_c[..., :2]
        uv_w   = (uv_01 + offset_c_01).clamp(0, 1)
        uvd_w  = torch.cat([uv_w, distorted_uvd[..., 2:3]], dim=-1)
        q_w    = self.point_mlp(uvd_w) + q
        _, raw_f = self.cross_fine(q_w, fine_feat, uv_w)

        raw = raw_c + raw_f
        return clamp_params(raw, self.img_size)   # (B,N,5)
