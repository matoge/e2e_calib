"""
model.py  –  Minimal Coarse-to-Fine UV correction model

Fix: offset head receives (attention_output || uv_position) so it can
     learn "attended_feature_position - query_position = offset".
     Coordinate system: UV normalized to [0,1] throughout, consistent
     with the 2D positional encoding. Output scaled to pixel space.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 128  # feature dimension


# ---------------------------------------------------------------------------
# 2-D Sinusoidal Positional Encoding  (positions in [0,1])
# ---------------------------------------------------------------------------
class PosEnc2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 4 == 0
        d_band = d_model // 4
        freq = 1.0 / (10000 ** (torch.arange(d_band, dtype=torch.float32) / d_band))
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D_, H, W = x.shape
        ys = torch.linspace(0, 1, H, device=x.device)
        xs = torch.linspace(0, 1, W, device=x.device)
        freq = self.freq
        ay = ys[:, None] * freq[None, :]   # (H, d_band)
        ax = xs[:, None] * freq[None, :]   # (W, d_band)
        sin_y = torch.sin(ay).T[:, :, None].expand(-1, -1, W)
        cos_y = torch.cos(ay).T[:, :, None].expand(-1, -1, W)
        sin_x = torch.sin(ax).T[:, None, :].expand(-1, H, -1)
        cos_x = torch.cos(ax).T[:, None, :].expand(-1, H, -1)
        pe = torch.cat([sin_y, cos_y, sin_x, cos_x], dim=0)  # (D, H, W)
        return x + pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# CNN Backbone
# ---------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    def __init__(self, d: int = D, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
        )
        self.fine_branch = nn.Sequential(
            nn.Conv2d(64, d, 3, stride=2, padding=1),   # 32x32
            nn.BatchNorm2d(d), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=1, padding=1),
            nn.BatchNorm2d(d), nn.GELU(),
        )
        self.coarse_branch = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1),    # 16x16
            nn.BatchNorm2d(d), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=1, padding=1),
            nn.BatchNorm2d(d), nn.GELU(),
        )
        self.pe_fine   = PosEnc2D(d)
        self.pe_coarse = PosEnc2D(d)

    def forward(self, x):
        s      = self.stem(x)
        fine   = self.fine_branch(s)
        coarse = self.coarse_branch(fine)
        return self.pe_coarse(coarse), self.pe_fine(fine)


# ---------------------------------------------------------------------------
# ConvNeXt Backbone
# ---------------------------------------------------------------------------
class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dw   = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1  = nn.Linear(dim, dim * expansion)
        self.act  = nn.GELU()
        self.pw2  = nn.Linear(dim * expansion, dim)

    def forward(self, x):
        h = self.dw(x).permute(0, 2, 3, 1)   # BCHW→BHWC
        h = self.pw2(self.act(self.pw1(self.norm(h))))
        return x + h.permute(0, 3, 1, 2)


class ConvNeXtBackbone(nn.Module):
    def __init__(self, d: int = D, in_channels: int = 3, n_blocks: int = 2):
        super().__init__()
        # 4x4 stride-4 patchify stem (ConvNeXt-Tiny standard) — img/4 → 64ch.
        # img=128 → 32, img=64 → 16. Below stages: fine /2, coarse /2 = ÷16 total.
        # Final feature shapes: img=128 → fine 16×16, coarse 8×8 (= same token
        # count as img=64 with the pre-2026-05-04 stride-2 stem).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=4),
            nn.BatchNorm2d(64), nn.GELU(),
        )
        # /2 → fine features
        self.fine_down   = nn.Conv2d(64, d, 3, stride=2, padding=1)
        self.fine_norm   = nn.BatchNorm2d(d)
        self.fine_blocks = nn.Sequential(*[_ConvNeXtBlock(d) for _ in range(n_blocks)])
        # /2 → coarse features
        self.coarse_down   = nn.Conv2d(d, d, 3, stride=2, padding=1)
        self.coarse_norm   = nn.BatchNorm2d(d)
        self.coarse_blocks = nn.Sequential(*[_ConvNeXtBlock(d) for _ in range(n_blocks)])
        self.pe_fine   = PosEnc2D(d)
        self.pe_coarse = PosEnc2D(d)

    def forward(self, x):
        s      = self.stem(x)
        fine   = self.fine_blocks(torch.relu(self.fine_norm(self.fine_down(s))))
        coarse = self.coarse_blocks(torch.relu(self.coarse_norm(self.coarse_down(fine))))
        return self.pe_coarse(coarse), self.pe_fine(fine)


# ---------------------------------------------------------------------------
# Point MLP  (UV in [0,1] → D)
# ---------------------------------------------------------------------------
class PointMLP(nn.Module):
    def __init__(self, d: int = D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.GELU(),
            nn.Linear(64, d),  nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
        )

    def forward(self, uv_01: torch.Tensor) -> torch.Tensor:
        return self.net(uv_01)


# ---------------------------------------------------------------------------
# Cross-Attention + Self-Attention Block
#
# Cross-attn : each point queries the image feature map
# Self-attn  : points communicate with each other → consensus on global shift
# Offset head: f(self-attn output || uv_position)
# ---------------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int = D, n_heads: int = 4):
        super().__init__()
        # Cross-attention: points → image
        self.cross_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q      = nn.LayerNorm(d)
        self.norm_kv     = nn.LayerNorm(d)

        # Self-attention: points → points (aggregate global shift consensus)
        self.self_attn   = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self   = nn.LayerNorm(d)

        self.drop        = nn.Dropout(0.1)
        self.ffn         = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.norm_ffn    = nn.LayerNorm(d)

        # offset head: conditioned on (features || my_position)
        self.proj_offset = nn.Linear(d + 2, 2)

    def forward(self, q: torch.Tensor, feat: torch.Tensor, uv_01: torch.Tensor):
        """
        q     : (B, N, D)
        feat  : (B, D, H, W)
        uv_01 : (B, N, 2)  in [0,1]
        Returns: q_out (B,N,D),  offset (B,N,2)  in [0,1] space
        """
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)          # (B, H*W, D)

        # 1. Cross-attention: each point looks at the image
        ca_out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(ca_out)

        # 2. Self-attention: points share information with each other
        sa_out, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q))
        q = q + self.drop(sa_out)

        # 3. FFN
        q = q + self.ffn(self.norm_ffn(q))

        # 4. Per-point offset conditioned on current position
        offset = self.proj_offset(torch.cat([q, uv_01], dim=-1))   # (B,N,2)
        return q, offset


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class CalibNet(nn.Module):
    def __init__(self, d: int = D, img_size: int = 128):
        super().__init__()
        self.img_size     = img_size
        self.cnn          = CNNBackbone(d)
        self.point_mlp    = PointMLP(d)
        self.cross_coarse = CrossAttentionBlock(d)
        self.cross_fine   = CrossAttentionBlock(d)

    def forward(self, image: torch.Tensor, distorted_uv: torch.Tensor) -> torch.Tensor:
        """
        image        : (B, 1, 128, 128)
        distorted_uv : (B, N, 2)  pixel coords [0, img_size)
        Returns      : predicted_offset (B, N, 2)  in pixel space
        """
        coarse_feat, fine_feat = self.cnn(image)

        # Normalize UV to [0,1] — consistent with PE coordinate system
        uv_01 = distorted_uv / self.img_size   # (B, N, 2) in [0,1]

        q = self.point_mlp(uv_01)              # (B, N, D)

        # --- Coarse pass ---
        q, offset_coarse_01 = self.cross_coarse(q, coarse_feat, uv_01)

        # --- Warp and Fine pass ---
        uv_warped_01 = (uv_01 + offset_coarse_01).clamp(0.0, 1.0)
        q_warped = self.point_mlp(uv_warped_01) + q
        _, offset_fine_01 = self.cross_fine(q_warped, fine_feat, uv_warped_01)

        # Sum coarse + fine, convert from [0,1]-space to pixel-space
        offset_01 = offset_coarse_01 + offset_fine_01   # (B, N, 2)

        # Each point independently regresses its own offset
        return offset_01 * self.img_size  # (B, N, 2) pixels
