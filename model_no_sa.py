"""Cross-Attention only, no Self-Attention, no mean pooling — ablation model."""
import torch
import torch.nn as nn
from model import CNNBackbone, PointMLP, PosEnc2D

D = 128

class CrossAttentionOnly(nn.Module):
    def __init__(self, d: int = D, n_heads: int = 4):
        super().__init__()
        self.cross_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q      = nn.LayerNorm(d)
        self.norm_kv     = nn.LayerNorm(d)
        self.drop        = nn.Dropout(0.1)
        self.ffn         = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.norm_ffn    = nn.LayerNorm(d)
        self.proj_offset = nn.Linear(d + 2, 2)

    def forward(self, q, feat, uv_01):
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)
        ca_out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(ca_out)
        q = q + self.ffn(self.norm_ffn(q))
        offset = self.proj_offset(torch.cat([q, uv_01], dim=-1))
        return q, offset


class CalibNetNoSA(nn.Module):
    def __init__(self, d: int = D, img_size: int = 128):
        super().__init__()
        self.img_size  = img_size
        self.cnn       = CNNBackbone(d)
        self.point_mlp = PointMLP(d)
        self.cross_coarse = CrossAttentionOnly(d)
        self.cross_fine   = CrossAttentionOnly(d)

    def forward(self, image, distorted_uv):
        coarse_feat, fine_feat = self.cnn(image)
        uv_01 = distorted_uv / self.img_size
        q = self.point_mlp(uv_01)
        q, offset_coarse_01 = self.cross_coarse(q, coarse_feat, uv_01)
        uv_warped_01 = (uv_01 + offset_coarse_01).clamp(0.0, 1.0)
        q_warped = self.point_mlp(uv_warped_01) + q
        _, offset_fine_01 = self.cross_fine(q_warped, fine_feat, uv_warped_01)
        return (offset_coarse_01 + offset_fine_01) * self.img_size
