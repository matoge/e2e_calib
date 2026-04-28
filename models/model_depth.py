"""
model_depth.py  –  Depth-aware CalibNet with covariance output

Input points: (U, V, D_norm)  — 3-channel instead of 2
Output per point: (tx, ty, log_sx, log_sy, rho_raw)

The depth channel lets the model learn:
  - obj points (D~0.2,0.4) → small confident covariance
  - bg  points (D~0.8)     → large/degenerate covariance
"""
import math, torch, torch.nn as nn
from models.model import CNNBackbone, ConvNeXtBackbone, D as D_DIM
from models.model_cov import CrossAttentionBlockCov, TransformerDecoderBlock, clamp_params, gaussian2d_nll


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

    def forward(self, query_uvd: torch.Tensor,
                full_uvd: torch.Tensor = None,
                full_pad_mask: torch.Tensor = None,
                query_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """Local-feature encoding.

        For each *query* point, find neighbors in *full_uvd* via box filter
        + top-k UV-nearest, MLP on relative (Δu, Δv, Δd), MaxPool.

        Args:
            query_uvd:      (B, N_q, 3)  U,V in [0,img_size], D in [0,1]
            full_uvd:       (B, N_kv, 3)  context point cloud. If None, falls
                             back to legacy single-set behavior (full = query).
            full_pad_mask:  (B, N_kv) bool — True = padded entry to ignore
            query_pad_mask: (B, N_q)  bool — kept for API compat; not used here

        Returns:
            (B, N_q, d_out) — local feature per query point
        """
        if full_uvd is None:
            full_uvd = query_uvd
            if full_pad_mask is None:
                full_pad_mask = query_pad_mask
        B, N_q, _ = query_uvd.shape
        N_kv = full_uvd.shape[1]

        # relative coords: rel[b, i, j] = full_uvd[b, j] - query_uvd[b, i]
        rel = full_uvd.unsqueeze(1) - query_uvd.unsqueeze(2)   # (B, N_q, N_kv, 3)

        in_box = ((rel[..., 0].abs() < self.r_uv) &
                  (rel[..., 1].abs() < self.r_uv) &
                  (rel[..., 2].abs() < self.r_d))
        # exclude exact self-coord matches (when query is a subset of full)
        self_match = ((rel[..., 0] == 0) & (rel[..., 1] == 0) & (rel[..., 2] == 0))
        in_box = in_box & ~self_match
        if full_pad_mask is not None:
            in_box = in_box & ~full_pad_mask.unsqueeze(1)        # (B, 1, N_kv)

        uv_d2 = rel[..., 0] ** 2 + rel[..., 1] ** 2              # (B, N_q, N_kv)
        uv_d2 = uv_d2.masked_fill(~in_box, 1e9)

        k = min(self.k, N_kv)
        _, topk_idx = uv_d2.topk(k, dim=-1, largest=False)       # (B, N_q, k)

        idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        topk_rel = rel.gather(2, idx_exp)                        # (B, N_q, k, 3)
        valid    = uv_d2.gather(2, topk_idx) < 1e8               # (B, N_q, k)

        feat = self.mlp(topk_rel)                                # (B, N_q, k, d_out)
        feat = feat.masked_fill(~valid.unsqueeze(-1), -1e9)
        feat, _ = feat.max(dim=2)                                # (B, N_q, d_out)
        feat = feat.masked_fill(~valid.any(dim=-1, keepdim=True), 0.0)
        return feat


class CalibNetDepth(nn.Module):
    def __init__(self, d: int = D_DIM, img_size: int = 128, in_channels: int = 1,
                 n_layers: int = 3, self_first: bool = False, kv_self_attn: bool = False,
                 cross_temp: float = 1.0, use_convnext: bool = False,
                 use_frustum: bool = False, r_uv: float = 8.0, r_d: float = 0.004, k_nb: int = 8,
                 deform_mode: str = 'none', deform_n_points: int = 4):
        """deform_mode: 'none' (standard cross-attn, cascaded coarse/fine),
                       'sl'  (single-level deformable, same cascade),
                       'ml'  (multi-level deformable — each block sees both
                              coarse and fine with learnable level embedding)."""
        super().__init__()
        assert deform_mode in ('none', 'sl', 'ml'), f'bad deform_mode: {deform_mode}'
        self.img_size    = img_size
        self.n_layers    = n_layers
        self.cnn         = (ConvNeXtBackbone(d, in_channels=in_channels)
                            if use_convnext else CNNBackbone(d, in_channels=in_channels))
        self.point_mlp   = PointMLP3(d)
        self.frustum_enc = FrustumLocalEncoder(d, r_uv=r_uv, r_d=r_d, k=k_nb) if use_frustum else None

        if deform_mode != 'none':
            assert not self_first, "deform_mode is incompatible with self_first=True"
            from model_deform import CrossAttentionBlockDeform, CrossAttentionBlockDeformML
            if deform_mode == 'sl':
                Block = CrossAttentionBlockDeform
                kw = dict(kv_self_attn=kv_self_attn, cross_temp=cross_temp,
                          n_points=deform_n_points)
            else:  # 'ml' — multi-level deformable
                Block = CrossAttentionBlockDeformML
                kw = dict(kv_self_attn=kv_self_attn, cross_temp=cross_temp,
                          n_levels=2, n_points=deform_n_points)
                # learnable resolution (level) embedding shared across blocks
                self.level_embed = nn.Parameter(torch.zeros(2, d))
                nn.init.normal_(self.level_embed, std=0.02)
        elif self_first:
            Block = TransformerDecoderBlock
            kw = {}
        else:
            Block = CrossAttentionBlockCov
            kw = dict(kv_self_attn=kv_self_attn, cross_temp=cross_temp)

        self.cross_coarse  = Block(d, **kw)
        self.cross_fine    = Block(d, **kw)
        if n_layers >= 3:
            self.cross_refine  = Block(d, **kw)
        if n_layers >= 4:
            self.cross_fine2   = Block(d, **kw)
        self._self_first = self_first
        self._deform_mode = deform_mode

    def set_cross_temp(self, t: float):
        for m in self.modules():
            if hasattr(m, '_cross_temp'):
                m._cross_temp = t

    def _block(self, block, q, feat, uv_01, mask):
        if self._deform_mode == 'ml':
            # feat here is (coarse_feat, fine_feat) tuple — ML block wants the list
            return block(q, list(feat), uv_01, self.level_embed,
                          key_padding_mask=mask, self_first=False)
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
            q = q + self.frustum_enc(distorted_uvd, query_pad_mask=key_padding_mask)

        # ML mode: every block sees both levels; SL / none: alternate coarse→fine
        if self._deform_mode == 'ml':
            feat_all = (coarse_feat, fine_feat)
            # same feat tuple at every layer; each block refines uv
            feats_by_layer = [feat_all] * max(self.n_layers, 2)
        else:
            feats_seq = [coarse_feat]
            if self.n_layers == 2:
                feats_seq += [fine_feat]
            elif self.n_layers == 3:
                feats_seq += [coarse_feat, fine_feat]
            else:  # 4
                feats_seq += [coarse_feat, fine_feat, fine_feat]
            feats_by_layer = feats_seq

        # Block order.
        # deform='none': restore the pre-refactor convention (cross_refine in
        #   the coarse half, cross_fine/fine2 in the fine half) so that
        #   pre-85b6ccc checkpoints load correctly.
        # deform='sl'/'ml': keep the refactor's order, because those weights
        #   were trained under it and swapping the list now would mis-wire
        #   them symmetrically.
        if self._deform_mode == 'none':
            blocks = [self.cross_coarse]
            if self.n_layers >= 3: blocks.append(self.cross_refine)
            blocks.append(self.cross_fine)
            if self.n_layers >= 4: blocks.append(self.cross_fine2)
        else:
            blocks = [self.cross_coarse, self.cross_fine]
            if self.n_layers >= 3: blocks.append(self.cross_refine)
            if self.n_layers >= 4: blocks.append(self.cross_fine2)

        # first layer (uv_01)
        q, raw_cum = self._block(blocks[0], q, feats_by_layer[0], uv_01, key_padding_mask)
        # refinement layers
        for i in range(1, min(self.n_layers, len(blocks))):
            uv_i = (uv_01 + raw_cum[..., :2]).clamp(0, 1)
            q    = self.point_mlp(torch.cat([uv_i, d3], dim=-1)) + q
            q, raw_i = self._block(blocks[i], q, feats_by_layer[i], uv_i, key_padding_mask)
            raw_cum = raw_cum + raw_i
        raw = raw_cum

        return clamp_params(raw, self.img_size)   # (B,N,5)
