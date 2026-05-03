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


class FrameTokenEncoder(nn.Module):
    """UV-pos-enc-only Q ← cross-attn over (coarse, fine) image tokens.

    Compresses image features into M frame tokens whose only identity is
    their UV position in [0,1]. No per-point info, no CNN-side bias —
    just a learned read-out at fixed UV slots.
    """
    def __init__(self, d: int = D_DIM, m_side: int = 8, n_heads: int = 4):
        super().__init__()
        self.m_side = m_side
        self.M = m_side * m_side
        # UV-only positional Q (Fourier-style projection of (u,v) ∈ [0,1])
        self.uv_proj = nn.Sequential(
            nn.Linear(2, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.cross   = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.drop    = nn.Dropout(0.1)
        self.ffn     = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d),
        )
        self.norm_ffn = nn.LayerNorm(d)

        u = torch.linspace(0, 1, m_side)
        v = torch.linspace(0, 1, m_side)
        gv, gu = torch.meshgrid(v, u, indexing='ij')
        uv = torch.stack([gu.flatten(), gv.flatten()], dim=-1)   # (M, 2)
        self.register_buffer('uv_grid', uv)

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        """feats: any number of (B, D, H, W) maps. KV = concat over flattened spatial.
        Returns (B, M, D) frame tokens."""
        kv_list = [f.flatten(2).permute(0, 2, 1) for f in feats]
        kv = torch.cat(kv_list, dim=1)                           # (B, sum_HW, D)
        B  = kv.size(0)
        q  = self.uv_proj(self.uv_grid).unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        ca, _ = self.cross(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(ca)
        q = q + self.ffn(self.norm_ffn(q))
        return q                                                  # (B, M, D)


class FrustumLocalEncoder(nn.Module):
    """Per-cell local-neighborhood feature for the LiDAR query stream.

    PURPOSE
    -------
    The cross-attention transformer is good at long-range image↔point reasoning,
    but a single query token (one LiDAR point per occupied cell) carries no
    information about *what's around it in 3D*. This encoder fills that gap by
    summarizing, for every query, the local 3D neighborhood structure visible
    in the dense raw point cloud of the same crop. Conceptually it is the
    PointNet++ "set-abstraction" trick: per-point local feature = MLP +
    permutation-invariant pool over a small neighborhood.

    GEOMETRY
    --------
    The crop has been resized to (img_size × img_size) pixels and binned into a
    grid_n × grid_n grid, so each cell is `cell_px = img_size / grid_n` wide.
    The dataset supplies:
      • query_uvd   (B, N_q, 3)  one LiDAR point per occupied cell, near cell center
      • full_uvd    (B, N_kv, 3) the dense raw LiDAR points inside the crop
                                  (≈ hundreds–thousands), padded to a fixed N_kv
      • full_pad_mask (B, N_kv)   True = padded slot to ignore
    Coordinates: u, v in crop pixels [0, img_size]; d = depth_meters / 100
    (matching dist_uvd's convention so r_d=0.004 ≈ 0.4m).

    NEIGHBORHOOD SELECTION
    ----------------------
    For each query Qi, build a 3D box around it of half-width
        Δu, Δv  ≤  r_uv_cells * cell_px         (image-plane radius)
        Δd      ≤  r_d                          (depth radius)
    `r_uv_cells` is in *cell units* (default 1.5 = own cell + ~half of each
    8-neighbor) so the same hyper-parameter behaves identically across S=64,
    S=128, S=256 once grid_n is fixed. From the points landing in that box,
    keep the `k` nearest in (Δu, Δv) — top-k on UV distance only.

    AGGREGATION
    -----------
    The k chosen neighbors carry features (Δu, Δv, Δd) relative to the query
    (intentionally relative, not absolute, so the MLP is translation-invariant
    in image plane). A 3-layer MLP lifts each to d_out, then channel-wise
    MaxPool over the k neighbors gives the per-query feature, summed into the
    transformer's query stream. Padded / no-neighbor cases produce a zero
    feature (graceful, but logged in vis tools).

    SCALING / ASSUMPTIONS
    ---------------------
    - r_uv_cells × cell_px is the only image-plane scale knob. Don't pass a
      fixed pixel value — it would silently break when img_size or grid_n
      changes (this is what bit v504r for a month).
    - full_uvd MUST be passed; silent fallback to query-self pooling has been
      removed and the call now raises if missing.
    - The encoder is geometry-only (no image features, no learned positions).
      Image context comes from cross-attention later in the network.
    """
    def __init__(self, d_out: int = D_DIM, r_uv_cells: float = 1.5, r_d: float = 0.004,
                 k: int = 32, grid_n: int = 16,
                 d_local: int = 32, n_heads: int = 2, n_layers: int = 2):
        super().__init__()
        self.r_uv_cells = r_uv_cells   # neighborhood radius in cell units (1.5 = own cell + 8-neighbors)
        self.grid_n = grid_n
        self.r_d  = r_d
        self.k    = k                  # random sample size; no stratified bookkeeping
        self.d_local = d_local
        self.n_heads = n_heads
        self.n_layers = n_layers
        assert d_local % n_heads == 0
        # Single down-projection D → d_local (once, at the start) and single up-
        # projection d_local → D (once, at the end). Everything in between runs
        # at d_local. PT blocks are residual + Pre-LN, K/V freshly projected from
        # rel each layer; KV fused into one 3 → 2·d_local matmul.
        self.in_proj  = nn.Linear(d_out, d_local)
        self.layers   = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_q  = nn.LayerNorm(d_local),
                q_proj = nn.Linear(d_local, d_local),
                kv_proj = nn.Linear(3, 2 * d_local),
                out_proj = nn.Linear(d_local, d_local),
            )) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_local, d_out)
        # learnable per-cell UV embedding for dense mode. Even cells with no
        # LiDAR get a position-informative token (so the model can ask "what's
        # around UV (u,v)?" without needing a real point there). Sized
        # (1, gh*gw, d_out) so it broadcasts across batch.
        self.cell_uv_embed = nn.Parameter(torch.zeros(1, grid_n * grid_n, d_out))
        nn.init.trunc_normal_(self.cell_uv_embed, std=0.02)

    def forward(self, query_uvd: torch.Tensor,
                full_uvd: torch.Tensor,
                full_pad_mask: torch.Tensor,
                query_token: torch.Tensor,
                query_pad_mask: torch.Tensor = None,
                img_size: int = 64) -> torch.Tensor:
        """Per-query mini cross-attn over k random in-box neighbors.

        Args:
            query_uvd:      (B, N_q, 3)  U,V in [0,img_size], D in [0,1]
            full_uvd:       (B, N_kv, 3)  REQUIRED dense raw point cloud.
            full_pad_mask:  (B, N_kv) bool — True = padded entry to ignore
            query_token:    (B, N_q, d_out)  per-point feature from upstream
                            (point_mlp). Used as Q in the local cross-attn.
            query_pad_mask: (B, N_q)  bool — kept for API compat; not used here
            img_size:       crop pixel side at the resolution `query_uvd` lives in.

        Returns:
            (B, N_q, d_out) — local feature per query point
        """
        if full_uvd is None or full_pad_mask is None:
            raise ValueError(
                "FrustumLocalEncoder.forward: full_uvd AND full_pad_mask are required. "
                "Silent fallback to query-self pooling was removed — it broke a month "
                "of v504r training (frustum encoded zero local context)."
            )
        cell_px = float(img_size) / float(self.grid_n)
        r_uv = self.r_uv_cells * cell_px

        B, N_q, _ = query_uvd.shape
        N_kv = full_uvd.shape[1]

        # relative coords: rel[b, i, j] = full_uvd[b, j] - query_uvd[b, i]
        rel = full_uvd.unsqueeze(1) - query_uvd.unsqueeze(2)   # (B, N_q, N_kv, 3)

        in_box = ((rel[..., 0].abs() < r_uv) &
                  (rel[..., 1].abs() < r_uv) &
                  (rel[..., 2].abs() < self.r_d))
        # exclude exact self-coord matches (when query is a subset of full)
        self_match = ((rel[..., 0] == 0) & (rel[..., 1] == 0) & (rel[..., 2] == 0))
        in_box = in_box & ~self_match
        if full_pad_mask is not None:
            in_box = in_box & ~full_pad_mask.unsqueeze(1)        # (B, 1, N_kv)

        # Density-invariant: k random points from the in-box set. Stratified
        # sub-cell guarantee was dropped because the 16-iter Python loop cost
        # ~25 ms in launch overhead, dwarfing the actual work; with k=32 random
        # samples the per-sub-cell coverage is statistically fine.
        rand_score = torch.rand_like(in_box, dtype=rel.dtype)
        rand_score = rand_score.masked_fill(~in_box, -1.0)
        kk = min(self.k, N_kv)
        _, topk_idx = rand_score.topk(kk, dim=-1, largest=True)  # (B, N_q, k)
        valid       = rand_score.gather(2, topk_idx) >= 0

        idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        topk_rel = rel.gather(2, idx_exp)                        # (B, N_q, k, 3)

        # Debug instrumentation: stash the live selection so vis tools can read
        # exactly what this forward picked. Always written, only consumed when
        # a caller looks at it. Detached so they don't pin compute graphs.
        self._last_topk_idx = topk_idx.detach()
        self._last_valid    = valid.detach()
        self._last_r_uv_px  = float(r_uv)
        self._last_cell_px  = float(cell_px)

        # ── 2-layer Point-Transformer-style local attention, all at d_local ──
        h, dl = self.n_heads, self.d_local
        dh = dl // h
        K_total = topk_rel.size(2)
        scale = dh ** -0.5
        any_valid = valid.any(dim=-1, keepdim=True)               # (B, Nq, 1)
        valid_mask = ~valid.unsqueeze(-1)                          # (B, Nq, K, 1)

        # one-time down-projection D → d_local
        x = self.in_proj(query_token)                              # (B, Nq, d_local)

        # stack of PT blocks: residual + Pre-LN, fresh K/V from rel each layer
        for layer in self.layers:
            x_n = layer['ln_q'](x)                                 # Pre-LN
            Q = layer['q_proj'](x_n).view(B, N_q, h, dh)           # (B, Nq, h, dh)
            KV = layer['kv_proj'](topk_rel)                        # (B, Nq, K, 2·dl)
            K_, V_ = KV.chunk(2, dim=-1)
            K_ = K_.view(B, N_q, K_total, h, dh)
            V_ = V_.view(B, N_q, K_total, h, dh)
            attn_logits = (Q.unsqueeze(2) * K_).sum(-1) * scale    # (B, Nq, K, h)
            # FP16-safe: dtype-appropriate -inf, otherwise -1e9 underflows on V100
            attn_logits = attn_logits.masked_fill(valid_mask, torch.finfo(attn_logits.dtype).min)
            attn = attn_logits.softmax(dim=2)
            out_h = (attn.unsqueeze(-1) * V_).sum(2)               # (B, Nq, h, dh)
            update = layer['out_proj'](out_h.flatten(-2))          # (B, Nq, d_local)
            x = x + update                                          # residual

        feat = self.out_proj(x)                                    # 32 → D, ONCE
        feat = feat.masked_fill(~any_valid, 0.0)                   # all-pad guard
        return feat

    def forward_dense(self, full_uvd: torch.Tensor,
                       full_pad_mask: torch.Tensor,
                       img_size: int = 64) -> torch.Tensor:
        """Dense gh×gw lidar map: query at every cell center, output (B, gh*gw, D).

        Empty cells (no LiDAR within ±r_uv_cells) get a zero geometry feature,
        but the learnable per-cell UV embedding is still added — so even
        no-LiDAR cells carry useful position information for downstream
        cross-attention. Use this when the lidar stream is treated as a 2D
        KV map (alongside coarse_feat / fine_feat) rather than scattered
        per-pivot tokens.

        Args:
            full_uvd, full_pad_mask: same as forward()
            img_size: crop pixel side
        Returns:
            (B, gh*gw, D)  with cell_uv_embed already added.
        """
        B = full_uvd.shape[0]
        device = full_uvd.device
        cell_px = float(img_size) / float(self.grid_n)
        # build (gh*gw, 3) cell-center queries: u,v at cell center, d=0
        cy, cx = torch.meshgrid(
            torch.arange(self.grid_n, device=device, dtype=full_uvd.dtype),
            torch.arange(self.grid_n, device=device, dtype=full_uvd.dtype),
            indexing='ij',
        )
        u = (cx + 0.5) * cell_px
        v = (cy + 0.5) * cell_px
        d = torch.zeros_like(u)
        cell_uvd = torch.stack([u, v, d], dim=-1).reshape(1, -1, 3).expand(B, -1, -1)
        # query_token: zero — we want pure local-pt aggregation, no upstream MLP.
        # The PT block will lift the local Δ-uvd into d_local. Empty cells fall
        # through the masked_fill at the end and stay zero.
        D = self.in_proj.in_features
        zero_q_token = torch.zeros(B, cell_uvd.shape[1], D,
                                    device=device, dtype=full_uvd.dtype)
        feat = self.forward(query_uvd=cell_uvd, full_uvd=full_uvd,
                             full_pad_mask=full_pad_mask,
                             query_token=zero_q_token, img_size=img_size)
        # add learnable per-cell UV embedding so even all-zero (empty) cells
        # carry their position
        feat = feat + self.cell_uv_embed
        return feat                                                # (B, gh*gw, D)


class LocalNeighborhood3D(nn.Module):
    """Multi-scale 3D ball-query encoder (PointNet++ MSG, point-cloud style).

    Replaces the FrustumLocalEncoder's UV-pixel-box neighborhood with a
    depth-scaled 3D ball: `||P_j - P_i|| < r` for each radius. Uses the
    pixel-to-3D approximation `P ~ z * (u_c/vfp, v_c/vfp, 1)` in the cam
    frame (z = z_norm * scale), so distances are in metric units (~m).

    Multi-scale grouping (= concat over radii) keeps full per-point
    resolution while letting each query see near + mid + far context.
    """
    def __init__(self, d_out: int = D_DIM,
                 radii=(1.0, 4.0, 16.0),
                 k_per_scale=(8, 8, 8),
                 z_scale: float = 50.0,
                 vfp_default: float = 64.0):
        super().__init__()
        assert len(radii) == len(k_per_scale)
        self.radii = tuple(float(r) for r in radii)
        self.k = tuple(int(k) for k in k_per_scale)
        self.z_scale = z_scale
        self.vfp_default = vfp_default
        d_per = max(d_out // len(radii), 32)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, 32), nn.GELU(),
                nn.Linear(32, d_per), nn.GELU(),
                nn.Linear(d_per, d_per),
            ) for _ in radii
        ])
        self.fuse = nn.Linear(d_per * len(radii), d_out)

    def _to_3d(self, uvd: torch.Tensor, vfp: torch.Tensor, img_size: int) -> torch.Tensor:
        """uvd: (B, N, 3) [u, v, z_norm]   vfp: (B,) or scalar.
        Returns (B, N, 3) cam-frame 3D positions in meters (approx)."""
        u = uvd[..., 0] - img_size * 0.5    # center: principal point ≈ patch center
        v = uvd[..., 1] - img_size * 0.5
        z = uvd[..., 2] * self.z_scale
        if vfp is None:
            f = uvd.new_full(uvd.shape[:-1], self.vfp_default)
        else:
            if vfp.dim() == 0:
                f = uvd.new_full(uvd.shape[:-1], float(vfp))
            else:
                f = vfp.unsqueeze(-1).expand_as(z)
        x = z * u / f.clamp(min=1.0)
        y = z * v / f.clamp(min=1.0)
        return torch.stack([x, y, z], dim=-1)        # (B, N, 3) in metres

    def forward(self, query_uvd: torch.Tensor,
                full_uvd: torch.Tensor,
                full_pad_mask: torch.Tensor,
                vfp: torch.Tensor = None,
                img_size: int = 64,
                query_pad_mask: torch.Tensor = None) -> torch.Tensor:
        if full_uvd is None or full_pad_mask is None:
            raise ValueError(
                "LocalNeighborhood3D.forward: full_uvd AND full_pad_mask are required. "
                "Silent fallback to query-self pooling was removed."
            )
        Pq = self._to_3d(query_uvd, vfp, img_size)         # (B, N_q, 3)
        Pk = self._to_3d(full_uvd,  vfp, img_size)         # (B, N_kv, 3)
        rel = Pk.unsqueeze(1) - Pq.unsqueeze(2)             # (B, N_q, N_kv, 3) metres
        dist = rel.norm(dim=-1)                             # (B, N_q, N_kv)
        # exclude self
        self_match = (dist == 0)
        dist = dist.masked_fill(self_match, 1e9)
        if full_pad_mask is not None:
            dist = dist.masked_fill(full_pad_mask.unsqueeze(1), 1e9)

        feats_per_scale = []
        for r, k, mlp in zip(self.radii, self.k, self.mlps):
            d_masked = dist.masked_fill(dist > r, 1e9)
            kk = min(k, dist.size(-1))
            _, idx = d_masked.topk(kk, dim=-1, largest=False)         # (B, N_q, k)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            rel_top = rel.gather(2, idx_exp)                          # (B, N_q, k, 3)
            valid = d_masked.gather(2, idx) < r                       # (B, N_q, k)
            feat = mlp(rel_top)                                       # (B, N_q, k, d_per)
            feat = feat.masked_fill(~valid.unsqueeze(-1), torch.finfo(feat.dtype).min)
            feat, _ = feat.max(dim=2)                                 # (B, N_q, d_per)
            feat = feat.masked_fill(
                ~valid.any(dim=-1, keepdim=True), 0.0)
            feats_per_scale.append(feat)
        return self.fuse(torch.cat(feats_per_scale, dim=-1))           # (B, N_q, d_out)


class PoseEmb(nn.Module):
    """Embed (relative SE3 6-DoF + log_vfp) into a per-sample D-dim bias.

    For single-frame calib, the SE3 part is constant zero — the network sees
    only log(vfp), giving it the scale anchor (vfp + depth → metric size).
    For cross-frame, SE3 carries the virtual-cam → virtual-cam pose change.
    """
    def __init__(self, d: int = D_DIM, in_dof: int = 7):
        super().__init__()
        self.in_dof = in_dof
        self.net = nn.Sequential(
            nn.Linear(in_dof, d), nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, D)


class CalibNetDepth(nn.Module):
    def __init__(self, d: int = D_DIM, img_size: int = 128, in_channels: int = 1,
                 n_layers: int = 3, self_first: bool = False, kv_self_attn: bool = False,
                 cross_temp: float = 1.0, use_convnext: bool = False,
                 use_frustum: bool = False, r_uv_cells: float = 1.5, r_d: float = 0.004,
                 k_nb: int = 16, frustum_grid_n: int = 16,
                 frustum_dense: bool = False,
                 deform_mode: str = 'none', deform_n_points: int = 4,
                 use_frame_token: bool = False, frame_token_side: int = 8,
                 use_lidar_kv: bool = False, use_pose_emb: bool = False,
                 use_3d_local: bool = False,
                 local_3d_radii=(1.0, 4.0, 16.0),
                 local_3d_k=(8, 8, 8)):
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
        if use_3d_local:
            # Replaces FrustumLocalEncoder's UV-pixel-box with 3D ball-query
            # MSG (PointNet++ style). Multi-scale via radii in metres.
            self.frustum_enc = LocalNeighborhood3D(d, radii=local_3d_radii,
                                                    k_per_scale=local_3d_k)
            self._is_3d_local = True
        else:
            self.frustum_enc = (FrustumLocalEncoder(d, r_uv_cells=r_uv_cells, r_d=r_d,
                                                     k=k_nb, grid_n=frustum_grid_n)
                                if use_frustum else None)
            self._is_3d_local = False
        self.frame_enc   = FrameTokenEncoder(d, m_side=frame_token_side) if use_frame_token else None
        self._use_frame_token  = use_frame_token
        self._frame_token_side = frame_token_side
        self._use_lidar_kv = use_lidar_kv
        self._frustum_dense = frustum_dense and use_frustum and not use_3d_local
        self._use_pose_emb = use_pose_emb
        self.pose_emb = PoseEmb(d) if use_pose_emb else None

        if deform_mode != 'none':
            assert not self_first, "deform_mode is incompatible with self_first=True"
            try:
                from .model_deform import CrossAttentionBlockDeform, CrossAttentionBlockDeformML
            except ImportError:
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

    def _block(self, block, q, feat, uv_01, mask, extra_kv=None, extra_kv_mask=None):
        if self._deform_mode == 'ml':
            # feat here is (coarse_feat, fine_feat) tuple — ML block wants the list
            return block(q, list(feat), uv_01, self.level_embed,
                          key_padding_mask=mask, self_first=False)
        if self._self_first:
            return block(q, feat, uv_01, key_padding_mask=mask)
        return block(q, feat, uv_01, key_padding_mask=mask, self_first=False,
                     extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)

    def forward(self, image: torch.Tensor, distorted_uvd: torch.Tensor,
                key_padding_mask=None, vfp: torch.Tensor = None,
                distorted_uvd_full: torch.Tensor = None,
                pad_full: torch.Tensor = None):
        """
        image           : (B, C, H, W)
        distorted_uvd   : (B, N, 3)  [U, V, D_norm]
        key_padding_mask: (B, N) bool  True = padding position
        Returns params  : (B, N, 5)  [tx, ty, log_sx, log_sy, rho]
        """
        coarse_feat, fine_feat = self.cnn(image)

        # Frame-token bottleneck: compress (coarse, fine) into M=m_side² tokens
        # whose Q is UV-pos-enc only. Both per-point layers then read out of
        # this same bank, replacing direct image-token cross-attn.
        if self.frame_enc is not None:
            ft = self.frame_enc(coarse_feat, fine_feat)                  # (B, M, D)
            B  = ft.size(0)
            ms = self._frame_token_side
            ft_map = ft.transpose(1, 2).reshape(B, -1, ms, ms)            # (B, D, m, m)
            coarse_feat = ft_map
            fine_feat   = ft_map

        uv_01    = distorted_uvd[..., :2] / self.img_size
        uvd_norm = torch.cat([uv_01, distorted_uvd[..., 2:3]], dim=-1)
        d3       = distorted_uvd[..., 2:3]

        q = self.point_mlp(uvd_norm)
        if self.frustum_enc is not None:
            if distorted_uvd_full is None or pad_full is None:
                raise ValueError(
                    "CalibNetDepth.forward: distorted_uvd_full + pad_full are REQUIRED "
                    "when use_frustum=True or use_3d_local=True. The dataset/collate "
                    "must surface the dense raw point cloud per crop. Silent fallback "
                    "removed (broken frustum invalidated v504r runs)."
                )
            if getattr(self, '_is_3d_local', False):
                q = q + self.frustum_enc(distorted_uvd,
                                          full_uvd=distorted_uvd_full,
                                          full_pad_mask=pad_full,
                                          vfp=vfp, img_size=self.img_size,
                                          query_pad_mask=key_padding_mask)
            elif self._frustum_dense:
                # dense gh*gw lidar map → keep separately, fed as extra_kv to
                # the cross-attention later (NOT added to q).
                self._lidar_kv_dense = self.frustum_enc.forward_dense(
                    full_uvd=distorted_uvd_full,
                    full_pad_mask=pad_full,
                    img_size=self.img_size)
            else:
                q = q + self.frustum_enc(distorted_uvd,
                                          full_uvd=distorted_uvd_full,
                                          full_pad_mask=pad_full,
                                          query_token=q,
                                          query_pad_mask=key_padding_mask,
                                          img_size=self.img_size)

        # pose_emb: per-sample (SE3=0 for calib) + log(vfp) → D-dim bias.
        # Broadcast added to Q (per-point) AND to KV (image tokens, lidar tokens).
        pose_emb_b = None
        if self._use_pose_emb:
            B = image.size(0)
            if vfp is None:
                vfp = torch.full((B,), float(self.img_size), device=image.device)
            log_vfp = torch.log(vfp.clamp(min=1.0)).unsqueeze(-1)            # (B, 1)
            zero_se3 = torch.zeros(B, 6, device=image.device, dtype=log_vfp.dtype)
            pose_emb_b = self.pose_emb(torch.cat([zero_se3, log_vfp], dim=-1))  # (B, D)
            q = q + pose_emb_b.unsqueeze(1)                                   # broadcast → (B, N, D)
            pe_2d = pose_emb_b.unsqueeze(-1).unsqueeze(-1)                    # (B, D, 1, 1)
            coarse_feat = coarse_feat + pe_2d
            fine_feat   = fine_feat   + pe_2d

        # extra_kv (lidar bank): two modes —
        #   _use_lidar_kv (legacy):   copy of initial Q (per-pivot features)
        #   _frustum_dense:           gh*gw dense lidar map from forward_dense,
        #                             every cell gets a token (zero LiDAR + UV emb
        #                             for empty cells). This is the "dense LiDAR
        #                             map as KV channel" route — empty cells
        #                             still carry their position embedding so DA
        #                             can attend by UV alone.
        if self._frustum_dense and getattr(self, '_lidar_kv_dense', None) is not None:
            extra_kv = self._lidar_kv_dense
            extra_kv_mask = None       # all cells valid (UV emb is always there)
        else:
            extra_kv = q if self._use_lidar_kv else None
            extra_kv_mask = key_padding_mask if self._use_lidar_kv else None

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
        q, raw_cum = self._block(blocks[0], q, feats_by_layer[0], uv_01, key_padding_mask,
                                  extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)
        # refinement layers
        for i in range(1, min(self.n_layers, len(blocks))):
            uv_i = (uv_01 + raw_cum[..., :2]).clamp(0, 1)
            q    = self.point_mlp(torch.cat([uv_i, d3], dim=-1)) + q
            q, raw_i = self._block(blocks[i], q, feats_by_layer[i], uv_i, key_padding_mask,
                                    extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)
            raw_cum = raw_cum + raw_i
        raw = raw_cum

        return clamp_params(raw, self.img_size)   # (B,N,5)
