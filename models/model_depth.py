"""
model_depth.py  –  Depth-aware CalibNet with covariance output

Input points: (U, V, D_norm)  — 3-channel instead of 2
Output per point: (tx, ty, log_sx, log_sy, rho_raw)

The depth channel lets the model learn:
  - obj points (D~0.2,0.4) → small confident covariance
  - bg  points (D~0.8)     → large/degenerate covariance
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from models.model import CNNBackbone, ConvNeXtBackbone, D as D_DIM
from models.model_cov import CrossAttentionBlockCov, TransformerDecoderBlock, clamp_params, gaussian2d_nll


class PointMLP3(nn.Module):
    """Point MLP for (U, V, D_norm, [intensity]) input.

    in_channels is 3 for legacy caches (no intensity) or 4 for V3-i caches
    (with per-point intensity). The downstream code is unchanged — only the
    first linear layer's input dim shifts.
    """
    def __init__(self, d: int = D_DIM, in_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Linear(self.in_channels, 64), nn.GELU(),
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
                kv_proj = nn.Linear(4, 2 * d_local),  # uvd + intensity
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
                bucket_uvd: torch.Tensor,
                bucket_valid: torch.Tensor,
                query_token: torch.Tensor,
                query_pad_mask: torch.Tensor = None,
                img_size: int = 64) -> torch.Tensor:
        """Per-query mini cross-attn over neighbors gathered from a (G², K)
        cell-bucketed lidar grid built in the dataloader.

        Args:
            query_uvd:    (B, N_q, 3)            U,V in [0,img_size], D in [0,1]
            bucket_uvd:   (B, G², K_per_cell, 3) lidar pts pre-binned by cell
            bucket_valid: (B, G², K_per_cell)    bool — True = real pt, False = pad
            query_token:  (B, N_q, d_out)        per-point feature (PointMLP)
            query_pad_mask: (B, N_q)             unused, kept for API compat
            img_size:     crop side in the resolution `query_uvd` lives in.

        Returns:
            (B, N_q, d_out) — local feature per query point.

        Pre-bucketing in the dataloader (vanilla numpy scatter) means we no
        longer materialize the (B, Nq, Nkv, 3) `rel` tensor or do brute-force
        topk over Nkv. Each query reads exactly 9·K candidates.
        """
        cell_px = float(img_size) / float(self.grid_n)
        G = self.grid_n
        K_pc = bucket_uvd.shape[2]
        C    = bucket_uvd.shape[-1]   # 3 (uvd) or 4 (uvd + intensity)
        B, N_q, _ = query_uvd.shape

        # ── Gather 3×3 neighbor cells per query ──
        q_cu = (query_uvd[..., 0] / cell_px).long().clamp(0, G - 1)   # (B, Nq)
        q_cv = (query_uvd[..., 1] / cell_px).long().clamp(0, G - 1)
        # offsets: (du, dv) for the 9 cells in row-major order
        du = torch.arange(-1, 2, device=query_uvd.device)             # (3,)
        dv = torch.arange(-1, 2, device=query_uvd.device)
        du9 = du.repeat(3)                                            # (9,)  -1,0,1,-1,0,1,-1,0,1
        dv9 = dv.repeat_interleave(3)                                 # (9,)  -1,-1,-1,0,0,0,1,1,1
        nb_cu = (q_cu.unsqueeze(-1) + du9).clamp(0, G - 1)            # (B, Nq, 9)
        nb_cv = (q_cv.unsqueeze(-1) + dv9).clamp(0, G - 1)
        nb_cid = nb_cv * G + nb_cu                                     # (B, Nq, 9)

        # gather (B, Nq, 9, K, C) and (B, Nq, 9, K)
        nb_idx_exp = nb_cid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, K_pc, C)
        bucket_uvd_exp = bucket_uvd.unsqueeze(1).expand(-1, N_q, -1, -1, -1)
        cands = bucket_uvd_exp.gather(2, nb_idx_exp)                   # (B, Nq, 9, K, C)
        nb_v_exp = nb_cid.unsqueeze(-1).expand(-1, -1, -1, K_pc)
        bucket_v_exp = bucket_valid.unsqueeze(1).expand(-1, N_q, -1, -1)
        cand_valid = bucket_v_exp.gather(2, nb_v_exp)                  # (B, Nq, 9, K)
        # flatten neighbor + slot dims
        cands = cands.reshape(B, N_q, 9 * K_pc, C)                     # (B, Nq, 72, C)
        cand_valid = cand_valid.reshape(B, N_q, 9 * K_pc)              # (B, Nq, 72)

        # rel relative to query
        rel_all = cands - query_uvd.unsqueeze(2)                       # (B, Nq, 72, C)

        # Density-invariant: k random points from the valid set.
        rand_score = torch.rand(B, N_q, 9 * K_pc, device=query_uvd.device, dtype=query_uvd.dtype)
        rand_score = rand_score.masked_fill(~cand_valid, -1.0)
        kk = min(self.k, 9 * K_pc)
        _, topk_idx = rand_score.topk(kk, dim=-1, largest=True)        # (B, N_q, k)
        valid = rand_score.gather(2, topk_idx) >= 0
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        topk_rel = rel_all.gather(2, idx_exp)                          # (B, N_q, k, C)

        # Debug instrumentation
        self._last_topk_idx = topk_idx.detach()
        self._last_valid    = valid.detach()
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

    def forward_dense(self, bucket_uvd: torch.Tensor,
                       bucket_valid: torch.Tensor,
                       img_size: int = 64) -> torch.Tensor:
        """Dense gh×gw lidar map: query at every cell center, output (B, gh*gw, D).

        Args:
            bucket_uvd:   (B, G², K_per_cell, 3)  pre-binned lidar grid
            bucket_valid: (B, G², K_per_cell)     bool valid mask
            img_size:     crop pixel side
        Returns:
            (B, gh*gw, D)  with cell_uv_embed already added.

        Empty cells (no lidar) fall through to zero geometry feature; the
        learnable per-cell UV embedding is added so even no-LiDAR cells carry
        positional info for downstream cross-attention.
        """
        B = bucket_uvd.shape[0]
        device = bucket_uvd.device
        cell_px = float(img_size) / float(self.grid_n)
        cy, cx = torch.meshgrid(
            torch.arange(self.grid_n, device=device, dtype=bucket_uvd.dtype),
            torch.arange(self.grid_n, device=device, dtype=bucket_uvd.dtype),
            indexing='ij',
        )
        u = (cx + 0.5) * cell_px
        v = (cy + 0.5) * cell_px
        d = torch.zeros_like(u)
        C = bucket_uvd.shape[-1]
        if C == 4:
            i = torch.zeros_like(u)  # zero intensity for empty cells
            cell_uvd = torch.stack([u, v, d, i], dim=-1).reshape(1, -1, C).expand(B, -1, -1)
        else:
            cell_uvd = torch.stack([u, v, d], dim=-1).reshape(1, -1, C).expand(B, -1, -1)
        D = self.in_proj.in_features
        zero_q_token = torch.zeros(B, cell_uvd.shape[1], D,
                                    device=device, dtype=bucket_uvd.dtype)
        feat = self.forward(query_uvd=cell_uvd, bucket_uvd=bucket_uvd,
                             bucket_valid=bucket_valid,
                             query_token=zero_q_token, img_size=img_size)
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


class CLSFramePoseHead(nn.Module):
    """CLS-style aggregator: learnable token cross-attends to per-pt features
    and outputs (μ, log σ) for an n_dof SE3 perturbation. Defaults to diagonal
    covariance for back-compat with old ckpts; pass full_cov=True to get the
    full n_dof×n_dof Cholesky factor (off-diagonal entries capture pose-dim
    correlations like yaw↔t_x ambiguity — crucial for multi-tile BA fusion).

    Output (full_cov=False, default): linear head dim 2*n_dof, returns (μ, log_σ, None).
    Output (full_cov=True):           linear head dim n_dof + n_dof*(n_dof+1)/2,
                                       returns (μ, log_σ_diag, L) where Σ = L Lᵀ.
    """
    def __init__(self, d: int, n_dof: int = 6, n_heads: int = 4, full_cov: bool = False):
        super().__init__()
        self.n_dof = n_dof
        self.full_cov = bool(full_cov)
        self.n_chol = n_dof * (n_dof + 1) // 2 if self.full_cov else 0
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.drop = nn.Dropout(0.1)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(4 * d, d),
        )
        out_dim = (n_dof + self.n_chol) if self.full_cov else (2 * n_dof)
        self.head = nn.Linear(d, out_dim)
        if self.full_cov:
            self.register_buffer('tril_idx', torch.tril_indices(n_dof, n_dof), persistent=False)

    def forward(self, q: torch.Tensor, key_padding_mask=None):
        """Returns (μ, log_σ, L_or_None)."""
        B = q.size(0)
        cls = self.cls.expand(B, -1, -1)
        cls_attn, _ = self.cross(self.norm_q(cls),
                                  self.norm_kv(q), self.norm_kv(q),
                                  key_padding_mask=key_padding_mask)
        cls = cls + self.drop(cls_attn)
        cls = cls + self.ffn(self.norm_ffn(cls))
        out = self.head(cls.squeeze(1))
        mu = out[:, :self.n_dof]
        if not self.full_cov:
            log_sigma = out[:, self.n_dof:]
            return mu, log_sigma, None
        chol_flat = out[:, self.n_dof:]
        L = q.new_zeros(B, self.n_dof, self.n_dof)
        L[:, self.tril_idx[0], self.tril_idx[1]] = chol_flat
        diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-4
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
        log_sigma = torch.log(diag)
        return mu, log_sigma, L


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
                 local_3d_k=(8, 8, 8),
                 use_frame_pose: bool = False, frame_pose_dof: int = 6,
                 frame_pose_full_cov: bool = False,
                 use_intensity: bool = True):
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
        # Point input dim: 3 (u, v, d) for legacy caches; 4 (+ intensity) for V3-i.
        self.use_intensity = bool(use_intensity)
        self.point_mlp   = PointMLP3(d, in_channels=4 if self.use_intensity else 3)
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
                # Stage 1 dual-DA: when frustum_dense, treat dense LiDAR map as
                # an additional level (in addition to coarse+fine image), so
                # both image and lidar go through the SAME deformable attention.
                # n_levels = 3 in that case, otherwise 2.
                ml_levels = 3 if frustum_dense else 2
                Block = CrossAttentionBlockDeformML
                kw = dict(kv_self_attn=kv_self_attn, cross_temp=cross_temp,
                          n_levels=ml_levels, n_points=deform_n_points)
                # learnable resolution (level) embedding shared across blocks
                self.level_embed = nn.Parameter(torch.zeros(ml_levels, d))
                nn.init.normal_(self.level_embed, std=0.02)
                self._ml_levels = ml_levels
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

        # CLS frame-level pose head: outputs (μ, log σ) for an n_dof SE3 patch
        # perturbation. Generic body (cross-attn aggregator) + DoF-specific head.
        # Adding intrinsic params later = swap head + jacobian, body reusable.
        self.frame_pose_head = (CLSFramePoseHead(d, n_dof=frame_pose_dof, full_cov=frame_pose_full_cov)
                                 if use_frame_pose else None)

        # DDP support: freeze parameters whose forward path is never taken in
        # the current config so find_unused_parameters=False doesn't crash.
        # Two known dead branches:
        #   1. CrossAttentionBlockDeform.{extra_kv_attn, norm_extra_kv} —
        #      only run when extra_kv != None, i.e. use_lidar_kv=True or
        #      frustum_dense=True. Unused otherwise.
        #   2. FrustumLocalEncoder.cell_uv_embed — only touched in
        #      forward_dense, gated on frustum_dense=True.
        # state_dict keys preserved (requires_grad isn't saved) so existing
        # ckpts load unchanged.
        extra_kv_used = self._use_lidar_kv or self._frustum_dense
        if not extra_kv_used:
            for block_name in ('cross_coarse', 'cross_fine', 'cross_refine', 'cross_fine2'):
                blk = getattr(self, block_name, None)
                if blk is None:
                    continue
                for attr in ('extra_kv_attn', 'norm_extra_kv'):
                    mod = getattr(blk, attr, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad_(False)
        if not self._frustum_dense and self.frustum_enc is not None:
            ce = getattr(self.frustum_enc, 'cell_uv_embed', None)
            if ce is not None:
                ce.requires_grad_(False)

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
                bucket_uvd: torch.Tensor = None,
                bucket_valid: torch.Tensor = None):
        """
        image           : (B, C, H, W)
        distorted_uvd   : (B, N, 3 or 4)  [U, V, D_norm, (intensity if use_intensity)]
        key_padding_mask: (B, N) bool  True = padding position
        bucket_uvd      : (B, G², K_per_cell, 3)  pre-binned lidar grid (geometric only)
        bucket_valid    : (B, G², K_per_cell)     bool valid mask
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
        d3       = distorted_uvd[..., 2:3]
        if self.use_intensity:
            # distorted_uvd has (u, v, d, intensity). uvd_norm becomes 4-D so the
            # point MLP (in_channels=4) embeds intensity alongside geometry.
            uvd_norm = torch.cat([uv_01, d3, distorted_uvd[..., 3:4]], dim=-1)
        else:
            uvd_norm = torch.cat([uv_01, d3], dim=-1)
        # Spatial+intensity view of the query, fed to frustum encoder so the
        # KV (uvd+intensity) and Q (uvd+intensity) match in channel count and
        # rel = cands - query stays well-defined.
        if self.use_intensity:
            distorted_uvd_geom = torch.cat([distorted_uvd[..., :3],
                                            distorted_uvd[..., 3:4]], dim=-1)
        else:
            distorted_uvd_geom = distorted_uvd[..., :3]

        # Stage 2 (mixed Q): randomly zero out depth for some queries so the
        # model learns to handle UV-only Q. query_drop_prob is set externally
        # (training loop) — 0 means no drop, 1 means all queries become UV-only.
        # Applied only in training mode.
        qdp = float(getattr(self, 'query_drop_prob', 0.0))
        if self.training and qdp > 0.0:
            # Bernoulli per (B, N) — independent per query position
            mask = torch.bernoulli(torch.full_like(d3, 1.0 - qdp)).to(d3.dtype)
            if self.use_intensity:
                uvd_norm = torch.cat([uv_01, d3 * mask, distorted_uvd[..., 3:4]], dim=-1)
            else:
                uvd_norm = torch.cat([uv_01, d3 * mask], dim=-1)

        q = self.point_mlp(uvd_norm)
        if self.frustum_enc is not None:
            if bucket_uvd is None or bucket_valid is None:
                raise ValueError(
                    "CalibNetDepth.forward: bucket_uvd + bucket_valid are REQUIRED "
                    "when use_frustum=True. The dataset/collate must produce "
                    "(B, G², K_per_cell, 3) cell-bucketed lidar (PandaSetCalibDatasetFull "
                    "post-2026-05-04). Old (full_uvd, pad_full) flat layout removed."
                )
            if getattr(self, '_is_3d_local', False):
                # 3D-local path still uses the flat layout — flatten bucket to (B, G²·K, C).
                B_, G2, Kpc, C_ = bucket_uvd.shape
                flat_uvd = bucket_uvd.reshape(B_, G2 * Kpc, C_)
                flat_pad = ~bucket_valid.reshape(B_, G2 * Kpc)
                q = q + self.frustum_enc(distorted_uvd_geom,
                                          full_uvd=flat_uvd,
                                          full_pad_mask=flat_pad,
                                          vfp=vfp, img_size=self.img_size,
                                          query_pad_mask=key_padding_mask)
            elif self._frustum_dense:
                self._lidar_kv_dense = self.frustum_enc.forward_dense(
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                    img_size=self.img_size)
            else:
                q = q + self.frustum_enc(distorted_uvd_geom,
                                          bucket_uvd=bucket_uvd,
                                          bucket_valid=bucket_valid,
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
            # In ML mode, the dense LiDAR is fed as an additional DA level (see
            # below) — extra_kv path is skipped so we don't double-count.
            if self._deform_mode == 'ml':
                extra_kv, extra_kv_mask = None, None
            else:
                extra_kv = self._lidar_kv_dense
                extra_kv_mask = None
        else:
            extra_kv = q if self._use_lidar_kv else None
            extra_kv_mask = key_padding_mask if self._use_lidar_kv else None

        # ML mode: every block sees both levels; SL / none: alternate coarse→fine
        if self._deform_mode == 'ml':
            # Stage 1 dual-DA: when frustum_dense is on, append dense LiDAR as
            # 3rd level. _lidar_kv_dense is (B, gh*gw, D); reshape to (B, D, gh, gw).
            if self._frustum_dense and getattr(self, '_lidar_kv_dense', None) is not None:
                lkv = self._lidar_kv_dense
                B_, NHW, D_ = lkv.shape
                gn = int(NHW ** 0.5)
                assert gn * gn == NHW, f"lidar_kv_dense not square grid: NHW={NHW}"
                lidar_2d = lkv.permute(0, 2, 1).reshape(B_, D_, gn, gn).contiguous()
                feat_all = (coarse_feat, fine_feat, lidar_2d)
            else:
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
            if self.use_intensity:
                refine_uvd = torch.cat([uv_i, d3, distorted_uvd[..., 3:4]], dim=-1)
            else:
                refine_uvd = torch.cat([uv_i, d3], dim=-1)
            q    = self.point_mlp(refine_uvd) + q
            q, raw_i = self._block(blocks[i], q, feats_by_layer[i], uv_i, key_padding_mask,
                                    extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)
            raw_cum = raw_cum + raw_i
        raw = raw_cum
        per_pt = clamp_params(raw, self.img_size)   # (B,N,5)

        if self.frame_pose_head is not None:
            # q here is the final per-pt feature stack (B, N, D) — input to CLS.
            mu, log_sigma, L = self.frame_pose_head(q, key_padding_mask=key_padding_mask)
            return per_pt, (mu, log_sigma, L)
        return per_pt
