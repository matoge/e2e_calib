"""
calibnet2.py — CalibNet2 (CND2): shared-weight DA cross-attn stack with
type-1 RoPE PoseEmb on Q.  q is the only state; readout happens once on Δ_cum.

Design (memory: project_calibnet2_design.md, north_star.md):

  * One Block, stacked n_iter times with shared weights.
  * Block (DA cross-attn over 3-level KV, then self-attn, then FFN — pure
    feature-space refinement):
        ca = MSDeformAttn(q, ref_q, kv_flat, spatial_shapes, level_start)
        q' = q + ca
        sa = self_attn(q')
        q'' = q' + sa
        ff = ffn(q'')
        delta  = ca + sa + ff
        q_next = q + delta
    The block emits NO (duv, σ).  Readout happens once at the end via
    final_head(Δ_cum), Δ_cum = Σ delta_i, NOT q_final — this is the
    abs-PE leak gate that prevents PointMLP3's absolute-position info
    from reaching the head.
  * KV always own-frame, MSDeformAttn over 3 flattened levels:
    coarse_img / fine_img / dense_lidar (2D, gn×gn from FrustumLocalEncoder).
    level_embed (3,D) separates the three levels.
  * Reference points for DA are derived from Q internally via a small
    Linear→sigmoid (zero-init → ref ≈ 0.5 at start).  No uv_01 is passed
    into the block — Q already carries the position info.
  * KV is built ONCE in CalibNet2.forward and reused across n_iter.
  * RoPE on Q at entry only:
        type-0 (D_s=8 scalar):  intrinsic additive (log_vfp via small MLP).
        type-1 (K=40 vectors of dim 3, total 120):  block-diag(R) action.
        Total D = 8 + 40·3 = 128.
  * Final readout: raw = final_head(Δ_cum), single Linear(D → 5).  Zero-init
    weight, bias[2,3] = log(2) for log_σ — initial output ≈ legacy
    clamp_params bias.
  * dpose_R = I (default) → forward degenerates to legacy single-frame calib.
  * forward_pair(A, B, R_AB):  pred_A = forward(A, R=I);
                               pred_A_to_B = forward(A, R=R_AB).
                               KV always own — A's image+lidar stay as KV.
                               (Cross-frame transport happens through Q-rotation.)

CalibNetDepth (model_depth.CalibNetDepth) is left untouched — existing ckpts
still load. CalibNet2 is a fresh net.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import ConvNeXtBackbone, CNNBackbone, D as D_DIM
from models.model_depth import PointMLP3, FrustumLocalEncoder
from models.model_deform import MSDeformAttn


# ---------------------------------------------------------------------------
# RoPE PoseEmb — type-0 scalar (intrinsic) + type-1 block-diag(R) on Q
# ---------------------------------------------------------------------------
class RoPEPoseEmb(nn.Module):
    """Apply SO(3) on the type-1 chunk of Q and (optionally) add an intrinsic
    bias on the type-0 chunk.

    Token layout (D=128):
        q[..., :D_s]                          type-0 scalar (rotation-invariant)
        q[..., D_s : D_s + K*3].view(K, 3)    type-1 vector group (K vectors of dim 3)

    Defaults: D_s=8, K=40  →  8 + 120 = 128.

    SO(3) action on type-1 is exact at float32 precision (1e-7 in the toy,
    docs/blog/2026-05-26_se3_token_rotation*.md). dpose_R = I → identity.
    """
    def __init__(self, d: int = D_DIM, d_scalar: int = 8, n_type1: int = 40):
        super().__init__()
        assert d_scalar + n_type1 * 3 == d, (
            f"D={d} must equal d_scalar({d_scalar}) + 3·n_type1({n_type1})")
        self.d = d
        self.d_scalar = d_scalar
        self.n_type1 = n_type1
        # Intrinsic bias on type-0 only: takes log_vfp scalar → d_scalar dims.
        self.intrinsic_mlp = nn.Sequential(
            nn.Linear(1, max(d_scalar, 8)), nn.GELU(),
            nn.Linear(max(d_scalar, 8), d_scalar),
        )
        # Zero-init final layer so PoseEmb(log_vfp) ≈ 0 at start (legacy calib).
        nn.init.zeros_(self.intrinsic_mlp[-1].weight)
        nn.init.zeros_(self.intrinsic_mlp[-1].bias)
        # Translation bias on type-0: 3D t (in scene units, e.g. metres) →
        # d_scalar. Separated from rotation (which acts on type-1) — block stack
        # downstream mixes the two chunks anyway, so keeping t in the scalar
        # path avoids physical-unit-vs-token-scale mismatch on type-1.
        self.translation_mlp = nn.Sequential(
            nn.Linear(3, max(d_scalar, 16)), nn.GELU(),
            nn.Linear(max(d_scalar, 16), d_scalar),
        )
        # Zero-init: PoseEmb(t) = 0 at start → SO(3)-only behaviour preserved
        # (existing kick #1 ckpts load-compatible).
        nn.init.zeros_(self.translation_mlp[-1].weight)
        nn.init.zeros_(self.translation_mlp[-1].bias)

    def forward(self, q: torch.Tensor, *,
                dpose_R: torch.Tensor = None,
                dpose_t: torch.Tensor = None,
                log_vfp: torch.Tensor = None) -> torch.Tensor:
        """
        q       : (B, N, D)
        dpose_R : (B, 3, 3) or None  — None ↔ identity (no rotation)
        dpose_t : (B, 3) or None     — None ↔ no translation bias
        log_vfp : (B,) or (B, 1) or None — None ↔ no intrinsic bias

        Returns (B, N, D).
        """
        B, N, D = q.shape
        Ds, K = self.d_scalar, self.n_type1
        q_s = q[..., :Ds]                                   # (B, N, Ds)
        q_v = q[..., Ds:].reshape(B, N, K, 3)               # (B, N, K, 3)

        if dpose_R is not None:
            q_v = torch.einsum('bij,bnkj->bnki', dpose_R, q_v)

        if log_vfp is not None:
            if log_vfp.dim() == 1:
                log_vfp = log_vfp.unsqueeze(-1)             # (B, 1)
            q_s = q_s + self.intrinsic_mlp(log_vfp).unsqueeze(1)

        if dpose_t is not None:
            q_s = q_s + self.translation_mlp(dpose_t).unsqueeze(1)

        q_out = torch.cat([q_s, q_v.reshape(B, N, K * 3)], dim=-1)
        return q_out


# ---------------------------------------------------------------------------
# DA cross-attn block — feature-space only.  q is the only state token.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Single shared-weight block.  q ← q + (DA-cross-attn + self-attn + ffn).

    Pure feature-space refinement.  The block emits NO (duv, σ) — readout
    happens once at the end of the model via `final_head(Δ_cum)`.

    Cross-attn = MSDeformAttn over n_levels=3 flattened KV (coarse_img,
    fine_img, dense_lidar_2d).  Reference points are projected from Q
    internally (zero-init → ref = sigmoid(0) = 0.5 at start; the network
    learns to focus from there as Q acquires geometric content).  uv_01 is
    NOT a block argument — Q already carries position via PointMLP3.

    The KV bundle (kv_flat, spatial_shapes, level_start_index) is built
    ONCE outside the n_iter loop and threaded in unchanged each iter.
    """
    def __init__(self, d: int = D_DIM, n_heads: int = 4, n_levels: int = 3,
                 n_points: int = 4):
        super().__init__()
        self.n_levels = n_levels
        self.norm_q   = nn.LayerNorm(d)
        self.norm_kv  = nn.LayerNorm(d)
        self.cross_attn = MSDeformAttn(d_model=d, n_levels=n_levels,
                                        n_heads=n_heads, n_points=n_points)
        # Q → reference points (one ref per level).  Zero-init weight + zero
        # bias → ref = sigmoid(0) = 0.5 at start (image center).  The network
        # learns to anchor as Q acquires per-point u/v info from cross-attn.
        self.ref_proj = nn.Linear(d, n_levels * 2)
        nn.init.zeros_(self.ref_proj.weight)
        nn.init.zeros_(self.ref_proj.bias)

        self.norm_self  = nn.LayerNorm(d)
        self.self_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True,
                                                 dropout=0.1)
        self.drop = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor,
                kv_flat: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                key_padding_mask=None):
        """
        q                 : (B, N, D)
        kv_flat           : (B, Σ HlWl, D)  — already + level_embed, NOT normed
        spatial_shapes    : (n_levels, 2) long
        level_start_index : (n_levels,)    long
        key_padding_mask  : (B, N) bool — True = pad (for self-attn).

        Returns:
            q_next : (B, N, D)  =  q + delta   (next-state token)
            delta  : (B, N, D)  =  ca + sa + ffn  (the residual the block adds —
                                  this is what the final head reads, NOT q.
                                  Gates abs-PE leakage from PointMLP3.)
        """
        B, N, D_ = q.shape
        nq  = self.norm_q(q)
        nkv = self.norm_kv(kv_flat)

        # reference points from Q — (B, N, n_levels, 2) in [0, 1].
        ref = self.ref_proj(nq).view(B, N, self.n_levels, 2)
        ref = torch.sigmoid(ref)

        ca = self.cross_attn(nq, ref, nkv, spatial_shapes, level_start_index,
                              input_padding_mask=None)
        ca = self.drop(ca)

        q1 = q + ca
        sq = self.norm_self(q1)
        sa, _ = self.self_attn(sq, sq, sq, key_padding_mask=key_padding_mask)
        sa = self.drop(sa)

        q2 = q1 + sa
        ff = self.ffn(self.norm_ffn(q2))

        delta  = ca + sa + ff
        q_next = q + delta
        return q_next, delta


# ---------------------------------------------------------------------------
# CalibNet2 (CND2) — main model
# ---------------------------------------------------------------------------
class CalibNet2(nn.Module):
    """
    Single shared DA block stacked `n_iter` times.

    Components per north_star.md / project_calibnet2_design.md:
      * ConvNeXt backbone (coarse + fine image features).
      * PointMLP3 (uvd + intensity).
      * FrustumLocalEncoder.forward_dense() → dense LiDAR map (gh*gw, D).
      * RoPEPoseEmb on Q at entry (type-0 intrinsic add, type-1 block-diag(R)).
      * Block (DA cross-attn over 3 levels) applied n_iter times with SHARED
        weights.  KV is built once and reused across all iterations.
      * Final output: final_head(Δ_cum) where Δ_cum = Σ delta_i (the abs-PE
        leak gate — final head NEVER reads q directly).

    forward(image, distorted_uvd, *, dpose_R=None, vfp=None,
            bucket_uvd, bucket_valid, key_padding_mask=None)
        dpose_R = None → I → degenerates to legacy single-frame calib.
        KV is ALWAYS own-frame (image/lidar of the input image).

    forward_pair(image_A, distorted_uvd_A, image_B, distorted_uvd_B, R_AB,
                 vfp_A, vfp_B, bucket_uvd_A/B, bucket_valid_A/B, ...)
        Calls forward(A, R=I) and forward(A, R=R_AB) — KV is A's own
        image/lidar in both cases.  R_AB rotates A's Q so the network
        is asked "where does A's points go in B's frame after R_AB?".
    """
    def __init__(self, d: int = D_DIM, img_size: int = 128, in_channels: int = 3,
                 use_intensity: bool = True,
                 frustum_grid_n: int = 16,
                 r_uv_cells: float = 1.5, r_d: float = 0.004, k_nb: int = 16,
                 n_iter: int = 3, n_heads: int = 4, n_points: int = 4,
                 d_scalar: int = 8, n_type1: int = 40,
                 convnext_n_blocks: int = 2,
                 use_info_head: bool = True,
                 kv_schedule: list[dict] | None = None,
                 fourier_head_n_freq: int = 0,
                 fourier_head_scale: float = 10.0,
                 point_mlp_fourier_n_freq: int = 0):
        super().__init__()
        self.img_size  = img_size
        self.cnn = ConvNeXtBackbone(d, in_channels=in_channels,
                                     n_blocks=convnext_n_blocks)
        self.use_intensity = bool(use_intensity)
        self.point_mlp = PointMLP3(d,
                                    in_channels=4 if self.use_intensity else 3,
                                    fourier_n_freq=int(point_mlp_fourier_n_freq))
        self.frustum_enc = FrustumLocalEncoder(
            d, r_uv_cells=r_uv_cells, r_d=r_d, k=k_nb, grid_n=frustum_grid_n)
        self.pose_emb = RoPEPoseEmb(d, d_scalar=d_scalar, n_type1=n_type1)
        # --- block stack ---------------------------------------------------
        # Two regimes:
        #  * kv_schedule=None (legacy / kick #1, #2): ONE shared block, KV
        #    is a fixed 3-level pyramid [coarse, fine, lidar], n_levels=3.
        #  * kv_schedule given (kick #3): per-layer blocks (own weights),
        #    each with its own (image_scale, lidar?, n_points) config.
        #    A/B共有 is enforced at the call site by replaying the same
        #    ModuleList on both KV_A and KV_B. Example:
        #      [
        #       {"image": "coarse",     "lidar": True, "n_points": 4},
        #       {"image": "coarse",     "lidar": True, "n_points": 4},
        #       {"image": "fine",       "lidar": True, "n_points": 4},
        #       {"image": "super_fine", "lidar": True, "n_points": 8},
        #      ]
        #    "super_fine" = stem output (img/4 = 32x32 for img=128).
        self.kv_schedule = kv_schedule
        if self.kv_schedule is None:
            self.n_iter = n_iter
            self.block = Block(d=d, n_heads=n_heads, n_levels=3, n_points=n_points)
            self.level_embed = nn.Parameter(torch.zeros(3, d))
            nn.init.normal_(self.level_embed, std=0.02)
        else:
            # Validate schedule.
            for li, cfg in enumerate(self.kv_schedule):
                assert cfg.get("image") in ("coarse", "fine", "super_fine", None), \
                    f"kv_schedule[{li}].image must be one of coarse/fine/super_fine/None"
                n_lvl = (1 if cfg.get("image") is not None else 0) + \
                        (1 if cfg.get("lidar", True) else 0)
                assert n_lvl >= 1, f"kv_schedule[{li}] has no levels"
                assert int(cfg.get("n_points", 4)) >= 1
            self.n_iter = len(self.kv_schedule)
            self.blocks = nn.ModuleList()
            for li, cfg in enumerate(self.kv_schedule):
                n_lvl = (1 if cfg.get("image") is not None else 0) + \
                        (1 if cfg.get("lidar", True) else 0)
                np_l = int(cfg.get("n_points", n_points))
                self.blocks.append(
                    Block(d=d, n_heads=n_heads, n_levels=n_lvl, n_points=np_l))
            # Per-layer level_embed sized to that layer's n_levels.
            self.level_embeds = nn.ParameterList()
            for cfg in self.kv_schedule:
                n_lvl = (1 if cfg.get("image") is not None else 0) + \
                        (1 if cfg.get("lidar", True) else 0)
                pe = nn.Parameter(torch.zeros(n_lvl, d))
                nn.init.normal_(pe, std=0.02)
                self.level_embeds.append(pe)
        # Final readout — Δ_cum → (duv_x, duv_y, log_sx, log_sy, rho).
        # Zero-init weights so initial output is bias only; bias log_s = log(2)
        # matches legacy clamp_params expectation.
        # Optionally prepend a Fourier-feature lift to delta_cum before the
        # head: γ(δ) = [sin(B δ), cos(B δ)] with B ~ N(0, σ²) frozen at init
        # (NeRF / Tancik 2020). Lifts MLP NTK out of its low-frequency regime
        # so the head can express sub-pixel-grain corrections that a plain
        # Linear(d, 5) tends to smooth out. Zero cost at inference (one matmul
        # + sin/cos), and γ(0) = 0 so the lift is a no-op at init — safe to
        # warm-start ckpts that didn't have it.
        self.fourier_head_n_freq = int(fourier_head_n_freq)
        if self.fourier_head_n_freq > 0:
            B = torch.randn(self.fourier_head_n_freq, d) * float(fourier_head_scale)
            self.register_buffer('fourier_B', B, persistent=True)
            head_in = d + 2 * self.fourier_head_n_freq
        else:
            head_in = d
        self.final_head = nn.Linear(head_in, 5)
        nn.init.zeros_(self.final_head.weight)
        nn.init.zeros_(self.final_head.bias)
        with torch.no_grad():
            self.final_head.bias[2] = math.log(2.0)
            self.final_head.bias[3] = math.log(2.0)

        if use_info_head:
            from models.model_depth import InfoHead2x2
            self.info_head = InfoHead2x2(d)
        else:
            self.info_head = None

    # ------------------------------------------------------------------
    def _build_Q(self, distorted_uvd: torch.Tensor,
                  bucket_uvd: torch.Tensor,
                  bucket_valid: torch.Tensor,
                  key_padding_mask=None) -> torch.Tensor:
        """Per-point token Q for one frame.

        PointMLP3([u/img_size, v/img_size, d, intensity?]) → (B, N, D), then
        add per-query frustum-local context. Used by BOTH calib path and
        cross-frame path (the cross-frame variant simply runs this once on
        the A side; B has no Q).
        """
        uv_01 = distorted_uvd[..., :2] / self.img_size
        d3    = distorted_uvd[..., 2:3]
        if self.use_intensity:
            q_in = torch.cat([uv_01, d3, distorted_uvd[..., 3:4]], dim=-1)
        else:
            q_in = torch.cat([uv_01, d3], dim=-1)
        q = self.point_mlp(q_in)
        # frustum_enc.forward subtracts query_uvd from cands (cands - query_uvd).
        # Bucket is 4-col [u,v,d,intensity]; pass query at the same width so the
        # subtract produces (Δu, Δv, Δd, Δintensity).
        q = q + self.frustum_enc(
            query_uvd=distorted_uvd[..., :bucket_uvd.shape[-1]],
            bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
            query_token=q, query_pad_mask=key_padding_mask,
            img_size=self.img_size)
        return q

    def _run_blocks(self, q: torch.Tensor,
                     kv_flat: torch.Tensor,
                     spatial_shapes: torch.Tensor,
                     level_start_index: torch.Tensor,
                     n_iters: int,
                     delta_cum: torch.Tensor = None,
                     key_padding_mask=None):
        """Apply the shared block n_iters times against a single KV bundle.
        (Legacy path; only called when kv_schedule is None.)

        Returns (q_out, delta_cum) where delta_cum is the running residual
        sum used by the final head. Caller passes in delta_cum if continuing
        from a previous half (cross-frame, after PoseEmb).
        """
        if delta_cum is None:
            delta_cum = torch.zeros_like(q)
        for _ in range(n_iters):
            q, delta_i = self.block(q, kv_flat, spatial_shapes,
                                     level_start_index,
                                     key_padding_mask=key_padding_mask)
            delta_cum = delta_cum + delta_i
        return q, delta_cum

    def _run_blocks_per_layer(self, q: torch.Tensor,
                                feats: dict,
                                lidar_2d: torch.Tensor,
                                layer_indices: list[int],
                                delta_cum: torch.Tensor = None,
                                key_padding_mask=None):
        """Per-layer block stack with kv_schedule.

        feats : dict {"coarse": (B,D,Hc,Wc), "fine": (B,D,Hf,Wf),
                       "super_fine": (B,D,Hs,Ws)}  -- some entries may be None
                       if no layer in `layer_indices` needs them.
        lidar_2d : (B, D, gn, gn)  -- always built; layers can opt out via
                                      kv_schedule[li].lidar = False.
        layer_indices : which schedule entries to run, in order.
        """
        device = q.device
        B = q.size(0)
        if delta_cum is None:
            delta_cum = torch.zeros_like(q)
        for li in layer_indices:
            cfg = self.kv_schedule[li]
            kvs, shapes, cum = [], [], [0]
            lvl = 0
            img_kind = cfg.get("image")
            if img_kind is not None:
                feat = feats[img_kind]
                _, _, H, W = feat.shape
                kv_l = feat.flatten(2).permute(0, 2, 1) \
                       + self.level_embeds[li][lvl][None, None, :]
                kvs.append(kv_l); shapes.append([H, W])
                cum.append(cum[-1] + H * W)
                lvl += 1
            if cfg.get("lidar", True):
                _, _, H, W = lidar_2d.shape
                kv_l = lidar_2d.flatten(2).permute(0, 2, 1) \
                       + self.level_embeds[li][lvl][None, None, :]
                kvs.append(kv_l); shapes.append([H, W])
                cum.append(cum[-1] + H * W)
                lvl += 1
            kv_flat = torch.cat(kvs, dim=1)
            spatial_shapes    = torch.as_tensor(shapes,   dtype=torch.long, device=device)
            level_start_index = torch.as_tensor(cum[:-1], dtype=torch.long, device=device)
            q, delta_i = self.blocks[li](q, kv_flat, spatial_shapes,
                                          level_start_index,
                                          key_padding_mask=key_padding_mask)
            delta_cum = delta_cum + delta_i
        return q, delta_cum

    def _readout(self, delta_cum: torch.Tensor):
        """Δ_cum → (per_pt[..., :5], W?). Single final head; abs-PE leak gate
        comes from feeding Δ_cum, not q itself.
        """
        from models.model_cov import clamp_params
        if self.fourier_head_n_freq > 0:
            # γ(δ) = [sin(B δ), cos(B δ)], B is registered buffer.
            proj = delta_cum @ self.fourier_B.t()       # (B, N, n_freq)
            gamma = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            head_in = torch.cat([delta_cum, gamma], dim=-1)
        else:
            head_in = delta_cum
        raw = self.final_head(head_in)
        per_pt = clamp_params(raw, self.img_size)
        if self.info_head is not None:
            # info_head still consumes raw delta_cum; it's a separate predictor
            # for the 2x2 information matrix and benefits from the smoother
            # representation (it does not have to chase sub-pixel residuals).
            W = self.info_head(delta_cum)
            return per_pt, W
        return per_pt

    def forward_frame(self,
                       image: torch.Tensor,
                       distorted_uvd: torch.Tensor,
                       bucket_uvd: torch.Tensor,
                       bucket_valid: torch.Tensor,
                       n_iters: int,
                       *,
                       q: torch.Tensor = None,
                       delta_cum: torch.Tensor = None,
                       key_padding_mask=None,
                       layer_indices: list[int] | None = None):
        """One frame of work — KV (own image+lidar), Q (per-point), block stack.

        Used by both calib and cross-frame paths:
          * calib: call ONCE with n_iters=self.n_iter, q=None.
          * cross-frame A side: call with n_iters=n_half, q=None.
          * cross-frame B side: call with q=<post-PoseEmb token>,
                                 delta_cum=<from A side>, n_iters=n_iter-n_half.
        If `q` is None it is rebuilt from distorted_uvd; if given (cross-frame
        B side), distorted_uvd may be None and we just run blocks against KV_B.
        delta_cum chains across A→B so the final readout sees the full A→B
        residual, not just the B-side increment.

        kv_schedule mode (kick #3+): pass `layer_indices=[li, li+1, ...]`
        instead of `n_iters`; the per-layer KV pyramid + per-layer Block
        weights are dispatched accordingly. n_iters is ignored.
        """
        if q is None:
            assert distorted_uvd is not None, \
                "forward_frame: distorted_uvd required when q is not provided"
            q = self._build_Q(distorted_uvd, bucket_uvd, bucket_valid,
                               key_padding_mask=key_padding_mask)
        if self.kv_schedule is not None:
            assert layer_indices is not None, \
                "kv_schedule mode: forward_frame requires layer_indices"
            feats = self._build_feats(image)
            lidar_2d = self._build_lidar_2d(bucket_uvd, bucket_valid)
            return self._run_blocks_per_layer(
                q, feats, lidar_2d, layer_indices=layer_indices,
                delta_cum=delta_cum, key_padding_mask=key_padding_mask)
        kv_flat, sh, lsi = self._build_kv(image, bucket_uvd, bucket_valid)
        return self._run_blocks(q, kv_flat, sh, lsi, n_iters=n_iters,
                                 delta_cum=delta_cum,
                                 key_padding_mask=key_padding_mask)

    # ------------------------------------------------------------------
    def _build_feats(self, image: torch.Tensor) -> dict:
        """Backbone forward → {coarse, fine, super_fine}. Only the scales
        that the kv_schedule actually uses are computed (super_fine is the
        cheap stem projection — always available)."""
        coarse, fine, super_fine = self.cnn(image, return_super_fine=True)
        return {"coarse": coarse, "fine": fine, "super_fine": super_fine}

    def _build_lidar_2d(self, bucket_uvd, bucket_valid):
        lidar_dense = self.frustum_enc.forward_dense(
            bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
            img_size=self.img_size)
        B = lidar_dense.size(0)
        gn = int(round(lidar_dense.size(1) ** 0.5))
        return lidar_dense.permute(0, 2, 1).reshape(B, -1, gn, gn).contiguous()

    # ------------------------------------------------------------------
    def _build_kv(self, image: torch.Tensor,
                   bucket_uvd: torch.Tensor,
                   bucket_valid: torch.Tensor):
        """Build the 3-level KV bundle once: (kv_flat, spatial_shapes,
        level_start_index).  kv_flat already has level_embed added.
        """
        coarse_feat, fine_feat = self.cnn(image)            # (B, D, Hc, Wc), (B, D, Hf, Wf)
        # dense lidar map → reshape to (B, D, gn, gn) — same level layout as image.
        lidar_dense = self.frustum_enc.forward_dense(
            bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
            img_size=self.img_size)                                  # (B, gn², D)
        B = lidar_dense.size(0)
        gn = int(round(lidar_dense.size(1) ** 0.5))
        assert gn * gn == lidar_dense.size(1), \
            f"lidar_dense not square: {lidar_dense.size(1)}"
        lidar_2d = lidar_dense.permute(0, 2, 1).reshape(B, -1, gn, gn).contiguous()

        feats = [coarse_feat, fine_feat, lidar_2d]
        device = image.device
        kvs, shapes, cum = [], [], [0]
        for lid, feat in enumerate(feats):
            _, _, H, W = feat.shape
            kv_l = feat.flatten(2).permute(0, 2, 1)              # (B, H*W, D)
            kv_l = kv_l + self.level_embed[lid][None, None, :]
            kvs.append(kv_l); shapes.append([H, W])
            cum.append(cum[-1] + H * W)
        kv_flat = torch.cat(kvs, dim=1)                          # (B, Σ HW, D)
        spatial_shapes    = torch.as_tensor(shapes,    dtype=torch.long, device=device)
        level_start_index = torch.as_tensor(cum[:-1],  dtype=torch.long, device=device)
        return kv_flat, spatial_shapes, level_start_index

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, distorted_uvd: torch.Tensor, *,
                dpose_R: torch.Tensor = None, vfp: torch.Tensor = None,
                bucket_uvd: torch.Tensor = None,
                bucket_valid: torch.Tensor = None,
                key_padding_mask=None,
                mode: str = 'calib',
                image_B: torch.Tensor = None,
                R_AB: torch.Tensor = None,
                t_AB: torch.Tensor = None,
                vfp_B: torch.Tensor = None,
                bucket_uvd_B: torch.Tensor = None,
                bucket_valid_B: torch.Tensor = None):
        """Dispatch entry. mode='calib' (default) → legacy single-frame path
        (image, distorted_uvd, dpose_R, vfp, bucket_uvd, bucket_valid).
        mode='cross' → cross-frame path with KV switching A→B mid-stack;
        in that case image is image_A, distorted_uvd is uvd_A, and the
        B-side tensors come in via image_B / bucket_uvd_B / bucket_valid_B
        / vfp_B / R_AB / t_AB. Routing through .forward (not a separate
        method) so DDP's grad-sync hook fires correctly.
        """
        if mode == 'cross':
            return self.forward_cross_frame(
                image, distorted_uvd, image_B, R_AB,
                t_AB=t_AB,
                vfp_A=vfp, vfp_B=vfp_B,
                bucket_uvd_A=bucket_uvd, bucket_valid_A=bucket_valid,
                bucket_uvd_B=bucket_uvd_B, bucket_valid_B=bucket_valid_B,
                key_padding_mask_A=key_padding_mask,
            )
        # legacy calib path
        """
        image           : (B, C, H, W)
        distorted_uvd   : (B, N, 3 or 4)  [U, V, D_norm, (intensity)]
        dpose_R         : (B, 3, 3) or None — None ↔ identity (legacy calib).
        vfp             : (B,) or None — virtual focal length in px (own frame).
        bucket_uvd      : (B, gn², K_per_cell, 3 or 4)  cell-bucketed lidar.
        bucket_valid    : (B, gn², K_per_cell)
        key_padding_mask: (B, N) bool — True = pad.

        Returns:
            per_pt: (B, N, 5)  cumulative (duv_x, duv_y, log_sx, log_sy, rho)
            W:     (B, N, 2, 2)  — only when use_info_head=True
        """
        # Calib = single forward_frame call. No PoseEmb: there's no ego motion
        # to inject (R=I implicit) and frame-token learns lidar↔image alignment
        # without a hint.
        if self.kv_schedule is not None:
            # In kv_schedule mode the entire schedule runs on the single frame.
            q, delta_cum = self.forward_frame(
                image, distorted_uvd, bucket_uvd, bucket_valid,
                n_iters=0, key_padding_mask=key_padding_mask,
                layer_indices=list(range(self.n_iter)))
        else:
            q, delta_cum = self.forward_frame(
                image, distorted_uvd, bucket_uvd, bucket_valid,
                n_iters=self.n_iter, key_padding_mask=key_padding_mask)
        return self._readout(delta_cum)

    # ------------------------------------------------------------------
    def forward_cross_frame(self,
                              image_A: torch.Tensor,
                              distorted_uvd_A: torch.Tensor,
                              image_B: torch.Tensor,
                              R_AB: torch.Tensor,
                              t_AB: torch.Tensor = None,
                              vfp_A: torch.Tensor = None,
                              vfp_B: torch.Tensor = None,
                              bucket_uvd_A=None, bucket_valid_A=None,
                              bucket_uvd_B=None, bucket_valid_B=None,
                              key_padding_mask_A=None):
        """Cross-frame forward: KV switches A→B mid-stack, RoPE(R_AB) once.

        Q is the LiDAR set of frame A (we want to predict where each A point
        lands in frame B image). The shared block runs n_iter//2 iterations
        on KV_A (= A's own image + dense lidar), then RoPE(R_AB) is applied
        to Q once, then the same block runs the remaining iterations on KV_B
        (= B's own image + dense lidar). All blocks share weights, identical
        to the legacy single-frame calib path.

        Args mirror `forward`, plus `image_B` / `vfp_B` / `bucket_uvd_B` /
        `bucket_valid_B` for the B side. B's LiDAR enters only via the dense-
        lidar level inside KV_B; there is no B-side Q.

        Returns:
            per_pt: (B, N_A, 5)  cumulative Δ_AtoB readout
            W:     (B, N_A, 2, 2)  — only when use_info_head=True
        """
        # Cross-frame = forward_frame(A) → SE(3) PoseEmb on Q → forward_frame(B,
        # q=carry, delta_cum=carry). Two forward_frame calls share the same
        # block stack; only the KV bundle and the iteration count differ.
        # q_frametoken is the per-A-point frame token: it carries the
        # accumulated Δuv prediction (via delta_cum) AND the contextualised
        # representation built by A-block cross-attn. After A blocks finish,
        # pose_emb rotates the token by R_AB → q_frametoken_pose_AB, which is
        # then fed to B blocks as Q. Same token, different observation frame.
        n_half = self.n_iter // 2
        if self.kv_schedule is not None:
            a_layers = list(range(0, n_half))
            b_layers = list(range(n_half, self.n_iter))
            q_frametoken, delta_cum = self.forward_frame(
                image_A, distorted_uvd_A, bucket_uvd_A, bucket_valid_A,
                n_iters=0, key_padding_mask=key_padding_mask_A,
                layer_indices=a_layers)
        else:
            q_frametoken, delta_cum = self.forward_frame(
                image_A, distorted_uvd_A, bucket_uvd_A, bucket_valid_A,
                n_iters=n_half, key_padding_mask=key_padding_mask_A)
        # Mid-stack readout for A-side calib supervision: same head, just
        # called on Δ_cum at the boundary between A blocks and pose_emb.
        # When the dataset puts ε_calib on the A inputs, the trainer can
        # then ask "did A blocks produce the right Δuv on the A image?".
        pred_A = self._readout(delta_cum)
        # SE(3) PoseEmb on q_frametoken at the stack midpoint:
        #   type-1 chunk (40 vec×3): block-diag(R_AB) action
        #   type-0 chunk (8 scalar): translation_mlp(t_AB) added (zero-init →
        #                            no-op when ckpt predates SE(3) extension)
        q_frametoken_pose_AB = self.pose_emb(
            q_frametoken, dpose_R=R_AB, dpose_t=t_AB, log_vfp=None)
        if self.kv_schedule is not None:
            q_frametoken, delta_cum = self.forward_frame(
                image_B, None, bucket_uvd_B, bucket_valid_B,
                n_iters=0,
                q=q_frametoken_pose_AB, delta_cum=delta_cum,
                key_padding_mask=key_padding_mask_A,
                layer_indices=b_layers)
        else:
            q_frametoken, delta_cum = self.forward_frame(
                image_B, None, bucket_uvd_B, bucket_valid_B,
                n_iters=self.n_iter - n_half,
                q=q_frametoken_pose_AB, delta_cum=delta_cum,
                key_padding_mask=key_padding_mask_A)
        pred_B = self._readout(delta_cum)
        # Cross-frame returns (pred_A, pred_B) so the trainer can apply two
        # losses (calib + pose) on the same forward pass. pred_A may be
        # ignored when ε_calib is not enabled on the A side (legacy behaviour).
        # Both follow the same _readout shape conventions (per_pt or
        # (per_pt, W) depending on use_info_head).
        return pred_A, pred_B
