"""CalibNetMultiFrame — N-frame cross-attention with per-frame pose bias.

Generalises the 2-frame pair-net to a chain (A, M, B). For the A→B head:

  Q  : pt tokens from frame A           (no pose embedding added)
  KV : concat(pt_M, img_M, pt_B, img_B) (each KV token tagged with origin frame)

  attention scores[q, k] += b_{A→frame(k)}      ← per-frame scalar bias
                                                  from PoseMLP(pose_{A→frame(k)})

  softmax over KV (M tokens + B tokens jointly).

This lets the model treat M as an auxiliary "second look" while keeping all
spatial tokens in their own frame's local uv (no absolute 3D leak — only the
*relative* pose enters as an attention bias). For wide A↔B baselines, the
mid-frame M provides shorter-baseline context; the model learns when to trust
each frame via the bias-modulated softmax.

PoC scope (v51):
  • Plain cross-attn (no deformable). Frustum local-encoder still feeds Q.
  • Single direction A→B (B→A trained symmetrically by swapping inputs).
  • Pose bias is per-head scalar (n_heads independent learnable scalars per pose).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import D as D_DIM
from models.model_cov import clamp_params, clamp_params_uvd
from models.cross_frame import (
    PoseMLP, SelfAttnFusionBlock,
)
from models.cross_frame import CalibNetCrossFrame  # to reuse _encode_frame


class PoseBiasMLP(nn.Module):
    """Encodes 6-DoF relative pose → (n_heads,) scalar attention bias."""

    def __init__(self, n_heads: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(),
            nn.Linear(64, n_heads),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, ypr_t: torch.Tensor) -> torch.Tensor:
        ypr = ypr_t[..., :3] / 5.0
        t   = ypr_t[..., 3:] / 2.0
        return self.net(torch.cat([ypr, t], dim=-1))   # (B, n_heads)


class MultiFrameCrossBlock(nn.Module):
    """Cross-attn from Q (frame_A pt tokens) into a multi-frame KV concat.

    Each KV "frame slice" carries an additive per-(q, k) bias derived from the
    relative pose A→frame_of_k. Implemented as manual scaled-dot-product so we
    can inject the bias before softmax.
    """

    def __init__(self, d: int = D_DIM, n_heads: int = 4, out_dim: int = 5):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_d  = d // n_heads
        assert d % n_heads == 0, f'd={d} not divisible by n_heads={n_heads}'

        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.q_proj  = nn.Linear(d, d)
        self.k_proj  = nn.Linear(d, d)
        self.v_proj  = nn.Linear(d, d)
        self.o_proj  = nn.Linear(d, d)

        self.norm_sa  = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.drop = nn.Dropout(0.1)

        self.proj = nn.Linear(d + 2, out_dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            if out_dim == 5:
                self.proj.bias[2] = 0.69
                self.proj.bias[3] = 0.69
            elif out_dim == 7:
                self.proj.bias[3] = 0.69
                self.proj.bias[4] = 0.69
                self.proj.bias[5] = 0.0

    def forward(self, q, kv_segments, uv_hat_01,
                q_pad_mask=None, kv_seg_pad_masks=None, kv_seg_biases=None):
        """
        q              : (B, N_q, D)   — query tokens (frame A pt tokens)
        kv_segments    : list of (B, N_k_i, D) tensors per KV frame
        kv_seg_pad_masks: list of (B, N_k_i) bool, True = pad (or None)
        kv_seg_biases   : list of (B, n_heads) per-frame attention bias scalars
        uv_hat_01      : (B, N_q, 2) — A→B hypothesis projection (for output head)
        Returns (q_updated, raw)
        """
        B, N_q, D = q.shape
        H = self.n_heads
        Dh = self.head_d
        kv = torch.cat(kv_segments, dim=1)       # (B, N_kv, D)
        N_kv = kv.shape[1]

        Qn = self.norm_q(q)
        Kn = self.norm_kv(kv)
        Qp = self.q_proj(Qn).view(B, N_q,  H, Dh).transpose(1, 2)   # (B, H, N_q, Dh)
        Kp = self.k_proj(Kn).view(B, N_kv, H, Dh).transpose(1, 2)
        Vp = self.v_proj(Kn).view(B, N_kv, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        scores = torch.matmul(Qp, Kp.transpose(-1, -2)) * scale     # (B, H, N_q, N_kv)

        # per-(q, k) bias: bias only depends on which KV frame k belongs to,
        # so we add it as (B, H, 1, N_kv) broadcast across queries.
        if kv_seg_biases is not None:
            bias_kv = torch.zeros(B, H, N_kv, device=q.device, dtype=scores.dtype)
            off = 0
            for seg, b in zip(kv_segments, kv_seg_biases):
                L = seg.shape[1]
                bias_kv[:, :, off:off + L] = b.unsqueeze(-1)        # (B,H,1)→(B,H,L)
                off += L
            scores = scores + bias_kv.unsqueeze(2)                  # broadcast queries

        if kv_seg_pad_masks is not None:
            kv_pad = torch.zeros(B, N_kv, dtype=torch.bool, device=q.device)
            off = 0
            for seg, m in zip(kv_segments, kv_seg_pad_masks):
                L = seg.shape[1]
                if m is not None:
                    kv_pad[:, off:off + L] = m
                off += L
            scores = scores.masked_fill(kv_pad.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        out  = torch.matmul(attn, Vp)                                # (B,H,N_q,Dh)
        out  = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out  = self.o_proj(out)

        q = q + self.drop(out)

        sa, _ = self.self_attn(self.norm_sa(q), self.norm_sa(q), self.norm_sa(q),
                                key_padding_mask=q_pad_mask)
        q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_hat_01], dim=-1))
        return q, raw


class MultiFrameCrossBlockDeform(nn.Module):
    """Cross-attn from Q (frame_A pt tokens) into N KV frames using a
    multi-level deformable attention on the IMAGE side, plus a plain
    cross-attn on the POINT side. Per-frame RPE bias is added to:
      - the deformable attention WEIGHTS (pre-softmax) per level
      - the plain attention SCORES (pre-softmax) per KV frame slice
    """

    def __init__(self, d: int = D_DIM, n_heads: int = 4, n_points: int = 4,
                 max_levels: int = 4, out_dim: int = 5):
        super().__init__()
        from models.model_deform import MSDeformAttn
        self.d = d
        self.n_heads = n_heads
        self.n_points = n_points
        self.max_levels = max_levels
        # Build a single MSDeformAttn supporting up to max_levels (we reuse it
        # for any actual N_kv ≤ max_levels by passing the right shapes).
        # The sampling_offsets weights are sized for max_levels.
        self.deform_img = MSDeformAttn(d_model=d, n_levels=max_levels,
                                        n_heads=n_heads, n_points=n_points)

        # Plain cross-attn for pt KV side (manual to inject per-frame bias)
        self.norm_q_img  = nn.LayerNorm(d)
        self.norm_kv_img = nn.LayerNorm(d)
        self.norm_q_pt   = nn.LayerNorm(d)
        self.norm_kv_pt  = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)

        self.norm_sa  = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.drop = nn.Dropout(0.1)

        self.proj = nn.Linear(d + 2, out_dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            if out_dim == 5:
                self.proj.bias[2] = 0.69
                self.proj.bias[3] = 0.69
            elif out_dim == 7:
                self.proj.bias[3] = 0.69
                self.proj.bias[4] = 0.69
                self.proj.bias[5] = 0.0

    def forward(self, q, img_2d_list, ref_uv_01_list,
                pt_kv_list, pt_pad_list,
                kv_biases, uv_hat_01,
                q_pad_mask=None,
                target_shift=None, kv_pose_shifts=None):
        """
        q                : (B, N_q, D) — query frame's tokens, ALREADY shifted
                            by `target_shift`. Used as the residual stream.
        img_2d_list      : list of (B, D, H, W) — one per KV frame
        ref_uv_01_list   : list of (B, N_q, 2) — reference uv per query per frame
        pt_kv_list       : list of (B, N_pt_i, D) — pt KV per frame
        pt_pad_list      : list of (B, N_pt_i) bool — pad masks per frame
        kv_biases        : list of (B, n_heads) per-frame bias (same for img & pt of that frame)
        uv_hat_01        : (B, N_q, 2) — A→target hypothesis (for output head)
        target_shift     : (B, 1, D) — the pose embedding already baked into
                            `q`. Used to "rebase" q for per-KV-frame Q
                            (option-b cross-attn, see docs/multi_frame_attention.md).
        kv_pose_shifts   : list of (B, 1, D) — pose embedding for the
                            (Q-frame, KV-frame_k) pair. If provided, the
                            Q used in cross-attn for each KV-frame_k becomes
                            `(q - target_shift) + kv_pose_shifts[k]`, so each
                            (Q_k, KV_k) sub-attention matches the 2-frame
                            baseline. If None / target_shift=None: legacy
                            behaviour (single Q for all KV).
        """
        B, N_q, D = q.shape
        L = len(img_2d_list)
        assert L <= self.max_levels, f'L={L} > max_levels={self.max_levels}'
        H, W = img_2d_list[0].shape[2], img_2d_list[0].shape[3]
        device = q.device

        # ── deformable image side: legacy concat single call ────────────────
        # Multi-level integration via softmax over (level × points) inside
        # MSDeformAttn. Frame discrimination on img-side is handled by the
        # per-level reference uv (where to sample) — frame-conditional Q on
        # img-side previously hurt because per-frame deformable splits then
        # averages, breaking the natural softmax mixing across levels.
        flats = [feat.flatten(2).permute(0, 2, 1) for feat in img_2d_list]
        kv_img_flat = torch.cat(flats, dim=1)
        spatial_shapes = torch.as_tensor([[H, W]] * L, dtype=torch.long, device=device)
        level_start_index = torch.as_tensor(
            [0] + [H * W * (i + 1) for i in range(L - 1)],
            dtype=torch.long, device=device,
        )
        ref = torch.stack(ref_uv_01_list, dim=2).clamp(0.0, 1.0)
        if L < self.max_levels:
            pad_L = self.max_levels - L
            ref = torch.cat([ref, ref[:, :, -1:].expand(-1, -1, pad_L, -1)], dim=2)
            spatial_shapes = torch.cat([spatial_shapes, spatial_shapes[-1:].expand(pad_L, -1)], dim=0)
            level_start_index = torch.cat([level_start_index, level_start_index[-1:].expand(pad_L)], dim=0)
        ca_img_full = self.deform_img(self.norm_q_img(q), ref,
                                       self.norm_kv_img(kv_img_flat),
                                       spatial_shapes, level_start_index,
                                       input_padding_mask=None)
        # NOTE: MSDeformAttn already softmaxes weights across (level × points)
        # and aggregates internally — so per-level RPE bias is delegated to
        # the pt-side plain cross-attn (where it can be cleanly injected).

        # ── plain cross-attn on pt KV side, with per-frame bias ──────────────
        # Per-KV-frame Q (option b in docs/multi_frame_attention.md): the same
        # pt tokens are re-projected with a frame-conditional pose shift for
        # every KV frame, so each (Q_k, KV_k) sub-attention matches the
        # 2-frame baseline pattern. When kv_pose_shifts is None, falls back
        # to the legacy single-Q-for-all-KV behaviour.
        kv_pt = torch.cat(pt_kv_list, dim=1)                  # (B, Σ N_pt_i, D)
        Hh = self.n_heads; Dh = D // Hh
        scale = Dh ** -0.5
        # In the per-frame-Q path the convention is: kv_pose_shifts[0] is the
        # *target* pose shift (already baked into `q` by the caller). Other
        # entries are the (anchor → KV-frame_k) shifts; the residual we add
        # to q is (kv_pose_shifts[k] - kv_pose_shifts[0]).
        if kv_pose_shifts is None or target_shift is None:
            # legacy: single Q (= q already includes the target pose shift)
            Qn = self.norm_q_pt(q)
            Qp = self.q_proj(Qn).view(B, N_q, Hh, Dh).transpose(1, 2)
            Kn = self.norm_kv_pt(kv_pt)
            Kp = self.k_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
            Vp = self.v_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
            scores = torch.matmul(Qp, Kp.transpose(-1, -2)) * scale
        else:
            # per-frame Q (option b): rebase q from `target_shift` to each
            # KV-frame_k's shift, recompute Q for each segment.
            assert len(kv_pose_shifts) == len(pt_kv_list), (
                f'kv_pose_shifts len {len(kv_pose_shifts)} != pt_kv_list '
                f'len {len(pt_kv_list)}')
            # K, V projections (one shot over concat KV — no Q dependence)
            Kn = self.norm_kv_pt(kv_pt)
            Kp = self.k_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
            Vp = self.v_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
            score_blocks = []
            off = 0
            for seg, ps in zip(pt_kv_list, kv_pose_shifts):
                Lpt = seg.shape[1]
                q_k = q - target_shift + ps              # rebase to frame_k
                Qn_k = self.norm_q_pt(q_k)
                Qp_k = self.q_proj(Qn_k).view(B, N_q, Hh, Dh).transpose(1, 2)
                Kp_k = Kp[:, :, off:off + Lpt]
                scores_k = torch.matmul(Qp_k, Kp_k.transpose(-1, -2)) * scale
                score_blocks.append(scores_k)
                off += Lpt
            scores = torch.cat(score_blocks, dim=-1)

        # per-frame bias on attention scores (segment-wise constant).
        # Skipped when per-KV-frame Q is enabled — the D-dim Q-rebase already
        # provides full frame discrimination, so the 4-scalar bias becomes
        # redundant (and competing) signal.
        if kv_pose_shifts is None or target_shift is None:
            N_kv = kv_pt.shape[1]
            bias_kv = torch.zeros(B, Hh, N_kv, device=device, dtype=scores.dtype)
            off = 0
            for seg, b in zip(pt_kv_list, kv_biases):
                Lpt = seg.shape[1]
                bias_kv[:, :, off:off + Lpt] = b.unsqueeze(-1)
                off += Lpt
            scores = scores + bias_kv.unsqueeze(2)
        N_kv = kv_pt.shape[1]

        # padding mask
        pt_pad = torch.zeros(B, N_kv, dtype=torch.bool, device=device)
        off = 0
        for seg, m in zip(pt_kv_list, pt_pad_list):
            Lpt = seg.shape[1]
            if m is not None:
                pt_pad[:, off:off + Lpt] = m
            off += Lpt
        scores = scores.masked_fill(pt_pad.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        ca_pt = torch.matmul(attn, Vp).transpose(1, 2).contiguous().view(B, N_q, D)
        ca_pt = self.o_proj(ca_pt)

        # combine + tail
        q = q + self.drop(ca_img_full) + self.drop(ca_pt)
        sa, _ = self.self_attn(self.norm_sa(q), self.norm_sa(q), self.norm_sa(q),
                                key_padding_mask=q_pad_mask)
        q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_hat_01], dim=-1))
        return q, raw


class CalibNetMultiFrame(CalibNetCrossFrame):
    """Unified pair / multi-frame model.

    KV is built from one or more frames (B alone = pair, M ‖ B = multi-frame).
    Cross-attn is deformable on the image side (one MSDeformAttn level per KV
    frame) and plain on the point side (per-frame RPE bias on attention
    scores). Q-shift via parent's pose_mlp gives explicit pose channel
    independent of the bias.
    """

    def __init__(self, *args, n_heads: int = 4, max_kv_frames: int = 2,
                 deform_n_points: int = 4, **kwargs):
        # we always build deformable cross_blocks ourselves; ignore parent's
        # deform_mode setting for the cross_blocks (parent still uses it for
        # other internals, but cross_blocks is overridden below).
        kwargs.setdefault('deform_mode', 'sl')
        super().__init__(*args, **kwargs)
        d = self.d
        out_dim = self._out_dim
        self.pose_bias = PoseBiasMLP(n_heads=n_heads)
        self.max_kv_frames = max_kv_frames
        n_cross = len(self.cross_blocks)
        self.cross_blocks = nn.ModuleList(
            [MultiFrameCrossBlockDeform(d, n_heads=n_heads,
                                          n_points=deform_n_points,
                                          max_levels=max_kv_frames,
                                          out_dim=out_dim)
             for _ in range(n_cross)]
        )

    def _multi_forward(self, q, img_2d_list, ref_uv_01_list,
                        pt_kv_list, pt_pad_list, kv_biases, uv_hat_01,
                        q_pad=None,
                        target_shift=None, kv_pose_shifts=None):
        raw_cum = None
        for blk in self.cross_blocks:
            q, raw = blk(q, img_2d_list, ref_uv_01_list,
                          pt_kv_list, pt_pad_list,
                          kv_biases, uv_hat_01,
                          q_pad_mask=q_pad,
                          target_shift=target_shift,
                          kv_pose_shifts=kv_pose_shifts)
            raw_cum = raw if raw_cum is None else raw_cum + raw
        return raw_cum

    def forward(self, patch_A, uvd_A, patch_B, uvd_B,
                pose_AB_6dof, pose_BA_6dof,
                uv_B_hat_of_A, uv_A_hat_of_B,
                pad_A=None, pad_B=None,
                uvd_A_full=None, uvd_B_full=None,
                pad_A_full=None, pad_B_full=None,
                # multi-frame extras (None → pair mode)
                patch_M=None, uvd_M=None, pad_M=None,
                uvd_M_full=None, pad_M_full=None,
                pose_AM_6dof=None,
                uv_M_hat_of_A=None, uv_M_hat_of_B=None,
                per_frame_emb=False):
        """Unified forward. Pair mode: M args = None, KV = single B frame
        with bias = pose_bias(pose_AB). Multi-frame mode: KV = M ‖ B with
        per-frame biases. RPE-style — no Q-shift, no absolute 3D leak.
        """
        # encode A and B (shared weights)
        pt_A, _, img_A_2d = self._encode_frame(
            patch_A, uvd_A, pad_A, uvd_full=uvd_A_full, pad_full=pad_A_full)
        pt_B, _, img_B_2d = self._encode_frame(
            patch_B, uvd_B, pad_B, uvd_full=uvd_B_full, pad_full=pad_B_full)
        has_M = patch_M is not None
        if has_M:
            pt_M, _, img_M_2d = self._encode_frame(
                patch_M, uvd_M, pad_M, uvd_full=uvd_M_full, pad_full=pad_M_full)

        # Q-shift via legacy pose_mlp — explicit pose channel that survives
        # softmax in pair mode (per-frame bias is degenerate at L=1).
        pose_emb_AB = self.pose_mlp(pose_AB_6dof).unsqueeze(1)
        pose_emb_BA = self.pose_mlp(pose_BA_6dof).unsqueeze(1)
        q_A = pt_A + pose_emb_AB
        q_B = pt_B + pose_emb_BA

        # Per-frame absolute embedding mode: anchor at A, every frame's
        # tokens carry their own pose (broadcast). Each (Q, K, V) ends up
        # with emb_X added before attention. No relative Q-shift used.
        if per_frame_emb:
            zero_pose   = torch.zeros_like(pose_AB_6dof)
            emb_A       = self.pose_mlp(zero_pose).unsqueeze(1)         # (B, 1, D)
            emb_B_in_A  = self.pose_mlp(pose_AB_6dof).unsqueeze(1)
            # broadcast over spatial axes for img features
            emb_A_img   = emb_A.permute(0, 2, 1).unsqueeze(-1)            # (B, D, 1, 1)
            emb_B_img   = emb_B_in_A.permute(0, 2, 1).unsqueeze(-1)
            # Pre-shift pt and img features per frame
            pt_A   = pt_A   + emb_A
            pt_B   = pt_B   + emb_B_in_A
            img_A_2d = img_A_2d + emb_A_img
            img_B_2d = img_B_2d + emb_B_img
            if has_M:
                emb_M_in_A = self.pose_mlp(pose_AM_6dof).unsqueeze(1)
                emb_M_img  = emb_M_in_A.permute(0, 2, 1).unsqueeze(-1)
                pt_M     = pt_M     + emb_M_in_A
                img_M_2d = img_M_2d + emb_M_img
            # No additional Q-shift in this mode — Q already carries its frame's emb
            q_A = pt_A
            q_B = pt_B

        # per-frame KV bias (per-head scalars). Constant in pair mode (single
        # frame on K axis), so cancels in softmax. In multi-frame mode it
        # gates between M and B according to relative pose distance.
        b_AB = self.pose_bias(pose_AB_6dof)
        b_BA = self.pose_bias(pose_BA_6dof)
        if has_M:
            pose_MB_6dof = _compose_pose_MB(pose_AB_6dof, pose_AM_6dof)
            pose_BM_6dof = _invert_6dof(pose_MB_6dof)
            b_AM = self.pose_bias(pose_AM_6dof)
            b_BM = self.pose_bias(pose_BM_6dof)

        # pose embeddings — used both as Q-shift and (when has_M) as the
        # per-KV-frame shift for option-b cross-attention.
        pose_emb_AM = self.pose_mlp(pose_AM_6dof).unsqueeze(1) if has_M else None
        pose_emb_MB = self.pose_mlp(pose_MB_6dof).unsqueeze(1) if has_M else None
        pose_emb_BM = self.pose_mlp(pose_BM_6dof).unsqueeze(1) if has_M else None
        # MA = inverse of AM, but we don't have it precomputed; cheap to recompute:
        pose_emb_MA = self.pose_mlp(_invert_6dof(pose_AM_6dof)).unsqueeze(1) if has_M else None

        # A → B  (deformable img KV per frame, plain pt KV per frame)
        if has_M:
            img_2d_list_AB = [img_M_2d, img_B_2d]
            ref_AB         = [uv_M_hat_of_A / self.img_size,
                              uv_B_hat_of_A / self.img_size]
            pt_kv_AB       = [pt_M, pt_B]
            pt_pad_AB      = [pad_M, pad_B]
            bias_AB        = [b_AM, b_AB]
            kv_shifts_AB   = [pose_emb_AM, pose_emb_AB]
        else:
            img_2d_list_AB = [img_B_2d]
            ref_AB         = [uv_B_hat_of_A / self.img_size]
            pt_kv_AB       = [pt_B]
            pt_pad_AB      = [pad_B]
            bias_AB        = [b_AB]
            kv_shifts_AB   = None
        raw_AtoB = self._multi_forward(
            q_A, img_2d_list_AB, ref_AB,
            pt_kv_AB, pt_pad_AB, bias_AB,
            uv_B_hat_of_A / self.img_size, q_pad=pad_A,
            target_shift=pose_emb_AB if has_M else None,
            kv_pose_shifts=kv_shifts_AB)

        # B → A
        if has_M:
            img_2d_list_BA = [img_M_2d, img_A_2d]
            ref_BA         = [uv_M_hat_of_B / self.img_size,
                              uv_A_hat_of_B / self.img_size]
            pt_kv_BA       = [pt_M, pt_A]
            pt_pad_BA      = [pad_M, pad_A]
            bias_BA        = [b_BM, b_BA]
            kv_shifts_BA   = [pose_emb_BM, pose_emb_BA]
        else:
            img_2d_list_BA = [img_A_2d]
            ref_BA         = [uv_A_hat_of_B / self.img_size]
            pt_kv_BA       = [pt_A]
            pt_pad_BA      = [pad_A]
            bias_BA        = [b_BA]
            kv_shifts_BA   = None
        raw_BtoA = self._multi_forward(
            q_B, img_2d_list_BA, ref_BA,
            pt_kv_BA, pt_pad_BA, bias_BA,
            uv_A_hat_of_B / self.img_size, q_pad=pad_B,
            target_shift=pose_emb_BA if has_M else None,
            kv_pose_shifts=kv_shifts_BA)

        clamp_fn = clamp_params if self._out_dim == 5 else clamp_params_uvd
        return (clamp_fn(raw_AtoB, self.img_size),
                clamp_fn(raw_BtoA, self.img_size))

    # ──────────────── N-frame stacked-tensor forward path ────────────────
    # Inputs are stacked along a frame axis so the same code handles
    # N=2 (pair), N=3 (triplet), and N=4+ later. For each (Q=i, T=j) with
    # i ≠ j, we cross-attend Q's tokens to KV built from all other frames
    # (target j first, others as auxiliary context) and emit a residual
    # against `uv_hat[i, j]`. Output is a (B, N, N, max_pts, out_dim) raw
    # tensor with diag(i=j) zero; step_n() masks those out plus pad_dir.

    def forward_n(self,
                  patches: torch.Tensor,        # (B, N, 3, H, W)
                  uvd: torch.Tensor,            # (B, N, max_pts, 3)
                  pad: torch.Tensor,            # (B, N, max_pts) bool
                  uvd_full: torch.Tensor,       # (B, N, N_FULL, 3)
                  pad_full: torch.Tensor,       # (B, N, N_FULL) bool
                  pose_hat_6dof: torch.Tensor,  # (B, N, N, 6)
                  uv_hat: torch.Tensor,         # (B, N, N, max_pts, 2) — local patch coords
                  mix_mode: str = 'mix',        # 'mix' = full N-frame KV (legacy);
                                                # 'pair' = each (i, j) sees only frame j as KV
                                                #          (= 6× independent pair forwards,
                                                #            equivalent to batch×6 of 2-frame
                                                #            calls; baseline before any
                                                #            multi-frame fusion is added).
                  ) -> torch.Tensor:
        B, N = patches.shape[0], patches.shape[1]
        assert N - 1 <= self.max_kv_frames, (
            f'N-1={N-1} KV frames > max_kv_frames={self.max_kv_frames}; '
            f'rebuild model with max_kv_frames>={N-1}')

        # 1. encode each frame once (shared encoder weights). Per-frame results
        #    cached so each (Q, T) pair below just reuses.
        pt_list, img_list = [], []
        for k in range(N):
            pt_k, _, img_k = self._encode_frame(
                patches[:, k], uvd[:, k], pad[:, k],
                uvd_full=uvd_full[:, k], pad_full=pad_full[:, k])
            pt_list.append(pt_k)
            img_list.append(img_k)

        # 2. for each (i, j), Q-shift by pose_hat[i,j], KV = all other frames
        clamp_fn = clamp_params if self._out_dim == 5 else clamp_params_uvd
        raws = torch.zeros(B, N, N, pt_list[0].shape[1], self._out_dim,
                           device=patches.device, dtype=pt_list[0].dtype)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pose_emb_ij = self.pose_mlp(pose_hat_6dof[:, i, j]).unsqueeze(1)
                q = pt_list[i] + pose_emb_ij
                if mix_mode == 'pair':
                    # KV = target j only → identical setup to 2-frame pair.
                    # 6 directions = 6 independent pair forwards. No internal
                    # multi-frame fusion. Each call gets v55-pair's behaviour.
                    kv_order = [j]
                else:
                    # KV order: target j first (so its uv_hat lives at level 0
                    # which is what the deformable head's bias-via-Q-conditioned
                    # weight MLP can favour). Aux frames follow.
                    kv_order = [j] + [k for k in range(N) if k != i and k != j]
                img_2d_list = [img_list[k]                   for k in kv_order]
                ref_list    = [uv_hat[:, i, k] / self.img_size for k in kv_order]
                pt_kv_list  = [pt_list[k]                    for k in kv_order]
                pt_pad_list = [pad[:, k]                     for k in kv_order]
                bias_list   = [self.pose_bias(pose_hat_6dof[:, i, k]) for k in kv_order]
                # per-KV-frame pose shifts (option-b cross-attn)
                kv_shifts   = [self.pose_mlp(pose_hat_6dof[:, i, k]).unsqueeze(1)
                               for k in kv_order]
                # In 'pair' mode KV is single → kv_shifts is single → per-frame
                # Q rebase reduces to identity, equivalent to legacy single-Q.
                use_pf_q   = (mix_mode != 'pair') and len(kv_order) > 1
                raw_ij = self._multi_forward(
                    q, img_2d_list, ref_list,
                    pt_kv_list, pt_pad_list, bias_list,
                    uv_hat[:, i, j] / self.img_size, q_pad=pad[:, i],
                    target_shift=pose_emb_ij if use_pf_q else None,
                    kv_pose_shifts=kv_shifts if use_pf_q else None)
                raws[:, i, j] = clamp_fn(raw_ij, self.img_size)
        return raws


def _ypr_t_to_T(pose_6dof: torch.Tensor) -> torch.Tensor:
    """(..., 6) [yaw, pitch, roll, tx, ty, tz] (degrees) → (..., 4, 4).

    zyx Euler convention to match `_ypr_t_to_mat` in pandaset_pair.py:
    R = R_z(yaw) @ R_y(pitch) @ R_x(roll), column-vector convention.
    """
    ypr = pose_6dof[..., :3] * (math.pi / 180.0)
    t   = pose_6dof[..., 3:]
    yaw, pitch, roll = torch.unbind(ypr, dim=-1)
    cy, sy = torch.cos(yaw),   torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll),  torch.sin(roll)
    # Closed-form R = Rz(yaw) Ry(pitch) Rx(roll):
    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr
    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr
    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr
    R = torch.stack([
        torch.stack([R00, R01, R02], dim=-1),
        torch.stack([R10, R11, R12], dim=-1),
        torch.stack([R20, R21, R22], dim=-1),
    ], dim=-2)
    T = torch.zeros(*pose_6dof.shape[:-1], 4, 4,
                    device=pose_6dof.device, dtype=pose_6dof.dtype)
    T[..., :3, :3] = R
    T[..., :3,  3] = t
    T[...,  3,  3] = 1.0
    return T


def _T_to_ypr_t(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) → (..., 6) [yaw, pitch, roll, tx, ty, tz] (degrees).

    Inverse of `_ypr_t_to_T` for non-singular pitch (|pitch| < 90°).
    """
    R = T[..., :3, :3]
    t = T[..., :3,  3]
    pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1.0 + 1e-6, 1.0 - 1e-6))
    yaw   = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll  = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    ypr = torch.stack([yaw, pitch, roll], dim=-1) * (180.0 / math.pi)
    return torch.cat([ypr, t], dim=-1)


def _invert_T(T: torch.Tensor) -> torch.Tensor:
    """SE(3) inverse: [R | t] → [R^T | -R^T t]."""
    R = T[..., :3, :3]
    t = T[..., :3,  3:4]
    Rinv = R.transpose(-1, -2)
    tinv = -Rinv @ t
    Tinv = torch.zeros_like(T)
    Tinv[..., :3, :3]   = Rinv
    Tinv[..., :3,  3:4] = tinv
    Tinv[...,  3,  3]   = 1.0
    return Tinv


def _invert_6dof(pose_XY: torch.Tensor) -> torch.Tensor:
    """T_YX = T_XY^{-1} via proper SE(3) inverse."""
    return _T_to_ypr_t(_invert_T(_ypr_t_to_T(pose_XY)))


def _compose_pose_MB(pose_AB: torch.Tensor, pose_AM: torch.Tensor) -> torch.Tensor:
    """T_MB = T_AB @ T_AM^{-1} (column-vector conv: p_B = T_AB p_A).

    Replaces the wrong linear approximation `pose_AB - pose_AM`.
    Required for any non-trivial yaw — front-camera baselines up to
    scene length on PandaSet routinely accumulate >5° yaw, where the
    linear form drifts enough to corrupt the bias MLP signal.
    """
    T_AB = _ypr_t_to_T(pose_AB)
    T_AM = _ypr_t_to_T(pose_AM)
    return _T_to_ypr_t(T_AB @ _invert_T(T_AM))
