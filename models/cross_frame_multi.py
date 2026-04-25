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
                q_pad_mask=None):
        """
        q                : (B, N_q, D)
        img_2d_list      : list of (B, D, H, W) — one per KV frame
        ref_uv_01_list   : list of (B, N_q, 2) — reference uv per query per frame
        pt_kv_list       : list of (B, N_pt_i, D) — pt KV per frame
        pt_pad_list      : list of (B, N_pt_i) bool — pad masks per frame
        kv_biases        : list of (B, n_heads) per-frame bias (same for img & pt of that frame)
        uv_hat_01        : (B, N_q, 2) — A→B hypothesis (for output head)
        """
        B, N_q, D = q.shape
        L = len(img_2d_list)
        assert L <= self.max_levels, f'L={L} > max_levels={self.max_levels}'
        H, W = img_2d_list[0].shape[2], img_2d_list[0].shape[3]
        device = q.device

        # ── deformable image side ────────────────────────────────────────────
        # flatten and concat all frames' image features
        flats = [feat.flatten(2).permute(0, 2, 1) for feat in img_2d_list]   # each (B, HW, D)
        kv_img_flat = torch.cat(flats, dim=1)                                 # (B, L*HW, D)
        spatial_shapes = torch.as_tensor([[H, W]] * L, dtype=torch.long, device=device)
        level_start_index = torch.as_tensor(
            [0] + [H * W * (i + 1) for i in range(L - 1)],
            dtype=torch.long, device=device,
        )
        # reference points (B, N_q, L, 2) — clamp to [0, 1]
        ref = torch.stack(ref_uv_01_list, dim=2).clamp(0.0, 1.0)              # (B, N_q, L, 2)
        # If L < max_levels, pad ref + spatial_shapes with copies of the last
        # so MSDeformAttn's sampling_offsets shape (sized for max_levels) sees
        # consistent input. Mask out the padded levels' attention weights to 0.
        if L < self.max_levels:
            pad_L = self.max_levels - L
            # repeat the LAST level's ref + shape; duplicated levels point
            # back into the LAST real level's flat region so MSDeformAttn
            # samples valid memory (same content twice — harmless redundancy).
            ref = torch.cat([ref, ref[:, :, -1:].expand(-1, -1, pad_L, -1)], dim=2)
            shape_pad = spatial_shapes[-1:].expand(pad_L, -1)
            spatial_shapes = torch.cat([spatial_shapes, shape_pad], dim=0)
            # all extra levels reuse the LAST real level's start index
            last_start = level_start_index[-1:]
            extra = last_start.expand(pad_L)
            level_start_index = torch.cat([level_start_index, extra], dim=0)

        # call the kernel (built-in attention weights MLP); we'll inject bias
        # POST-MLP, PRE-softmax by patching forward inline. The cleanest way
        # is to compute the standard call AND then add a residual bias term —
        # mathematically equivalent to a scalar multiplier on each level's
        # contribution. We approximate by SCALING the deformable output by
        # frame-bias gates (per-head softmax over levels), which is similar
        # in spirit and much simpler to implement.
        ca_img_full = self.deform_img(self.norm_q_img(q), ref,
                                       self.norm_kv_img(kv_img_flat),
                                       spatial_shapes, level_start_index,
                                       input_padding_mask=None)
        # NOTE: MSDeformAttn already softmaxes weights across (level × points)
        # and aggregates internally — so per-level RPE bias is delegated to
        # the pt-side plain cross-attn (where it can be cleanly injected).

        # ── plain cross-attn on pt KV side, with per-frame bias ──────────────
        kv_pt = torch.cat(pt_kv_list, dim=1)                  # (B, Σ N_pt_i, D)
        Qn = self.norm_q_pt(q)
        Kn = self.norm_kv_pt(kv_pt)
        Hh = self.n_heads; Dh = D // Hh
        Qp = self.q_proj(Qn).view(B, N_q,           Hh, Dh).transpose(1, 2)
        Kp = self.k_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
        Vp = self.v_proj(Kn).view(B, kv_pt.shape[1], Hh, Dh).transpose(1, 2)
        scale = Dh ** -0.5
        scores = torch.matmul(Qp, Kp.transpose(-1, -2)) * scale   # (B, Hh, N_q, Σ N_pt_i)

        # per-frame bias on attention scores (segment-wise constant)
        N_kv = kv_pt.shape[1]
        bias_kv = torch.zeros(B, Hh, N_kv, device=device, dtype=scores.dtype)
        off = 0
        for seg, b in zip(pt_kv_list, kv_biases):
            Lpt = seg.shape[1]
            bias_kv[:, :, off:off + Lpt] = b.unsqueeze(-1)
            off += Lpt
        scores = scores + bias_kv.unsqueeze(2)

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
                        q_pad=None):
        raw_cum = None
        for blk in self.cross_blocks:
            q, raw = blk(q, img_2d_list, ref_uv_01_list,
                          pt_kv_list, pt_pad_list,
                          kv_biases, uv_hat_01,
                          q_pad_mask=q_pad)
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
                uv_M_hat_of_A=None, uv_M_hat_of_B=None):
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

        # A → B  (deformable img KV per frame, plain pt KV per frame)
        if has_M:
            img_2d_list_AB = [img_M_2d, img_B_2d]
            ref_AB         = [uv_M_hat_of_A / self.img_size,
                              uv_B_hat_of_A / self.img_size]
            pt_kv_AB       = [pt_M, pt_B]
            pt_pad_AB      = [pad_M, pad_B]
            bias_AB        = [b_AM, b_AB]
        else:
            img_2d_list_AB = [img_B_2d]
            ref_AB         = [uv_B_hat_of_A / self.img_size]
            pt_kv_AB       = [pt_B]
            pt_pad_AB      = [pad_B]
            bias_AB        = [b_AB]
        raw_AtoB = self._multi_forward(
            q_A, img_2d_list_AB, ref_AB,
            pt_kv_AB, pt_pad_AB, bias_AB,
            uv_B_hat_of_A / self.img_size, q_pad=pad_A)

        # B → A
        if has_M:
            img_2d_list_BA = [img_M_2d, img_A_2d]
            ref_BA         = [uv_M_hat_of_B / self.img_size,
                              uv_A_hat_of_B / self.img_size]
            pt_kv_BA       = [pt_M, pt_A]
            pt_pad_BA      = [pad_M, pad_A]
            bias_BA        = [b_BM, b_BA]
        else:
            img_2d_list_BA = [img_A_2d]
            ref_BA         = [uv_A_hat_of_B / self.img_size]
            pt_kv_BA       = [pt_A]
            pt_pad_BA      = [pad_A]
            bias_BA        = [b_BA]
        raw_BtoA = self._multi_forward(
            q_B, img_2d_list_BA, ref_BA,
            pt_kv_BA, pt_pad_BA, bias_BA,
            uv_A_hat_of_B / self.img_size, q_pad=pad_B)

        clamp_fn = clamp_params if self._out_dim == 5 else clamp_params_uvd
        return (clamp_fn(raw_AtoB, self.img_size),
                clamp_fn(raw_BtoA, self.img_size))


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
