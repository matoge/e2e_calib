"""Deformable cross-attention variant of the CalibNet block.

Drop-in replacement for `CrossAttentionBlockCov`: same `forward(q, feat, uv_01,
key_padding_mask, self_first)` signature. Internally replaces the full
image-token cross-attn with MS-Deformable-Attention: each point samples K
learned offsets around its reference_point (u, v) — O(N·K) instead of O(N·HW).

Uses the pure-PyTorch core (`ms_deform_attn_core_pytorch`) by default. If the
compiled CUDA extension `MultiScaleDeformableAttention` is importable, switches
to the fast CUDA path transparently.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

try:
    import MultiScaleDeformableAttention as _MSDA  # CUDA extension
    _HAVE_CUDA_KERNEL = True
except Exception:
    _MSDA = None
    _HAVE_CUDA_KERNEL = False


# ── pure-pytorch core (from Deformable-DETR) ─────────────────────────────────
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations,
                                  attention_weights):
    """value: (N, S, M, D_h)  sampling_locations: (N, Lq, M, L, P, 2) in [0,1]."""
    N_, _, M_, D_h = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        v = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_h, H_, W_)
        g = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)   # (N*M, Lq, P, 2)
        sampling_value_list.append(
            F.grid_sample(v, g, mode='bilinear', padding_mode='zeros', align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    out = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * aw).sum(-1).view(
        N_, M_ * D_h, Lq_)
    return out.transpose(1, 2).contiguous()


class _MSDAFunctionCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                 sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _MSDA.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, im2col_step)
        ctx.save_for_backward(value, spatial_shapes, level_start_index,
                               sampling_locations, attention_weights)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = \
            ctx.saved_tensors
        grad_value, grad_sl, grad_aw = _MSDA.ms_deform_attn_backward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sl, grad_aw, None


# ── MSDeformAttn module (ported, kernel-optional) ────────────────────────────
class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention. Matches Deformable-DETR semantics.

    reference_points (B, Lq, L, 2) — normalized (0..1) top-left origin, per level.
    input_flatten    (B, Σ HlWl, C) — feature maps concatenated, flattened along H*W.
    """
    def __init__(self, d_model=64, n_levels=1, n_heads=4, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_levels, self.n_heads, self.n_points = \
            d_model, n_levels, n_heads, n_points
        # Kernel asserts value.size(0) % im2col_step == 0. value.size(0) == B
        # (batch size) in this implementation — reshape happens AFTER value_proj
        # via value.view(B, Lin, ...), so it's just B. Training batch sizes are
        # multiples of 64 (64, 128, 256...), so 64 divides cleanly. Using 1
        # here — as a previous defensive tweak did — launches the im2col kernel
        # once per sample and drops SPS by ~30× on PandaSet.
        self.im2col_step = 64

        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.output_proj       = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) \
            .view(self.n_heads, 1, 1, 2) \
            .repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.); constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data);     constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data);    constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        query              : (B, Lq, D)
        reference_points   : (B, Lq, n_levels, 2)  normalized
        input_flatten      : (B, Σ HlWl, D)
        input_spatial_shapes: (L, 2) long
        input_level_start_index: (L,) long
        """
        B, Lq, _ = query.shape
        B, Lin, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, Lin, self.n_heads, self.d_model // self.n_heads)

        off = self.sampling_offsets(query).view(B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        aw  = self.attention_weights(query).view(B, Lq, self.n_heads, self.n_levels * self.n_points)
        aw  = F.softmax(aw, -1).view(B, Lq, self.n_heads, self.n_levels, self.n_points)

        # reference_points (B, Lq, L, 2) → (B, Lq, 1, L, 1, 2)
        # offset_normalizer (L, 2) — (W, H)
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).to(query.dtype)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                             + off / offset_normalizer[None, None, None, :, None, :]

        if _HAVE_CUDA_KERNEL and value.is_cuda:
            # Kernel dispatches fp32/fp16/bf16, but requires all inputs to share
            # dtype + be contiguous. reference_points arrives in fp32 from outside
            # autocast scope, so align to value's dtype.
            dt = value.dtype
            out = _MSDAFunctionCUDA.apply(
                value.contiguous(), input_spatial_shapes, input_level_start_index,
                sampling_locations.to(dt).contiguous(), aw.to(dt).contiguous(),
                self.im2col_step)
        else:
            # pure pytorch (works on CPU + CUDA; differentiable)
            shapes_list = [(int(h), int(w)) for h, w in input_spatial_shapes.tolist()]
            out = ms_deform_attn_core_pytorch(value, shapes_list, sampling_locations, aw)

        return self.output_proj(out)


# ── drop-in cross-attn block using deformable attn ───────────────────────────
class CrossAttentionBlockDeform(nn.Module):
    """Same interface as `CrossAttentionBlockCov` but uses MSDeformAttn for the
    point→image cross-attention. Self-attn / FFN / proj are unchanged.

    NOTE: single-level per block — keeps the cascaded coarse/fine structure.
    """
    def __init__(self, d: int = 64, n_heads: int = 4, n_points: int = 4,
                 cross_temp: float = 1.0, kv_self_attn: bool = False):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model=d, n_levels=1, n_heads=n_heads,
                                        n_points=n_points)
        self.norm_q   = nn.LayerNorm(d)
        self.norm_kv  = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.norm_ffn = nn.LayerNorm(d)
        self.proj = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69

        if kv_self_attn:
            self.kv_sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            self.norm_kv_sa = nn.LayerNorm(d)
        self._kv_self_attn = kv_self_attn
        self._cross_temp   = cross_temp  # unused for deformable, kept for API parity
        # extra-KV branch: regular softmax cross-attention on a sparse / dense
        # token bank (e.g. dense LiDAR map at gh×gw, or radar tokens). Output
        # is summed with the DA output before the SA stage. Created always so
        # .to(device) works correctly; if extra_kv is never passed at forward
        # time the module just sits unused (~30k params, negligible).
        self.extra_kv_attn = nn.MultiheadAttention(d, n_heads, batch_first=True,
                                                    dropout=0.1)
        self.norm_extra_kv = nn.LayerNorm(d)

    def forward(self, q, feat, uv_01, key_padding_mask=None, self_first=False,
                 extra_kv=None, extra_kv_mask=None):
        """q (B,N,D), feat (B,D,H,W), uv_01 (B,N,2) in [0,1].

        extra_kv (B,N_kv,D): optional secondary KV bank attended to with regular
            softmax cross-attention. Result summed with the DA output. Use for
            dense LiDAR map (gh*gw, with empty cells carrying UV emb) or radar
            tokens — anything that should attend by content rather than via
            uv-anchored deformable sampling.
        """
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)    # (B, HW, D)

        if self._kv_self_attn:
            k = self.norm_kv_sa(kv)
            kv = kv + self.drop(self.kv_sa(k, k, k)[0])

        nq = self.norm_q(q)
        nkv = self.norm_kv(kv)

        spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=feat.device)
        level_start_index = torch.as_tensor([0], dtype=torch.long, device=feat.device)
        ref = uv_01.unsqueeze(2).clamp(0.0, 1.0)    # (B, N, 1, 2)

        ca = self.cross_attn(nq, ref, nkv, spatial_shapes, level_start_index,
                              input_padding_mask=None)
        # parallel softmax CA on extra_kv (e.g. dense LiDAR map gh×gw cells).
        if extra_kv is not None:
            n_extra = self.norm_extra_kv(extra_kv)
            ca_extra, _ = self.extra_kv_attn(nq, n_extra, n_extra,
                                              key_padding_mask=extra_kv_mask)
            ca = ca + ca_extra        # sum DA + regular attn outputs

        if self_first:
            # (keep parity with CrossAttentionBlockCov: under self_first the SA
            # runs first and CA uses the SA-updated q. The original uses pre-SA
            # q for both. We mirror the flag by swapping order here.)
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q),
                                    self.norm_self(q), key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
            q = q + self.drop(ca)
        else:
            q = q + self.drop(ca)
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q),
                                    self.norm_self(q), key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)

        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))
        return q, raw


# ── multi-level deformable block with level (resolution) embedding ───────────
class CrossAttentionBlockDeformML(nn.Module):
    """Multi-level deformable cross-attn. Each query samples from ALL feature
    levels simultaneously, with a learnable level (resolution) embedding added
    to image tokens to let the model tell levels apart.

    Forward signature: `forward(q, feats, uv_01, level_embed, key_padding_mask)`
      - feats:       list[Tensor(B, D, H_l, W_l)], length = n_levels
      - level_embed: Tensor(n_levels, D)  — owned by parent model
    """
    def __init__(self, d: int = 64, n_heads: int = 4, n_levels: int = 2,
                 n_points: int = 4, kv_self_attn: bool = False,
                 cross_temp: float = 1.0):
        super().__init__()
        self.n_levels = n_levels
        self.cross_attn = MSDeformAttn(d_model=d, n_levels=n_levels,
                                        n_heads=n_heads, n_points=n_points)
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.norm_ffn = nn.LayerNorm(d)
        self.proj = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69

        if kv_self_attn:
            self.kv_sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            self.norm_kv_sa = nn.LayerNorm(d)
        self._kv_self_attn = kv_self_attn
        self._cross_temp   = cross_temp

    def forward(self, q, feats, uv_01, level_embed, key_padding_mask=None,
                self_first=False):
        B, N, D_ = q.shape
        assert len(feats) == self.n_levels
        device = q.device

        # build concatenated KV with level embedding
        kvs, shapes = [], []
        cum = [0]
        for lid, feat in enumerate(feats):
            _, _, H, W = feat.shape
            kv = feat.flatten(2).permute(0, 2, 1)           # (B, HW, D)
            kv = kv + level_embed[lid][None, None, :]        # + level embed
            kvs.append(kv); shapes.append([H, W])
            cum.append(cum[-1] + H * W)
        kv = torch.cat(kvs, dim=1)                          # (B, Σ HW, D)

        if self._kv_self_attn:
            k = self.norm_kv_sa(kv)
            kv = kv + self.drop(self.kv_sa(k, k, k)[0])

        nkv = self.norm_kv(kv)
        nq  = self.norm_q(q)

        spatial_shapes    = torch.as_tensor(shapes, dtype=torch.long, device=device)
        level_start_index = torch.as_tensor(cum[:-1], dtype=torch.long, device=device)
        # reference_points: same uv replicated across levels
        ref = uv_01.unsqueeze(2).expand(-1, -1, self.n_levels, -1).clamp(0.0, 1.0)

        ca = self.cross_attn(nq, ref, nkv, spatial_shapes, level_start_index,
                              input_padding_mask=None)

        if self_first:
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q),
                                    self.norm_self(q), key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
            q = q + self.drop(ca)
        else:
            q = q + self.drop(ca)
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q),
                                    self.norm_self(q), key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)

        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))
        return q, raw


if __name__ == '__main__':
    B, N, D = 2, 64, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q = torch.randn(B, N, D, device=device, requires_grad=True)
    uv = torch.rand(B, N, 2, device=device)
    print('CUDA kernel available:', _HAVE_CUDA_KERNEL)

    # SL
    feat_c = torch.randn(B, D, 8, 8, device=device)
    blk_sl = CrossAttentionBlockDeform(d=D, n_heads=4, n_points=4).to(device)
    q_out, raw = blk_sl(q, feat_c, uv)
    print(f'SL  q {tuple(q_out.shape)} raw {tuple(raw.shape)} params {sum(p.numel() for p in blk_sl.parameters())/1e3:.1f}k')

    # ML
    feat_f = torch.randn(B, D, 4, 4, device=device)
    blk_ml = CrossAttentionBlockDeformML(d=D, n_heads=4, n_levels=2, n_points=4).to(device)
    level_embed = nn.Parameter(torch.zeros(2, D, device=device))
    nn.init.normal_(level_embed, std=0.02)
    q_out, raw = blk_ml(q, [feat_c, feat_f], uv, level_embed)
    print(f'ML  q {tuple(q_out.shape)} raw {tuple(raw.shape)} params {sum(p.numel() for p in blk_ml.parameters())/1e3:.1f}k')
    raw.sum().backward()
    print('backward OK')
