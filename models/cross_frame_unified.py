"""CalibNetUnifiedFrame — frame-token unified architecture.

Per-frame representation collapses image + LiDAR onto the same 16x16 grid
("frame_token"). Empty cells (no point observation) get pt = 0 plus a
"has_pt" mask channel. The unified frame_token is then the only modality
the cross-frame block sees → MSDeformAttn applies symmetrically over both
image and LiDAR content.

Per-point query is bilinear-sampled from the anchor frame_token at the
point's uv (so Q already carries anchor-frame appearance + LiDAR context),
plus a small PointMLP encoding of the point's 3D coordinate, plus the
pose embedding to the target frame.

Multi-frame: each KV frame's grid gets a per-frame absolute pose embedding
(broadcast across the grid) so the model can tell levels apart. Single
softmax over (level × n_points) inside MSDeformAttn handles all mixing.

Drop-in replacement for CalibNetMultiFrame.forward / forward_n.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import CNNBackbone, ConvNeXtBackbone, D as D_DIM
from models.model_cov import clamp_params, clamp_params_uvd
from models.model_depth import PointMLP3, FrustumLocalEncoder
from models.model_deform import MSDeformAttn
from models.cross_frame import SelfAttnFusionBlock, PoseMLP
from models.cross_frame_multi import _invert_6dof, _compose_pose_MB


def _build_2d_pos_emb(Hg: int, Wg: int, d: int) -> torch.Tensor:
    """Sin/cos 2D positional encoding, (1, Hg*Wg, d)."""
    assert d % 4 == 0, f'd={d} must be divisible by 4 for 2D sin/cos'
    d4 = d // 4
    freq = 1.0 / (10000 ** (torch.arange(d4, dtype=torch.float32) / d4))  # (d/4,)
    yy, xx = torch.meshgrid(
        torch.arange(Hg, dtype=torch.float32) / max(1, Hg - 1),
        torch.arange(Wg, dtype=torch.float32) / max(1, Wg - 1),
        indexing='ij')
    angy = yy.flatten().unsqueeze(-1) * freq               # (HW, d/4)
    angx = xx.flatten().unsqueeze(-1) * freq               # (HW, d/4)
    emb = torch.cat([
        torch.sin(angy), torch.cos(angy),
        torch.sin(angx), torch.cos(angx)], dim=-1)         # (HW, d)
    return emb.unsqueeze(0)                                # (1, HW, d)


def scatter_pt_to_grid(pt_feat: torch.Tensor, uv: torch.Tensor,
                       pad_mask, Hg: int, Wg: int, img_size: int):
    """(B,P,D) pt features → (B,D,Hg,Wg) avg-per-cell grid + (B,1,Hg,Wg) mask.

    Multiple pts in the same cell are averaged. Empty cells: pt = 0, mask = 0.
    Padded points (pad_mask=True) are dropped.
    """
    B, P, D = pt_feat.shape
    device = pt_feat.device
    cu = (uv[..., 0] / img_size * Wg).long().clamp(0, Wg - 1)
    cv = (uv[..., 1] / img_size * Hg).long().clamp(0, Hg - 1)
    cell_idx = cv * Wg + cu                                         # (B, P)

    if pad_mask is not None:
        valid_f = (~pad_mask).to(pt_feat.dtype).unsqueeze(-1)        # (B, P, 1)
    else:
        valid_f = torch.ones(B, P, 1, device=device, dtype=pt_feat.dtype)
    pt_feat_v = pt_feat * valid_f

    grid_flat  = torch.zeros(B, Hg * Wg, D, device=device, dtype=pt_feat.dtype)
    count_flat = torch.zeros(B, Hg * Wg, 1, device=device, dtype=pt_feat.dtype)
    cell_idx_exp = cell_idx.unsqueeze(-1).expand(-1, -1, D)          # (B, P, D)
    grid_flat.scatter_add_(1, cell_idx_exp, pt_feat_v)
    count_flat.scatter_add_(1, cell_idx.unsqueeze(-1), valid_f)
    grid_flat = grid_flat / count_flat.clamp(min=1.0)

    pt_grid = grid_flat.permute(0, 2, 1).reshape(B, D, Hg, Wg)
    mask    = (count_flat > 0).to(pt_feat.dtype) \
                .permute(0, 2, 1).reshape(B, 1, Hg, Wg)
    return pt_grid, mask


def sample_grid_at_uv(grid: torch.Tensor, uv_01: torch.Tensor) -> torch.Tensor:
    """(B,D,Hg,Wg), (B,P,2) in [0,1] → (B,P,D)."""
    g = (2 * uv_01.clamp(0.0, 1.0) - 1).unsqueeze(2)                 # (B, P, 1, 2)
    sampled = F.grid_sample(grid, g, mode='bilinear',
                             padding_mode='zeros', align_corners=False)
    return sampled.squeeze(-1).permute(0, 2, 1)                      # (B, P, D)


class ModalityXAttnBlock(nn.Module):
    """One round of Q ← cross-attn ← KV(image⊕lidar concat).

    Per user spec: encoder repeats `Q (UV positional slots) ← KV (lidar +
    camera concat, with modality_emb to distinguish)` N times. Q starts
    as a fixed grid of UV slots, absorbs multimodal info each round.
    Output = refined Q grid → unified frame_token.
    """
    def __init__(self, d, n_heads=4):
        super().__init__()
        # cross-attn (Q → KV)
        self.norm_q    = nn.LayerNorm(d)
        self.norm_kv   = nn.LayerNorm(d)
        self.attn      = nn.MultiheadAttention(
            d, n_heads, batch_first=True, dropout=0.1)
        # self-attn (Q → Q) — Perceiver-style mixing inside the latent set
        self.norm_sa   = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            d, n_heads, batch_first=True, dropout=0.1)
        # FFN
        self.norm_ffn  = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(0.1)

    def forward(self, q, kv, kv_pad_mask=None):
        # Q ← cross-attn ← KV
        attn, _ = self.attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=kv_pad_mask)
        q = q + self.drop(attn)
        # Q ← self-attn ← Q (latent mixing)
        sa, _ = self.self_attn(
            self.norm_sa(q), self.norm_sa(q), self.norm_sa(q))
        q = q + self.drop(sa)
        # FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class FrameTokenEncoder(nn.Module):
    """Per-frame: image + sparse pts → unified (B, D, Hg, Wg) frame_token.

    `n_xattn_modality` ≥ 1 inserts that many ModalityXAttnBlock rounds
    before the scatter+fuse step → image and lidar tokens directly mix
    via cross-attention multiple times. 0 (default) = legacy single-pass
    scatter-then-fuse path.
    """

    def __init__(self, d=D_DIM, in_channels=3, use_convnext=False,
                 r_uv=4.0, r_d=0.006, k_nb=8, use_frustum=True,
                 n_intra_layers=2, img_size=128,
                 n_xattn_modality=0, kv_image_only=False):
        super().__init__()
        self.cnn_d = d
        self.cnn = (ConvNeXtBackbone(d, in_channels=in_channels)
                    if use_convnext else CNNBackbone(d, in_channels=in_channels))
        self.point_mlp = PointMLP3(d)
        self.frustum_enc = (FrustumLocalEncoder(d, r_uv=r_uv, r_d=r_d, k=k_nb)
                            if use_frustum else None)
        # Fusion: [img_grid | pt_grid | mask] → frame_token
        # Init so img branch is identity and pt branch is zero — model
        # starts as "img-only", learns to use pt as needed.
        self.fuse = nn.Conv2d(2 * d + 1, d, kernel_size=1)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            for c in range(d):
                self.fuse.weight[c, c, 0, 0] = 1.0
        self.intra = nn.ModuleList(
            [SelfAttnFusionBlock(d) for _ in range(n_intra_layers)])
        # Ablation: when True, frame_token is CNN(image) only — pt is NOT
        # mixed in via scatter/fuse/Perceiver. The decoder Q (PointMLP+
        # frustum) then attends directly to image features → reproduces
        # the old CalibNet pattern (camera-as-KV, lidar-as-Q with frustum).
        self.kv_image_only = kv_image_only
        # Multi-round Perceiver-style cross-attn:
        #   Q (uv positional slots, initially fixed grid) ← cross ← KV
        # where KV = (image_tokens ⊕ pt_tokens) concat, each modality
        # tagged with its own learned modality_emb. After N rounds, Q is
        # the unified frame_token grid.
        self.xattn_modality = nn.ModuleList(
            [ModalityXAttnBlock(d) for _ in range(n_xattn_modality)])
        if n_xattn_modality > 0:
            # Shared 2D sin/cos positional encoding used for BOTH:
            #   - image_tokens (added before concat → tells transformer
            #     which UV cell each image token came from)
            #   - encoder Q init (uv-slot grid; same UV system as image)
            # Same coords → cross-attn can naturally align Q[uv] with
            # image_tok[uv]. Modality is identified via feature distribution
            # (CNN vs PointMLP outputs are clearly different) — no learned
            # modality_emb needed.
            Hg = Wg = img_size // 8
            self.register_buffer(
                'pos_emb_2d',
                _build_2d_pos_emb(Hg, Wg, d),    # (1, HW, D) fixed
                persistent=False)
            # Q init: learnable, starts as the same 2D sin/cos as image_tok
            # then specializes via training.
            self.q_init = nn.Parameter(
                _build_2d_pos_emb(Hg, Wg, d), requires_grad=True)
        self.img_size = img_size

    def forward(self, image, uvd, pad_mask=None,
                uvd_full=None, pad_full=None, modality='mm'):
        """modality: 'mm' (camera+LiDAR), 'cam' (LiDAR-zeroed), 'lidar' (image-zeroed).

        For 'cam'/'lidar' modes the missing modality's grid contribution is
        zeroed so the same encoder can produce a frame_token from any single
        modality — enables calibration training where one frame is camera-only
        and the other is LiDAR-only.
        """
        if modality == 'lidar':
            B = uvd.shape[0]
            Hg = Wg = self.img_size // 8
            coarse = torch.zeros(B, self.cnn_d, Hg, Wg,
                                  device=uvd.device, dtype=uvd.dtype)
        else:
            coarse, _ = self.cnn(image)                              # (B, D, Hg, Wg)
        B, D, Hg, Wg = coarse.shape

        pt_feat = None  # set in non-cam branch; remains None for camera-only
        if modality == 'cam':
            pt_grid = torch.zeros(B, D, Hg, Wg,
                                   device=coarse.device, dtype=coarse.dtype)
            mask = torch.zeros(B, 1, Hg, Wg,
                                device=coarse.device, dtype=coarse.dtype)
        else:
            uv_01 = uvd[..., :2] / self.img_size
            uvd_n = torch.cat([uv_01, uvd[..., 2:3]], dim=-1)
            pt_feat = self.point_mlp(uvd_n)
            if self.frustum_enc is not None:
                pt_feat = pt_feat + self.frustum_enc(
                    uvd, full_uvd=uvd_full,
                    full_pad_mask=pad_full, query_pad_mask=pad_mask)
            # Persist the per-point feature so the model can reuse it as
            # decoder Q (same identity-encoded feature in Q and KV → the
            # cross-attn becomes a self-consistency check).
            self._last_pt_feat = pt_feat

            pt_grid, mask = scatter_pt_to_grid(
                pt_feat, uvd[..., :2], pad_mask, Hg, Wg, self.img_size)

        if self.kv_image_only:
            # Ablation: frame_token = CNN(image) only. pt_feat is computed
            # (and returned for decoder Q) but NOT mixed into frame_token.
            # Decoder cross-attn becomes: Q (pt+frustum) → KV (image-only) —
            # the old CalibNet direct pattern.
            mask = torch.zeros(B, 1, Hg, Wg,
                                device=coarse.device, dtype=coarse.dtype)
            x = coarse.flatten(2).permute(0, 2, 1)
            for blk in self.intra:
                x = blk(x)
            frame_token = x.permute(0, 2, 1).reshape(B, D, Hg, Wg).contiguous()
            return frame_token, mask, pt_feat

        if len(self.xattn_modality) > 0:
            # Perceiver-style: Q = uv positional slots, KV = image_tok ⊕ pt_tok.
            # 2D sin/cos pos_emb is added to image_tok (same coord system as Q
            # init) → cross-attn learns "Q[uv] ↔ image_tok[uv]" alignment from
            # day 1. pt_tok already has uv info via PointMLP+frustum.
            # Modality is implicit in feature distribution (CNN vs MLP outputs).
            img_tok = coarse.flatten(2).permute(0, 2, 1)            # (B, HW, D)
            kv_img = img_tok + self.pos_emb_2d                      # +2D pos emb
            if modality == 'cam':
                kv = kv_img
                kv_pad = None
            else:
                kv = torch.cat([kv_img, pt_feat], dim=1)             # (B, HW+P, D)
                if pad_mask is not None:
                    img_pad = torch.zeros(B, kv_img.shape[1],
                                           dtype=torch.bool, device=kv.device)
                    kv_pad = torch.cat([img_pad, pad_mask], dim=1)
                else:
                    kv_pad = None
            q = self.q_init.expand(B, -1, -1)                        # (B, HW, D)
            for blk in self.xattn_modality:
                q = blk(q, kv, kv_pad_mask=kv_pad)
            x = q
            mask = (mask if modality != 'cam'
                    else torch.zeros(B, 1, Hg, Wg,
                                      device=coarse.device, dtype=coarse.dtype))
            for blk in self.intra:
                x = blk(x)
            frame_token = x.permute(0, 2, 1).reshape(B, D, Hg, Wg).contiguous()
            return frame_token, mask, pt_feat

        fused = self.fuse(torch.cat([coarse, pt_grid, mask], dim=1))

        x = fused.flatten(2).permute(0, 2, 1)                        # (B, Hg*Wg, D)
        for blk in self.intra:
            x = blk(x)
        frame_token = x.permute(0, 2, 1).reshape(B, D, Hg, Wg).contiguous()
        return frame_token, mask, pt_feat


class UnifiedCrossBlock(nn.Module):
    """Cross-frame deformable attention over multi-level frame_token KV."""

    def __init__(self, d=D_DIM, n_heads=4, n_points=4,
                 max_levels=2, out_dim=5, sigma_bias_share=1.0):
        """`sigma_bias_share` ∈ (0, 1]: each block contributes
        `sigma_bias_share * log(2)` to the cumulative log_σ output. The parent
        model passes `1/n_cross_layers` so the C blocks sum to log(2) ⇒ initial
        σ = 2 px regardless of depth. Without this, C=5 starts σ at ~31 px =
        the MAX_SIGMA clamp ceiling, killing gradients on σ permanently."""
        super().__init__()
        self.deform = MSDeformAttn(d_model=d, n_levels=max_levels,
                                    n_heads=n_heads, n_points=n_points)
        self.norm_q   = nn.LayerNorm(d)
        self.norm_kv  = nn.LayerNorm(d)
        self.norm_sa  = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Linear(d + 2, out_dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        sb = 0.69 * sigma_bias_share
        with torch.no_grad():
            if out_dim == 5:
                self.proj.bias[2] = sb
                self.proj.bias[3] = sb
            elif out_dim == 7:
                self.proj.bias[3] = sb
                self.proj.bias[4] = sb
                self.proj.bias[5] = 0.0
        self.max_levels = max_levels

    def forward(self, q, kv_grids, ref_uv_01_list, uv_hat_01, q_pad_mask=None):
        """
        q              : (B, P, D)
        kv_grids       : list of (B, D, Hg, Wg) — one per KV frame
        ref_uv_01_list : list of (B, P, 2) — reference uv per query per frame
        uv_hat_01      : (B, P, 2) — primary target hypothesis (for output head)
        """
        B, P, D = q.shape
        L = len(kv_grids)
        assert L <= self.max_levels, f'L={L} > max_levels={self.max_levels}'
        Hg, Wg = kv_grids[0].shape[2], kv_grids[0].shape[3]
        device = q.device

        flats = [g.flatten(2).permute(0, 2, 1) for g in kv_grids]
        kv_flat = torch.cat(flats, dim=1)
        spatial_shapes = torch.as_tensor([[Hg, Wg]] * L,
                                          dtype=torch.long, device=device)
        level_start_index = torch.as_tensor(
            [0] + [Hg * Wg * (i + 1) for i in range(L - 1)],
            dtype=torch.long, device=device)
        ref = torch.stack(ref_uv_01_list, dim=2).clamp(0.0, 1.0)
        if L < self.max_levels:
            # Pad ref + spatial_shapes + level_start_index AND kv_flat together
            # by duplicating the last real level. Pure-pytorch deformable core
            # does `value.split([H*W for ...])` so the sum must match Σ tokens —
            # padding only the metadata desyncs that. Duplication is benign:
            # the duplicated level contributes via its own softmax weight slice
            # but with identical content, so the marginal effect is captured by
            # the original level's weight.
            pad_L = self.max_levels - L
            last_flat = flats[-1]                                 # (B, H*W, D)
            kv_flat = torch.cat([kv_flat] + [last_flat] * pad_L, dim=1)
            ref = torch.cat([ref, ref[:, :, -1:].expand(-1, -1, pad_L, -1)], dim=2)
            spatial_shapes = torch.cat(
                [spatial_shapes, spatial_shapes[-1:].expand(pad_L, -1)], dim=0)
            extra_starts = torch.as_tensor(
                [Hg * Wg * (L + i) for i in range(pad_L)],
                dtype=torch.long, device=device)
            level_start_index = torch.cat([level_start_index, extra_starts], dim=0)

        ca = self.deform(self.norm_q(q), ref, self.norm_kv(kv_flat),
                          spatial_shapes, level_start_index,
                          input_padding_mask=None)
        q = q + self.drop(ca)

        sa, _ = self.self_attn(self.norm_sa(q), self.norm_sa(q), self.norm_sa(q),
                                key_padding_mask=q_pad_mask)
        q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_hat_01], dim=-1))
        return q, raw


class CalibNetUnifiedFrame(nn.Module):
    """Frame-token unified architecture — drop-in for CalibNetMultiFrame."""

    def __init__(self, d=D_DIM, n_heads=4, n_intra_layers=2, n_cross_layers=2,
                 in_channels=3, img_size=128, max_kv_frames=2,
                 deform_n_points=4, use_convnext=False,
                 r_uv=4.0, r_d=0.006, k_nb=8, use_frustum=True, out_dim=5,
                 uv_only_query=False, q_uv_pure=False, n_xattn_modality=0,
                 kv_image_only=False):
        """uv_only_query: drop the bilinear sample of anchor frame_token from
        Q construction. Q becomes (PointMLP(uvd) + pose_emb) only — purely
        positional. Required for calib mode (anchor-frame may be camera-only
        or LiDAR-only, so the mixed bilinear sample doesn't carry uniform
        modality content). Cross-frame still works with uv_only_query=True;
        the model just relies entirely on cross-attention to pull context."""
        super().__init__()
        self.d = d
        self.img_size = img_size
        self.max_kv_frames = max_kv_frames
        self._out_dim = out_dim
        self.uv_only_query = uv_only_query
        self.q_uv_pure = q_uv_pure
        self.encoder = FrameTokenEncoder(
            d, in_channels=in_channels, use_convnext=use_convnext,
            r_uv=r_uv, r_d=r_d, k_nb=k_nb, use_frustum=use_frustum,
            n_intra_layers=n_intra_layers, img_size=img_size,
            n_xattn_modality=n_xattn_modality,
            kv_image_only=kv_image_only)
        self.point_mlp_q = PointMLP3(d)
        # Pure-uv positional Q (DETR-style): sin/cos basis at multi-freqs
        # over (u,v) ∈ [0, 1] only. No depth, no anchor sample. Used when
        # q_uv_pure=True — the model has to interpret 'which 3D point' from
        # what it pulls out of the unified frame_token via cross-attn.
        n_uv_freq = 6
        self.register_buffer(
            'uv_freqs',
            (2.0 ** torch.arange(n_uv_freq, dtype=torch.float32)) * 3.14159265,
            persistent=False)
        self.uv_pure_mlp = nn.Sequential(
            nn.Linear(2 + 2 * n_uv_freq * 2, 64), nn.GELU(),
            nn.Linear(64, d),
        )
        self.pose_mlp = PoseMLP(d)
        # Per-point feature embed (rcs, vx_world, vy_world). Zero for LiDAR,
        # real for radar — initialised so the contribution starts at zero so
        # existing LiDAR-only training is unchanged at init, then the model
        # learns to use radar features only when present.
        self.feat_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(),
            nn.Linear(32, d),
        )
        nn.init.zeros_(self.feat_mlp[-1].weight)
        nn.init.zeros_(self.feat_mlp[-1].bias)
        # Per-sample virtual-focal-length embedding. Encodes log(vfl) at 4
        # frequencies (sin+cos basis). When vfl is None, the embedding is
        # skipped — back-compat with checkpoints trained without it.
        # Last-layer zero-init so newly-added vfl conditioning is a no-op
        # until the model learns to use it.
        n_freq = 4
        self.register_buffer(
            'vfl_freqs',
            2.0 ** torch.arange(n_freq, dtype=torch.float32) * 0.5,  # 0.5,1,2,4
            persistent=False)
        self.vfl_mlp = nn.Sequential(
            nn.Linear(2 * n_freq + 1, 32), nn.GELU(),
            nn.Linear(32, d),
        )
        nn.init.zeros_(self.vfl_mlp[-1].weight)
        nn.init.zeros_(self.vfl_mlp[-1].bias)
        self.cross_blocks = nn.ModuleList([
            UnifiedCrossBlock(d, n_heads=n_heads, n_points=deform_n_points,
                              max_levels=max_kv_frames, out_dim=out_dim,
                              sigma_bias_share=1.0 / n_cross_layers)
            for _ in range(n_cross_layers)
        ])

    def _build_query(self, frame_token_anchor, uvd_anchor, pose_emb_to_tgt,
                     feats_anchor=None, pt_feat_anchor=None):
        """Q = (optional bilinear(frame_token_A, uv_A)) + PointMLP_q(uvd_A)
             + pose_emb_AB + (optional FeatMLP(feats_A)).

        feats_anchor is optional per-point sensor side-channel (rcs, vx, vy
        for radar; zeros for lidar). Zero-initialised contribution so the
        added term is no-op until the model learns to use it.

        pt_feat_anchor: when supplied AND uv_only_query=True, REUSE the
        encoder's per-point feature (PointMLP+frustum) directly as Q's
        identity component instead of computing a separate point_mlp_q.
        Same identity in Q and KV (the frame_token built from the same
        pt_feat) → decoder cross-attn becomes a self-consistency check
        which is the right framing for "where did point i go?".
        """
        uv_01 = uvd_anchor[..., :2] / self.img_size
        if self.q_uv_pure:
            # Pure-uv positional Q: sin/cos basis on (u, v), no depth, no
            # anchor sample. Q is purely a positional slot — the model
            # must pull '3D content' from KV via cross-attn.
            ang = uv_01.unsqueeze(-1) * self.uv_freqs                # (B, P, 2, F)
            sc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1) # (B, P, 2, 2F)
            sc = sc.flatten(-2)                                       # (B, P, 4F)
            feat = torch.cat([uv_01, sc], dim=-1)                     # (B, P, 2+4F)
            return self.uv_pure_mlp(feat) + pose_emb_to_tgt
        uvd_n = torch.cat([uv_01, uvd_anchor[..., 2:3]], dim=-1)
        # Pick point identity feature: either reuse encoder's pt_feat
        # (PointMLP+frustum) or recompute via the decoder's own point_mlp_q.
        if pt_feat_anchor is not None and self.uv_only_query:
            ptq = pt_feat_anchor
        else:
            ptq = self.point_mlp_q(uvd_n)
        if feats_anchor is not None:
            ptq = ptq + self.feat_mlp(feats_anchor)
        if self.uv_only_query:
            return ptq + pose_emb_to_tgt
        ctx = sample_grid_at_uv(frame_token_anchor, uv_01)
        return ctx + ptq + pose_emb_to_tgt

    def _multi_forward(self, q, kv_grids, ref_uv_01_list, uv_hat_01, q_pad=None):
        raw_cum = None
        for blk in self.cross_blocks:
            q, raw = blk(q, kv_grids, ref_uv_01_list, uv_hat_01, q_pad_mask=q_pad)
            raw_cum = raw if raw_cum is None else raw_cum + raw
        return raw_cum

    @staticmethod
    def _emb_to_grid(emb_b1d, Hg, Wg):
        """(B, 1, D) → (B, D, Hg, Wg) broadcast."""
        return emb_b1d.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, Hg, Wg)

    def _vfl_emb(self, vfl):
        """vfl: (B,) — virtual focal length (float).
        Returns (B, 1, d) embedding via log + sin/cos basis + 2-layer MLP.
        """
        # log(vfl), normalised by log(64) so range ~[-2, 4] for typical
        # vfl ∈ [10, 4000].
        x = torch.log(vfl.clamp(min=1e-3)).unsqueeze(-1) / 4.0   # (B, 1)
        # sin/cos basis at multiple frequencies
        ang = x * self.vfl_freqs                                # (B, n_freq)
        feat = torch.cat([x, torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, 2*nf+1)
        return self.vfl_mlp(feat).unsqueeze(1)                  # (B, 1, d)

    def forward(self, patch_A, uvd_A, patch_B, uvd_B,
                pose_AB_6dof, pose_BA_6dof,
                uv_B_hat_of_A, uv_A_hat_of_B,
                pad_A=None, pad_B=None,
                uvd_A_full=None, uvd_B_full=None,
                pad_A_full=None, pad_B_full=None,
                patch_M=None, uvd_M=None, pad_M=None,
                uvd_M_full=None, pad_M_full=None,
                pose_AM_6dof=None,
                uv_M_hat_of_A=None, uv_M_hat_of_B=None,
                # M2 (quad mode) — second mid-frame
                patch_M2=None, uvd_M2=None, pad_M2=None,
                uvd_M2_full=None, pad_M2_full=None,
                pose_AM2_6dof=None,
                uv_M2_hat_of_A=None, uv_M2_hat_of_B=None,
                # M3 (quint mode) — third mid-frame
                patch_M3=None, uvd_M3=None, pad_M3=None,
                uvd_M3_full=None, pad_M3_full=None,
                pose_AM3_6dof=None,
                uv_M3_hat_of_A=None, uv_M3_hat_of_B=None,
                modality_A='mm', modality_B='mm',
                feats_A=None, feats_B=None,
                vfl=None,
                **_ignored):
        """Pair / triplet (M) / quad (M+M2) modes via optional kwargs.

        modality_A / modality_B: 'mm' (camera+LiDAR), 'cam' (LiDAR-zeroed),
        'lidar' (image-zeroed). Calib uses ('cam', 'lidar') so frame A is
        a camera-only frame_token and frame B is a LiDAR-only frame_token —
        same encoder, same downstream cross-attention path.
        """
        ft_A, _, ptf_A = self.encoder(patch_A, uvd_A, pad_A,
                                       uvd_A_full, pad_A_full, modality=modality_A)
        ft_B, _, ptf_B = self.encoder(patch_B, uvd_B, pad_B,
                                       uvd_B_full, pad_B_full, modality=modality_B)
        has_M = patch_M is not None
        has_M2 = patch_M2 is not None
        if has_M:
            ft_M, _, ptf_M = self.encoder(patch_M, uvd_M, pad_M, uvd_M_full, pad_M_full)
        if has_M2:
            ft_M2, _, ptf_M2 = self.encoder(patch_M2, uvd_M2, pad_M2, uvd_M2_full, pad_M2_full)
        has_M3 = patch_M3 is not None
        if has_M3:
            ft_M3, _, ptf_M3 = self.encoder(patch_M3, uvd_M3, pad_M3, uvd_M3_full, pad_M3_full)

        Hg, Wg = ft_A.shape[2], ft_A.shape[3]

        # vfl conditioning: same vfl per sample → broadcast as a global
        # additive bias to every frame_token. Last-layer zero init means this
        # is a no-op until the model learns to use it.
        if vfl is not None:
            vfl_grid = self._emb_to_grid(self._vfl_emb(vfl), Hg, Wg)
            ft_A = ft_A + vfl_grid
            ft_B = ft_B + vfl_grid
            if has_M:  ft_M  = ft_M  + vfl_grid
            if has_M2: ft_M2 = ft_M2 + vfl_grid
            if has_M3: ft_M3 = ft_M3 + vfl_grid
        emb_AB = self.pose_mlp(pose_AB_6dof).unsqueeze(1)
        emb_BA = self.pose_mlp(pose_BA_6dof).unsqueeze(1)

        kv_B_for_A = ft_B + self._emb_to_grid(emb_AB, Hg, Wg)
        kv_A_for_B = ft_A + self._emb_to_grid(emb_BA, Hg, Wg)
        if has_M:
            pose_MB_6dof = _compose_pose_MB(pose_AB_6dof, pose_AM_6dof)
            pose_BM_6dof = _invert_6dof(pose_MB_6dof)
            emb_AM = self.pose_mlp(pose_AM_6dof).unsqueeze(1)
            emb_BM = self.pose_mlp(pose_BM_6dof).unsqueeze(1)
            kv_M_for_A = ft_M + self._emb_to_grid(emb_AM, Hg, Wg)
            kv_M_for_B = ft_M + self._emb_to_grid(emb_BM, Hg, Wg)
        if has_M2:
            pose_M2B_6dof = _compose_pose_MB(pose_AB_6dof, pose_AM2_6dof)
            pose_BM2_6dof = _invert_6dof(pose_M2B_6dof)
            emb_AM2 = self.pose_mlp(pose_AM2_6dof).unsqueeze(1)
            emb_BM2 = self.pose_mlp(pose_BM2_6dof).unsqueeze(1)
            kv_M2_for_A = ft_M2 + self._emb_to_grid(emb_AM2, Hg, Wg)
            kv_M2_for_B = ft_M2 + self._emb_to_grid(emb_BM2, Hg, Wg)
        if has_M3:
            pose_M3B_6dof = _compose_pose_MB(pose_AB_6dof, pose_AM3_6dof)
            pose_BM3_6dof = _invert_6dof(pose_M3B_6dof)
            emb_AM3 = self.pose_mlp(pose_AM3_6dof).unsqueeze(1)
            emb_BM3 = self.pose_mlp(pose_BM3_6dof).unsqueeze(1)
            kv_M3_for_A = ft_M3 + self._emb_to_grid(emb_AM3, Hg, Wg)
            kv_M3_for_B = ft_M3 + self._emb_to_grid(emb_BM3, Hg, Wg)

        q_A = self._build_query(ft_A, uvd_A, emb_AB,
                                  feats_anchor=feats_A, pt_feat_anchor=ptf_A)
        kv_AB  = [kv_B_for_A]
        ref_AB = [uv_B_hat_of_A / self.img_size]
        if has_M:
            kv_AB.append(kv_M_for_A)
            ref_AB.append(uv_M_hat_of_A / self.img_size)
        if has_M2:
            kv_AB.append(kv_M2_for_A)
            ref_AB.append(uv_M2_hat_of_A / self.img_size)
        if has_M3:
            kv_AB.append(kv_M3_for_A)
            ref_AB.append(uv_M3_hat_of_A / self.img_size)
        raw_AtoB = self._multi_forward(
            q_A, kv_AB, ref_AB,
            uv_B_hat_of_A / self.img_size, q_pad=pad_A)

        q_B = self._build_query(ft_B, uvd_B, emb_BA,
                                  feats_anchor=feats_B, pt_feat_anchor=ptf_B)
        kv_BA  = [kv_A_for_B]
        ref_BA = [uv_A_hat_of_B / self.img_size]
        if has_M:
            kv_BA.append(kv_M_for_B)
            ref_BA.append(uv_M_hat_of_B / self.img_size)
        if has_M2:
            kv_BA.append(kv_M2_for_B)
            ref_BA.append(uv_M2_hat_of_B / self.img_size)
        if has_M3:
            kv_BA.append(kv_M3_for_B)
            ref_BA.append(uv_M3_hat_of_B / self.img_size)
        raw_BtoA = self._multi_forward(
            q_B, kv_BA, ref_BA,
            uv_A_hat_of_B / self.img_size, q_pad=pad_B)

        clamp_fn = clamp_params if self._out_dim == 5 else clamp_params_uvd
        return (clamp_fn(raw_AtoB, self.img_size),
                clamp_fn(raw_BtoA, self.img_size))

    def forward_n(self, patches, uvd, pad, uvd_full, pad_full,
                  pose_hat_6dof, uv_hat, mix_mode='mix'):
        """Stacked-tensor N-frame forward (same signature as CalibNetMultiFrame)."""
        B, N = patches.shape[0], patches.shape[1]
        assert N - 1 <= self.max_kv_frames, (
            f'N-1={N-1} > max_kv_frames={self.max_kv_frames}')

        ft_list = []
        ptf_list = []
        for k in range(N):
            ft_k, _, ptf_k = self.encoder(
                patches[:, k], uvd[:, k], pad[:, k],
                uvd_full[:, k], pad_full[:, k])
            ft_list.append(ft_k)
            ptf_list.append(ptf_k)
        Hg, Wg = ft_list[0].shape[2], ft_list[0].shape[3]
        P = uvd.shape[2]

        clamp_fn = clamp_params if self._out_dim == 5 else clamp_params_uvd
        raws = torch.zeros(B, N, N, P, self._out_dim,
                           device=patches.device, dtype=ft_list[0].dtype)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                emb_ij = self.pose_mlp(pose_hat_6dof[:, i, j]).unsqueeze(1)
                q = self._build_query(ft_list[i], uvd[:, i], emb_ij,
                                        pt_feat_anchor=ptf_list[i])
                if mix_mode == 'pair':
                    kv_order = [j]
                else:
                    kv_order = [j] + [k for k in range(N) if k != i and k != j]
                kv_grids, ref_list = [], []
                for k in kv_order:
                    emb_ik = self.pose_mlp(pose_hat_6dof[:, i, k]).unsqueeze(1)
                    kv_grids.append(ft_list[k] + self._emb_to_grid(emb_ik, Hg, Wg))
                    ref_list.append(uv_hat[:, i, k] / self.img_size)
                raw_ij = self._multi_forward(
                    q, kv_grids, ref_list,
                    uv_hat[:, i, j] / self.img_size, q_pad=pad[:, i])
                raws[:, i, j] = clamp_fn(raw_ij, self.img_size)
        return raws


if __name__ == '__main__':
    # Smoke test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, P, H, W = 2, 64, 128, 128
    img = torch.randn(B, 3, H, W, device=device)
    uvd = torch.cat([torch.rand(B, P, 2, device=device) * H,
                      torch.rand(B, P, 1, device=device) * 30 + 5], dim=-1)
    pad = torch.zeros(B, P, dtype=torch.bool, device=device)
    pose_AB = torch.randn(B, 6, device=device) * 0.1
    pose_BA = -pose_AB
    uv_hat = torch.rand(B, P, 2, device=device) * H

    m = CalibNetUnifiedFrame(img_size=H, max_kv_frames=1).to(device)
    raw_AB, raw_BA = m(img, uvd, img, uvd, pose_AB, pose_BA, uv_hat, uv_hat,
                        pad_A=pad, pad_B=pad,
                        uvd_A_full=uvd, uvd_B_full=uvd,
                        pad_A_full=pad, pad_B_full=pad)
    print(f'pair: AB {tuple(raw_AB.shape)} BA {tuple(raw_BA.shape)}')

    # multi-frame
    m3 = CalibNetUnifiedFrame(img_size=H, max_kv_frames=2).to(device)
    pose_AM = torch.randn(B, 6, device=device) * 0.1
    raw_AB, raw_BA = m3(img, uvd, img, uvd, pose_AB, pose_BA, uv_hat, uv_hat,
                         pad_A=pad, pad_B=pad,
                         uvd_A_full=uvd, uvd_B_full=uvd,
                         pad_A_full=pad, pad_B_full=pad,
                         patch_M=img, uvd_M=uvd, pad_M=pad,
                         uvd_M_full=uvd, pad_M_full=pad,
                         pose_AM_6dof=pose_AM,
                         uv_M_hat_of_A=uv_hat, uv_M_hat_of_B=uv_hat)
    print(f'triplet: AB {tuple(raw_AB.shape)} BA {tuple(raw_BA.shape)}')

    # forward_n
    N = 3
    patches_n = img.unsqueeze(1).expand(-1, N, -1, -1, -1).contiguous()
    uvd_n = uvd.unsqueeze(1).expand(-1, N, -1, -1).contiguous()
    pad_n = pad.unsqueeze(1).expand(-1, N, -1).contiguous()
    pose_hat = torch.zeros(B, N, N, 6, device=device)
    uv_hat_n = uv_hat.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1, -1).contiguous()
    raws = m3.forward_n(patches_n, uvd_n, pad_n, uvd_n, pad_n,
                         pose_hat, uv_hat_n, mix_mode='mix')
    print(f'forward_n: {tuple(raws.shape)}')
    raws.sum().backward()
    print('backward OK')
