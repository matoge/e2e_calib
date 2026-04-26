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


class FrameTokenEncoder(nn.Module):
    """Per-frame: image + sparse pts → unified (B, D, Hg, Wg) frame_token."""

    def __init__(self, d=D_DIM, in_channels=3, use_convnext=False,
                 r_uv=4.0, r_d=2.0, k_nb=8, use_frustum=True,
                 n_intra_layers=2, img_size=128):
        super().__init__()
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
        self.img_size = img_size

    def forward(self, image, uvd, pad_mask=None,
                uvd_full=None, pad_full=None):
        coarse, _ = self.cnn(image)                                  # (B, D, Hg, Wg)
        B, D, Hg, Wg = coarse.shape

        uv_01 = uvd[..., :2] / self.img_size
        uvd_n = torch.cat([uv_01, uvd[..., 2:3]], dim=-1)
        pt_feat = self.point_mlp(uvd_n)
        if self.frustum_enc is not None:
            pt_feat = pt_feat + self.frustum_enc(
                uvd, full_uvd=uvd_full,
                full_pad_mask=pad_full, query_pad_mask=pad_mask)

        pt_grid, mask = scatter_pt_to_grid(
            pt_feat, uvd[..., :2], pad_mask, Hg, Wg, self.img_size)

        fused = self.fuse(torch.cat([coarse, pt_grid, mask], dim=1))

        x = fused.flatten(2).permute(0, 2, 1)                        # (B, Hg*Wg, D)
        for blk in self.intra:
            x = blk(x)
        frame_token = x.permute(0, 2, 1).reshape(B, D, Hg, Wg).contiguous()
        return frame_token, mask


class UnifiedCrossBlock(nn.Module):
    """Cross-frame deformable attention over multi-level frame_token KV."""

    def __init__(self, d=D_DIM, n_heads=4, n_points=4,
                 max_levels=2, out_dim=5):
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
        with torch.no_grad():
            if out_dim == 5:
                self.proj.bias[2] = 0.69
                self.proj.bias[3] = 0.69
            elif out_dim == 7:
                self.proj.bias[3] = 0.69
                self.proj.bias[4] = 0.69
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
                 r_uv=4.0, r_d=2.0, k_nb=8, use_frustum=True, out_dim=5):
        super().__init__()
        self.d = d
        self.img_size = img_size
        self.max_kv_frames = max_kv_frames
        self._out_dim = out_dim
        self.encoder = FrameTokenEncoder(
            d, in_channels=in_channels, use_convnext=use_convnext,
            r_uv=r_uv, r_d=r_d, k_nb=k_nb, use_frustum=use_frustum,
            n_intra_layers=n_intra_layers, img_size=img_size)
        self.point_mlp_q = PointMLP3(d)
        self.pose_mlp = PoseMLP(d)
        self.cross_blocks = nn.ModuleList([
            UnifiedCrossBlock(d, n_heads=n_heads, n_points=deform_n_points,
                              max_levels=max_kv_frames, out_dim=out_dim)
            for _ in range(n_cross_layers)
        ])

    def _build_query(self, frame_token_anchor, uvd_anchor, pose_emb_to_tgt):
        """Q = bilinear(frame_token_A, uv_A) + PointMLP_q(uvd_A) + pose_emb_AB.

        The bilinear sample carries A-frame's local image+LiDAR context at
        the point's projection. PointMLP encodes the 3D position/depth.
        Pose embedding bridges to the target frame. Same construction shape
        in pair / triplet — only `pose_emb_to_tgt` changes per (Q-frame, T-frame).
        """
        uv_01 = uvd_anchor[..., :2] / self.img_size
        ctx = sample_grid_at_uv(frame_token_anchor, uv_01)
        uvd_n = torch.cat([uv_01, uvd_anchor[..., 2:3]], dim=-1)
        ptq = self.point_mlp_q(uvd_n)
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
                **_ignored):
        """Pair mode: M args = None. Multi-frame: M args provided.

        `_ignored` swallows leftover kwargs (e.g. `per_frame_emb`) the legacy
        trainer might still pass — the unified design always uses per-frame
        absolute embedding on KV grids, so the flag is a no-op here.
        """
        ft_A, _ = self.encoder(patch_A, uvd_A, pad_A, uvd_A_full, pad_A_full)
        ft_B, _ = self.encoder(patch_B, uvd_B, pad_B, uvd_B_full, pad_B_full)
        has_M = patch_M is not None
        if has_M:
            ft_M, _ = self.encoder(patch_M, uvd_M, pad_M, uvd_M_full, pad_M_full)

        Hg, Wg = ft_A.shape[2], ft_A.shape[3]
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

        q_A = self._build_query(ft_A, uvd_A, emb_AB)
        if has_M:
            kv_AB  = [kv_B_for_A, kv_M_for_A]
            ref_AB = [uv_B_hat_of_A / self.img_size,
                      uv_M_hat_of_A / self.img_size]
        else:
            kv_AB  = [kv_B_for_A]
            ref_AB = [uv_B_hat_of_A / self.img_size]
        raw_AtoB = self._multi_forward(
            q_A, kv_AB, ref_AB,
            uv_B_hat_of_A / self.img_size, q_pad=pad_A)

        q_B = self._build_query(ft_B, uvd_B, emb_BA)
        if has_M:
            kv_BA  = [kv_A_for_B, kv_M_for_B]
            ref_BA = [uv_A_hat_of_B / self.img_size,
                      uv_M_hat_of_B / self.img_size]
        else:
            kv_BA  = [kv_A_for_B]
            ref_BA = [uv_A_hat_of_B / self.img_size]
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
        for k in range(N):
            ft_k, _ = self.encoder(
                patches[:, k], uvd[:, k], pad[:, k],
                uvd_full[:, k], pad_full[:, k])
            ft_list.append(ft_k)
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
                q = self._build_query(ft_list[i], uvd[:, i], emb_ij)
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
