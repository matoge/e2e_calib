"""CalibNetDepth + cross-frame forward path (ps_v12).

Same backbone & blocks as CalibNetDepth (CNN, PointMLP, PoseEmb, the
CrossAttentionBlockCov per layer). Difference: forward_pair() takes a
SECOND frame and adds image-B + lidar-B to the per-layer KV.

VCPE is the absolute-frame tag (single MLP):
  VCPE_A = MLP([SE3=0,        log(vfp)])
  VCPE_B = MLP([rel_AB_6dof,  log(vfp)])
Query is a frame-A point asking "where do I land in B" → tagged with VCPE_B
(target frame). Image_A / lidar_A KV gets VCPE_A, Image_B / lidar_B KV
gets VCPE_B. Cross-attn computes relative geometry from tag-difference.
"""
from __future__ import annotations

import torch
from models.model_depth import CalibNetDepth
from models.model_cov import clamp_params


class CalibNetDepthPair(CalibNetDepth):
    """Adds forward_pair() for cross-frame training.

    Calib (single-frame) forward() is unchanged — instances trained as
    ps_v11 still work without modification.
    """

    def _vcpe(self, se3_6dof: torch.Tensor, vfp: torch.Tensor) -> torch.Tensor:
        """Build VCPE = MLP([SE3 6-DoF, log(vfp)]). (B, D)."""
        log_vfp = torch.log(vfp.clamp(min=1.0)).unsqueeze(-1)
        return self.pose_emb(torch.cat([se3_6dof, log_vfp], dim=-1))

    def _point_q(self, uvd: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Per-point feature = PointMLP(uvd_norm) + Frustum (if enabled)."""
        uv_01 = uvd[..., :2] / self.img_size
        uvd_norm = torch.cat([uv_01, uvd[..., 2:3]], dim=-1)
        q = self.point_mlp(uvd_norm)
        if self.frustum_enc is not None:
            q = q + self.frustum_enc(uvd, query_pad_mask=pad)
        return q

    def forward_pair(self,
                     image_A: torch.Tensor, image_B: torch.Tensor,
                     uvd_A: torch.Tensor,   uvd_B: torch.Tensor,
                     uv_B_naive: torch.Tensor,
                     pose_AB_6dof: torch.Tensor,
                     vfp: torch.Tensor,
                     pad_A: torch.Tensor = None,
                     pad_B: torch.Tensor = None,
                     query_pad: torch.Tensor = None) -> torch.Tensor:
        """Cross-frame forward.

        image_A, image_B: (B, C, H, W)
        uvd_A: (B, N_A, 3)  — A's points in A's crop  (= lidar bank A)
        uvd_B: (B, N_B, 3)  — B's points in B's crop  (= lidar bank B)
        uv_B_naive: (B, N_A, 2)  — A's points projected to B via current
            pose estimate (= per-point Q's "starting position in B" hint)
        pose_AB_6dof: (B, 6)  — the (perturbed) virtual-cam SE3 from A to B
        vfp: (B,)  — focal pixel scalar (assumed shared A=B per pair)
        pad_A: (B, N_A) bool  — True = padded entry to skip
        pad_B: (B, N_B) bool  — True = padded entry to skip
        query_pad: (B, N_A) bool  — query mask (defaults to pad_A)

        Returns params (B, N_A, 5)  — predicted (Δu, Δv, log_σx, log_σy, ρ)
        in B's local crop. Target = uv_B_gt - uv_B_naive.
        """
        if query_pad is None:
            query_pad = pad_A
        B = image_A.size(0)
        D = image_A.device

        # 1) Two CNN passes (separate Streams — same weights)
        coarse_A, fine_A = self.cnn(image_A)
        coarse_B, fine_B = self.cnn(image_B)

        # 2) VCPE per frame (absolute frame tag)
        zero_se3 = torch.zeros(B, 6, device=D, dtype=pose_AB_6dof.dtype)
        vcpe_A = self._vcpe(zero_se3,         vfp)        # (B, D)
        vcpe_B = self._vcpe(pose_AB_6dof,     vfp)        # (B, D)
        pe_A_2d = vcpe_A.unsqueeze(-1).unsqueeze(-1)      # (B, D, 1, 1)
        pe_B_2d = vcpe_B.unsqueeze(-1).unsqueeze(-1)
        coarse_A = coarse_A + pe_A_2d
        fine_A   = fine_A   + pe_A_2d
        coarse_B = coarse_B + pe_B_2d
        fine_B   = fine_B   + pe_B_2d

        # 3) Per-frame point bank features (= lidar KV concat)
        kvL_A = self._point_q(uvd_A, pad_A) + vcpe_A.unsqueeze(1)
        kvL_B = self._point_q(uvd_B, pad_B) + vcpe_B.unsqueeze(1)
        extra_kv = torch.cat([kvL_A, kvL_B], dim=1)
        if pad_A is not None and pad_B is not None:
            extra_kv_mask = torch.cat([pad_A, pad_B], dim=1)
        else:
            extra_kv_mask = None

        # 4) Q = A's points asking "where in B" — input position = uv_B_naive
        d3 = uvd_A[..., 2:3]
        uv_q = uv_B_naive / self.img_size                  # (B, N_A, 2) in [0,1]
        q_in = torch.cat([uv_q, d3], dim=-1)
        q = self.point_mlp(q_in)
        if self.frustum_enc is not None:
            # frustum context = lidar bank of TARGET frame (B), since query is in B
            q = q + self.frustum_enc(uvd_B, query_pad_mask=pad_B,
                                     full_uvd=uvd_B if False else None)
        q = q + vcpe_B.unsqueeze(1)                         # query in target frame

        # 5) Layer-wise cross-attn — concat A-side and B-side image tokens along W
        #    so each block sees BOTH frames' spatial maps in one flat KV.
        coarse_pair = torch.cat([coarse_A, coarse_B], dim=-1)   # (B, D, H, 2W)
        fine_pair   = torch.cat([fine_A,   fine_B],   dim=-1)
        feats_seq = [coarse_pair, fine_pair][:max(self.n_layers, 2)]

        q, raw_cum = self._block(self.cross_coarse, q, feats_seq[0], uv_q,
                                  query_pad,
                                  extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)
        for i in range(1, min(self.n_layers, len(feats_seq))):
            uv_i = (uv_q + raw_cum[..., :2]).clamp(0, 1)
            q = self.point_mlp(torch.cat([uv_i, d3], dim=-1)) + q
            block = self.cross_fine if i == 1 else self.cross_refine
            q, raw_i = self._block(block, q, feats_seq[i], uv_i,
                                    query_pad,
                                    extra_kv=extra_kv, extra_kv_mask=extra_kv_mask)
            raw_cum = raw_cum + raw_i

        return clamp_params(raw_cum, self.img_size)            # (B, N_A, 5)
