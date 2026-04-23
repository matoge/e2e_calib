"""CalibNetCrossFrame — 2-frame cross-attention residual net.

Architecture:

  frame A                                   frame B
     │                                         │
  [intra: concat(pt, img_coarse)            [intra: same, shared weights]
   → self-attn stack → split back                  │
   into pt_A and img_A tokens]                     │
     │                                             │
     ▼                                             ▼
  pt_A_tok , img_A_tok                    pt_B_tok , img_B_tok
  (A-frame)                               (B-frame)

  A→B direction (symmetric swap gives B→A):

    pose_emb_AB = PoseMLP(T_AB_hat)   # bridges A-frame → B-frame
    Q_A  = pt_A_tok + pose_emb_AB
    KV_B = concat(pt_B_tok, img_B_tok)
    ─ cross-attn + self-attn + FFN ─►  Δ_AtoB  (per-point 2D gaussian in B-patch coords)

Output head takes [token, uv_hat_01] so the predicted Δ is a correction on the
hypothesis projection. Dual-direction loss: A→B + B→A.
"""
import torch
import torch.nn as nn

from models.model import CNNBackbone, ConvNeXtBackbone, D as D_DIM
from models.model_cov import clamp_params
from models.model_depth import PointMLP3, FrustumLocalEncoder


# ─── self-attention block (intra-frame fusion over concat(pt, img)) ──────────

class SelfAttnFusionBlock(nn.Module):
    """Standard pre-norm self-attn + FFN block.  No output head."""

    def __init__(self, d: int = D_DIM, n_heads: int = 4):
        super().__init__()
        self.norm_sa  = nn.LayerNorm(d)
        self.sa       = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x, key_padding_mask=None):
        h, _ = self.sa(self.norm_sa(x), self.norm_sa(x), self.norm_sa(x),
                        key_padding_mask=key_padding_mask)
        x = x + self.drop(h)
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ─── pose embedding (A-frame → B-frame token-space shift) ─────────────────────

class PoseMLP(nn.Module):
    """Encodes 6-DoF relative pose (ypr_deg, t_m) → D-dim token shift."""

    def __init__(self, d_out: int = D_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(),
            nn.Linear(64, d_out), nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, ypr_t: torch.Tensor) -> torch.Tensor:
        # roughly normalise: ypr ±10° → /5 , t ±5m → /2
        ypr = ypr_t[..., :3] / 5.0
        t   = ypr_t[..., 3:] / 2.0
        return self.net(torch.cat([ypr, t], dim=-1))


# ─── cross-frame block ────────────────────────────────────────────────────────

class CrossFrameBlock(nn.Module):
    """Cross-attention A-query → B-KV (or vice versa).

    Q : (B, N_Q, D)    — pt tokens of the query frame, already pose-shifted
    KV: (B, N_KV, D)   — concat(pt_tokens_other, img_tokens_other)
    Output: (B, N_Q, 5) — (Δu, Δv, log σu, log σv, ρ_raw)
    """

    def __init__(self, d: int = D_DIM, n_heads: int = 4):
        super().__init__()
        self.norm_q   = nn.LayerNorm(d)
        self.norm_kv  = nn.LayerNorm(d)
        self.norm_sa  = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.self_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d),
        )
        self.drop = nn.Dropout(0.1)
        # output head: [token ‖ uv_hat_01] → 5
        self.proj = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69   # log σu  ≈ log 2 px
            self.proj.bias[3] = 0.69   # log σv

    def forward(self, q, kv, uv_hat_01, q_pad_mask=None, kv_pad_mask=None):
        ca, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                                 key_padding_mask=kv_pad_mask)
        q = q + self.drop(ca)
        sa, _ = self.self_attn(self.norm_sa(q), self.norm_sa(q), self.norm_sa(q),
                                key_padding_mask=q_pad_mask)
        q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_hat_01], dim=-1))
        return q, raw


# ─── full model ───────────────────────────────────────────────────────────────

class CalibNetCrossFrame(nn.Module):
    def __init__(self, d: int = D_DIM, img_size: int = 64, in_channels: int = 3,
                 use_convnext: bool = True, use_frustum: bool = True,
                 r_uv: float = 8.0, r_d: float = 0.004, k_nb: int = 8,
                 n_intra_layers: int = 2, n_cross_layers: int = 1):
        super().__init__()
        self.img_size = img_size
        self.d = d

        # shared per-frame encoder
        self.cnn = (ConvNeXtBackbone(d, in_channels=in_channels)
                    if use_convnext else CNNBackbone(d, in_channels=in_channels))
        self.point_mlp   = PointMLP3(d)
        self.frustum_enc = FrustumLocalEncoder(d, r_uv=r_uv, r_d=r_d, k=k_nb) if use_frustum else None

        # intra-frame fusion: self-attn over concat(pt, img_coarse), shared weights across A/B
        self.intra_blocks = nn.ModuleList(
            [SelfAttnFusionBlock(d) for _ in range(n_intra_layers)]
        )

        # pose embedding (shared across A→B and B→A via inverse input)
        self.pose_mlp = PoseMLP(d)

        # cross-frame stack
        self.cross_blocks = nn.ModuleList(
            [CrossFrameBlock(d) for _ in range(n_cross_layers)]
        )

    # ------------------------------------------------------------------ helpers

    def _encode_frame(self, image, uvd, pad_mask=None):
        """Per-frame intra encoding.

        Returns:
            pt_tok  : (B, N_pt, D)   — point-derived tokens after fusion
            img_tok : (B, N_img, D)  — image-derived tokens after fusion
        """
        coarse, _ = self.cnn(image)                         # coarse: (B, D, H/8, W/8)
        B, D_, Hc, Wc = coarse.shape
        img_tok = coarse.flatten(2).permute(0, 2, 1)        # (B, H*W, D)  PE already baked in

        uv_01 = uvd[..., :2] / self.img_size
        uvd_n = torch.cat([uv_01, uvd[..., 2:3]], dim=-1)
        pt_tok = self.point_mlp(uvd_n)
        if self.frustum_enc is not None:
            pt_tok = pt_tok + self.frustum_enc(uvd, pad_mask=pad_mask)

        # Concatenate: [pt tokens ‖ img tokens]
        N_pt  = pt_tok.shape[1]
        N_img = img_tok.shape[1]
        combined = torch.cat([pt_tok, img_tok], dim=1)       # (B, N_pt+N_img, D)

        # Padding mask: pad_mask for pt side, zeros for img tokens
        if pad_mask is not None:
            img_pad  = torch.zeros(B, N_img, dtype=torch.bool, device=pad_mask.device)
            full_pad = torch.cat([pad_mask, img_pad], dim=1)
        else:
            full_pad = None

        # Self-attn stack over concatenated tokens (modality fusion)
        for blk in self.intra_blocks:
            combined = blk(combined, key_padding_mask=full_pad)

        # Split back
        pt_tok_out  = combined[:, :N_pt]
        img_tok_out = combined[:, N_pt:]
        return pt_tok_out, img_tok_out

    def _cross_forward(self, q_in, kv, uv_hat_01, q_pad=None, kv_pad=None):
        raw_cum = None
        for blk in self.cross_blocks:
            q_in, raw = blk(q_in, kv, uv_hat_01, q_pad_mask=q_pad, kv_pad_mask=kv_pad)
            raw_cum = raw if raw_cum is None else raw_cum + raw
        return raw_cum

    # --------------------------------------------------------------- forward(s)

    def forward(self, patch_A, uvd_A, patch_B, uvd_B,
                pose_AB_6dof, pose_BA_6dof,
                uv_B_hat_of_A, uv_A_hat_of_B,
                pad_A=None, pad_B=None):
        """Returns (raw_AtoB (B,N_A,5), raw_BtoA (B,N_B,5))."""
        # 1. intra-frame encode (shared weights)
        pt_A, img_A = self._encode_frame(patch_A, uvd_A, pad_A)
        pt_B, img_B = self._encode_frame(patch_B, uvd_B, pad_B)

        # 2. pose embeddings
        pose_emb_AB = self.pose_mlp(pose_AB_6dof).unsqueeze(1)     # (B, 1, D)
        pose_emb_BA = self.pose_mlp(pose_BA_6dof).unsqueeze(1)

        # 3. A → B
        q_A = pt_A + pose_emb_AB
        kv_AB = torch.cat([pt_B, img_B], dim=1)
        kv_pad_AB = self._make_kv_pad(pad_B, img_B.shape[1])
        raw_AtoB = self._cross_forward(
            q_A, kv_AB, uv_B_hat_of_A / self.img_size,
            q_pad=pad_A, kv_pad=kv_pad_AB)

        # 4. B → A
        q_B = pt_B + pose_emb_BA
        kv_BA = torch.cat([pt_A, img_A], dim=1)
        kv_pad_BA = self._make_kv_pad(pad_A, img_A.shape[1])
        raw_BtoA = self._cross_forward(
            q_B, kv_BA, uv_A_hat_of_B / self.img_size,
            q_pad=pad_B, kv_pad=kv_pad_BA)

        return (clamp_params(raw_AtoB, self.img_size),
                clamp_params(raw_BtoA, self.img_size))

    def _make_kv_pad(self, pt_pad, n_img_tokens):
        if pt_pad is None:
            return None
        img_pad = torch.zeros(pt_pad.shape[0], n_img_tokens,
                               dtype=torch.bool, device=pt_pad.device)
        return torch.cat([pt_pad, img_pad], dim=1)


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from datasets.pandaset_pair import PandaSetCrossFrameDataset

    ds = PandaSetCrossFrameDataset(
        scene_root='/mnt/mininas/datasets/pandaset/015',
        img_size=64, max_points=64,
    )
    s = ds[0]
    batch = {k: v.unsqueeze(0) for k, v in s.items() if torch.is_tensor(v)}

    model = CalibNetCrossFrame(img_size=64)
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M')

    raw_AB, raw_BA = model(
        patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
        patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
        pose_AB_6dof=batch['pose_AB_6dof'], pose_BA_6dof=batch['pose_BA_6dof'],
        uv_B_hat_of_A=batch['uv_B_hat_of_A'], uv_A_hat_of_B=batch['uv_A_hat_of_B'],
        pad_A=batch['pad_A'], pad_B=batch['pad_B'],
    )
    print('raw_AB:', raw_AB.shape, '  |Δ| mean:', raw_AB[..., :2].abs().mean().item())
    print('raw_BA:', raw_BA.shape)
