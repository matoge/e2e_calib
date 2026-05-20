"""
model_cov.py  –  CalibNet with per-point covariance output

Output per point: (tx, ty, log_sx, log_sy, rho_raw)
  tx, ty    : predicted offset (pixel space)
  log_sx    : log(sigma_x), ensures sigma_x > 0
  log_sy    : log(sigma_y)
  rho_raw   : pre-tanh correlation, ensures |rho| < 1

Parameterisation prevents:
  - variance collapse: clamp log_s >= log(MIN_SIGMA)
  - variance explosion: clamp log_s <= log(MAX_SIGMA)
  - invalid correlation: rho = tanh(rho_raw) * 0.99
"""
import torch
import torch.nn as nn
from models.model import CNNBackbone, PointMLP, D

MIN_SIGMA = 0.7   # px  — floor prevents NLL collapsing to -inf; raised from 0.3
                  #       after v13-v16 showed σ overfits to train-time err ~1px
                  #       leaving val z=err/σ ≈ 6-8× (overconfidence). A higher
                  #       floor costs a little train NLL but keeps σ BA-usable.
MAX_SIGMA = 30.0  # px  — ceiling prevents explosion


class CrossAttentionBlockCov(nn.Module):
    """Cross-attn block. Order: [kv_self_attn →] cross → [self_first? sa :] sa → FFN.
    kv_self_attn=True enriches image tokens with self-attn before cross-attn.
    """
    def __init__(self, d: int = D, n_heads: int = 4, kv_self_attn: bool = False,
                 cross_temp: float = 1.0):
        super().__init__()
        self.cross_attn  = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q      = nn.LayerNorm(d)
        self.norm_kv     = nn.LayerNorm(d)
        self.self_attn   = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self   = nn.LayerNorm(d)
        self.drop        = nn.Dropout(0.1)
        self.ffn         = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d),
        )
        self.norm_ffn    = nn.LayerNorm(d)
        self.proj        = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69
        # optional image encoder self-attn
        if kv_self_attn:
            self.kv_sa      = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            self.norm_kv_sa = nn.LayerNorm(d)
        self._kv_self_attn = kv_self_attn
        self._cross_temp   = cross_temp

    def forward(self, q, feat, uv_01, key_padding_mask=None, self_first=False,
                extra_kv=None, extra_kv_mask=None):
        """extra_kv: (B, M, D) — appended to flattened image KV.
        extra_kv_mask: (B, M) bool, True = padded position to ignore.
        Used for the lidar-token-bank in the per-modality KV concat path."""
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)   # (B, H*W, D)
        HW = H * W

        # image self-attn (optional, image tokens only — does not touch extra_kv)
        if self._kv_self_attn:
            img_sa, _ = self.kv_sa(self.norm_kv_sa(kv), self.norm_kv_sa(kv), self.norm_kv_sa(kv))
            kv = kv + self.drop(img_sa)

        # append extra (e.g. lidar) KV tokens
        if extra_kv is not None:
            kv = torch.cat([kv, extra_kv], dim=1)   # (B, HW + M, D)
            if extra_kv_mask is not None:
                cam_mask = torch.zeros(B, HW, dtype=torch.bool, device=kv.device)
                cross_kv_mask = torch.cat([cam_mask, extra_kv_mask], dim=1)
            else:
                cross_kv_mask = None
        else:
            cross_kv_mask = None

        # temperature scaling: divide Q by T → softmax(QK^T / (T·√d_k))
        nq = self.norm_q(q) / self._cross_temp

        if self_first:
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                   key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
            ca, _ = self.cross_attn(nq, self.norm_kv(kv), self.norm_kv(kv),
                                    key_padding_mask=cross_kv_mask)
            q = q + self.drop(ca)
        else:
            ca, _ = self.cross_attn(nq, self.norm_kv(kv), self.norm_kv(kv),
                                    key_padding_mask=cross_kv_mask)
            q = q + self.drop(ca)
            sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                   key_padding_mask=key_padding_mask)
            q = q + self.drop(sa)
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))   # (B,N,5)
        return q, raw


class TransformerDecoderBlock(nn.Module):
    """Standard Transformer Decoder block:
    image self-attn → point self-attn → cross-attn → FFN
    """
    def __init__(self, d: int = D, n_heads: int = 4):
        super().__init__()
        # encoder (image) self-attn
        self.img_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_img      = nn.LayerNorm(d)
        # decoder (point) self-attn
        self.self_attn     = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_self     = nn.LayerNorm(d)
        # cross-attn
        self.cross_attn    = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
        self.norm_q        = nn.LayerNorm(d)
        self.norm_kv       = nn.LayerNorm(d)
        self.drop          = nn.Dropout(0.1)
        self.ffn           = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d),
        )
        self.norm_ffn      = nn.LayerNorm(d)
        self.proj          = nn.Linear(d + 2, 5)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.proj.bias[2] = 0.69
            self.proj.bias[3] = 0.69

    def forward(self, q, feat, uv_01, key_padding_mask=None):
        B, D_, H, W = feat.shape
        kv = feat.flatten(2).permute(0, 2, 1)          # (B, H*W, D)

        # 1. image self-attn
        img_sa, _ = self.img_self_attn(self.norm_img(kv), self.norm_img(kv), self.norm_img(kv))
        kv = kv + self.drop(img_sa)

        # 2. point self-attn
        pt_sa, _ = self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q),
                                  key_padding_mask=key_padding_mask)
        q = q + self.drop(pt_sa)

        # 3. cross-attn
        ca, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(ca)

        # 4. FFN
        q = q + self.ffn(self.norm_ffn(q))
        raw = self.proj(torch.cat([q, uv_01], dim=-1))  # (B,N,5)
        return q, raw


def clamp_params(raw: torch.Tensor, img_size: int = 128):
    """
    raw: (B, N, 5)  →  (B, N, 5) with valid parameterisation
    Returns tensor with columns: [tx, ty, log_sx, log_sy, rho]
    """
    tx, ty      = raw[..., 0], raw[..., 1]
    import math
    log_sx      = raw[..., 2].clamp(min=math.log(MIN_SIGMA), max=math.log(MAX_SIGMA))
    log_sy      = raw[..., 3].clamp(min=math.log(MIN_SIGMA), max=math.log(MAX_SIGMA))
    rho         = torch.tanh(raw[..., 4]) * 0.99   # |rho| < 0.99
    # scale mean to pixel space
    tx = tx * img_size
    ty = ty * img_size
    return torch.stack([tx, ty, log_sx, log_sy, rho], dim=-1)


# ---------------------------------------------------------------------------
# 3-D (uvd) extension — per-point 3D gaussian (diag in depth, uv with ρ)
# ---------------------------------------------------------------------------
MIN_SIGMA_D = 0.05   # m — floor for depth σ
MAX_SIGMA_D = 50.0   # m
D_SCALE     = 50.0   # m — raw output's Δd is scaled by this (same normalization as dataset z_norm)


def clamp_params_uvd(raw: torch.Tensor, img_size: int = 128):
    """
    raw: (B, N, 7)  [tx, ty, td, log_sx, log_sy, log_sd, rho_uv]
    Returns tensor with pixel-/meter-scaled (tx, ty, td) and clamped sigmas / rho.
    """
    import math
    tx = raw[..., 0] * img_size
    ty = raw[..., 1] * img_size
    td = raw[..., 2] * D_SCALE
    log_sx = raw[..., 3].clamp(min=math.log(MIN_SIGMA),   max=math.log(MAX_SIGMA))
    log_sy = raw[..., 4].clamp(min=math.log(MIN_SIGMA),   max=math.log(MAX_SIGMA))
    log_sd = raw[..., 5].clamp(min=math.log(MIN_SIGMA_D), max=math.log(MAX_SIGMA_D))
    rho    = torch.tanh(raw[..., 6]) * 0.99
    return torch.stack([tx, ty, td, log_sx, log_sy, log_sd, rho], dim=-1)


def gaussian_uvd_nll(params: torch.Tensor,
                     target_uv: torch.Tensor, target_d: torch.Tensor) -> torch.Tensor:
    """params: (B,N,7)  target_uv: (B,N,2)  target_d: (B,N,)  → scalar NLL.

    Depth is modelled as independent of (u,v) (diagonal block); ρ only couples u and v.
    This is the simplest statistically-valid extension: full 3x3 requires 3 ρs and
    PD parameterisation we skip for now.
    """
    tx, ty, td  = params[..., 0], params[..., 1], params[..., 2]
    log_sx, log_sy, log_sd = params[..., 3], params[..., 4], params[..., 5]
    rho = params[..., 6]

    dx = target_uv[..., 0] - tx
    dy = target_uv[..., 1] - ty
    dd = target_d - td

    sx, sy, sd = log_sx.exp(), log_sy.exp(), log_sd.exp()
    zx, zy, zd = dx / sx, dy / sy, dd / sd
    r2 = (1.0 - rho * rho).clamp(min=1e-6)

    # 2D uv Mahalanobis
    maha_uv = (zx * zx - 2 * rho * zx * zy + zy * zy) / r2
    log_det_uv = 2 * log_sx + 2 * log_sy + torch.log(r2)
    # 1D depth
    maha_d = zd * zd
    log_det_d = 2 * log_sd

    nll = 0.5 * (log_det_uv + maha_uv + log_det_d + maha_d)
    return nll.mean()


class CalibNetCov(nn.Module):
    def __init__(self, d: int = D, img_size: int = 128):
        super().__init__()
        self.img_size     = img_size
        self.cnn          = CNNBackbone(d)
        self.point_mlp    = PointMLP(d)
        self.cross_coarse = CrossAttentionBlockCov(d)
        self.cross_fine   = CrossAttentionBlockCov(d)

    def forward(self, image, distorted_uv):
        """
        Returns params (B, N, 5): [tx, ty, log_sx, log_sy, rho]
        All in pixel space.
        """
        coarse_feat, fine_feat = self.cnn(image)
        uv_01 = distorted_uv / self.img_size

        q = self.point_mlp(uv_01)
        q, raw_c = self.cross_coarse(q, coarse_feat, uv_01)

        # warp using coarse mean (columns 0,1 are in [0,1] before scaling → use raw)
        offset_c_01 = raw_c[..., :2]   # still un-scaled at this point
        uv_w = (uv_01 + offset_c_01).clamp(0, 1)
        q_w  = self.point_mlp(uv_w) + q
        _, raw_f = self.cross_fine(q_w, fine_feat, uv_w)

        # sum coarse + fine raw, then clamp/scale
        raw = raw_c + raw_f
        return clamp_params(raw, self.img_size)   # (B,N,5)


# ---------------------------------------------------------------------------
# 2-D Gaussian NLL loss
# ---------------------------------------------------------------------------

def gaussian2d_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    params : (B, N, 5)  [tx, ty, log_sx, log_sy, rho]
    target : (B, N, 2)  [gt_tx, gt_ty]
    Returns scalar mean NLL.
    """
    tx, ty       = params[..., 0], params[..., 1]
    log_sx, log_sy = params[..., 2], params[..., 3]
    rho          = params[..., 4]

    dx = target[..., 0] - tx
    dy = target[..., 1] - ty
    sx = log_sx.exp()
    sy = log_sy.exp()

    # normalised residuals
    zx = dx / sx
    zy = dy / sy
    r2 = 1.0 - rho**2

    # log det of covariance = log(sx²·sy²·(1-ρ²)) = 2log_sx + 2log_sy + log(1-ρ²)
    log_det = 2*log_sx + 2*log_sy + torch.log(r2.clamp(min=1e-6))

    # Mahalanobis
    maha = (zx**2 - 2*rho*zx*zy + zy**2) / r2.clamp(min=1e-6)

    nll = 0.5 * (log_det + maha)  # + const (log 2π, ignored)
    return nll.mean()


def compute_calibnet_loss(model_out, gt_uv, valid_mask, *, pert_vec=None,
                           frame_pose_weight: float = 0.5):
    """Unified per-pt + frame-pose loss for CalibNetDepth.

    Used by both train_ps_v3 (single-GPU) and train_ps_v3_ddp (Accelerate).

    Args:
        model_out: model(...) return value. Either `params` (B, N, 5) for
            per-pt only, or `(params, head_out)` where `head_out` is
            either `(μ, log_σ)` or `(μ, log_σ, L)` from the CLS frame-pose
            head.
        gt_uv:       (B, N, 2) target Δuv.
        valid_mask:  (B, N) bool, True where the point is valid.
        pert_vec:    (B, K) perturbation label for the frame-pose head.
            Only required when `use_frame_pose=True`. K must be ≥ n_dof
            of the head; the leading n_dof entries are taken as the
            target.
        frame_pose_weight: scalar weight on the frame-pose NLL term.

    Returns:
        (loss, loss_pt, loss_fr) — `loss_fr` is None when the frame-pose
        head is disabled or pert_vec is missing.
    """
    frame_mu = frame_logsig = frame_L = None
    if isinstance(model_out, tuple):
        params, head_out = model_out
        if len(head_out) == 3:
            frame_mu, frame_logsig, frame_L = head_out
        else:
            frame_mu, frame_logsig = head_out
    else:
        params = model_out

    loss_pt = gaussian2d_nll(params[valid_mask], gt_uv[valid_mask])
    loss = loss_pt

    loss_fr = None
    if frame_mu is not None and pert_vec is not None:
        resid = pert_vec[..., :frame_mu.shape[-1]] - frame_mu  # (B, n_dof)
        if frame_L is not None:
            z = torch.linalg.solve_triangular(
                frame_L, resid.unsqueeze(-1), upper=False).squeeze(-1)
            fr_nll = 0.5 * (z * z).sum(dim=-1) + frame_logsig.sum(dim=-1)
        else:
            inv_var = torch.exp(-2.0 * frame_logsig)
            fr_nll  = 0.5 * (resid * resid * inv_var).sum(dim=-1) \
                       + frame_logsig.sum(dim=-1)
        loss_fr = fr_nll.mean()
        loss = loss + float(frame_pose_weight) * loss_fr

    return loss, loss_pt, loss_fr, params
