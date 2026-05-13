"""WFPS (Variance-Weighted Farthest Point Sampling) gate.

Standalone module per the 2026-05-13 patent draft: "multimodal-fusion-time
residual aleatoric uncertainty gating". The block is consumed by a model
*before* heavy LiDAR×Camera fusion to deterministically condense N candidate
tokens down to K, where K is chosen so the downstream Cross-Attention cost
becomes practically constant in N.

Design (3-step gate):
  1. SharedMuSigma — two parallel linears on a shared per-point feature →
     (μ, log σ²). μ is for training-time aleatoric supervision; the gate
     path discards μ and reads only σ².
  2. wfps_indices  — vectorized FPS where the score at each step is
     σ²(i) × min_{j∈Selected} ||pos_i − pos_j||. First pick = arg max σ².
     Pure tensor ops, deterministic, no Python-side random state.
  3. WFPSGate      — wires the head + the sampler. Two training paths:
       * Deep Supervision  : caller adds aleatoric_loss(μ, log σ², y) over
         the FULL N-point output. Canonical signal for σ². No STE needed.
       * Straight-Through  : optional. After hard gather, multiply gathered
         features by a normalized soft σ² weight using the additive identity
         trick so the forward value is the un-modulated hard gather but
         the backward routes a gradient into σ² (and through it the
         backbone) from any downstream loss.

Not yet wired into any model — `python models/wfps_gate.py` runs a smoke
test that exercises gather/grad/determinism.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMuSigma(nn.Module):
    """Two parallel linears on a shared backbone feature.

    Returns (μ, log σ²) of shape (B, N, head_out_dim). Default init biases
    log σ² → 0 so σ² ≈ 1 at iteration 0; the aleatoric loss then has stable
    inverse-variance weighting on its first few steps.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.log_sigma2 = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(self.log_sigma2.bias)
        nn.init.normal_(self.log_sigma2.weight, std=1e-3)

    def forward(self, feats: torch.Tensor):
        return self.mu(feats), self.log_sigma2(feats)


def aleatoric_loss(mu: torch.Tensor, log_sigma2: torch.Tensor,
                    target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Heteroscedastic regression NLL (Kendall & Gal, NIPS 2017).

        L_i = ½ exp(−log σ²_i) · (y_i − μ_i)² + ½ log σ²_i

    Summed over the last (output) dim; reduced across batch/points per
    `reduction`. Input shapes broadcast as long as the trailing output dim
    matches.
    """
    inv_var = torch.exp(-log_sigma2)
    elem = 0.5 * inv_var * (target - mu).pow(2) + 0.5 * log_sigma2
    if reduction == 'mean':
        return elem.mean()
    if reduction == 'sum':
        return elem.sum()
    return elem


@torch.no_grad()
def wfps_indices(sigma2: torch.Tensor, pos: torch.Tensor, K: int) -> torch.Tensor:
    """Variance-weighted FPS, pure tensor ops, deterministic.

    Args:
        sigma2: (B, N) scalar per-point variance (>0, post-exp).
        pos:    (B, N, D) per-point coords for distance (D ∈ {2, 3, …}).
        K:      number of points to select. K must be ≤ N.

    Returns:
        idx: (B, K) long — selected indices. For identical (sigma2, pos)
            inputs the output is bit-identical across calls (FuSa hook).
    """
    B, N = sigma2.shape
    assert pos.shape[:2] == (B, N), f"shape mismatch: σ²={sigma2.shape} pos={pos.shape}"
    assert K <= N, f"K={K} > N={N}"
    device = sigma2.device
    NEG_INF = torch.finfo(sigma2.dtype).min

    sel = torch.empty(B, K, dtype=torch.long, device=device)
    sel[:, 0] = sigma2.argmax(dim=-1)

    sel_pos = pos.gather(1, sel[:, :1, None].expand(-1, 1, pos.size(-1)))   # (B,1,D)
    min_dist = torch.linalg.norm(pos - sel_pos, dim=-1)                     # (B,N)

    for k in range(1, K):
        score = sigma2 * min_dist
        score.scatter_(1, sel[:, :k], NEG_INF)                              # mask prior picks
        sel[:, k] = score.argmax(dim=-1)
        sel_pos = pos.gather(1, sel[:, k:k+1, None].expand(-1, 1, pos.size(-1)))
        new_dist = torch.linalg.norm(pos - sel_pos, dim=-1)
        min_dist = torch.minimum(min_dist, new_dist)
    return sel


def gather_tokens(feats: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """(B, N, D), (B, K) → (B, K, D)."""
    return feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)))


class WFPSGate(nn.Module):
    """LiDAR-only residual-uncertainty predictor + WFPS condenser.

    Forward:
        gathered, idx, mu, log_sigma2 = self(feats, pos, k=None)
            feats: (B, N, D_feat)
            pos:   (B, N, D_pos)   D_pos ∈ {2, 3, …}
            k:     override module default

    Training paths (use one or both):
        - Deep supervision: outside, `aleatoric_loss(mu, log_sigma2, y).backward()`
          on full N points — direct supervision of σ² head.
        - STE: if `ste=True` and in train mode, gradients from downstream
          losses on `gathered` propagate into σ² (and through it the
          backbone). Forward value of `gathered` is unchanged (identity).
    """
    def __init__(self, in_dim: int, head_out_dim: int = 4,
                 k: int = 16, ste: bool = True):
        super().__init__()
        self.head = SharedMuSigma(in_dim, head_out_dim)
        self.k = int(k)
        self.ste = bool(ste)

    def forward(self, feats: torch.Tensor, pos: torch.Tensor,
                k: int | None = None):
        K = int(k) if k is not None else self.k
        mu, log_sigma2 = self.head(feats)                                   # (B,N,O)
        # Reduce variance to a scalar gate per point. Trace (mean over output
        # dims) flags points uncertain on ANY dim. clamp for numerical safety.
        sigma2 = torch.exp(log_sigma2).mean(dim=-1).clamp(min=1e-6)         # (B,N)

        idx = wfps_indices(sigma2, pos, K)                                  # (B,K)
        gathered = gather_tokens(feats, idx)                                # (B,K,D)

        if self.training and self.ste:
            # Soft σ² gate over the K picks, normalized to sum to 1. Then
            # use the standard additive-identity STE so the forward value
            # equals the un-modulated `gathered` but the backward routes a
            # signal into σ²:
            #     out = gathered + soft_w * gathered.detach()
            #                   − (soft_w * gathered.detach()).detach()
            sig_picked = sigma2.gather(1, idx).clamp(min=1e-6)              # (B,K)
            soft_w = (sig_picked /
                      sig_picked.sum(dim=-1, keepdim=True)).unsqueeze(-1)   # (B,K,1)
            mod = soft_w * gathered.detach()
            gathered = gathered + mod - mod.detach()

        return gathered, idx, mu, log_sigma2


# ── smoke test (numerical sanity, gradient flow, determinism) ────────────────
if __name__ == '__main__':
    torch.manual_seed(0)
    B, N, D_feat, D_out, D_pos, K = 2, 1024, 64, 4, 3, 16

    gate = WFPSGate(in_dim=D_feat, head_out_dim=D_out, k=K, ste=True)
    feats = torch.randn(B, N, D_feat, requires_grad=True)
    pos = torch.randn(B, N, D_pos)
    y = torch.randn(B, N, D_out)

    # forward
    gate.train()
    gathered, idx, mu, log_sigma2 = gate(feats, pos)
    assert gathered.shape == (B, K, D_feat)
    assert idx.shape == (B, K)
    assert mu.shape == log_sigma2.shape == (B, N, D_out)

    # backward — Deep Supervision + downstream both
    L_aleatoric = aleatoric_loss(mu, log_sigma2, y)
    L_downstream = gathered.pow(2).mean()
    (L_aleatoric + L_downstream).backward()
    assert feats.grad is not None and feats.grad.abs().max().item() > 0
    head_grad = gate.head.log_sigma2.weight.grad
    assert head_grad is not None and head_grad.abs().max().item() > 0
    print(f"OK  feats.grad max={feats.grad.abs().max().item():.4e}  "
          f"σ²-head.weight grad max={head_grad.abs().max().item():.4e}")

    # determinism: same inputs → identical idx (FuSa hook)
    gate.eval()
    torch.manual_seed(42)
    _, idx1, *_ = gate(feats.detach(), pos)
    torch.manual_seed(999)
    _, idx2, *_ = gate(feats.detach(), pos)
    assert torch.equal(idx1, idx2), "non-determinism detected"
    print("determinism OK (idx1 ≡ idx2 under different RNG seeds)")

    # FPS effect check: avg pairwise distance of selected pts > random K
    sel_pos = pos.gather(1, idx1.unsqueeze(-1).expand(-1, -1, D_pos))
    pdist = torch.cdist(sel_pos, sel_pos).mean(dim=(-1, -2))
    rand_idx = torch.randperm(N)[:K].unsqueeze(0).expand(B, -1)
    rand_pos = pos.gather(1, rand_idx.unsqueeze(-1).expand(-1, -1, D_pos))
    rand_pdist = torch.cdist(rand_pos, rand_pos).mean(dim=(-1, -2))
    print(f"diversity check: WFPS avg pairwise dist {pdist.mean():.3f}  "
          f"vs random {rand_pdist.mean():.3f}  (WFPS should be ≥)")
