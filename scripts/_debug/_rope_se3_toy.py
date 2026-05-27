"""SE(3) token-rotation toy: can the network rotate its OWN tokens?

Setup
-----
N=64 random points P ∈ R^3. Sample R ∈ SO(3). Targets = R·P (and t-task: P+t).

Encoder f: P → X ∈ R^(N×D)        (Linear / optional self-attn)
Token-space rotation T(R): X → X'  (architecturally specified, not learned)
Decoder g: X' → P̂ ∈ R^(N×3)        (Linear)

Loss = ‖g(T(R) f(P)) − R·P‖²   (per-point MSE)

The point: PE+加算 では T(R) を architecturally 構成できない (PE(R·p)−PE(p) is
nonlinear in p, the network has to *learn* a different correction per p).
SVD-style rigid alignment cannot be expressed by a fixed T(R) acting on PE-tokens.

Conditions (all share same N, D, optimizer; only token encoding differs):

  (a) abs-PE-add + learned-T(R)
        f(P)[i]    = MLP_id(p_i) + sinPE(p_i)
        T(R) X     = MLP_R([X, vec(R)])                  ← R を network が token 空間で覚える
        Predicted: small |R| ok, large |R| degrades — token space cannot represent SO(3)

  (b) **type-1 RoPE-3D (block-diag(R) on K type-1 chunks)**   ← key proposed method
        D = D_s + 3K
        f(P)[i] = (MLP_s(p_i) ∈ R^{D_s},  W_v ⊗ p_i ∈ R^{3K})
        T(R) X[i] = (X_s[i],  block-diag(R,...,R) · X_v[i])
        i.e. the type-1 vector chunks rotate by R itself (no power scaling needed for
        decode-equivariance — per-channel ω scaling matters when we want *frequency
        diversity* for matching, but for the decode toy plain R already gives an exact
        equivariant solution: g_v(R·v) = R·g_v(v) when g_v is linear & type-preserving).
        Predicted: error ≈ 0 for all |R|, by construction.

  (c) per-axis 1D RoPE-3D (the wrong-but-popular form)
        f(P)[i]    = MLP_id(p_i)
        T(R) X     = (no direct R; instead phase rotation tied to p_i, with KV-side
                      using R·p_i — same as previous toy.)  Decoder cannot recover R·P
                      because per-axis SO(2) is not SO(3).
        Predicted: degrades with |R|, similar to (a).

  (d) sanity translation: T(t) X = X + W_t·t  (learned)
        Predicted: trivial — flat near-zero for all |t|.

Figure: 4-panel — (a/b/c) decode error vs |R|° and (d) decode error vs |t|.

Run
---
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_rope_se3_toy.py
  → prints tables and saves rope_se3_decode.png
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_POINTS = 64
D_MODEL  = 96      # divisible by 6 (per-axis RoPE) and by 3 (type-1 chunks)
D_SCALAR = 24      # type-0 part for (b)
D_VEC    = D_MODEL - D_SCALAR  # = 72 = 24 type-1 chunks × 3
K_VEC    = D_VEC // 3
BATCH    = 256
N_STEPS  = 1500
LR       = 3e-3
SEED     = 0


# ── data ─────────────────────────────────────────────────────────────
def random_points(B: int, N: int) -> torch.Tensor:
    p = torch.randn(B, N, 3, device=DEVICE)
    p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    p = p * torch.rand(B, N, 1, device=DEVICE) ** (1/3)
    return p


def random_rot(B: int, max_deg: float) -> torch.Tensor:
    axis = F.normalize(torch.randn(B, 3, device=DEVICE), dim=-1)
    ang  = torch.rand(B, device=DEVICE) * (max_deg * math.pi / 180.0)
    K = torch.zeros(B, 3, 3, device=DEVICE)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]; K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] =  axis[:, 0]
    I = torch.eye(3, device=DEVICE).expand(B, 3, 3)
    s = ang.sin().view(B, 1, 1); c = ang.cos().view(B, 1, 1)
    return I + s * K + (1 - c) * (K @ K)


def fixed_axis_rot(B: int, deg: float, axis_idx: int = 1) -> torch.Tensor:
    """Eval-time helper: rotate exactly by `deg` around a fixed axis (for clean curves)."""
    ang = torch.full((B,), deg * math.pi / 180.0, device=DEVICE)
    axis = torch.zeros(B, 3, device=DEVICE); axis[:, axis_idx] = 1.0
    K = torch.zeros(B, 3, 3, device=DEVICE)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]; K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] =  axis[:, 0]
    I = torch.eye(3, device=DEVICE).expand(B, 3, 3)
    s = ang.sin().view(B, 1, 1); c = ang.cos().view(B, 1, 1)
    return I + s * K + (1 - c) * (K @ K)


# ── encodings ────────────────────────────────────────────────────────
def sinusoidal_pe(p: torch.Tensor, d: int) -> torch.Tensor:
    B, N, _ = p.shape
    d_axis = d // 6
    freqs = torch.arange(d_axis, device=p.device).float()
    freqs = 1.0 / (10000 ** (freqs / d_axis))
    out = []
    for ax in range(3):
        a = p[..., ax:ax+1] * freqs
        out += [a.sin(), a.cos()]
    pe = torch.cat(out, dim=-1)
    if pe.shape[-1] < d:
        pe = F.pad(pe, (0, d - pe.shape[-1]))
    return pe[..., :d]


def rope_apply_3d_per_axis(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Per-axis 1D RoPE on (x,y,z): wrong-but-popular SO(2)^3, NOT SO(3)."""
    B, N, D = x.shape
    d_axis = D // 6
    assert d_axis * 6 == D
    x = x.view(B, N, 3, 2 * d_axis)
    freqs = 1.0 / (10000 ** (torch.arange(d_axis, device=x.device).float() / d_axis))
    ang = p.unsqueeze(-1) * freqs                             # (B,N,3,d_axis)
    cs = ang.cos(); sn = ang.sin()
    a = x[..., :d_axis]; b = x[..., d_axis:]
    a2 = a * cs - b * sn
    b2 = a * sn + b * cs
    return torch.cat([a2, b2], dim=-1).reshape(B, N, D)


def apply_block_R_on_type1(x_vec: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """x_vec (B,N,3K) treated as K type-1 chunks; apply R to each chunk.
    R is (B,3,3) per-sample."""
    B, N, D3K = x_vec.shape
    K = D3K // 3
    x = x_vec.view(B, N, K, 3)                               # (B,N,K,3)
    # (B,1,1,3,3) @ (B,N,K,3,1) → (B,N,K,3,1)
    Rb = R.view(B, 1, 1, 3, 3)
    x_rot = (Rb @ x.unsqueeze(-1)).squeeze(-1)               # (B,N,K,3)
    return x_rot.reshape(B, N, D3K)


# ── three+one models ─────────────────────────────────────────────────
class AbsPELearnedT(nn.Module):
    """(a) abs-PE encode + learned T(R). The MLP_R must learn how to rotate
    PE-tokens — predicted to fail at large |R|."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.id  = nn.Linear(3, d)
        self.dec = nn.Linear(d, 3)
        self.tR  = nn.Sequential(
            nn.Linear(d + 9, 2 * d), nn.GELU(),
            nn.Linear(2 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, d),
        )
    def forward(self, P, R):
        X = self.id(P) + sinusoidal_pe(P, self.id.out_features)
        Rf = R.view(P.shape[0], 1, 9).expand(-1, P.shape[1], -1)
        X2 = self.tR(torch.cat([X, Rf], dim=-1))
        return self.dec(X2)


class Type1RoPE(nn.Module):
    """(b) type-1 chunks: token = (scalar D_s, K type-1 vectors). T(R) rotates
    each type-1 chunk by R. Decoder is type-preserving linear:
      g_v(R·v) = R·g_v(v)  for any K×K linear g_v on type-1 channels.
    Predicted: error ≈ 0 by construction."""
    def __init__(self, d_s=D_SCALAR, k=K_VEC):
        super().__init__()
        self.d_s, self.k = d_s, k
        self.mlp_s = nn.Linear(3, d_s)
        self.W_v   = nn.Linear(3, k, bias=False)             # K type-1 chunks: each = (W_v p)_k · (canonical 3D-frame applied via outer product)
        # decoder type-preserving: type-1 → 3 via K-mix
        self.dec_v = nn.Linear(k, 1, bias=False)
        # (scalar part is unused for decode; could add residual but not needed)
    def forward(self, P, R):
        B, N, _ = P.shape
        # encode: K type-1 vectors per token. f_v(p)[k,:] = (W_v p)_k · p̂_k  →
        # simpler: build K vectors by W_v ∈ R^{K×3} acting per-channel:
        # X_v[i, k, :] = (alpha_k(p_i)) · p_i, where alpha_k(p) = (W_v p)_k.
        # This makes f equivariant: f_v(R p)[k,:] = alpha_k(R p) · R p, but
        # alpha_k is NOT invariant. So instead use a TRULY equivariant form:
        # X_v[i, k, :] = U_k · p_i where U_k ∈ R^{3×3} is a *channel-mix* on
        # type-1 features... but p has only 1 type-1 channel. We promote first:
        # f_v(p) = W_v ⊗ p, treating p itself as the only type-1 input.
        # Concretely: X_v[i, k, :] = w_k · p_i, with w_k ∈ R learnable scalar.
        # Then T(R) X_v[i,k,:] = w_k · (R p_i) — equivariant by construction.
        # Decoder: g_v(X_v)[i] = sum_k a_k · X_v[i,k,:] ∈ R^3 — equivariant linear.
        # Compose: g_v(T(R) f_v(p)) = sum_k a_k w_k · R p = (sum a_k w_k) · R p.
        # If sum a_k w_k = 1 (one trainable degree of freedom) → exact identity.
        w = self.W_v.weight  # (K, 3) — but we only need a per-k scalar; we'll reduce
        # actually let's just use a per-k scalar for clarity:
        # rebuild: ignore W_v's matrix structure, treat its per-row L2 norm as scalar
        # SIMPLER: define X_v[i,k,:] = beta_k · p_i, beta_k learnable
        beta = self.W_v.weight.norm(dim=-1)                   # (K,) ≥ 0
        X_v = beta.view(1, 1, self.k, 1) * P.unsqueeze(2)     # (B,N,K,3)
        # T(R): rotate each type-1 chunk by R
        Rb = R.view(B, 1, 1, 3, 3)
        X_v_rot = (Rb @ X_v.unsqueeze(-1)).squeeze(-1)        # (B,N,K,3)
        # decoder: K→1 mix on type-1 channels, share across the 3 spatial dims
        a = self.dec_v.weight.view(self.k)                    # (K,)
        out = (a.view(1, 1, self.k, 1) * X_v_rot).sum(dim=2)  # (B,N,3)
        return out


class PerAxisRoPE(nn.Module):
    """(c) per-axis 1D RoPE. f(P)=Linear(P); T(R) replaces phase p_i by R p_i
    on the KV side (here: just on the input side). Per-axis SO(2)^3 is not SO(3),
    so decode of R·P from rotated phases will degrade for general R."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.id  = nn.Linear(3, d)
        self.dec = nn.Linear(d, 3)
    def forward(self, P, R):
        # encoded with phase = P, then we ask for output at phase R·P.
        # Implementation: encode at P, then re-phase token by (R·P − P) using per-axis RoPE.
        X = self.id(P)
        X = rope_apply_3d_per_axis(X, P)                      # bring to "frame P"
        P_rot = (R @ P.transpose(1, 2)).transpose(1, 2)
        # to move from frame P to frame R·P, apply rope by (P_rot − P) is equivalent
        # to applying rope by P_rot after un-applying P (which is the conjugate):
        X = rope_apply_3d_per_axis(X, -P)                     # un-rotate phase P
        X = rope_apply_3d_per_axis(X, P_rot)                  # apply phase R·P
        return self.dec(X)


class TransAddPE(nn.Module):
    """(d) translation sanity: T(t) X = X + (PE(p+t) − PE(p)) ≈ learn linear shift.
    The decoder reads p+t directly. Predicted: trivially flat."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.id  = nn.Linear(3, d)
        self.dec = nn.Linear(d, 3)
    def forward(self, P, t):
        X = self.id(P) + sinusoidal_pe(P, self.id.out_features)
        # Move PE phase from P to P+t
        Pt = P + t
        X = X - sinusoidal_pe(P, self.id.out_features) + sinusoidal_pe(Pt, self.id.out_features)
        return self.dec(X)


# ── train / eval ─────────────────────────────────────────────────────
def train_rot(model, n_steps=N_STEPS, lr=LR, max_deg=30.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(n_steps):
        P = random_points(BATCH, N_POINTS)
        R = random_rot(BATCH, max_deg=max_deg)
        target = (R @ P.transpose(1, 2)).transpose(1, 2)
        pred = model(P, R)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"      step {step:4d}  loss={loss.item():.4f}")
    return model


def train_trans(model, n_steps=N_STEPS, lr=LR, t_max=2.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(n_steps):
        P = random_points(BATCH, N_POINTS)
        t = (torch.rand(BATCH, 1, 3, device=DEVICE) * 2 - 1) * t_max
        target = P + t
        pred = model(P, t)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"      step {step:4d}  loss={loss.item():.4f}")
    return model


@torch.no_grad()
def eval_rot_curve(model, degs):
    model.eval()
    out = []
    for deg in degs:
        P = random_points(BATCH, N_POINTS)
        R = random_rot(BATCH, max_deg=deg + 1e-3)             # uniform [0,deg]
        target = (R @ P.transpose(1, 2)).transpose(1, 2)
        pred = model(P, R)
        rmse = (pred - target).pow(2).sum(-1).sqrt().mean().item()
        out.append((deg, rmse))
    return out


@torch.no_grad()
def eval_trans_curve(model, mags):
    model.eval()
    out = []
    for m in mags:
        P = random_points(BATCH, N_POINTS)
        t = F.normalize(torch.randn(BATCH, 1, 3, device=DEVICE), dim=-1) * m
        target = P + t
        pred = model(P, t)
        rmse = (pred - target).pow(2).sum(-1).sqrt().mean().item()
        out.append((m, rmse))
    return out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    out_dir = Path(__file__).parent / '_outputs'
    out_dir.mkdir(exist_ok=True)

    degs = [0.5, 5, 15, 30, 60, 90, 120, 150, 180]
    mags = [0.0, 0.1, 0.5, 1.0, 2.0]

    TRAIN_DEG = 30.0
    print(f'[a] abs-PE + learned T(R)   (train |R| ∈ [0,{TRAIN_DEG:.0f}°])')
    a_m = AbsPELearnedT().to(DEVICE);  train_rot(a_m, max_deg=TRAIN_DEG)
    a_curve = eval_rot_curve(a_m, degs)

    print(f'[b] type-1 RoPE-3D (block-diag R on type-1 chunks)  (train |R| ∈ [0,{TRAIN_DEG:.0f}°])')
    b_m = Type1RoPE().to(DEVICE);      train_rot(b_m, max_deg=TRAIN_DEG)
    b_curve = eval_rot_curve(b_m, degs)

    print(f'[c] per-axis 1D RoPE-3D    (train |R| ∈ [0,{TRAIN_DEG:.0f}°])')
    c_m = PerAxisRoPE().to(DEVICE);    train_rot(c_m, max_deg=TRAIN_DEG)
    c_curve = eval_rot_curve(c_m, degs)

    print('[d] translation sanity')
    d_m = TransAddPE().to(DEVICE);     train_trans(d_m)
    d_curve = eval_trans_curve(d_m, mags)

    print()
    print(f'   |R| (deg) |   abs-PE+T   |  type-1 RoPE |  per-axis')
    print(f'  -----------+--------------+--------------+-----------')
    for (d_, a_a), (_, a_b), (_, a_c) in zip(a_curve, b_curve, c_curve):
        print(f'   {d_:>9.1f} |   {a_a:.4f}     |   {a_b:.4f}     |   {a_c:.4f}')
    print()
    print(f'   |t|       | trans decode rmse')
    print(f'  -----------+-------------------')
    for m, e in d_curve:
        print(f'   {m:>9.2f} |   {e:.4f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].plot([d for d,_ in a_curve], [v for _,v in a_curve], 'o-', label='(a) abs-PE + learned T(R)')
        ax[0].plot([d for d,_ in b_curve], [v for _,v in b_curve], 's-', label='(b) type-1 RoPE-3D (R on token)')
        ax[0].plot([d for d,_ in c_curve], [v for _,v in c_curve], '^-', label='(c) per-axis 1D RoPE-3D')
        ax[0].set_xlabel('|R|  (deg, sampled uniform [0,|R|])')
        ax[0].set_ylabel('decode RMSE  ‖g(T(R)f(P)) − R·P‖')
        ax[0].axvline(30, color='gray', ls=':', label='train |R| upper bound')
        ax[0].set_title('SE(3) token-rotation: extrapolation beyond train range')
        ax[0].legend(); ax[0].grid(True, alpha=0.3); ax[0].set_yscale('log')

        ax[1].plot([m for m,_ in d_curve], [v for _,v in d_curve], '^-', color='C2', label='(d) translation')
        ax[1].set_xlabel('|t|'); ax[1].set_ylabel('decode RMSE')
        ax[1].set_title('translation sanity'); ax[1].legend(); ax[1].grid(True, alpha=0.3)

        png = out_dir / 'rope_se3_decode.png'
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f'\n[saved] {png.absolute()}')
    except Exception as e:
        print(f'[plot] skipped: {e}')


if __name__ == '__main__':
    main()
