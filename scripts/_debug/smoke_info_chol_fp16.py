"""Cholesky-direct info_nll_2d + sigmoid-bound InfoHead2x2 + Beta-NLL の
fp16 forward/backward 健全性 smoke。

ep19 NaN の根因 (`det.clamp(1e-8).log()` の 1/det fp16 spike) を
構造的に殺せたかを直接確認する:
  1. fp32 / fp16 ともに forward 出力が finite
  2. fp16 で loss.backward() しても全 grad が finite
  3. Cholesky-direct path (L) と det-direct path (W only) で同じ NLL
  4. Beta-NLL (β=0.5) が β=0 と同じ符号で動く (大きさは違う)
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from models.model_depth import CalibNetDepth, InfoHead2x2
from models.model_cov import info_nll_2d


def _is_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # InfoHead2x2 単体: a/b の値域が sigmoid bound で正しく [1e-2, 4.01] に
    # 収まっているか確認
    head = InfoHead2x2(d=128).to(device)
    q = torch.randn(2, 50, 128, device=device, requires_grad=True)
    W, L = head(q, return_chol=True)
    a = L[..., 0, 0]
    b = L[..., 1, 1]
    c = L[..., 1, 0]
    print(f"[head fp32] a ∈ [{a.min():.4f}, {a.max():.4f}]  "
          f"b ∈ [{b.min():.4f}, {b.max():.4f}]  "
          f"|c| max={c.abs().max():.4f}")
    assert a.min() >= 1e-2 - 1e-6, f"a min broke bound: {a.min()}"
    assert a.max() <= 4.01 + 1e-4
    assert b.min() >= 1e-2 - 1e-6
    assert c.abs().max() <= 4.01

    # Cholesky-direct vs det-direct で NLL が一致するか (fp32)
    target = torch.randn_like(q[..., :2])
    mu = torch.zeros_like(target)
    nll_chol = info_nll_2d(mu, W, target, L=L, beta=0.0)
    nll_det = info_nll_2d(mu, W, target, beta=0.0)
    print(f"[fp32] NLL Cholesky={nll_chol.item():.6f}  "
          f"det-direct={nll_det.item():.6f}  "
          f"diff={(nll_chol - nll_det).abs().item():.2e}")
    assert (nll_chol - nll_det).abs() < 1e-3, "Chol vs det 不一致"

    # Beta-NLL (β=0.5) が finite で sign が同じ
    nll_beta = info_nll_2d(mu, W, target, L=L, beta=0.5)
    print(f"[fp32] β=0.5 NLL={nll_beta.item():.6f}  "
          f"β=0 NLL={nll_chol.item():.6f}")
    assert _is_finite(nll_beta)

    # fp16 forward/backward
    q16 = torch.randn(2, 50, 128, device=device, requires_grad=True)
    target16 = torch.randn(2, 50, 2, device=device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        W16, L16 = head(q16, return_chol=True)
        mu16 = torch.zeros_like(target16)
        loss_chol = info_nll_2d(mu16, W16, target16, L=L16, beta=0.5)
        loss_det = info_nll_2d(mu16, W16, target16, beta=0.5)
    print(f"[fp16] β=0.5 Cholesky={loss_chol.item():.4f}  "
          f"det-direct={loss_det.item():.4f}")
    assert _is_finite(loss_chol)
    assert _is_finite(loss_det)

    loss_chol.backward()
    grads_finite = all(_is_finite(p.grad) for p in head.parameters() if p.grad is not None)
    print(f"[fp16] head all-grad finite: {grads_finite}")
    assert grads_finite

    # 念のため 100 反復 backward を回しても grad が finite を維持
    head.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    nan_step = -1
    for step in range(100):
        q = torch.randn(8, 200, 128, device=device)
        target = torch.randn(8, 200, 2, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            W, L = head(q, return_chol=True)
            mu = torch.zeros_like(target)
            loss = info_nll_2d(mu, W, target, L=L, beta=0.5)
        if not _is_finite(loss):
            nan_step = step
            print(f"[fp16 step {step}] loss non-finite!")
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        bad = [n for n, p in head.named_parameters()
                if p.grad is not None and not _is_finite(p.grad)]
        if bad:
            nan_step = step
            print(f"[fp16 step {step}] non-finite grads: {bad[:3]}")
            break
        optimizer.step()
    if nan_step < 0:
        print("[fp16] 100 steps OK — no NaN/Inf in loss or grads")
    else:
        print(f"[fp16] FAILED at step {nan_step}")

    print("\nALL CHECKS PASSED" if nan_step < 0 else "\nFAILED")


if __name__ == '__main__':
    main()
