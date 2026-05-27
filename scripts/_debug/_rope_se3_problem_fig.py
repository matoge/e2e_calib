"""Problem-setup diagram for the SE(3) token-rotation report.

Two panels:
  Left  — concept: P (Q side) and R·P (KV side), with PE injected on each side
  Right — three ways to inject R into the network and where they break

Saves: docs/assets/2026-05-26_se3_token_rotation/rope_se3_problem.png
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def make_random_points(n=24, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p = p / np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-6)
    p = p * rng.uniform(0.3, 1.0, size=(n, 1)) ** (1 / 3)
    return p


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)],
                     [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]])


def panel_concept(ax):
    P = make_random_points(n=20, seed=0)
    R = rot_y(60)
    Q = P
    KV = P @ R.T

    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    # Q-side blob
    q_cx, q_cy, q_r = 1.7, 4.3, 1.1
    ax.add_patch(plt.Circle((q_cx, q_cy), q_r, fill=False, ec='C0', lw=1.5))
    ax.text(q_cx, q_cy + q_r + 0.25, 'Q-frame: P', ha='center', fontsize=11, color='C0', weight='bold')
    for p in Q:
        ax.plot(q_cx + p[0] * q_r * 0.9, q_cy + p[1] * q_r * 0.9, 'o', color='C0', ms=3)

    # KV-side blob (rotated)
    k_cx, k_cy, k_r = 7.5, 4.3, 1.1
    ax.add_patch(plt.Circle((k_cx, k_cy), k_r, fill=False, ec='C1', lw=1.5))
    ax.text(k_cx, k_cy + k_r + 0.25, 'KV-frame: R · P', ha='center', fontsize=11, color='C1', weight='bold')
    for p in KV:
        ax.plot(k_cx + p[0] * k_r * 0.9, k_cy + p[1] * k_r * 0.9, 'o', color='C1', ms=3)

    # rotation arrow
    arr = FancyArrowPatch((q_cx + q_r + 0.2, q_cy), (k_cx - k_r - 0.2, k_cy),
                          arrowstyle='->', mutation_scale=18, color='black', lw=1.4)
    ax.add_patch(arr)
    ax.text((q_cx + k_cx) / 2, q_cy + 0.3, 'R ∈ SO(3)', ha='center', fontsize=11, weight='bold')
    ax.text((q_cx + k_cx) / 2, q_cy - 0.3, '(R is given to the network as input)', ha='center', fontsize=9, color='gray')

    # PE boxes
    ax.add_patch(FancyBboxPatch((q_cx - 1.0, 2.3), 2.0, 0.7, boxstyle='round,pad=0.05',
                                fc='#e8f0ff', ec='C0', lw=1))
    ax.text(q_cx, 2.65, 'PosEmbed(P)', ha='center', va='center', fontsize=10)
    ax.add_patch(FancyBboxPatch((k_cx - 1.0, 2.3), 2.0, 0.7, boxstyle='round,pad=0.05',
                                fc='#fff0e0', ec='C1', lw=1))
    ax.text(k_cx, 2.65, 'PosEmbed(R · P)', ha='center', va='center', fontsize=10)

    # cross-attn box
    ax.add_patch(FancyBboxPatch((3.5, 0.7), 3.0, 1.0, boxstyle='round,pad=0.05',
                                fc='#f3f3f3', ec='black', lw=1.2))
    ax.text(5.0, 1.4, 'Cross-Attention', ha='center', va='center', fontsize=11, weight='bold')
    ax.text(5.0, 1.0, 'Q · Kᵀ → softmax → assignment', ha='center', va='center', fontsize=9, color='gray')

    # arrows down
    ax.add_patch(FancyArrowPatch((q_cx, 2.3), (4.3, 1.7), arrowstyle='->', mutation_scale=14, color='C0'))
    ax.add_patch(FancyArrowPatch((k_cx, 2.3), (5.7, 1.7), arrowstyle='->', mutation_scale=14, color='C1'))

    # task statement
    ax.text(5.0, 0.3, 'Task: with R given as input, decode R · P from token  (or match correspondence)',
            ha='center', va='center', fontsize=10, style='italic')
    ax.text(5.0, 5.7, 'Setup: same point cloud P, rotated by KNOWN R on the KV side',
            ha='center', va='center', fontsize=11, weight='bold')


def panel_methods(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5.0, 5.7, 'Three ways to inject R into the transformer',
            ha='center', va='center', fontsize=11, weight='bold')

    # (a) abs-PE add + learn
    ax.add_patch(FancyBboxPatch((0.3, 3.5), 9.4, 1.6, boxstyle='round,pad=0.05',
                                fc='#fff0f0', ec='C3', lw=1.2))
    ax.text(0.5, 4.85, '(a) abs-PE  +  R as 9-vec → MLP  (learn T(R) end-to-end)',
            fontsize=10, weight='bold', va='center')
    ax.text(0.6, 4.4, 'token  ←  PE(p_i)  +  MLP([token, vec(R)])',
            fontsize=9.5, family='monospace', va='center')
    ax.text(0.6, 4.0, '✗ Network memorises T(R) on training |R|.  Extrapolation breaks (74× error at 180°).',
            fontsize=9, color='C3', va='center')

    # (b) type-1 block-diag
    ax.add_patch(FancyBboxPatch((0.3, 1.9), 9.4, 1.5, boxstyle='round,pad=0.05',
                                fc='#f0fff0', ec='C2', lw=1.2))
    ax.text(0.5, 3.15, '(b) Type-1 RoPE-3D:  rotate token CHUNKS by R  (block-diag(R, R, …, R))',
            fontsize=10, weight='bold', va='center')
    ax.text(0.6, 2.7, 'token  =  (D_s scalar  ⊕  K type-1 vectors)',
            fontsize=9.5, family='monospace', va='center')
    ax.text(0.6, 2.35, 'T(R) · token  =  (scalar,  block-diag(R) · type-1 vectors)',
            fontsize=9.5, family='monospace', va='center')
    ax.text(0.6, 2.0, '✓ R acts as a real rotation on the token. SO(3) is built into the architecture.',
            fontsize=9, color='C2', va='center')

    # (c) per-axis 1D RoPE
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 9.4, 1.4, boxstyle='round,pad=0.05',
                                fc='#f0f8ff', ec='C0', lw=1.2))
    ax.text(0.5, 1.55, '(c) Per-axis 1D RoPE on (x, y, z)  (popular but wrong)',
            fontsize=10, weight='bold', va='center')
    ax.text(0.6, 1.15, 'phase rotates each (x, y, z) coordinate independently — SO(2)³, not SO(3)',
            fontsize=9.5, va='center')
    ax.text(0.6, 0.75, '~ Translation-invariant only.  Mild degradation for general R.',
            fontsize=9, color='C0', va='center')


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    panel_concept(axes[0])
    panel_methods(axes[1])
    out = Path('/home/hfunaya/git/e2e_calib/docs/assets/2026-05-26_se3_token_rotation/rope_se3_problem.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'[saved] {out.absolute()}')


if __name__ == '__main__':
    main()
