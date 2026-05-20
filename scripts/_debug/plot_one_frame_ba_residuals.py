"""Plot 5-scene residuals for the 2026-05-18 1-frame BA writeup.

Two sub-plots (pitch / yaw), 5 scenes on x-axis, two solvers as lines.
Output: docs/assets/2026-05-18_one_frame_ba_residuals.png
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Tabulated from this run (yaw=+1°, pitch=+0.5° applied uniformly).
SCENES = ['d005_3000', 'd006_800', 'IWATESAN', 'd005_510', 'd002_350']
GT_PITCH = 0.500
GT_YAW   = 1.000

# (pitch, yaw) per scene
cf = {
    'pitch': np.array([0.504, 0.437, 0.407, 0.503, 0.467]),
    'yaw':   np.array([1.011, 0.945, 0.923, 0.930, 0.917]),
}
kbcf = {
    'pitch': np.array([0.503, 0.478, 0.479, 0.478, 0.509]),
    'yaw':   np.array([1.076, 1.179, 1.062, 1.042, 1.030]),
}

fig, (ax_p, ax_y) = plt.subplots(2, 1, figsize=(8, 5.4), dpi=120, sharex=True)
x = np.arange(len(SCENES))

ax_p.axhline(GT_PITCH, color='gray', linestyle=':', linewidth=1, label=f'GT = {GT_PITCH:.3f}°')
ax_p.plot(x, cf['pitch'],   'o-', color='#1f77b4', label='pinhole closed-form (2-DoF, centre band)')
ax_p.plot(x, kbcf['pitch'], 's-', color='#d62728', label='KB closed-form (2-DoF, full frame)')
ax_p.set_ylabel('pitch δ̂ [deg]')
ax_p.set_ylim(0.35, 0.55)
ax_p.legend(loc='lower right', fontsize=8)
ax_p.set_title('1-frame BA — pitch estimate (GT=+0.500°)', fontsize=10)
ax_p.grid(alpha=0.3)

ax_y.axhline(GT_YAW, color='gray', linestyle=':', linewidth=1, label=f'GT = {GT_YAW:.3f}°')
ax_y.plot(x, cf['yaw'],   'o-', color='#1f77b4', label='pinhole closed-form (2-DoF, centre band)')
ax_y.plot(x, kbcf['yaw'], 's-', color='#d62728', label='KB closed-form (2-DoF, full frame)')
ax_y.set_ylabel('yaw δ̂ [deg]')
ax_y.set_ylim(0.85, 1.22)
ax_y.legend(loc='upper right', fontsize=8)
ax_y.set_title('1-frame BA — yaw estimate (GT=+1.000°)', fontsize=10)
ax_y.grid(alpha=0.3)
ax_y.set_xticks(x)
ax_y.set_xticklabels(SCENES, rotation=15, fontsize=9)
ax_y.set_xlabel('scene')

fig.tight_layout()
out = REPO / 'docs' / 'assets' / '2026-05-18_one_frame_ba_residuals.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'wrote → {out}')
