"""Pull val NLL/MSE curves from DGX2 ClearML for n4 / n6 / resume runs and
plot them so the depth-scaling story (n4 vs n6, plus the resume) is
self-evident. Output: docs/assets/2026-05-18_one_frame_ba/learning_curves.png
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

os.environ.setdefault('CLEARML_CONFIG_FILE', '/home/hfunaya/clearml-dgx2.conf')

from clearml import Task
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

NAMES = [
    ('n6, DGX2-8gpu',           'km_wv_wm_n6_img128_cs256_512_200ep_dgx2_8gpu',         '#1f77b4'),
    ('n4, DGX1-16gpu (resume)', 'km_wv_wm_n4_img128_cs256_512_200ep_dgx1_16gpu_resume', '#d62728'),
]

fig, (ax_nll, ax_mse) = plt.subplots(2, 1, figsize=(8, 6), dpi=120, sharex=True)

for label, name, color in NAMES:
    ts = Task.get_tasks(project_name='e2e_calib/calib', task_name=name)
    if not ts:
        print(f'  {name}: not found')
        continue
    t = ts[0]
    sc = t.get_reported_scalars()
    if 'nll' not in sc or 'val' not in sc.get('nll', {}):
        print(f'  {name}: no nll/val curve')
        continue
    # Each series dict has 'x' (epoch) and 'y' (value)
    nll_v = sc['nll']['val']; nll_x = nll_v['x']; nll_y = nll_v['y']
    mse_v = sc['mse_px']['val']; mse_x = mse_v['x']; mse_y = mse_v['y']
    print(f'  {name}: ep {nll_x[0]:.0f}–{nll_x[-1]:.0f}, '
          f'val nll {nll_y[0]:.3f}→{nll_y[-1]:.3f}, '
          f'val mse {mse_y[0]:.2f}→{mse_y[-1]:.2f}')
    ax_nll.plot(nll_x, nll_y, '-', color=color, lw=1.4, label=f'{label}')
    ax_mse.plot(mse_x, mse_y, '-', color=color, lw=1.4, label=f'{label}')

ax_nll.set_ylabel('val NLL')
ax_nll.set_title('Training curves — depth ablation (n4 vs n6, ConvNeXt, img128, cs256-512, 200ep)', fontsize=10)
ax_nll.legend(loc='upper right', fontsize=9)
ax_nll.grid(alpha=0.3)

ax_mse.set_ylabel('val MSE [px]')
ax_mse.set_xlabel('epoch')
ax_mse.legend(loc='upper right', fontsize=9)
ax_mse.grid(alpha=0.3)

fig.tight_layout()
out = REPO / 'docs' / 'assets' / '2026-05-18_one_frame_ba' / 'learning_curves.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'wrote → {out}')
