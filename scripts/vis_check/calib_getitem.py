"""vis_check: PandaSetCalibDatasetFull (calib mode, pair_mode=False) の
__getitem__ が emission する built tuple を可視化して ClearML Debug Samples
panel に流す。

Subproject layout:
  e2e_calib/calib/vis_check/<run_name>

各 frame について 1 panel:
  - img (crop, 128x128)
  - true_uvd (●lime, GT calib 投影)
  - dist_uvd (×red, calib 摂動入り、model 入力相当)
  - pert_vec の (t, ypr) を caption

Run inside docker:
  docker run --rm --gpus '"device=10"' --shm-size=8g \
    -v /home/hfunaya/git/e2e_calib:/workspace \
    -v /mnt/fsx:/mnt/fsx \
    -e PYTHONPATH=/workspace \
    -w /workspace e2e-calib-train:np2 \
    /opt/conda/bin/python scripts/vis_check/calib_getitem.py
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# matplotlib 3.7+ + ClearML IPython auto-detect → AttributeError on
# partially-initialised IPython module. Set Agg backend BEFORE any other
# import touches pyplot.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Stub IPython attribute that matplotlib's _fix_ipython_backend2gui expects.
import IPython as _ipy
if not hasattr(_ipy, 'version_info'):
    _ipy.version_info = (0, 0, 0)

from clearml import Task as _Task

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.pandaset_full import PandaSetCalibDatasetFull  # noqa: E402

CACHE   = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
IMG_SIZE = 128
N_SAMPLES = 8
SEED = 0


def main() -> int:
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE,
        img_size=IMG_SIZE,
        oversample=1,
        max_offset_m=0.20, max_rot_deg=1.0,
        pair_mode=False,
        grid_n=16,
        split='train',
    )
    print(f'dataset len = {len(ds)}', flush=True)

    # ClearML init
    run_name = f'cnd2_calib_getitem_{datetime.now().strftime("%m%d_%H%M")}'
    task = _Task.init(project_name='e2e_calib/calib/vis_check',
                       task_name=run_name,
                       task_type=_Task.TaskTypes.qc,
                       auto_connect_frameworks={'matplotlib': False})
    task.connect({'cache': CACHE, 'img_size': IMG_SIZE,
                   'n_samples': N_SAMPLES, 'pair_mode': False,
                   'rot_deg': 1.0, 't_m': 0.20})
    logger = task.get_logger()

    rng = np.random.default_rng(SEED)
    n_drawn = 0; tries = 0
    while n_drawn < N_SAMPLES and tries < N_SAMPLES * 8:
        tries += 1
        idx = int(rng.integers(0, len(ds)))
        try:
            sample = ds[idx]
        except Exception as e:
            print(f'[idx={idx}] err: {e}'); continue
        if sample is None: continue
        # __getitem__ returns list of (built,) tuples in calib mode
        # (or list of (built_A, built_B, dpose) in pair mode).
        if isinstance(sample, list):
            if not sample: continue
            sample = sample[0]
        # calib mode: sample is the built tuple itself.
        built = sample
        img      = built[0].numpy().transpose(1, 2, 0).astype(np.uint8)
        true_uvd = built[1].numpy()
        dist_uvd = built[2].numpy()
        pert_vec = built[6].numpy()

        in_img = ((true_uvd[:, 0] >= 0) & (true_uvd[:, 0] < IMG_SIZE) &
                  (true_uvd[:, 1] >= 0) & (true_uvd[:, 1] < IMG_SIZE))
        N_ok = int(in_img.sum())

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img); ax.axis('off')
        ax.scatter(true_uvd[in_img, 0], true_uvd[in_img, 1],
                    s=8, c='lime', label='true_uvd (GT calib)')
        ax.scatter(dist_uvd[in_img, 0], dist_uvd[in_img, 1],
                    s=8, c='red', alpha=0.5, marker='x',
                    label='dist_uvd (HAT, model input)')
        for j in np.where(in_img)[0]:
            ax.plot([true_uvd[j, 0], dist_uvd[j, 0]],
                    [true_uvd[j, 1], dist_uvd[j, 1]],
                    '-', color='yellow', lw=0.4, alpha=0.5)
        title = (f'idx={idx}  N={N_ok}\n'
                 f'pert: t=({pert_vec[0]:+.2f},{pert_vec[1]:+.2f},'
                 f'{pert_vec[2]:+.2f})m  '
                 f'ypr=({pert_vec[3]:+.2f},{pert_vec[4]:+.2f},'
                 f'{pert_vec[5]:+.2f})°')
        ax.set_title(title, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

        fig.tight_layout()
        out_path = Path('/tmp') / f'calib_getitem_{n_drawn:02d}_idx{idx}.png'
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        # Upload to ClearML Debug Samples
        logger.report_image(title='calib_getitem',
                             series=f'sample_{n_drawn:02d}_idx{idx}',
                             iteration=0, local_path=str(out_path))
        print(f'wrote {out_path}  N_ok={N_ok}')
        n_drawn += 1

    print(f'\ndone: {n_drawn} samples uploaded to '
           f'project=e2e_calib/calib/vis_check task={run_name}')
    print(f'task id={task.id}')
    print(f'url={task.get_output_log_web_page()}')
    task.close()
    return 0 if n_drawn == N_SAMPLES else 1


if __name__ == '__main__':
    sys.exit(main())
