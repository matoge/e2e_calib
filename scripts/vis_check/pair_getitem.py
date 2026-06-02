"""vis_check: PandaSetCalibDatasetFull (pair_mode=True) の __getitem__ が emission
する (built_A, built_B, dpose_AB) tuple を可視化して ClearML Debug Samples
panel に流す。

Subproject layout:
  e2e_calib/cross-frame/vis_check/<run_name>

各 sample について 2-panel:
  - 左 (A): img_A + numbered ○ at true_uvd_A[i] (= pts_A[cand_idx_A[i]] の A 投影、tab10)
  - 右 (B): img_B + same-color ○ at true_uvd_B[i] (GT) + × at dist_uvd_B[i] (HAT)
  - ConnectionPatch dotted lines: A の点 i ↔ B GT 点 i (commit bf328de で対応)

Caption: pose_HAT ε (B 側 pert_vec) と dpose_AB GT。

Run inside docker:
  docker run --rm --gpus '"device=10"' --shm-size=8g \
    -v /home/hfunaya/git/e2e_calib:/workspace \
    -v /home/hfunaya:/home/hfunaya \
    -v /mnt/fsx:/mnt/fsx \
    -v /home/hfunaya/.clearml.conf:/root/clearml.conf:ro \
    -e CLEARML_CONFIG_FILE=/root/clearml.conf \
    -e PYTHONPATH=/workspace \
    -w /workspace e2e-calib-train:np2 \
    /opt/conda/bin/python scripts/vis_check/pair_getitem.py
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

# matplotlib first (Agg) + IPython stub before clearml import.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401
import IPython as _ipy
if not hasattr(_ipy, 'version_info'):
    _ipy.version_info = (0, 0, 0)

from clearml import Task as _Task
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.pandaset_full import PandaSetCalibDatasetFull  # noqa: E402
from scripts.vis_check._pair_render import render_one_pair  # noqa: E402

CACHE   = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
IMG_SIZE = 128
N_SAMPLES = 8
K_SHOW = 12
SEED = 0


def main() -> int:
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE,
        img_size=IMG_SIZE,
        oversample=1,
        max_offset_m=0.20, max_rot_deg=1.0,
        pair_mode=True, pair_stride=10,
        grid_n=16, same_frame_self_sup=False,
        split='train',
    )
    print(f'dataset len = {len(ds)}', flush=True)

    run_name = f'cnd2_pair_getitem_{datetime.now().strftime("%m%d_%H%M")}'
    task = _Task.init(project_name='e2e_calib/cross-frame/vis_check',
                       task_name=run_name,
                       task_type=_Task.TaskTypes.qc,
                       auto_connect_frameworks={'matplotlib': False})
    task.connect({'cache': CACHE, 'img_size': IMG_SIZE,
                   'n_samples': N_SAMPLES, 'pair_mode': True,
                   'pair_stride': 10, 'rot_deg': 1.0, 't_m': 0.20,
                   'k_show': K_SHOW})
    logger = task.get_logger()

    rng = np.random.default_rng(SEED)
    n_drawn = 0; tries = 0
    while n_drawn < N_SAMPLES and tries < N_SAMPLES * 10:
        tries += 1
        idx = int(rng.integers(0, len(ds)))
        try:
            sample = ds[idx]
        except Exception as e:
            print(f'[idx={idx}] err: {e}'); continue
        if sample is None: continue
        if isinstance(sample, list):
            if not sample: continue
            sample = sample[0]
        built_A, built_B, dpose_AB = sample
        out_path = Path('/tmp') / f'pair_getitem_{n_drawn:02d}_idx{idx}.png'
        res = render_one_pair(built_A, built_B, dpose_AB,
                                out_path=out_path, img_size=IMG_SIZE,
                                k_show=K_SHOW,
                                suptitle_prefix=f'pair idx={idx}  ')
        if res is None:
            print(f'[idx={idx}] too few in-image pts, skip'); continue
        path, stats = res
        logger.report_image(title='pair_getitem',
                             series=f'sample_{n_drawn:02d}_idx{idx}',
                             iteration=0, local_path=str(path))
        print(f"wrote {path}  N_ok={stats['n_ok']}  HAT→GT={stats['err_hyp']:.1f}px")
        n_drawn += 1

    print(f'\ndone: {n_drawn} samples uploaded to '
           f'project=e2e_calib/cross-frame/vis_check task={run_name}')
    print(f'task id={task.id}')
    print(f'url={task.get_output_log_web_page()}')
    task.close()
    return 0 if n_drawn == N_SAMPLES else 1


if __name__ == '__main__':
    sys.exit(main())
