"""ps_v24: cross-frame on PandaSet 103 scenes, mirror of ps_v13 recipe.

Same cross-frame setup that landed val_err=1.59 px in v13 (sigma 0.5°/0.05m,
front_camera, 20ep, virtual_epoch=30000). Now a clean re-run with the new
PD scenes pool + cuboid-aware adapter for any post-fix calib expectation.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v12_cross_frame import main as v12_main, CFG as V12

import argparse, copy
CFG = copy.deepcopy(V12)
CFG.update(
    name         = 'ps_v24_cross_frame_panda',
    epochs       = 50,           # was 20 in v13; let it converge fully
    batch_size   = 32,
    sigma_ypr    = 1.0,
    sigma_t      = 0.10,
    cameras      = 'front_camera',
    scenes_root  = '/mnt/nvme6t/pandaset',
    virtual_epoch= 20000,
    crop_min     = 128,
    crop_max     = 384,
    n_overfit    = 0,
    baseline_min = 1,
    baseline_max = 10,
)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/cross-frame')
    ap.add_argument('--queue', default=None)
    args = ap.parse_args()
    v12_main(cfg=CFG, clearml=args.clearml, clearml_project=args.clearml_project, queue=args.queue)
