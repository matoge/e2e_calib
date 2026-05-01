"""ps_v19: PandaSet sigma=1°/0.2m sanity check before sigma=3°.

Same recipe as ps_v17 but at moderate perturbation. Cross-frame v31 (AV2)
hit val_err=3.12px at this sigma; if v19 doesn't comfortably beat that
baseline-shift drop on PandaSet, the architecture has a deeper problem
than just sigma scaling.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v11_lidar_kv import main, CFG as BASE

CFG = dict(BASE)
CFG.update(
    name             = 'ps_v19_panda_sigma1',
    scenes_root_pair = '/mnt/nvme6t/pandaset',
    cameras          = 'all',
    sigma_ypr        = 1.0,
    sigma_t          = 0.2,
    use_pose_emb     = True,
    use_lidar_kv     = True,
    epochs           = 100,
    crop_min         = 128,
    crop_max         = 384,
    max_points       = 256,
    virtual_epoch    = 10000,
)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/calib')
    ap.add_argument('--queue', default=None)
    args = ap.parse_args()
    main(cfg=CFG, clearml=args.clearml, clearml_project=args.clearml_project, queue=args.queue)
