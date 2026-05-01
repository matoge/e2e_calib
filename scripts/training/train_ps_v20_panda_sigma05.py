"""ps_v20: PandaSet sigma=0.5°/0.2m sanity at v9_lazy baseline equivalence.

Same recipe as ps_v17 but at sigma matching ps_v9_lazy (val_nll=1.8141 /
val_mse=1.40px obj). If v20 fails to reach a comparable point, the new
pair-path + pose_emb + lidar_kv stack has a regression vs lazy-cache path,
not a sigma-scaling issue.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v11_lidar_kv import main, CFG as BASE

CFG = dict(BASE)
CFG.update(
    name             = 'ps_v20_panda_sigma05',
    scenes_root_pair = '/mnt/nvme6t/pandaset',
    cameras          = 'all',
    sigma_ypr        = 0.5,
    sigma_t          = 0.2,
    use_pose_emb     = True,
    use_lidar_kv     = True,
    epochs           = 100,
    crop_min         = 128,
    crop_max         = 384,
    max_points       = 256,
    virtual_epoch    = 10000,
    uniform_perturb  = True,    # match ps_v9_lazy uniform [-sigma, +sigma]
)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/calib')
    ap.add_argument('--queue', default=None)
    args = ap.parse_args()
    main(cfg=CFG, clearml=args.clearml, clearml_project=args.clearml_project, queue=args.queue)
