"""ps_v18: AV2-only calib at sigma=3°/0.3m, all 7 ring cameras.

Same recipe as ps_v17 (PD sigma3) but on the freshly-cuboid-annotated
Argoverse 2 PS conversion. Sanity check that AV2 alone calibrates with
obj/bg split surfaced by the adapter fix.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v11_lidar_kv import main, CFG as BASE


CFG = dict(BASE)
CFG.update(
    name             = 'ps_v18_av2_sigma3',
    scenes_root_pair = '/mnt/nvme6t/av2_ps',
    cameras          = 'all',
    sigma_ypr        = 3.0,
    sigma_t          = 0.3,
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
    main(cfg=CFG, clearml=args.clearml, clearml_project=args.clearml_project,
         queue=args.queue)
