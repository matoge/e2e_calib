"""ps_v17: PandaSet-only at sigma=3.0°/0.3m via the pair _try_one_calib path
            with adapter-fixed obj + vfl plumbing + use_pose_emb=True.

Purpose: nail down the obj/bg split at the same sigma as ps_v15/v16, so we
can compare apples-to-apples whether NLL≈2 is just BG saturating at the
wide-σ regime (cuboid points should sit way below NLL≈2 if model learns).
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v11_lidar_kv import main, CFG as BASE


CFG = dict(BASE)
CFG.update(
    name             = 'ps_v17_panda_sigma3',
    scenes_root_pair = '/mnt/nvme6t/pandaset',   # PandaSet only — obj annotations exist
    cameras          = 'all',                     # 6 cameras
    sigma_ypr        = 3.0,
    sigma_t          = 0.3,
    use_pose_emb     = True,                      # vfp conditioning ON
    use_lidar_kv     = True,
    epochs           = 100,                       # was 30 in v15/v16 — too short
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
