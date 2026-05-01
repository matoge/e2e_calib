"""ps_v21: fast sanity — 39-scene PandaSet (mininas), front_camera, sigma=0.5° uniform.

Goal: shortest path to v9_lazy-equivalent baseline. If v21 fails to hit
~val_mse 1.4-2 px obj in 30ep, the new pair-path stack has a regression.
ETA ~15-20min/run vs the 50min for the all-103-scene 6-cam configurations.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v11_lidar_kv import main, CFG as BASE

CFG = dict(BASE)
CFG.update(
    name             = 'ps_v21_panda39_sigma05_uniform',
    scenes_root_pair = '/mnt/mininas/datasets/pandaset',   # 39 scenes
    cameras          = 'front_camera',                      # match v9_lazy
    sigma_ypr        = 0.5,
    sigma_t          = 0.2,
    uniform_perturb  = True,
    use_pose_emb     = True,
    use_lidar_kv     = True,
    epochs           = 30,                                  # was 100
    crop_min         = 128,
    crop_max         = 384,
    max_points       = 256,
    virtual_epoch    = 2000,                                # was 10000 → 5x faster
    train_frac       = 0.85,                                # 33 train / 6 val on 39
)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--clearml-project', default='e2e_calib/calib')
    ap.add_argument('--queue', default=None)
    args = ap.parse_args()
    main(cfg=CFG, clearml=args.clearml, clearml_project=args.clearml_project, queue=args.queue)
