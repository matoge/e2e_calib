"""ps_v9_repro_check: run the exact ps_v9_lazy script with current model_depth
(now has unused use_pose_emb / use_lidar_kv flags, both default False).

Goal: confirm val_nll≈1.81 / val_mse_obj≈1.40 still reproduces. If yes, the
new pair-path is the regression locus, not the model. If no, model_depth.py
has drift.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.training.train_ps_v9_lazy import main as v9_main, CFG as V9_CFG
import argparse

CFG = dict(V9_CFG)
CFG['name'] = 'ps_v9_repro_check'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    v9_main(CFG)
