"""Ablation: no frustum encoding (plain PointMLP on (u, v, d)).

Identical to ps_v9_objsplit except use_frustum=False.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from train_ps_v9 import main, CFG

if __name__ == "__main__":
    cfg = {**CFG, "name": "ps_v9_objsplit_nofrust", "use_frustum": False}
    main(cfg)
