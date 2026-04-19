"""Ablation: no frustum encoding (plain PointMLP on (u, v, d)).

Identical to ps_v9_objsplit except use_frustum=False.
"""
from train_ps_v9 import main, CFG

if __name__ == "__main__":
    cfg = {**CFG, "name": "ps_v9_objsplit_nofrust", "use_frustum": False}
    main(cfg)
