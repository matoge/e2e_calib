"""Ablation: self-first decoder (self-attn → cross-attn per block).

Identical to ps_v9_objsplit except self_first=True.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from train_ps_v9 import main, CFG

if __name__ == "__main__":
    cfg = {**CFG, "name": "ps_v9_objsplit_sf", "self_first": True}
    main(cfg)
