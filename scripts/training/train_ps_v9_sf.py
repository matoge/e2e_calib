"""Ablation: self-first decoder (self-attn → cross-attn per block).

Identical to ps_v9_objsplit except self_first=True.
"""
from train_ps_v9 import main, CFG

if __name__ == "__main__":
    cfg = {**CFG, "name": "ps_v9_objsplit_sf", "self_first": True}
    main(cfg)
