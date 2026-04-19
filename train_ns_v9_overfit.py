"""Overfit smoke test: 500 dense samples (>=80 pts), 1000 ep.
Expected: NLL descends, MSE floors ~3px due to LiDAR sparsity (grid cell ≈ 4px).
"""
from train_ns_v9 import main, CFG

if __name__ == "__main__":
    cfg = {**CFG,
           "name":        "ns_v9_overfit500",
           "min_pts":     80,
           "subset_size": 500,
           "epochs":      1000,
           "batch_size":  32}
    main(cfg)
