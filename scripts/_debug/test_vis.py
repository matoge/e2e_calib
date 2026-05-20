"""Smoke test for scripts/util/vis.py — confirm err matches val pass."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.util.vis import visualize
from scripts.inference.infer_calib import load_calib_model
from scripts.inference.infer_pipeline import make_ds

EXP   = 'km_overfit_cnx_n2_os16_40ep'
CACHE = '/cache/kamikado_v3_tiled'

# build ds_kw the SAME way trainer does (no img_size key — see infer_pipeline.make_ds)
_, c = make_ds(EXP, CACHE, split='val', oversample=1)
ds_kw = dict(
    # mirror trainer's defaults exactly (train_ps_v3_ddp.py line 215-221)
    min_crop_px  = c.get('min_crop_px', 128),
    max_crop_px  = c.get('max_crop_px', 512),
    max_rot_deg  = c.get('max_rot_deg', 0.5),
    max_offset_m = c.get('max_offset_m', 0.20),
    max_fx_pct   = c.get('max_fx_pct', 0.0),
    max_fy_pct   = c.get('max_fy_pct', 0.0),
    pose_frame   = c.get('pose_frame', 'orig'),
    grid_n       = c.get('grid_n', 16),
    n_full       = c.get('n_full', 1024),
    k_per_cell   = c.get('k_per_cell', 8),
    oversample   = 1,
)
model = load_calib_model(EXP).eval()
exp_dir = REPO_ROOT / 'experiments' / EXP
visualize(model, exp_dir, CACHE, epoch=999, ds_kw=ds_kw, n=10)
