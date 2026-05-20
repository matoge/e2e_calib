import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
from scripts.util.vis import visualize
from scripts.inference.infer_calib import load_calib_model

EXP = 'km_only_15deg_06m_n2_img128_fp16_dgx2'
CACHE = '/cache/kamikado_v3_tiled'
ds_kw = dict(
    img_size=128, min_crop_px=256, max_crop_px=384,
    max_rot_deg=1.5, max_offset_m=0.6,
    grid_n=16, oversample=1, center_band=0.5,
)
model = load_calib_model(EXP).eval()
exp_dir = REPO / 'experiments' / EXP / '_eval_vis'
exp_dir.mkdir(parents=True, exist_ok=True)
visualize(model, exp_dir, CACHE, epoch=999, ds_kw=ds_kw, n=10)
print('done →', exp_dir)
