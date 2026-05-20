"""Run the trainer's epoch_loop verbatim on the TRAIN split, using
DataLoader + collate_full + train mode. Print mean nll/mse so we can
compare directly with the running km_overfit_cnx train.log values."""
import sys, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np, torch
from torch.utils.data import DataLoader, Subset

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model
from models.model_cov import gaussian2d_nll


EXP   = 'km_overfit_cnx_n2_os16_40ep'
CACHE = '/cache/kamikado_v3_tiled'
N     = 1000

# Replicate trainer's ds_kw EXACTLY (no img_size key — matches the trainer).
ds = PandaSetCalibDatasetFull(CACHE, split='train',
    max_offset_m=0.20, max_rot_deg=0.5,
    min_crop_px=256, max_crop_px=384,
    grid_n=16, oversample=16)
loader = DataLoader(Subset(ds, list(range(N))),
                     batch_size=64, num_workers=4,
                     collate_fn=collate_full, shuffle=False)

model = load_calib_model(EXP).eval()  # CalibNetDepth, eval mode
device = torch.device('cuda')
nll_s = mse_s = 0.0; n = 0
t0 = time.time()
for batch in loader:
    if len(batch) == 8:
        imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v, _ = batch
    else:
        imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch
    imgs     = imgs.to(device).float().div_(255.0)
    true_uvd = true_uvd.to(device); dist_uvd = dist_uvd.to(device)
    pad_mask = pad_mask.to(device); vfp = vfp.to(device)
    b_uvd = b_uvd.to(device); b_v = b_v.to(device)
    gt = true_uvd[..., :2] - dist_uvd[..., :2]
    use_intensity = bool(getattr(model, 'use_intensity', False))
    point_in = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
                  if use_intensity else dist_uvd[..., :3])
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        params = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                        bucket_uvd=b_uvd, bucket_valid=b_v)
    valid = ~pad_mask
    if valid.any():
        loss = gaussian2d_nll(params[valid], gt[valid])
        err  = (params[valid][..., :2].float() - gt[valid]).norm(dim=-1).mean()
        nll_s += loss.item(); mse_s += err.item(); n += 1
print(f'replay TRAIN split  N={N}  batches={n}  '
      f'mean_nll={nll_s/max(n,1):+.4f}  mean_mse(=mean px err)={mse_s/max(n,1):.3f}  '
      f'elapsed={time.time()-t0:.1f}s')
