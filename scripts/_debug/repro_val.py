"""Reproduce trainer val pass on 800 first val samples → expect val_mse 3.88, val_nll 2.78."""
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model
from models.model_cov import gaussian2d_nll

EXP   = 'km_only_15deg_06m_n2_img128_fp16_dgx2'
CACHE = '/cache/kamikado_v3_tiled'
device = torch.device('cuda')

ds = PandaSetCalibDatasetFull(CACHE, split='val',
        img_size=128, min_crop_px=256, max_crop_px=384,
        max_rot_deg=1.5, max_offset_m=0.6,
        grid_n=16, oversample=16, center_band=0.5)
val_subset = Subset(ds, list(range(800)))
loader = DataLoader(val_subset, batch_size=64, num_workers=4,
                     collate_fn=collate_full, shuffle=False)

model = load_calib_model(EXP).eval()
nll_s = mse_s = 0.0; n = 0
t0 = time.time()
with torch.no_grad():
    for batch in loader:
        if len(batch) == 8:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v, _ = batch
        else:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch
        imgs = imgs.to(device).float().div_(255.0)
        true_uvd = true_uvd.to(device); dist_uvd = dist_uvd.to(device)
        pad_mask = pad_mask.to(device); vfp = vfp.to(device)
        b_uvd = b_uvd.to(device); b_v = b_v.to(device)
        gt = true_uvd[..., :2] - dist_uvd[..., :2]
        use_int = bool(getattr(model, 'use_intensity', False))
        pin = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
                if use_int else dist_uvd[..., :3])
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            params = model(imgs, pin, key_padding_mask=pad_mask, vfp=vfp,
                            bucket_uvd=b_uvd, bucket_valid=b_v)
        valid = ~pad_mask
        if valid.any():
            loss = gaussian2d_nll(params[valid], gt[valid])
            err  = (params[valid][..., :2].float() - gt[valid]).norm(dim=-1).mean()
            nll_s += loss.item(); mse_s += err.item(); n += 1
print(f'val 800: batches={n}  val_nll={nll_s/n:+.4f}  val_mse={mse_s/n:.3f}  elapsed={time.time()-t0:.1f}s')
print('train.log target (ep30): val_nll=2.78  val_mse=3.88')
