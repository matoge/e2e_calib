"""Isolate the nan: is _ba_pose_loss FORWARD finite on batch 1, and is its
BACKWARD grad finite? Warm-start ckpt, one real batch."""
import sys; sys.path.insert(0, '/work')
import torch
from torch.utils.data import DataLoader
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.calibnet2 import CalibNet2
from datasets.train_cnd2_ddp import _ba_pose_loss
from models.model_cov import gaussian2d_nll

dev = torch.device('cuda')
ds = PandaSetCalibDatasetFull('/data', split='train', u_band=0.0, img_size=128, grid_n=16,
                              min_crop_px=128, max_crop_px=512, max_rot_deg=1.0, max_offset_m=0.20,
                              oversample=4, split_pert=False, pair_mode=False)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_full)
batch = next(iter(dl))
batch = [b.to(dev) if torch.is_tensor(b) else b for b in batch]
imgs, true_uvd, dist_uvd, pad, vfp, b_uvd, b_v = batch[:7]
imgs = imgs.float().div(255.0)
point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
m = CalibNet2(d=128, img_size=128, in_channels=3, use_intensity=True, frustum_grid_n=16,
              n_iter=4, n_heads=4, d_scalar=8, n_type1=40, use_info_head=True).to(dev)
sd = torch.load('/work/experiments/cnd2_ps_calib_repro_sk2/best_model.pt', map_location='cpu', weights_only=False)
sd = {k.removeprefix('module.'): v for k, v in sd.items()}
m.load_state_dict(sd, strict=False); m.train()
out = m(imgs, point_in, dpose_R=None, vfp=vfp, bucket_uvd=b_uvd, bucket_valid=b_v, key_padding_mask=pad)
per_pt = out[0] if isinstance(out, tuple) else out
print('per_pt finite:', torch.isfinite(per_pt).all().item())
for it in (1, 4):
    ba_l, diag = _ba_pose_loss(per_pt, dist_uvd, pad, batch, ba_iter=it, damping=1e-3, loss_type='nll')
    print(f'[ba_iter={it}] forward ba_l={ba_l.item():.4f} finite={torch.isfinite(ba_l).item()} diag={diag}')
# backward through ba_iter=1
ba_l, diag = _ba_pose_loss(per_pt, dist_uvd, pad, batch, ba_iter=1, damping=1e-3, loss_type='nll')
m.zero_grad(); ba_l.backward()
g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
print(f'backward: grad finite={torch.isfinite(g).all().item()} norm={g.norm().item():.3e} '
      f'n_nan={torch.isnan(g).sum().item()}/{g.numel()}')
