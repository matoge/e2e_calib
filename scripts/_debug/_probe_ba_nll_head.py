"""Smoke-probe the BA pose-NLL head on one PS front-cam calib batch.

Validates (no training):
  1. collate_full carries K_orig/cs/u0/v0 (else _ba_pose_loss is a silent no-op)
  2. CalibNet2 forward → per_pt
  3. _ba_pose_loss(loss_type='nll') returns a FINITE scalar + sane diagnostics
  4. backward() produces finite grads (E2E path is differentiable)
"""
import sys; sys.path.insert(0, '/home/hiro/git/e2e_calib')
import torch
from torch.utils.data import DataLoader
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.calibnet2 import CalibNet2
from datasets.train_cnd2_ddp import _ba_pose_loss

CACHE = '/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full'
dev = torch.device('cuda:0')

ds = PandaSetCalibDatasetFull(CACHE, split='train', u_band=0.0,
                              img_size=128, grid_n=16,
                              min_crop_px=128, max_crop_px=512,
                              max_rot_deg=1.0, max_offset_m=0.20,
                              oversample=2, split_pert=False, pair_mode=False)
dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                collate_fn=collate_full, drop_last=True)
batch = next(iter(dl))
names = ['imgs','true_p','dist_p','pad','vfps','b_uvds','b_valids','pert',
         'pts_cam_orig','duv_orig','K_orig','cs_t','delta1','u0_t','v0_t']
print(f'[batch] len={len(batch)}')
for i in (7, 10, 11, 13, 14):
    v = batch[i]
    print(f'  batch[{i}] {names[i]:12s}: '
          + ('None  <-- HEAD WOULD BE NO-OP!' if v is None
             else f'shape={tuple(v.shape)}'))

batch = [b.to(dev) if torch.is_tensor(b) else b for b in batch]
imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
imgs = imgs.float().div(255.0)
point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)

model = CalibNet2(d=128, img_size=128, in_channels=3, use_intensity=True,
                  frustum_grid_n=16, n_iter=4, n_heads=4,
                  d_scalar=8, n_type1=40, use_info_head=True).to(dev)
out = model(imgs, point_in, dpose_R=None, vfp=vfp,
            bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
            key_padding_mask=pad_mask)
per_pt = out[0] if isinstance(out, tuple) else out
print(f'[fwd] per_pt {tuple(per_pt.shape)}  finite={torch.isfinite(per_pt).all().item()}')

for lt in ('nll', 'mse'):
    loss, diag = _ba_pose_loss(per_pt, dist_uvd, pad_mask, batch,
                               ba_iter=4, damping=1e-3, loss_type=lt)
    if loss is None:
        print(f'[{lt}] loss=None  <-- _ba_pose_loss bailed (missing K/cs/u0)')
        continue
    print(f'[{lt}] loss={loss.item():+.4f}  finite={torch.isfinite(loss).item()}  diag={diag}')
    if lt == 'nll':
        model.zero_grad()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()
                       if p.grad is not None])
        print(f'      grad: finite={torch.isfinite(g).all().item()}  '
              f'norm={g.norm().item():.3e}  max|g|={g.abs().max().item():.3e}')
print('DONE')
