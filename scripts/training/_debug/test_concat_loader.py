"""Standalone test: 2-cache concat + DataLoader workers."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from torch.utils.data import DataLoader, ConcatDataset, Subset

ds_kw = dict(img_size=64, min_crop_px=128, max_crop_px=256,
             max_offset_m=0.2, max_rot_deg=0.5, frame_stride=1,
             grid_n=16, oversample=1)
caches = ['/cache/kamikado_v3_tiled', '/cache/woven_v3_tile']
trs = []
for c in caches:
    tr = PandaSetCalibDatasetFull(c, split='train', **ds_kw)
    va = PandaSetCalibDatasetFull(c, split='val',   **ds_kw)
    print(f'{c}: train={len(tr)} val={len(va)}', flush=True)
    trs.extend([tr, va])
full = ConcatDataset(trs)
print(f'concat={len(full)}', flush=True)
sub = Subset(full, list(range(0, len(full), max(1, len(full)//200))))
loader = DataLoader(sub, batch_size=16, num_workers=4, collate_fn=collate_full,
                    persistent_workers=True, pin_memory=False)
print('starting iter...', flush=True)
t0 = time.time()
for i, batch in enumerate(loader):
    if i == 0:
        names = ['imgs','true_uvd','dist_uvd','pad_mask','vfp','bucket_uvd','bucket_valid','pert']
        for n, t in zip(names, batch):
            print(f'  {n}: {tuple(t.shape) if hasattr(t,"shape") else type(t).__name__}', flush=True)
    if i >= 5:
        break
print(f'OK 5 batches in {time.time()-t0:.1f}s', flush=True)
