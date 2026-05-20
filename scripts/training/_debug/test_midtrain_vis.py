"""Standalone test: run vis_pretrain_run + midtrain_vis once each, on a 3-cache
ConcatDataset alongside a live LMDB train DataLoader. Verifies the parent
process .pt vis path doesn't poison the worker LMDB envs.
"""
import sys, pathlib, time, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from torch.utils.data import DataLoader
from models.model_depth import CalibNetDepth
from scripts.util.midtrain_vis import vis_pretrain_run, midtrain_vis
from pathlib import Path

caches = ['/cache/kamikado_v3_tiled', '/cache/woven_v3_tile', '/cache/waymo_v3_tiled_i']
exp = Path('/tmp/test_vis_exp'); exp.mkdir(exist_ok=True, parents=True)

# ── 1. parent vis (was crashing the workers) ──
for cp in caches:
    print(f'=== vis_pretrain {cp} ===', flush=True)
    sub_exp = exp / Path(cp).name
    sub_exp.mkdir(exist_ok=True)
    vis_pretrain_run(sub_exp, cp, n=3)

# ── 2. now spin up a real train_loader with workers + LMDB ──
ds_kw = dict(img_size=64, min_crop_px=128, max_crop_px=256,
             max_offset_m=0.2, max_rot_deg=0.5, oversample=1)
trs = []
for c in caches:
    trs.append(PandaSetCalibDatasetFull(c, split='train', **ds_kw))
from torch.utils.data import ConcatDataset, Subset
full = ConcatDataset(trs)
sub = Subset(full, list(range(0, len(full), max(1, len(full)//200))))
loader = DataLoader(sub, batch_size=4, num_workers=4, collate_fn=collate_full,
                    persistent_workers=True, pin_memory=False,
                    multiprocessing_context='spawn')
print('iterating loader...', flush=True)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 10: break
print(f'OK 10 batches in {time.time()-t0:.1f}s', flush=True)

# ── 3. midtrain_vis (parent .pt path again) ──
print('=== midtrain_vis pass ===', flush=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CalibNetDepth(img_size=64, in_channels=3, n_layers=2).to(device)
for cp in caches:
    sub_exp = exp / Path(cp).name
    midtrain_vis(model, sub_exp, cp, epoch=10,
                 img_size=64, min_crop_px=128, max_crop_px=256,
                 device=device, n=3)

# ── 4. iterate loader AGAIN — this is the case that crashed before ──
print('=== iterating loader after midtrain_vis ===', flush=True)
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= 10: break
print(f'OK 10 more batches in {time.time()-t0:.1f}s', flush=True)
print('ALL OK', flush=True)
