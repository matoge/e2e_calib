"""Disk-backed lazy version of PandaSetCalibDataset.

Per __init__: loads only the small meta.pt index (~MB).
Per __getitem__: torch.load's a single per-instance .pt file (~170KB).

OS page cache absorbs repeated reads; memory stays small regardless of
cache size. Compatible with the existing build_sample / collate_pandaset
in datasets/pandaset.py.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.pandaset import build_sample, CACHE_IMG  # reuse existing helpers


class PandaSetCalibDatasetLazy(Dataset):
    """Lazy disk-backed PandaSet calib dataset.

    Args:
        cache_dir: directory containing meta.pt + inst/*.pt
        split: 'train' or 'val'
        img_size: output patch size (px)
        max_offset_m: extrinsic translation perturbation half-range (m)
        max_rot_deg: extrinsic YPR perturbation half-range (deg)
        min_pts: minimum LiDAR points required per sample
        max_tries: max perturbation re-samples before falling back
    """
    def __init__(self,
                 cache_dir: str,
                 split: str = 'train',
                 img_size: int = 64,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5,
                 min_pts: int = 8,
                 max_tries: int = 20,
                 min_sub_px: int = None,
                 max_sub_px: int = None):
        self.cache_dir    = Path(cache_dir)
        self.inst_dir     = self.cache_dir / 'inst'
        meta              = torch.load(self.cache_dir / 'meta.pt', weights_only=False)
        self.fnames       = meta[split]                     # list of '<id>.pt'
        self.img_size     = img_size
        self.max_offset_m = max_offset_m
        self.max_rot_deg  = max_rot_deg
        self.min_pts      = min_pts
        self.max_tries    = max_tries
        # cache_img read from meta when present (v2 = 384), else legacy default 192
        self.cache_img    = int(meta.get('side', 192)) if isinstance(meta, dict) else 192
        # sub_px = side of the random sub-window cropped from the cache patch
        # before resize-to-img_size. min..max gives the (zoom-out) difficulty
        # range. Pass max_sub_px=192 with the v2 384-cache to match v1 cache's
        # difficulty distribution (= same residual stats as ps_v9_lazy baseline)
        # while still benefiting from the wider 384 context for crop selection.
        self.min_sub_px   = int(min_sub_px) if min_sub_px is not None else img_size
        self.max_sub_px   = (min(int(max_sub_px), self.cache_img)
                             if max_sub_px is not None else self.cache_img)

    def __len__(self):
        return len(self.fnames)

    def _load_inst(self, idx: int) -> dict:
        return torch.load(self.inst_dir / self.fnames[idx], weights_only=False)

    def _sample_sub(self, inst):
        C = self.cache_img
        s_min = max(self.img_size, int(self.min_sub_px))
        s_max = min(C, int(self.max_sub_px))
        s = int(np.random.randint(s_min, s_max + 1))
        u = int(np.random.randint(0, C - s + 1))
        v = int(np.random.randint(0, C - s + 1))
        return (u, v, s)

    def __getitem__(self, idx):
        inst = self._load_inst(idx % len(self))
        for _ in range(self.max_tries):
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            sub     = self._sample_sub(inst)
            out = build_sample(inst, ypr, t_delta,
                                img_size=self.img_size, min_pts=self.min_pts,
                                sub=sub)
            if out is not None:
                img_crop, true_uvd, dist_uvd, _, _ = out
                return img_crop, true_uvd, dist_uvd
        # all retries failed → recurse into a random other instance
        return self[random.randint(0, len(self) - 1)]


class PandaSetCalibDatasetLazyVFP(PandaSetCalibDatasetLazy):
    """Returns 4-tuple (img, true_uvd, dist_uvd, vfp). vfp = scale anchor."""
    def __getitem__(self, idx):
        inst = self._load_inst(idx % len(self))
        for _ in range(self.max_tries):
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            sub     = self._sample_sub(inst)
            out = build_sample(inst, ypr, t_delta,
                                img_size=self.img_size, min_pts=self.min_pts,
                                sub=sub)
            if out is not None:
                img_crop, true_uvd, dist_uvd, _, vfp = out
                return img_crop, true_uvd, dist_uvd, torch.tensor(vfp, dtype=torch.float32)
        return self[random.randint(0, len(self) - 1)]


def collate_pandaset_vfp(batch):
    """5-tuple OR 7-tuple collate.

    5-tuple input → 5-tuple output: (imgs, true_t, dist_t, mask, vfp_t).
    6-tuple input (img, true, dist, vfp, uvd_full, pad_full) → 7-tuple output:
        (imgs, true_t, dist_t, mask, vfp_t, uvd_full_t, pad_full_t).

    The 7-tuple form is used by MultiDSCalibDataset, which surfaces the dense
    ~2048-pt lidar context for the frustum encoder's per-cell PointNet pool.
    """
    sample0 = batch[0]
    has_full = len(sample0) >= 6
    if has_full:
        imgs, true_uvds, dist_uvds, vfps, uvd_fulls, pad_fulls = zip(*batch)
    else:
        imgs, true_uvds, dist_uvds, vfps = zip(*batch)
    max_n = max(t.shape[0] for t in true_uvds)
    def pad(tensors):
        out  = torch.zeros(len(tensors), max_n, tensors[0].shape[1])
        mask = torch.ones(len(tensors), max_n, dtype=torch.bool)
        for i, t in enumerate(tensors):
            out[i, :len(t)] = t; mask[i, :len(t)] = False
        return out, mask
    true_t, _    = pad(true_uvds)
    dist_t, mask = pad(dist_uvds)
    if has_full:
        # uvd_full / pad_full are already padded to fixed N_full per sample;
        # MultiDSCalibDataset's pair source pads at _pack_full_local time.
        return (torch.stack(imgs), true_t, dist_t, mask, torch.stack(vfps),
                torch.stack(uvd_fulls), torch.stack(pad_fulls))
    return torch.stack(imgs), true_t, dist_t, mask, torch.stack(vfps)


if __name__ == '__main__':
    ds = PandaSetCalibDatasetLazy(
        '/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy', split='train')
    print(f"instances: {len(ds)}")
    img, t, d = ds[0]
    print(f"img={img.shape}  true_uvd={t.shape}")
    print(f"shift mean: {(t[:, :2] - d[:, :2]).norm(dim=1).mean():.2f}px")
