"""img256/grid32 で LiDAR ↔ image ↔ grid cell の対応がずれてないか audit.

motivation: InfoHead 200ep が ep19 で全 NaN 爆発した。データ側に
スケール整合の壊れがあるか確認するための audit (NLL/InfoHead の
数式問題とは別軸)。

dump:
  1. dist_uvd[..., :2] の値域            (期待 [0, 256])
  2. bucket_uvd[..., :2] の値域           (期待 [0, 256])
  3. vfp が fx_orig * S/cs と一致するか
  4. (true_uv - dist_uv) の値域           (摂動 ±1.5° ±0.2m が orig→local px で何 px)
  5. *** dist_uvd の各 query 点が bucket_uvd の対応 cell に存在するか ***
     query 点 (u, v) → cell_id = floor(v/cell_S)*G + floor(u/cell_S)
     その cell の bucket が空だったり (u, v) と離れすぎてたら整合性破綻
  6. cs=128 (2× upscale) と cs=384 (0.66× downscale) で同 inst を流し、
     duv_orig (orig 単位 px) が cs に依存しないか
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full


CACHE = '/home/hfunaya/cache_v4/kamikado_v3_tiled'
IMG_SIZE = 256
GRID_N = 32
CELL_S = IMG_SIZE / GRID_N      # 8.0 px


def dump_batch_stats(name: str, batch):
    (imgs, true_p, dist_p, pad, vfps, b_uvds, b_valids, pert_6vec,
     pts_cam_orig, duv_orig, K_orig, cs_t) = batch
    B = imgs.shape[0]
    print(f"\n=== {name}  B={B} ===")

    # dtypes
    print(f"  imgs.dtype         = {imgs.dtype}")
    print(f"  dist_p.shape       = {tuple(dist_p.shape)}  (B, Nmax, 5)")
    print(f"  b_uvds.shape       = {tuple(b_uvds.shape)}  (B, G²={GRID_N*GRID_N}, K, 4)")

    # 1. dist_uvd[..., :2] 値域
    valid = ~pad
    dist_uv = dist_p[..., :2][valid]
    print(f"  dist_uv u: [{dist_uv[:, 0].min():.2f}, {dist_uv[:, 0].max():.2f}]  "
          f"v: [{dist_uv[:, 1].min():.2f}, {dist_uv[:, 1].max():.2f}]   "
          f"(target [0, {IMG_SIZE}])")

    # 2. bucket_uvd[..., :2] 値域 (valid のみ)
    b_uv = b_uvds[..., :2]
    b_v_mask = b_valids
    bv_uv = b_uv[b_v_mask]                     # (n_valid_pts, 2)
    if bv_uv.numel() > 0:
        print(f"  bucket_uv u: [{bv_uv[:, 0].min():.2f}, {bv_uv[:, 0].max():.2f}]  "
              f"v: [{bv_uv[:, 1].min():.2f}, {bv_uv[:, 1].max():.2f}]")
    else:
        print(f"  bucket_uv: 全 cell 空 !!!")

    # 3. vfp = fx * S / cs ?
    if K_orig is not None:
        fx = K_orig[:, 0, 0]
        cs = cs_t
        vfp_expect = fx * IMG_SIZE / cs
        diff = (vfps - vfp_expect).abs()
        print(f"  cs:           [{cs.min():.0f}, {cs.max():.0f}] px")
        print(f"  fx_orig:      [{fx.min():.1f}, {fx.max():.1f}]")
        print(f"  vfp:          [{vfps.min():.1f}, {vfps.max():.1f}]")
        print(f"  vfp expect:   [{vfp_expect.min():.1f}, {vfp_expect.max():.1f}]   "
              f"|diff| max={diff.max():.4f}")

    # 4. (true_uv - dist_uv) の値域 (= 摂動による Δuv, local 256-px scale)
    duv_local = (true_p[..., :2] - dist_p[..., :2])[valid]
    duv_norm = duv_local.norm(dim=-1)
    print(f"  Δuv (local px):  mean={duv_norm.mean():.2f}  med={duv_norm.median():.2f}  "
          f"max={duv_norm.max():.2f}")

    # 5. dist_uvd の各 query 点が bucket_uvd の対応 cell に存在するか
    # cell_id_query = floor(v/cell_S) * G + floor(u/cell_S)
    # 対応 cell の bucket に valid=True が 1 個以上あるか確認 (= 自分自身が
    # 必ず代表として入っているはず — そうでなければスケール整合バグ)
    G = GRID_N
    for bi in range(min(2, B)):
        v_mask = valid[bi]
        if v_mask.sum() == 0:
            continue
        uvs = dist_p[bi, v_mask, :2]                                # (n, 2)
        ci_u = uvs[:, 0].div(CELL_S).floor().clamp(0, G - 1).long()
        ci_v = uvs[:, 1].div(CELL_S).floor().clamp(0, G - 1).long()
        cell_ids = ci_v * G + ci_u                                   # (n,)
        n_total = uvs.shape[0]
        n_cell_empty = 0
        n_far_from_bucket = 0
        for k in range(n_total):
            cid = cell_ids[k].item()
            bk_uvs = b_uvds[bi, cid][b_valids[bi, cid]]              # (≤K, 4)
            if bk_uvs.numel() == 0:
                n_cell_empty += 1
                continue
            # query 点と cell 内 bucket pt の最小 uv 距離
            d = (bk_uvs[:, :2] - uvs[k:k+1]).norm(dim=-1).min()
            if d > CELL_S * 1.5:    # 8 px cell で 1.5×=12 px 以上離れてたら異常
                n_far_from_bucket += 1
        print(f"  [b={bi}] query→cell test  N={n_total}  "
              f"cell_empty={n_cell_empty}  far_from_bucket={n_far_from_bucket}")


def main():
    # cs=128 (2× upscale) と cs=384 (downscale) を強制する 2 つの dataset
    common = dict(img_size=IMG_SIZE, grid_n=GRID_N,
                   max_offset_m=0.2, max_rot_deg=1.5,
                   oversample=4, frame_stride=1)

    # 通常 mix (cs ∈ [128, 384])
    ds_mix = PandaSetCalibDatasetFull(CACHE, split='val',
                                       min_crop_px=128, max_crop_px=384,
                                       **common)
    # 強制 cs=128 (= 2× upscale)
    ds_up = PandaSetCalibDatasetFull(CACHE, split='val',
                                       min_crop_px=128, max_crop_px=129,
                                       **common)
    # 強制 cs=384 (= 0.667× downscale)
    ds_dn = PandaSetCalibDatasetFull(CACHE, split='val',
                                       min_crop_px=383, max_crop_px=384,
                                       **common)

    for name, ds in [('mix cs∈[128,384]', ds_mix),
                      ('forced cs=128 (2× up)', ds_up),
                      ('forced cs=384 (0.67× dn)', ds_dn)]:
        torch.manual_seed(0); np.random.seed(0)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                             collate_fn=collate_full)
        for batch in loader:
            dump_batch_stats(name, batch)
            break

    # 6. 同 inst を cs=128 と cs=384 で取り、duv_orig が cs 不変か
    print("\n=== same inst, different cs ===")
    np.random.seed(42)
    inst_idx = 17 if len(ds_mix) > 20 else 0
    # apply_perturbation_explicit を使うと crop 位置が tile_cutter 全体になり
    # cs 比較ができないので、build_window 直接呼びで cs を変えて見る。
    # ds 自身を min/max で固定して 1 サンプル取得する手で代用。
    ypr = np.array([0.5, -0.3, 0.0])
    t = np.array([0.05, -0.02, 0.10])
    for cs_target, ds_x in [(128, ds_up), (384, ds_dn)]:
        torch.manual_seed(0); np.random.seed(0)
        s = ds_x[inst_idx % len(ds_x)]
        if s is None:
            print(f"  cs={cs_target}: sample None")
            continue
        # tuple layout: (img, true, dist, vfp, b_uvd, b_valid, pert,
        #                pts_cam_orig, duv_orig, K_orig, cs)
        cs_actual = float(s[10].item())
        duv_o = s[8]                               # (n, 2) orig px
        true_uv = s[1][:, :2]; dist_uv = s[2][:, :2]
        local_dpx = (true_uv - dist_uv).norm(dim=-1)
        orig_dpx  = duv_o.norm(dim=-1)
        print(f"  cs={cs_actual:.0f}  vfp={s[3].item():.1f}  "
              f"local Δuv: mean={local_dpx.mean():.2f}  "
              f"orig Δuv: mean={orig_dpx.mean():.2f}  "
              f"ratio (local/orig)={local_dpx.mean()/orig_dpx.mean().clamp(min=1e-6):.3f}  "
              f"expect={IMG_SIZE/cs_actual:.3f}")


if __name__ == '__main__':
    main()
