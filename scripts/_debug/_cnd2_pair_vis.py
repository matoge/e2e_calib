"""Visualize CalibNet2 cross-frame predictions on a few PandaSet val pairs.

Loads experiments/<exp>/best_model.pt, runs forward_cross_frame on a handful
of validation pairs, and emits per-pair PNGs with:
  - left  panel: A image + A's lidar uv at GT (cyan)
  - right panel: B image + uv_B_HAT (red ○) / uv_B_GT (lime ×) / pred (yellow ★)

The pred is computed by adding the network's per-point Δuv to the HAT anchor
in B-local px (= dist_uvd_B + Δ̂). hyp_err = HAT vs GT, pred_err = pred vs GT.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import math

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_pair  # noqa: E402
from torch.utils.data import DataLoader, Subset                              # noqa: E402
from models.calibnet2 import CalibNet2                                       # noqa: E402
from scripts.util.pair_vis import render_pair_correspondence                  # noqa: E402

# match the trainer's ZYX intrinsic helper
from datasets.train_cnd2_ddp import _R_from_zyx_deg                          # noqa: E402

CACHE = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
EXP   = REPO / 'experiments' / 'cnd2_ps_pair_0531_1359'
CKPT  = EXP / 'best_model.pt'
OUT   = REPO / 'scripts/_debug/_outputs/cnd2_pair_vis'

NUM_SHOW = 8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(0); torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = PandaSetCalibDatasetFull(
        CACHE, split='val', img_size=128,
        min_crop_px=256, max_crop_px=512,
        max_offset_m=0.20, max_rot_deg=1.0,
        grid_n=16, preload=False,
        pair_mode=True, oversample=1,
    )
    print(f'pair_index={len(ds.pair_index)}  total={len(ds)}')
    ds = Subset(ds, list(range(min(NUM_SHOW * 4, len(ds)))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=collate_pair)

    # match trainer hyper-params (n_iter=4, info_head=True for matching ckpt)
    model = CalibNet2(d=128, img_size=128, in_channels=3,
                      use_intensity=True, frustum_grid_n=16,
                      n_iter=4, n_heads=4, d_scalar=8, n_type1=40,
                      use_info_head=True).to(device)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f'  missing keys (first 5): {missing[:5]}')
    if unexpected: print(f'  unexpected keys (first 5): {unexpected[:5]}')
    model.eval()

    rendered = 0
    for k_idx, batch in enumerate(loader):
        if rendered >= NUM_SHOW:
            break
        A = batch['A']; B = batch['B']
        dpose_AB = batch['dpose_AB']
        imgs_A, _trueA, distA, padA, vfpA, buA, bvA = A[:7]
        imgs_B, trueB, distB, padB, vfpB, buB, bvB = B[:7]
        pertB = B[7]

        imgs_A_f = imgs_A.float().div(255.0).to(device)
        imgs_B_f = imgs_B.float().div(255.0).to(device)
        distA_d = distA.to(device); padA_d = padA.to(device)
        vfpA_d = vfpA.to(device); vfpB_d = vfpB.to(device)
        buA_d = buA.to(device); bvA_d = bvA.to(device)
        buB_d = buB.to(device); bvB_d = bvB.to(device)

        ypr_GT = dpose_AB[..., 3:6]
        ypr_eps = pertB[..., 3:6].to(ypr_GT)
        ypr_HAT = ypr_GT + ypr_eps
        R_AB = _R_from_zyx_deg(ypr_HAT).to(device, dtype=imgs_A_f.dtype)

        point_in_A = torch.cat([distA_d[..., :3], distA_d[..., 4:5]], dim=-1)

        with torch.no_grad():
            out = model(imgs_A_f, point_in_A, mode='cross',
                         image_B=imgs_B_f, R_AB=R_AB,
                         vfp=vfpA_d, vfp_B=vfpB_d,
                         bucket_uvd=buA_d, bucket_valid=bvA_d,
                         bucket_uvd_B=buB_d, bucket_valid_B=bvB_d,
                         key_padding_mask=padA_d)
        per_pt = out[0] if isinstance(out, tuple) else out
        per_pt = per_pt.cpu()  # (1, N_A, 5)

        Nmin = min(per_pt.shape[1], distB.shape[1], trueB.shape[1])
        mu = per_pt[0, :Nmin, :2].numpy()                       # (Nmin, 2) Δuv pred
        # Pred uv on B (in B-local px) = dist_uvd_B + mu (legacy convention).
        uv_B_hat = distB[0, :Nmin, :2].numpy()                  # HAT
        uv_B_gt  = trueB[0, :Nmin, :2].numpy()                  # GT
        uv_B_pred = uv_B_hat + mu

        valid = (~padA[0, :Nmin].numpy()) & (~padB[0, :Nmin].numpy())
        if valid.sum() < 5:
            continue

        # Build minimal A/B 12-tuples for render_pair_correspondence.
        # collate_full layout (line 1654): (imgs, true_p, dist_p, pad, vfps,
        # b_uvds, b_valids, pert_6vec, pts_cam_orig, duv_orig, K_orig, cs_t,
        # delta1_se3). pair_vis expects build_window's 12-tuple, which is:
        # (img, true, dist, vfp, b_uvd, b_valid, pert, pts_orig, duv_orig,
        # K_orig, cs, delta1) — so we drop pad and re-shuffle.
        def _per_sample_12(C, b=0, n=Nmin):
            return (
                C[0][b],                                              # 0 img
                C[1][b, :n],                                          # 1 true
                C[2][b, :n],                                          # 2 dist
                C[4][b],                                              # 3 vfp
                C[5][b],                                              # 4 b_uvd
                C[6][b],                                              # 5 b_valid
                C[7][b],                                              # 6 pert
                C[8][b, :n] if C[8] is not None else torch.zeros(n,3), # 7 pts_orig
                C[9][b, :n] if C[9] is not None else torch.zeros(n,2), # 8 duv_orig
                C[10][b] if C[10] is not None else torch.eye(3),       # 9 K_orig
                C[11][b] if C[11] is not None else torch.tensor(128.0),# 10 cs
                C[12][b],                                              # 11 delta1
            )
        A_t = _per_sample_12(A)
        B_t = _per_sample_12(B)

        hyp_err  = np.linalg.norm(uv_B_hat[valid] - uv_B_gt[valid], axis=-1).mean()
        pred_err = np.linalg.norm(uv_B_pred[valid] - uv_B_gt[valid], axis=-1).mean()

        out_path = OUT / f'pair_{k_idx:03d}_hyp{hyp_err:.1f}_pred{pred_err:.2f}.png'
        title = (f'cnd2_pair  idx={k_idx}  hyp={hyp_err:.2f} px → '
                 f'pred={pred_err:.2f} px  '
                 f'(yrp_eps={ypr_eps[0].tolist()})')
        render_pair_correspondence(
            A_t, B_t, dpose_AB[0],
            pred_duv_B=torch.from_numpy(mu),
            out_path=out_path,
            title=title,
            max_pts_draw=100,
        )
        rendered += 1
        print(f'[{rendered}/{NUM_SHOW}] hyp={hyp_err:.2f} pred={pred_err:.2f} → {out_path}')

    print(f'\nwrote {rendered} pngs to {OUT.absolute()}')


if __name__ == '__main__':
    main()
