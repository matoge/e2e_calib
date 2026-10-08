"""推論経路の回帰テスト (PandaSet)。

推論は scripts/inference/infer_calib.py の 1 本で、学習と同じ dataset の
__init__・同じ forward_calib・同じ ba_solve を通る。ここではその経路が
学習と同じ答えを出すことを固定する。

  モデルに依存しない (経路の正しさ):
    test_external_perturbation_matches_internal
        外からポーズをずらす (perturb_T) のと、dataset が中で摂動を掛けるのが
        同じ投影になるか。
    test_perfect_residual_recovers_pose
        正しい残差を与えたら predict_pose と同じ GN + apply_delta で
        正解ポーズに戻るか。
  モデルを使う:
    test_val_nll_matches_log    段1 の ckpt で val の NLL を再生し train.log と比べる
    test_predict_pose_smoke     推論 1 フレームで δ と H が有限、H が正定値

  python -m pytest tests/test_inference.py -v -s
以前のこのファイルは CalibNetDepth + kamikado + ba_multicam_corr.infer_tiles
用で、その推論経路を畳んだ (2026-10-07) ので書き直した。
"""
from __future__ import annotations

import os, re, sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from datasets.train_cnd2_ddp import forward_calib, predict_pose
from models.model_cov import gaussian2d_nll
from scripts.ba.gn_pose import solve_pose
from scripts.ba.ba_torch import make_info_from_sigma_rho
from scripts.inference.infer_calib import (load_model, build_inst, make_dataset,
                                           perturb_T, apply_delta, pose_error)

CACHE = os.environ.get('CACHE', '/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full')
NLL_TOL = 0.20      # val の摂動は eval ごとに引き直すので完全一致はしない
STAGE1 = ['ps_s1_noleak']                     # BA なしの ckpt (val_nll = 点ごとの NLL)
ANY = ['ps_s2_noleak', 'ps_s1_noleak', 'ps_grid_ba_infohead']
dev = 'cuda'

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA が要る')
needs_cache = pytest.mark.skipif(not Path(CACHE, 'meta.pt').exists(),
                                 reason=f'キャッシュが無い: {CACHE}')


def _have(exp):
    return (REPO / 'experiments' / exp / 'best_model.pt').is_file()


def _src():
    return PandaSetCalibDatasetFull(cache_dir=CACHE, split='val', img_size=256,
                                    max_offset_m=0.2, max_rot_deg=0.5, min_crop_px=256,
                                    max_crop_px=256, grid_n=16, oversample=1)


def _raw(inst):
    import cv2
    img = cv2.imdecode(np.frombuffer(inst['jpg_bytes'], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    K = inst['K_full'].numpy().astype(np.float64)
    R = inst['R_gt'].numpy().astype(np.float64); cp = inst['cam_pos'].numpy().astype(np.float64)
    T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ cp
    pts = np.concatenate([inst['pts'].numpy(), inst['intensity'].numpy()[:, None]], 1)
    return img, pts, K, T


_CFG = dict(img_size=256, grid_n=16, min_crop_px=256, max_crop_px=256,
            t_m=0.2, rot_deg=0.5, oversample=40)


# ── モデルに依存しない ───────────────────────────────────────────────────

@needs_cache
def test_external_perturbation_matches_internal():
    from scipy.spatial.transform import Rotation
    inst = _src()._load_inst(0); img, pts, K, T = _raw(inst)
    g = np.random.default_rng(0)
    t = (g.random(3) * 2 - 1) * 0.2; r = (g.random(3) * 2 - 1) * 0.5
    iA = build_inst(img=img, pts=pts, K=K, T_cam_lidar=T)
    iB = build_inst(img=img, pts=pts, K=K, T_cam_lidar=perturb_T(T, rot_deg=r, t_m=t))
    # dataset (build_crop) の式: R_off = R_gt @ Rot(ypr), cp_off = cp + R_gt @ t
    Rg = iA['R_gt'].numpy().astype(np.float64); cp = iA['cam_pos'].numpy().astype(np.float64)
    R_off = Rg @ Rotation.from_euler('zyx', r, degrees=True).as_matrix()
    cp_off = cp + Rg @ t
    assert np.abs(iB['R_gt'].numpy() - R_off).max() < 1e-6
    assert np.abs(iB['cam_pos'].numpy() - cp_off).max() < 1e-5
    W = iA['pts'].numpy().astype(np.float64)
    pa = (((W - cp) @ Rg) - Rg.T @ (cp_off - cp)) @ (Rg.T @ R_off)
    pb = (W - iB['cam_pos'].numpy().astype(np.float64)) @ iB['R_gt'].numpy().astype(np.float64)
    m = (pa[:, 2] > 0.5) & (pb[:, 2] > 0.5)
    pj = lambda P: P[:, :2] / P[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    d = np.linalg.norm(pj(pa[m]) - pj(pb[m]), axis=1)
    print(f'\n  外と中の摂動  投影差 max {d.max():.2e} px  ({m.sum()} 点)')
    assert d.max() < 1e-3


@needs_cache
@needs_gpu
def test_perfect_residual_recovers_pose():
    """正しい μ (= 正解の投影 − ずれた投影) を ba_solve と同じ +μ で解き、
    apply_delta で正解ポーズに戻る。推論経路の符号と当て方を固定する。"""
    src = _src()
    for fi in (0, 170):
        img, pts, K, Tg = _raw(src._load_inst(fi))
        g = np.random.default_rng(fi)
        t = (g.random(3) * 2 - 1) * 0.2; r = (g.random(3) * 2 - 1) * 0.5
        Tp = perturb_T(Tg, rot_deg=r, t_m=t)
        ds = make_dataset(_CFG, [build_inst(img=img, pts=pts, K=K, T_cam_lidar=Tp)])
        w = ds[0]; b = [x.to(dev) if torch.is_tensor(x) else x for x in collate_full(w)]
        P0 = b[8].double().clone(); v = (~b[3]) & (P0[..., 2] > 0.5)
        P0[~v] = torch.tensor([0., 0., 10.], dtype=P0.dtype, device=P0.device)
        Pn = P0.cpu().numpy().reshape(-1, 3)
        Xw = (Tp[:3, :3].T @ (Pn - Tp[:3, 3]).T).T
        Pt = (Tg[:3, :3] @ Xw.T).T + Tg[:3, 3]
        pj = lambda P: np.stack([P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2],
                                 P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]], 1)
        mu = (pj(Pt) - pj(Pn)); mu[~v.cpu().numpy().reshape(-1)] = 0
        B, N = P0.shape[:2]
        o = torch.ones(1, B * N, dtype=torch.float64, device=P0.device)
        d, _ = solve_pose(P0.reshape(1, B * N, 3),
                          torch.tensor(mu, device=P0.device).reshape(1, B * N, 2),
                          make_info_from_sigma_rho(o, o, 0 * o), b[10][:1].double(),
                          valid=v.reshape(1, B * N), n_iter=10, damping=0.0)
        e = pose_error(apply_delta(Tp, d[0].cpu().numpy()), Tg)
        e0 = pose_error(Tp, Tg)
        print(f'\n  frame {fi}  ずれ {e0[0]:.4f} deg {e0[1]:.4f} m → 補正後 {e[0]:.2e} deg {e[1]:.2e} m')
        assert e[0] < 1e-4 and e[1] < 1e-5


# ── モデルを使う ─────────────────────────────────────────────────────────

def _trainlog_best_val(exp):
    log = REPO / 'experiments' / exp / 'train.log'
    if not log.is_file():
        return None
    m = re.findall(r'best val_nll=([-\d.]+)', log.read_text(errors='ignore'))
    return float(m[-1]) if m else None


@needs_cache
@needs_gpu
@pytest.mark.parametrize('exp', [e for e in STAGE1 if _have(e)])
def test_val_nll_matches_log(exp):
    model, c = load_model(exp)
    ds = PandaSetCalibDatasetFull(cache_dir=CACHE, split='val', center_band=0.5,
                                  img_size=c['img_size'], grid_n=c['grid_n'],
                                  min_crop_px=c['min_crop_px'], max_crop_px=c['max_crop_px'],
                                  max_offset_m=c['t_m'], max_rot_deg=c['rot_deg'],
                                  oversample=c['oversample'], crop_grid=bool(c.get('crop_grid')),
                                  share_pert=bool(c.get('share_pert')), n_full=0)
    loader = DataLoader(Subset(ds, list(range(len(ds)))), batch_size=4, num_workers=4,
                        collate_fn=collate_full, shuffle=False)
    s = n = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(dev) if torch.is_tensor(x) else x for x in batch]
            per_pt, _ = forward_calib(model, batch)
            gt = batch[1][..., :2] - batch[2][..., :2]
            v = ~batch[3]
            if v.any():
                s += gaussian2d_nll(per_pt[v], gt[v]).item(); n += 1
    nll = s / max(n, 1); expected = _trainlog_best_val(exp)
    print(f'\n  [{exp}] 再生 val_nll={nll:+.4f}  train.log best={expected}')
    if expected is None:
        pytest.skip('train.log に best val_nll が無い')
    assert abs(nll - expected) <= NLL_TOL


@needs_cache
@needs_gpu
@pytest.mark.parametrize('exp', [e for e in ANY if _have(e)][:1])
def test_predict_pose_smoke(exp):
    model, c = load_model(exp)
    img, pts, K, T = _raw(_src()._load_inst(0))
    ds = make_dataset(c, [build_inst(img=img, pts=pts, K=K, T_cam_lidar=T)])
    w = ds[0]; b = [x.to(dev) if torch.is_tensor(x) else x for x in collate_full(w)]
    d, H = predict_pose(model, b, img_size=model.img_size, group=len(w))
    d = d[0].cpu().numpy(); H = H[0].cpu().numpy()
    print(f'\n  [{exp}] 窓 {len(w)}  δ rot={np.round(d[:3], 4)} deg  t={np.round(d[3:], 5)} m')
    assert d.shape == (6,) and np.all(np.isfinite(d)) and np.all(np.isfinite(H))
    assert np.all(np.linalg.eigvalsh(0.5 * (H + H.T)) > 0), 'H が正定値でない'
