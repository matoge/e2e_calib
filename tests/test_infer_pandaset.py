"""PandaSet で推論経路が学習経路と同じ答えを出すか。

  A 学習経路   正しいポーズ + dataset の中で摂動 (fixed_pert)
               → δ_pred と δ_gt (学習時の val と同じ量)
  B 推論経路   perturb_T で外からずらしたポーズ + 摂動なし → predict_pose → δ
               → 補正後のポーズ = apply_delta(T_ずれ, δ)

B はポーズ同士で評価する (回転は測地角、並進はカメラ中心の距離)。δ の成分
どうしを比べると並びや表現の違いに引っかかる。

  CKPT_EXP=ps_s1_noleak N=8 python tests/test_infer_pandaset.py
  ZERO=1 ...   摂動ゼロ (正しいポーズのまま) で δ がゼロ付近に出るか
"""
import os, sys
from pathlib import Path
import numpy as np, cv2, torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from datasets.train_cnd2_ddp import _ba_pose_loss, forward_calib, predict_pose
from scripts.inference.infer_calib import (load_model, build_inst, make_dataset,
                                           perturb_T, apply_delta, pose_error)

CACHE = os.environ.get('CACHE', '/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full')
EXP = os.environ.get('CKPT_EXP', 'ps_grid_ba_infohead')
N = int(os.environ.get('N', '8'))
ZERO = os.environ.get('ZERO', '0') == '1'
dev = 'cuda'


def raw_of(inst):
    img = cv2.imdecode(np.frombuffer(inst['jpg_bytes'], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    K = inst['K_full'].numpy().astype(np.float64)
    R = inst['R_gt'].numpy().astype(np.float64); cp = inst['cam_pos'].numpy().astype(np.float64)
    T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ cp
    pts = np.concatenate([inst['pts'].numpy(), inst['intensity'].numpy()[:, None]], 1)
    return img, pts, K, T


def batch_of(ds):
    w = ds[0]; w = w if isinstance(w, list) else [w]
    return [t.to(dev) if torch.is_tensor(t) else t for t in collate_full(w)], len(w)


def main():
    src = PandaSetCalibDatasetFull(cache_dir=CACHE, split='val', img_size=256, max_offset_m=0.2,
                                   max_rot_deg=0.5, min_crop_px=256, max_crop_px=256,
                                   grid_n=16, oversample=1)
    model, c = load_model(EXP)
    rows = []
    idx = np.linspace(0, len(src) - 1, N).astype(int)
    print(f'{EXP}  {"ZERO" if ZERO else "注入あり"}  {N} フレーム')
    print(f'{"frame":>8} | {"補正前 rot[deg] t[m]":>20} | {"B 推論 rot t":>20} | {"A 学習経路 rot t":>18}')
    for fi in idx:
        inst = src._load_inst(int(fi))
        img, pts, K, T = raw_of(inst)
        g = np.random.default_rng(int(fi))
        t_inj = np.zeros(3) if ZERO else (g.random(3) * 2 - 1) * c['t_m']
        r_inj = np.zeros(3) if ZERO else (g.random(3) * 2 - 1) * c['rot_deg']
        # A: 学習経路 (中で摂動)。δ_pred と δ_gt の差 = 学習時の val 指標
        dsA = make_dataset(c, [build_inst(img=img, pts=pts, K=K, T_cam_lidar=T)],
                           fixed_pert=np.concatenate([t_inj, r_inj]))
        bA, G = batch_of(dsA)
        with torch.no_grad():
            per_pt, W = forward_calib(model, bA)
            _l, dg = _ba_pose_loss(per_pt, bA[2], bA[3], bA, img_size=model.img_size,
                                   group=G, W_head=W, is_grid=bA[15], w_active=bA[16],
                                   detach_mu=True, return_pose=True)
        eA = (dg['delta_pred'] - dg['delta_gt'])[0].cpu().numpy()
        # B: 推論経路 (外でずらして、補正後のポーズを正解と比べる)
        Tp = perturb_T(T, rot_deg=r_inj, t_m=t_inj)
        dsB = make_dataset(c, [build_inst(img=img, pts=pts, K=K, T_cam_lidar=Tp)])
        bB, _ = batch_of(dsB)
        d, _H = predict_pose(model, bB, img_size=model.img_size, group=G)
        Tc = apply_delta(Tp, d[0].cpu().numpy())
        e0 = pose_error(Tp, T); e1 = pose_error(Tc, T)
        rows.append((e0[0], e0[1], e1[0], e1[1], np.abs(eA[:3]).mean(), np.abs(eA[3:]).mean()))
        r = rows[-1]
        print(f"{inst['scene']}/{inst['frame']:<3} | {r[0]:8.4f} {r[1]:9.5f}  | {r[2]:8.4f} {r[3]:9.5f}  | "
              f"{r[4]:7.4f} {r[5]:8.5f}", flush=True)
    a = np.array(rows)
    print(f'\n中央値  補正前 rot {np.median(a[:,0]):.4f} deg t {np.median(a[:,1]):.5f} m'
          f'   B 推論 rot {np.median(a[:,2]):.4f} t {np.median(a[:,3]):.5f}'
          f'   A 学習経路 rot {np.median(a[:,4]):.4f} t {np.median(a[:,5]):.5f}')
    print(f'最悪    B 推論 rot {a[:,2].max():.4f} deg  t {a[:,3].max():.5f} m')
    return a


if __name__ == '__main__':
    main()
