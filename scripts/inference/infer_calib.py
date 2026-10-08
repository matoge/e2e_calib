"""外部データ 1 フレーム → 外部パラメータの補正量 δ。

学習と同じ経路を通す。推論用のパイプラインを別に持たない。

  学習  渡すポーズ = 正しい  + dataset が摂動を掛ける  → モデルは「ずれた投影」を見る
  推論  渡すポーズ = ずれてる + 摂動なし (fixed_pert=0) → モデルは「ずれた投影」を見る

dataset の R_gt / cam_pos は「渡されたポーズ」でしかないので、摂動と実際の
ずれは同じ場所から入る。だから推論側で足すものは無く、inst を 1 つ組んで
PandaSetCalibDatasetFull に食わせれば、窓の切り方・ジッタ固定・複製タイルの
除外・InfoHead の W・ソルバまで全部が学習時と同一コードになる。

以前はここに CalibNetDepth 用の単一タイル forward があり、別に
services/calib_api/raw_pipeline.py が kamikado 専用の raw 経路を持っていた。
どちらも学習側 (CalibNet2 + crop_grid + InfoHead) から取り残されていたので、
2026-10-07 に両方畳んでこの 1 本にした。

使い方 (ライブラリ):
    from scripts.inference.infer_calib import load_model, build_inst, infer
    model, ds = load_model('ps_grid_ba_infohead')
    inst = build_inst(img=rgb_uint8, pts=pts_xyzi, K=K, T_cam_lidar=T)
    delta = infer(model, ds)            # (6,) [ωx, ωy, ωz (deg), tx, ty, tz (m)]

使い方 (CLI):
    python scripts/inference/infer_calib.py ps_grid_ba_infohead \
        --image frame.png --points pts.txt --calib calib.json
    # 評価: 正しいポーズに外から摂動を掛けて、同じ経路で解き直す
    python scripts/inference/infer_calib.py ps_grid_ba_infohead \
        --image ... --points ... --calib ... --pert-rot-deg 0.5 --pert-t-m 0.2
"""
import argparse, importlib.util, io, json, math, os, sys
import cv2
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.calibnet2 import CalibNet2
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full

from scripts.util.projection import project_lidar_into_image

# キャッシュと同じ: 画像の外 FOV_PAD px、深度 Z_MIN m まで点を残す (build_*_v3.py)。
# 最終的な判定はデータセット側の z_off > 0.5 と窓の範囲。
Z_MIN = 0.1
FOV_PAD = 256


# ── 入力 ─────────────────────────────────────────────────────────────────

def load_points(data, name: str = '') -> np.ndarray:
    """(N, 4) float32 [x, y, z, intensity]。LiDAR 座標系。

    テキストを第一級にしてある。curl で投げられて中身が目で読めるのが利点で、
    npy/bin は先頭を見ないと列数も dtype も分からない。
    """
    if isinstance(data, (str, Path)):
        name = str(data); data = Path(data).read_bytes()
    suf = Path(name).suffix.lower()
    if suf == '.npy':
        a = np.load(io.BytesIO(data))
    elif suf == '.bin':
        # KITTI 系 (N,4) / nuScenes (N,5) のどちらも通す。点数が 20 の倍数だと長さだけでは
        # 決まらない (以前は 4 列に読んで黙って壊れていた)。nuScenes の 5 列目はリング番号 (0〜127 の整数)。
        f = np.frombuffer(data, np.float32)
        ok5, ok4 = f.size % 5 == 0, f.size % 4 == 0
        if ok5 and ok4:
            r = f.reshape(-1, 5)[:, 4]
            ok5 = bool(np.all((r == np.round(r)) & (r >= 0) & (r < 128)))
            ok4 = not ok5
        if ok5:
            a = f.reshape(-1, 5)
        elif ok4:
            a = f.reshape(-1, 4)
        else:
            raise ValueError(f'.bin: {f.size} float32 values is divisible by neither 4 nor 5: {name}')
    else:
        txt = data.decode('utf-8', 'replace') if isinstance(data, (bytes, bytearray)) else data
        rows = [r.replace(',', ' ').split() for r in txt.splitlines()
                if r.strip() and not r.lstrip().startswith('#')]
        if not rows:
            raise ValueError(f'点が 0 行: {name}')
        a = np.array([[float(x) for x in r] for r in rows], np.float32)
    a = np.asarray(a, np.float32)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f'点群の形が (N,>=3) でない: {a.shape}  ({name})')
    if a.shape[1] == 3:                      # intensity 無しは 0 で埋める
        a = np.concatenate([a, np.zeros((len(a), 1), np.float32)], 1)
    # 学習のキャッシュの intensity は 0〜255 を /255 した [0,1]。1 を超える値があれば 0〜255 とみなす
    if a.shape[1] >= 4 and len(a) and float(np.nanmax(a[:, 3])) > 1.0:
        a = a.copy(); a[:, 3] = a[:, 3] / 255.0
    return np.ascontiguousarray(a[:, :4])


def load_calib(data, name: str = ''):
    """(K, dist, T_cam_lidar, is_fisheye)。

    素の JSON   {"K": [[...]], "T_cam_lidar": [[...]], "dist": [k1..k4]}
    kamikado    calib.calib（intrinsic/extrinsic のネスト）
    """
    if isinstance(data, (str, Path)):
        name = str(data); data = Path(data).read_bytes()
    d = json.loads(data.decode('utf-8') if isinstance(data, (bytes, bytearray)) else data)
    if 'K' in d:
        K = np.asarray(d['K'], np.float64).reshape(3, 3)
        # 外部パラメータが無いときに単位行列で黙って進めない (以前は d.get(..., np.eye(4)))
        key = 'T_cam_lidar' if 'T_cam_lidar' in d else ('T' if 'T' in d else None)
        if key is None:
            raise ValueError(f'calib JSON has no T_cam_lidar (or T). keys: {sorted(d)}')
        T = np.asarray(d[key], np.float64).reshape(4, 4)
        dist = np.asarray(d.get('dist', [0, 0, 0, 0]), np.float64).ravel()
        # 魚眼 (Kannala-Brandt k1..k4) は is_fisheye で明示したときだけ。以前は dist が非ゼロなら
        # 魚眼扱いで、OpenCV の radtan 係数を KB として投影していた。
        fe = bool(d.get('is_fisheye', False))
        if fe:
            if dist.size != 4:
                raise ValueError(f'with is_fisheye, dist must be Kannala-Brandt k1..k4 (4 values), got {dist.size}')
        elif np.any(dist != 0):
            raise ValueError('pinhole (radtan) distortion is not supported: undistort the image and set dist to 0, '
                             'or add is_fisheye: true for Kannala-Brandt k1..k4')
        else:
            dist = np.zeros(4)
        return K, dist, T, fe
    # kamikado
    intr = d['intrinsic'] if 'intrinsic' in d else d
    K = np.asarray(intr['camera_model']['pinhole_parameters']
                     ['matrix_image_camera']['matrix'], np.float64).T
    K = np.ascontiguousarray(K)
    kb = intr['camera_model'].get('kannala_parameters', {})
    dist = np.asarray([kb.get(f'k{i}', 0.0) for i in (1, 2, 3, 4)], np.float64)
    T = np.asarray(d['extrinsic']['matrix'], np.float64).reshape(4, 4)
    return K, dist, T, bool(np.any(dist != 0))


# ── inst ─────────────────────────────────────────────────────────────────

def build_inst(*, img: np.ndarray, pts: np.ndarray, K: np.ndarray,
               T_cam_lidar: np.ndarray, dist=None, is_fisheye: bool = False,
               scene: str = 'infer', frame: int = 0) -> dict:
    """LMDB インスタンスと同じスキーマの dict を 1 つ作る。

    T_cam_lidar は「渡されたポーズ」。推論ではこれがずれている外部パラメータで、
    評価では正しいポーズに外から摂動を掛けたもの。uv_full はこのポーズで計算する。
    学習時に uv_full が GT だったのは、そのとき渡されたポーズが GT だったという
    だけの話で、経路は同じ。
    """
    import cv2
    img = np.ascontiguousarray(img)
    IH, IW = img.shape[:2]
    pts = np.asarray(pts, np.float32)
    dist = np.zeros(4) if dist is None else np.asarray(dist, np.float64).reshape(4)
    keep, pts_cam, uv, z, inten = project_lidar_into_image(
        pts, np.asarray(K, np.float64), np.asarray(T_cam_lidar, np.float64),
        IW, IH, is_fisheye=is_fisheye, dist=dist if is_fisheye else None,
        z_min=Z_MIN, pad_px=FOV_PAD)
    if len(uv) < 8:
        raise RuntimeError(f'画像に落ちた点が {len(uv)} 個しかない')
    # R_gt / cam_pos はカメラ姿勢。pts は world 相当 = LiDAR 座標のままでよく、
    # dataset 側が pts_cam = (pts - cam_pos) @ R_gt で camera 系に移す。
    Tcl = np.asarray(T_cam_lidar, np.float64)
    R_wc = Tcl[:3, :3].T                       # camera->"world"(=lidar) の回転
    cam_pos = -R_wc @ Tcl[:3, 3]
    ok, buf = cv2.imencode('.jpg', img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok, 'jpg encode に失敗'
    n = int(keep.sum())
    return dict(
        IW=int(IW), IH=int(IH), tile_u0=0, tile_v0=0,
        jpg_bytes=buf.tobytes(), scene=str(scene), frame=int(frame),
        K_full=torch.tensor(np.asarray(K, np.float32)),
        cam_pos=torch.tensor(cam_pos.astype(np.float32)),
        R_gt=torch.tensor(R_wc.astype(np.float32)),
        T_gt=torch.tensor(np.linalg.inv(Tcl).astype(np.float32)),
        pts=torch.tensor(pts[keep, :3].astype(np.float32)),
        uv_full=torch.tensor(uv.astype(np.float32)),
        z_cam=torch.tensor(z.astype(np.float32)),
        is_obj=torch.zeros(n, dtype=torch.uint8),
        intensity=torch.tensor(inten.astype(np.float32)),
        is_fisheye=bool(is_fisheye), distortion=dist.astype(np.float32),
    )


def perturb_T(T_cam_lidar: np.ndarray, rot_deg=(0., 0., 0.), t_m=(0., 0., 0.)) -> np.ndarray:
    """評価用。dataset の外側でポーズをずらす。

    掛け方は dataset (__getitem__ -> build_crop) と同一にする。あちらは
        R_off  = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True)
        cp_off = cp + R_gt @ t
    で、t は「カメラ自身の軸で測った並進」。T の並進に直接足すのとは別物。
    """
    from scipy.spatial.transform import Rotation
    Tcl = np.asarray(T_cam_lidar, np.float64)
    R_gt = Tcl[:3, :3].T                       # camera->world の回転 = dataset の R_gt
    cp = -R_gt @ Tcl[:3, 3]                    # カメラ原点 (world)
    R_off = R_gt @ Rotation.from_euler('zyx', np.asarray(rot_deg, float),
                                        degrees=True).as_matrix()
    cp_off = cp + R_gt @ np.asarray(t_m, float)
    out = np.eye(4)
    out[:3, :3] = R_off.T
    out[:3, 3] = -R_off.T @ cp_off
    return out


def delta_to_T(delta) -> np.ndarray:
    """δ (ωx, ωy, ωz [deg], tx, ty, tz [m]) → 4x4。ba_torch._apply_extrinsic と
    同じく P' = R(ω)·P + t で、ω は回転ベクトル (度)。"""
    from scipy.spatial.transform import Rotation
    d = np.asarray(delta, np.float64).reshape(6)
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(np.radians(d[:3])).as_matrix()
    M[:3, 3] = d[3:]
    return M


def apply_delta(T_cam_lidar: np.ndarray, delta) -> np.ndarray:
    """推論で返った δ を渡したポーズに当てる。補正後 = E(δ) @ T。
    正解の残差で解いた δ を当てると正解ポーズに 0.00000 deg / 0.00000 m で
    戻ることを確認済み (inv(E(δ)) は逆向きで、ずれが倍になる)。"""
    return delta_to_T(delta) @ np.asarray(T_cam_lidar, np.float64)


def pose_error(T_est: np.ndarray, T_gt: np.ndarray):
    """(回転 [deg], 並進 [m])。回転は測地角、並進はカメラ中心の距離。
    δ の成分どうしを比べると並びや表現 (euler / 回転ベクトル) の違いに
    引っかかるので、評価はポーズ同士で行う。"""
    from scipy.spatial.transform import Rotation
    Ta = np.asarray(T_est, np.float64); Tb = np.asarray(T_gt, np.float64)
    D = Ta[:3, :3] @ Tb[:3, :3].T
    ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(D).as_rotvec())))
    ca = -Ta[:3, :3].T @ Ta[:3, 3]; cb = -Tb[:3, :3].T @ Tb[:3, 3]
    return ang, float(np.linalg.norm(ca - cb))


def pose_error_axes(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    """誤差を軸ごとに。カメラ座標 (x 右, y 下, z 前) で
      yaw   = y 軸 (下向き) まわり = 左右の首振り   [deg]
      pitch = x 軸 (右向き) まわり = 上下の傾き     [deg]
      roll  = z 軸 (光軸) まわり   = 画像の回転     [deg]
      x, y, z = カメラ中心のずれをカメラの軸で測ったもの [m]
    回転の差は perturb_T と同じ向き (R_est = R_gt @ dR) で取る。"""
    from scipy.spatial.transform import Rotation
    Ta = np.asarray(T_est, np.float64); Tb = np.asarray(T_gt, np.float64)
    R_gt, R_est = Tb[:3, :3].T, Ta[:3, :3].T                  # camera -> world
    dR = R_gt.T @ R_est
    rx, ry, rz = Rotation.from_matrix(dR).as_euler('xyz', degrees=True)
    ca = -R_est @ Ta[:3, 3]; cb = -R_gt @ Tb[:3, 3]
    dx, dy, dz = R_gt.T @ (ca - cb)
    return dict(yaw_deg=float(ry), pitch_deg=float(rx), roll_deg=float(rz),
                x_m=float(dx), y_m=float(dy), z_m=float(dz))


PS_CACHE = os.environ.get('E2E_PS_CACHE', '/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full')


def load_val_frame(i: int, cache: str = PS_CACHE):
    """キャッシュの val の i 番目のフレーム → load_pandaset_val の 6 つ + (dist, is_fisheye)。
    魚眼のキャッシュ (kamikado / woven / TSS4) は Kannala-Brandt の係数を返す。
    以前の load_pandaset_val は歪みを返さず、魚眼をピンホールで投影していた。"""
    from datasets.pandaset_full import PandaSetCalibDatasetFull
    src = PandaSetCalibDatasetFull(cache_dir=cache, split='val', img_size=256, grid_n=16,
                                   min_crop_px=256, max_crop_px=256, oversample=1)
    inst = src._load_inst(int(i))
    fe = bool(inst.get('is_fisheye', False))
    dist = (inst['distortion'].numpy().astype(np.float64) if fe and 'distortion' in inst else None)
    return (*load_pandaset_val(i, cache, _inst=inst), dist, fe)


def load_pandaset_val(i: int, cache: str = PS_CACHE, _inst=None):
    """PandaSet のキャッシュの val の i 番目のフレーム →
    (画像 RGB, 点 xyz+強度, K, 正しい T_cam_lidar, 'シーン/フレーム', JPEG のバイト列)。
    魚眼のキャッシュには load_val_frame (歪み係数も返す) を使う。"""
    from datasets.pandaset_full import PandaSetCalibDatasetFull
    if _inst is None:
        src = PandaSetCalibDatasetFull(cache_dir=cache, split='val', img_size=256, grid_n=16,
                                       min_crop_px=256, max_crop_px=256, oversample=1)
        _inst = src._load_inst(int(i))
    inst = _inst
    img = cv2.imdecode(np.frombuffer(inst['jpg_bytes'], np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    K = inst['K_full'].numpy().astype(np.float64)
    R = inst['R_gt'].numpy().astype(np.float64); cp = inst['cam_pos'].numpy().astype(np.float64)
    T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ cp
    pts = np.concatenate([inst['pts'].numpy(), inst['intensity'].numpy()[:, None]], 1).astype(np.float32)
    return img, pts, K, T, f"{inst['scene']}/{inst['frame']}", inst['jpg_bytes']


def render_overlay(img, pts, K, poses: dict, dist=None, is_fisheye=False) -> np.ndarray:
    """画像に、各ポーズで投影した LiDAR の点を重ねて保存する。
    poses の色: true=緑, input=赤, corrected=水色。左右に 2 枚 (入力 / 補正後) 並べ、どちらにも正解を重ねる。
    右 (補正後) には入力も重ね、赤→水色でどれだけ動いたかを見せる。"""
    from scripts.util.projection import project_lidar_into_image
    IH, IW = img.shape[:2]
    col = {'true': (52, 199, 89), 'input': (255, 59, 48), 'corrected': (0, 194, 255)}
    def proj(T):
        keep, _pc, uv, *_ = project_lidar_into_image(pts, K, T, IW, IH, is_fisheye=is_fisheye, dist=dist)
        return uv
    base = (img.astype(np.float32) * 0.6).astype(np.uint8)
    panels = []
    for name in [k for k in ('input', 'corrected') if k in poses]:
        p = base.copy()
        layers = (['input'] if name == 'corrected' and 'input' in poses else []) + \
                 (['true'] if 'true' in poses else []) + [name]
        for k in layers:
            for u, v in proj(poses[k]).astype(int):
                cv2.circle(p, (int(u), int(v)), 2, col[k], -1)
        cv2.putText(p, {'input': 'input (red)', 'corrected': 'input (red)  /  corrected (cyan)'}[name] +
                    ('  /  true (green)' if 'true' in poses else ''),
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        panels.append(p)
    return np.concatenate(panels, 1) if len(panels) > 1 else panels[0]


def draw_overlay(img, pts, K, poses: dict, path: str, dist=None, is_fisheye=False):
    out = render_overlay(img, pts, K, poses, dist=dist, is_fisheye=is_fisheye)
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return path


# ── モデル ────────────────────────────────────────────────────────────────

def _cfg_of(exp: str) -> dict:
    p = REPO_ROOT / 'experiments' / exp / 'config.py'
    if not p.exists():
        raise FileNotFoundError(f'config が無い: {p}')
    spec = importlib.util.spec_from_file_location('_cfg', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return dict(m.CFG)


def load_model(exp: str, device='cuda', ckpt: str = 'best_model.pt'):
    """experiments/{exp}/config.py の値そのままで CalibNet2 を組んで重みを入れ、
    同じ設定の dataset も返す。手で hparam を並べ直さない。"""
    c = _cfg_of(exp)
    from datasets.train_cnd2_ddp import KV_SCHEDULES
    kv = KV_SCHEDULES.get(c.get('kv_schedule')) if c.get('kv_schedule') else None
    model = CalibNet2(d=128, img_size=c['img_size'], in_channels=3,
                      use_intensity=True, frustum_grid_n=c['grid_n'],
                      n_iter=c['n_iter'], n_heads=c.get('n_heads', 4),
                      d_scalar=8, n_type1=40, kv_schedule=kv,
                      fourier_head_n_freq=int(c.get('fourier_head_n_freq', 0)),
                      fourier_head_scale=float(c.get('fourier_head_scale', 10.0)),
                      point_mlp_fourier_n_freq=int(c.get('point_mlp_fourier_n_freq', 0)),
                      use_info_head=bool(c.get('use_info_head', True)),
                      ref_from_query=bool(c.get('ref_from_query', False)),
                      cross_attn=c.get('cross_attn', 'deform'),
                      ref_mode=(c.get('ref_mode') or None),
                      frustum_nb=c.get('frustum_nb', '3x3'))
    sd = torch.load(REPO_ROOT / 'experiments' / exp / ckpt, map_location='cpu',
                    weights_only=False)
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f'  state_dict: missing={len(miss)} unexpected={len(unexp)}', flush=True)
    model.to(device).eval()

    model.cfg = c
    return model, c


def make_dataset(c: dict, insts: list, fixed_pert=None):
    """学習と同じ __init__ を通す。窓の切り方・ジッタ・複製タイルの扱いが
    構造上一致する。cache_dir は渡さず insts= で inst を直接入れる。"""
    return PandaSetCalibDatasetFull(
        insts=insts, split='val',
        img_size=c['img_size'], grid_n=c['grid_n'],
        min_crop_px=c['min_crop_px'], max_crop_px=c['max_crop_px'],
        max_offset_m=c['t_m'], max_rot_deg=c['rot_deg'],
        # 窓の数は画像を覆う格子の数。学習の oversample (PandaSet なら 40) のままだと、
        # 大きな画像では格子の先頭 40 枚 (上の行) しか使わなかった。
        oversample=max(int(c['oversample']),
                       max(math.ceil(int(i['IW']) / int(c['min_crop_px'])) *
                           math.ceil(int(i['IH']) / int(c['min_crop_px'])) for i in insts)),
        crop_grid=True,
        k_per_cell=int(c.get('k_per_cell', 8)),
        share_pert=True, split_pert=False, n_full=0,
        # 推論は摂動ゼロ。ずれは渡されたポーズ側に入っている。
        # 検証で学習経路 (中で摂動) と比べるときだけ (t[3], ypr[3]) を渡す。
        fixed_pert=np.zeros(6) if fixed_pert is None else fixed_pert,
    )


# ── 推論 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, ds, device='cuda'):
    """inst → δ (6,) [ωx, ωy, ωz (deg), tx, ty, tz (m)]。並びと単位は
    scripts/ba/gn_pose.DOF6 / ba_torch._apply_extrinsic のまま。学習の eval と同じ predict_pose を
    呼ぶだけ。forward も GN もここには書かない。"""
    from datasets.train_cnd2_ddp import predict_pose
    wins = ds[0]
    if not isinstance(wins, list):
        wins = [wins]
    batch = [t.to(device) if torch.is_tensor(t) else t for t in collate_full(wins)]
    delta, H = predict_pose(model, batch, img_size=model.img_size, group=len(wins))
    return delta[0].cpu().numpy(), H[0].cpu().numpy(), len(wins)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('exp')
    ap.add_argument('--pandaset-val', type=int, default=None,
                    help='ファイルの代わりに PandaSet のキャッシュの val の i 番目のフレームを使う (calib は正しいポーズ)')
    ap.add_argument('--image', default=None)
    ap.add_argument('--points', default=None)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--overlay', default=None,
                    help='画像に点を重ねて保存する (左: 入力のポーズ、右: 補正後。正しいポーズがあれば緑で重ねる)')
    ap.add_argument('--pert-rot-deg', type=float, default=0.0,
                    help='評価用。正しいポーズに外から掛ける回転 (3 軸同じ値)')
    ap.add_argument('--pert-t-m', type=float, default=0.0,
                    help='評価用。同じく並進')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rot', type=str, default=None,
                    help='評価用。正しいポーズに掛ける回転 JSON [3] (ZYX: z=ロール, y=ヨー, x=ピッチ, deg)')
    ap.add_argument('--t', type=str, default=None,
                    help='評価用。正しいポーズに掛ける並進 JSON [3] (カメラ軸 x 右, y 下, z 前, m)')
    a = ap.parse_args()

    if a.pandaset_val is not None:
        img, pts, K, T, label, _, dist, fe = load_val_frame(a.pandaset_val)
        print(f'PandaSet val #{a.pandaset_val}  シーン/フレーム {label}  点 {len(pts)}')
    else:
        if not (a.image and a.points and a.calib):
            ap.error('--pandaset-val か、--image --points --calib の 3 つを指定する')
        img = cv2.imread(a.image)[:, :, ::-1]
        pts = load_points(a.points)
        K, dist, T, fe = load_calib(a.calib)
    inj = None
    if a.rot is not None or a.t is not None:
        import json as _json
        r = np.asarray(_json.loads(a.rot) if a.rot else [0, 0, 0], float).reshape(3)
        t = np.asarray(_json.loads(a.t) if a.t else [0, 0, 0], float).reshape(3)
        inj = (t, r)
        T_true = T
        T = perturb_T(T, rot_deg=r, t_m=t)
        print(f'注入 t={t} m  ZYX={r} deg')
    elif a.pert_rot_deg or a.pert_t_m:
        g = np.random.default_rng(a.seed)
        inj = ((g.random(3) * 2 - 1) * a.pert_t_m,
               (g.random(3) * 2 - 1) * a.pert_rot_deg)
        T_true = T
        T = perturb_T(T, rot_deg=inj[1], t_m=inj[0])
        print(f'注入 t={inj[0]} m  ypr={inj[1]} deg')

    model, c = load_model(a.exp)
    inst = build_inst(img=img, pts=pts, K=K, T_cam_lidar=T, dist=dist, is_fisheye=fe)
    ds = make_dataset(c, [inst])
    delta, H, nwin = infer(model, ds)
    # δ の並びは (ωx, ωy, ωz [deg], tx, ty, tz [m])
    T_corr = apply_delta(T, delta)
    np.set_printoptions(precision=6, suppress=True)
    print(f'窓 {nwin}   δ rot={delta[:3]} deg  t={delta[3:]} m')
    print('補正後の T_cam_lidar:'); print(T_corr)
    if inj is not None:
        e0 = pose_error(T, T_true); e1 = pose_error(T_corr, T_true)
        print(f'正解との差  補正前 {e0[0]:.4f} deg {e0[1]:.4f} m   補正後 {e1[0]:.4f} deg {e1[1]:.4f} m')
        a0, a1 = pose_error_axes(T, T_true), pose_error_axes(T_corr, T_true)
        print(f'{"":8s}' + ''.join(f'{k:>11s}' for k in ('yaw[deg]', 'pitch[deg]', 'roll[deg]', 'x[m]', 'y[m]', 'z[m]')))
        for lab, e in (('補正前', a0), ('補正後', a1)):
            print(f'{lab:6s}' + ''.join(f'{e[k]:+11.4f}' for k in ('yaw_deg', 'pitch_deg', 'roll_deg', 'x_m', 'y_m', 'z_m')))
    if a.overlay:
        poses = {'input': T, 'corrected': T_corr}
        if inj is not None:
            poses['true'] = T_true
        print('重ねた画像:', draw_overlay(img, pts, K, poses, a.overlay, dist=dist, is_fisheye=fe))

if __name__ == '__main__':
    main()
