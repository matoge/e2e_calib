"""LiDAR→カメラ外部パラメータ補正 API。学習と同じ推論経路を通す。

推論は scripts/inference/infer_calib.py の 1 本だけ:
    build_inst (画像 + 点群 + K + ポーズ) → PandaSetCalibDatasetFull(insts=...)
    → 格子窓 → predict_pose (学習の eval と同じ forward と同じ GN)
    → δ → 補正後 T = E(δ) @ T_入力
以前のサーバは kamikado のタイルキャッシュと旧モデル (σ 由来の W、
4 サブ窓、KB 魚眼ソルバ) で組まれた別経路だった (2026-10-07 に置き換え)。

Endpoints
  GET  /api/health           モデルと設定
  POST /api/calibrate_frame  multipart: image, points, calib  → δ, 補正後 T, σ
  POST /api/eval_frame       multipart: image, points, calib(正しいポーズ),
                             rot_deg, t_m (外から掛ける摂動) → 補正前後の誤差
  GET  /calibrate            アップロード用ページ

点群: LiDAR 座標の x y z [intensity]。txt (空白/カンマ区切り、# はコメント)、
      npy、bin (float32 の N x 4 または N x 5)。
calib: {"K": 3x3, "T_cam_lidar": 4x4, "dist": [k1..k4] (省略可)}
       または kamikado の calib.calib。
δ:   (ωx, ωy, ωz [deg], tx, ty, tz [m])。補正後 = E(δ) @ T_入力、
     E は P' = R(ω)·P + t (ω は回転ベクトル)。

起動
  E2E_EXP=ps_s1_noleak python -m uvicorn services.calib_api.server:app \\
      --host 0.0.0.0 --port 5002
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.inference.infer_calib import (load_model, build_inst, make_dataset,
                                           infer, load_points, load_calib,
                                           perturb_T, apply_delta, pose_error,
                                           pose_error_axes)
from services.calib_api import __version__ as API_VERSION

HERE = Path(__file__).resolve().parent
EXP = os.environ.get('E2E_EXP', 'ps_s1_noleak')
DEVICE = os.environ.get('E2E_DEVICE', 'cuda')
CKPT = os.environ.get('E2E_CKPT', 'best_model.pt')
# 魚眼 (kamikado / woven / TSS4) は別のモデルで解ける。未指定なら E2E_EXP を全部に使う
EXP_FE = os.environ.get('E2E_EXP_FISHEYE', '')
CKPT_FE = os.environ.get('E2E_CKPT_FISHEYE', 'best_model.pt')

app = FastAPI(title='e2e_calib calibration API', version=API_VERSION)
_state: dict = {}


@app.on_event('startup')
def _startup():
    model, cfg = load_model(EXP, device=DEVICE, ckpt=CKPT)
    _state.update(model=model, cfg=cfg)
    print(f'[boot] exp={EXP}/{CKPT} img_size={cfg["img_size"]} grid_n={cfg["grid_n"]} '
          f'crop={cfg["min_crop_px"]} device={DEVICE}', flush=True)
    if EXP_FE:
        m2, c2 = load_model(EXP_FE, device=DEVICE, ckpt=CKPT_FE)
        _state.update(model_fe=m2, cfg_fe=c2)
        print(f'[boot] fisheye exp={EXP_FE}/{CKPT_FE} crop={c2["min_crop_px"]}', flush=True)


def _sigma_from_H(H: np.ndarray):
    """H (6x6, GN の情報行列) → 各成分の 1σ。並びは δ と同じ。"""
    try:
        cov = np.linalg.inv(np.asarray(H, np.float64))
        s = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {'rot_deg': s[:3].tolist(), 't_m': s[3:].tolist()}
    except np.linalg.LinAlgError:
        return None


async def _read_inputs(image: UploadFile, points: UploadFile, calib: UploadFile):
    try:
        img = np.array(Image.open(io.BytesIO(await image.read())).convert('RGB'))
    except Exception as e:
        raise HTTPException(400, f'cannot read image: {type(e).__name__}: {e}')
    try:
        pts = load_points(await points.read(), points.filename or '')
    except Exception as e:
        raise HTTPException(400, f'cannot read points: {type(e).__name__}: {e}')
    try:
        K, dist, T, fe = load_calib(await calib.read(), calib.filename or '')
    except Exception as e:
        raise HTTPException(400, f'cannot read calib: {type(e).__name__}: {e}')
    return img, pts, K, dist, T, fe


def _solve(img, pts, K, dist, T, fe):
    if fe and 'model_fe' in _state:
        model, cfg = _state['model_fe'], _state['cfg_fe']
    else:
        model, cfg = _state['model'], _state['cfg']
    t0 = time.time()
    try:
        inst = build_inst(img=img, pts=pts, K=K, T_cam_lidar=T, dist=dist, is_fisheye=fe)
        delta, H, nwin = infer(model, make_dataset(cfg, [inst]), device=DEVICE)
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    T_corr = apply_delta(T, delta)
    return dict(
        delta={'rot_deg': delta[:3].tolist(), 't_m': delta[3:].tolist()},
        T_cam_lidar_corrected=T_corr.tolist(),
        sigma=_sigma_from_H(H),
        n_windows=int(nwin), image_size=[int(img.shape[1]), int(img.shape[0])],
        n_points=int(len(pts)), took_s=round(time.time() - t0, 3),
    ), T_corr


def _overlay(img, pts, K, dist, fe, poses: dict, n_max: int = 4000, seed: int = 0):
    """ページで画像に重ねる用。同じ点 (最大 n_max 点、カメラ前方のもの) を各ポーズで投影した
    画素位置を返す。{name: [[u, v] or None, ...]}。点の選び方はポーズによらず共通。"""
    from scripts.util.projection import project_lidar_into_image
    IH, IW = img.shape[:2]
    first = next(iter(poses.values()))
    keep0, *_ = project_lidar_into_image(pts, K, first, IW, IH, is_fisheye=fe, dist=dist)
    idx = np.where(keep0)[0]
    if len(idx) > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_max, replace=False))
    sub = pts[idx]
    out = {}
    for name, T in poses.items():
        keep, _pc, uv, *_ = project_lidar_into_image(sub, K, T, IW, IH, is_fisheye=fe, dist=dist)
        full = [None] * len(sub)
        for j, (u, v) in zip(np.where(keep)[0], uv):
            full[j] = [round(float(u), 1), round(float(v), 1)]
        out[name] = full
    return out


@app.get('/api/health')
def health():
    cfg = _state.get('cfg', {})
    return {'api_version': API_VERSION, 'exp': EXP, 'device': DEVICE,
            'img_size': cfg.get('img_size'), 'grid_n': cfg.get('grid_n'),
            'crop_px': cfg.get('min_crop_px'),
            'trained_pert': {'rot_deg': cfg.get('rot_deg'), 't_m': cfg.get('t_m')}}


@app.post('/api/calibrate_frame')
async def calibrate_frame(image: UploadFile = File(...), points: UploadFile = File(...),
                          calib: UploadFile = File(...)):
    """渡されたポーズ (calib の T_cam_lidar) の補正量を返す。"""
    img, pts, K, dist, T, fe = await _read_inputs(image, points, calib)
    out, T_corr = _solve(img, pts, K, dist, T, fe)
    out['overlay'] = _overlay(img, pts, K, dist, fe, {'input': T, 'corrected': T_corr})
    return out


@app.post('/api/eval_frame')
async def eval_frame(image: UploadFile = File(...), points: UploadFile = File(...),
                     calib: UploadFile = File(...),
                     rot_deg: str = Form('[0,0,0]', description='JSON [3] ZYX euler, deg'),
                     t_m: str = Form('[0,0,0]', description='JSON [3] カメラ軸の並進, m')):
    """calib は正しいポーズ。外から (rot_deg, t_m) でずらして推論し、補正前後の
    誤差を返す。ずらし方は学習時に dataset が中で掛けるのと同じ式 (perturb_T)。"""
    img, pts, K, dist, T_true, fe = await _read_inputs(image, points, calib)
    try:
        r = np.asarray(json.loads(rot_deg), np.float64).reshape(3)
        t = np.asarray(json.loads(t_m), np.float64).reshape(3)
    except Exception as e:
        raise HTTPException(400, f'rot_deg / t_m must be JSON lists of 3 numbers: {e}')
    T_in = perturb_T(T_true, rot_deg=r, t_m=t)
    out, T_corr = _solve(img, pts, K, dist, T_in, fe)
    e0 = pose_error(T_in, T_true); e1 = pose_error(T_corr, T_true)
    out['injected'] = {'rot_deg': r.tolist(), 't_m': t.tolist()}
    out['error_before'] = {'rot_deg': e0[0], 't_m': e0[1]}
    out['error_after'] = {'rot_deg': e1[0], 't_m': e1[1]}
    # 軸ごと (カメラ座標: yaw = y 軸まわり、pitch = x 軸まわり、roll = 光軸まわり、x/y/z = カメラ中心のずれ)
    out['error_before_axes'] = pose_error_axes(T_in, T_true)
    out['error_after_axes'] = pose_error_axes(T_corr, T_true)
    out['overlay'] = _overlay(img, pts, K, dist, fe,
                              {'true': T_true, 'input': T_in, 'corrected': T_corr})
    return out


# ── val からフレームを選んで評価する (アップロード無しで試す用)。PandaSet と nuScenes ────────────
# キャッシュは学習に使ったものと合わせる (intensity の扱い・画像外の点の有無がキャッシュで違う)。
VAL_CACHES = {
    'pandaset': os.environ.get('E2E_PS_CACHE', '/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full'),
    'nuscenes': os.environ.get('E2E_NS_CACHE', '/mnt/ssd2t/work/e2e_calib/cache/ns_850x4_pad256'),
    'kamikado': os.environ.get('E2E_KM_CACHE', '/mnt/ssd2t/work/e2e_calib/cache/kamikado_pad256'),
    'tss4': os.environ.get('E2E_TSS4_CACHE', '/mnt/ssd2t/work/e2e_calib/cache/tss4_pad256'),
}
PS_CACHE = VAL_CACHES['pandaset']


def _cache_of(ds: str) -> str:
    if ds not in VAL_CACHES:
        raise HTTPException(400, f'ds must be one of {sorted(VAL_CACHES)}, got {ds!r}')
    return VAL_CACHES[ds]


def _val_src(ds: str):
    key = f'val_src_{ds}'
    if key not in _state:
        from datasets.pandaset_full import PandaSetCalibDatasetFull
        c = _state['cfg']
        _state[key] = PandaSetCalibDatasetFull(
            cache_dir=_cache_of(ds), split='val', img_size=c['img_size'], grid_n=c['grid_n'],
            min_crop_px=c['min_crop_px'], max_crop_px=c['max_crop_px'],
            max_offset_m=c['t_m'], max_rot_deg=c['rot_deg'], oversample=1)
    return _state[key]


def _val_raw(ds: str, i: int):
    """キャッシュの inst → (画像 RGB, 点 xyz+強度, K, 正しい T_cam_lidar, ラベル, JPEG, dist, is_fisheye)。
    CLI と同じ関数。魚眼のキャッシュは Kannala-Brandt の係数も返す。"""
    from scripts.inference.infer_calib import load_val_frame
    n = len(_val_src(ds))
    if not 0 <= int(i) < n:
        raise HTTPException(400, f'{ds} val index must be 0..{n - 1}, got {i}')
    return load_val_frame(i, _cache_of(ds))


def _parse_pert(rot_deg: str, t_m: str):
    try:
        r = np.asarray(json.loads(rot_deg), np.float64).reshape(3)
        t = np.asarray(json.loads(t_m), np.float64).reshape(3)
    except Exception as e:
        raise HTTPException(400, f'rot_deg / t_m must be JSON lists of 3 numbers: {e}')
    return r, t


def _val_eval(ds: str, i: int, rot_deg: str, t_m: str):
    img, pts, K, T_true, label, _, dist, fe = _val_raw(ds, i)
    r, t = _parse_pert(rot_deg, t_m)
    T_in = perturb_T(T_true, rot_deg=r, t_m=t)
    out, T_corr = _solve(img, pts, K, dist, T_in, fe)
    return img, pts, K, T_true, T_in, T_corr, label, r, t, out, dist, fe


@app.get('/api/val/frames')
def val_frames(ds: str = 'pandaset'):
    """val のフレーム数 (index は 0..n-1)。ds = pandaset | nuscenes | kamikado | tss4"""
    return {'ds': ds, 'n': len(_val_src(ds)), 'cache': _cache_of(ds), 'split': 'val'}


@app.get('/api/val/image/{i}')
def val_image(i: int, ds: str = 'pandaset'):
    from fastapi.responses import Response
    jpg = _val_raw(ds, i)[5]
    return Response(content=bytes(jpg), media_type='image/jpeg')


@app.post('/api/val/eval')
def val_eval(i: int = Form(...), ds: str = Form('pandaset'),
             rot_deg: str = Form('[0,0,0]'), t_m: str = Form('[0,0,0]')):
    """val の i 番目のフレームを、正しいポーズから (rot_deg, t_m) ずらして推論し、補正前後を返す。
    点の重ね描きは間引かず全点。"""
    img, pts, K, T_true, T_in, T_corr, label, r, t, out, dist, fe = _val_eval(ds, i, rot_deg, t_m)
    e0 = pose_error(T_in, T_true); e1 = pose_error(T_corr, T_true)
    out.update(ds=ds, frame=label, index=int(i), injected={'rot_deg': r.tolist(), 't_m': t.tolist()},
               error_before={'rot_deg': e0[0], 't_m': e0[1]}, error_after={'rot_deg': e1[0], 't_m': e1[1]},
               error_before_axes=pose_error_axes(T_in, T_true), error_after_axes=pose_error_axes(T_corr, T_true),
               overlay=_overlay(img, pts, K, dist, fe,
                                {'true': T_true, 'input': T_in, 'corrected': T_corr}, n_max=10 ** 9))
    return out


def _png_response(img_rgb: np.ndarray, info: dict):
    import cv2
    from fastapi.responses import Response
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    hdr = {'X-Error-Before': json.dumps(info.get('error_before_axes')),
           'X-Error-After': json.dumps(info.get('error_after_axes')),
           'X-Frame': str(info.get('frame', ''))}
    return Response(content=buf.tobytes(), media_type='image/png', headers=hdr)


@app.post('/api/val/eval_image')
def val_eval_image(i: int = Form(...), ds: str = Form('pandaset'),
                   rot_deg: str = Form('[0,0,0]'), t_m: str = Form('[0,0,0]')):
    """/api/val/eval と同じことをして、補正前 (左) と補正後 (右) の重ね画像を PNG で返す。
    誤差 (ヨー・ピッチ・ロール・x・y・z) はヘッダ X-Error-Before / X-Error-After (JSON)。"""
    from scripts.inference.infer_calib import render_overlay
    img, pts, K, T_true, T_in, T_corr, label, r, t, out, dist, fe = _val_eval(ds, i, rot_deg, t_m)
    info = dict(frame=f'{ds} {label}', error_before_axes=pose_error_axes(T_in, T_true),
                error_after_axes=pose_error_axes(T_corr, T_true))
    return _png_response(render_overlay(img, pts, K, {'true': T_true, 'input': T_in, 'corrected': T_corr},
                                        dist=dist, is_fisheye=fe), info)


# 以前の PandaSet 専用の URL (ds=pandaset 固定の別名)
@app.get('/api/pandaset/frames')
def ps_frames():
    return val_frames('pandaset')


@app.get('/api/pandaset/image/{i}')
def ps_image(i: int):
    return val_image(i, 'pandaset')


@app.post('/api/pandaset/eval')
def ps_eval(i: int = Form(...), rot_deg: str = Form('[0,0,0]'), t_m: str = Form('[0,0,0]')):
    return val_eval(i, 'pandaset', rot_deg, t_m)


@app.post('/api/pandaset/eval_image')
def ps_eval_image(i: int = Form(...), rot_deg: str = Form('[0,0,0]'), t_m: str = Form('[0,0,0]')):
    return val_eval_image(i, 'pandaset', rot_deg, t_m)


@app.post('/api/eval_frame_image')
async def eval_frame_image(image: UploadFile = File(...), points: UploadFile = File(...),
                           calib: UploadFile = File(...),
                           rot_deg: str = Form('[0,0,0]'), t_m: str = Form('[0,0,0]')):
    """/api/eval_frame と同じ入力。補正前 (左) と補正後 (右) の重ね画像を PNG で返す。"""
    from scripts.inference.infer_calib import render_overlay
    img, pts, K, dist, T_true, fe = await _read_inputs(image, points, calib)
    r = np.asarray(json.loads(rot_deg), np.float64).reshape(3)
    t = np.asarray(json.loads(t_m), np.float64).reshape(3)
    T_in = perturb_T(T_true, rot_deg=r, t_m=t)
    out, T_corr = _solve(img, pts, K, dist, T_in, fe)
    info = dict(error_before_axes=pose_error_axes(T_in, T_true), error_after_axes=pose_error_axes(T_corr, T_true))
    return _png_response(render_overlay(img, pts, K, {'true': T_true, 'input': T_in, 'corrected': T_corr},
                                        dist=dist, is_fisheye=fe), info)


@app.get('/calibrate', response_class=HTMLResponse)
def page():
    return (HERE / 'static' / 'upload.html').read_text(encoding='utf-8')
