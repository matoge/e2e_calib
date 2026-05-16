"""CaaaS — Calibration As A Service (Triple-A).

Minimal Flask app for in-house demo. POST a single image + LiDAR point cloud
+ intrinsics → returns the predicted extrinsic correction (δR, δT) plus
per-point uncertainty. Sequence input is the next iteration; for now we
process one frame at a time so the BA aggregator has a Σ to work with.

Endpoints:
  GET  /health                       liveness
  GET  /api/models                   list available checkpoints
  POST /api/calibrate                multipart upload, returns JSON
  POST /api/calibrate_sequence       N images + N point clouds → BA-fused δ

Multipart fields for /api/calibrate:
  image      : JPEG/PNG bytes
  pts        : .npy or .bin (N, 4) [x, y, z, intensity]
  K          : JSON [[fx,0,cx],[0,fy,cy],[0,0,1]]
  T_cam_lidar: JSON 4x4 (optional, defaults to identity = pts already in cam frame)
  exp        : checkpoint name (optional, defaults to env CAAAS_EXP)
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
from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.infer_calib import load_calib_model
from scripts.ba.vcam_aggregator import aggregate_vcam_to_orig

app = Flask(__name__)

DEFAULT_EXP = os.environ.get(
    'CAAAS_EXP',
    'km_wv_wm_dgx2_n2_img128_v2',  # the 3-dataset n=2 best as of 2026-05-16
)
_MODEL_CACHE: dict = {}
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_model(exp: str):
    if exp not in _MODEL_CACHE:
        _MODEL_CACHE[exp] = load_calib_model(exp, device=_DEVICE).eval()
    return _MODEL_CACHE[exp]


def _decode_image(buf: bytes) -> np.ndarray:
    """JPEG/PNG bytes → RGB uint8 (H, W, 3)."""
    try:
        from turbojpeg import TurboJPEG
        return TurboJPEG().decode(buf)
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(buf)).convert('RGB'))


def _decode_pts(name: str, buf: bytes) -> np.ndarray:
    if name.endswith('.npy'):
        return np.load(io.BytesIO(buf), allow_pickle=False)
    # raw .bin assumed (N, 4) float32: x, y, z, intensity
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f'pts .bin size {arr.size} not divisible by 4')
    return arr.reshape(-1, 4)


@app.route('/health')
def health():
    return jsonify(status='ok', device=_DEVICE,
                   loaded_exps=list(_MODEL_CACHE.keys()),
                   default_exp=DEFAULT_EXP)


@app.route('/api/models')
def list_models():
    exps = []
    for d in (REPO_ROOT / 'experiments').iterdir():
        if (d / 'best_model.pt').is_file() and (d / 'config.py').is_file():
            exps.append(d.name)
    return jsonify(models=sorted(exps), default=DEFAULT_EXP)


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Full-frame calibration: sliding-tile inference + BA.

    multipart fields:
      image       JPEG/PNG bytes (any size)
      pts         .npy or .bin (N, 4) [x, y, z, intensity]  in LiDAR frame
      K           JSON 3×3 pinhole intrinsics
      T_cam_lidar JSON 4×4 LiDAR→camera (optional, default identity =
                  pts already in cam frame)
      exp         checkpoint name (optional, defaults to env CAAAS_EXP)
      huber_k     IRLS Huber k in Mahalanobis units (optional, 0=off)
      n_iter      IRLS iterations (optional, default 1)
      sigma_max   pre-filter: drop pts with σ_pt > sigma_max in image-px
                  (optional, 0 = keep all)
      is_fisheye  '1' to use Kannala projection (requires `dist`)
      dist        JSON 4-vec Kannala k1..k4 (only when is_fisheye=1)
    """
    from scripts.util.projection import (
        project_lidar_into_image, project_pinhole, project_kannala)
    from scripts.ba.ba_multicam_corr import (
        infer_tiles, solve_dofs, _DOF_PRESETS)

    t0 = time.time()
    if 'image' not in request.files or 'pts' not in request.files:
        return jsonify(error='image and pts files required'), 400
    if 'K' not in request.form:
        return jsonify(error='K (3x3 intrinsics) form field required'), 400

    img = _decode_image(request.files['image'].read())
    pts4 = _decode_pts(request.files['pts'].filename or 'p.bin',
                        request.files['pts'].read())  # (N, 4)
    K = np.asarray(json.loads(request.form['K']), dtype=np.float64)
    T_cl = (np.asarray(json.loads(request.form['T_cam_lidar']),
                        dtype=np.float64)
            if 'T_cam_lidar' in request.form else np.eye(4))
    exp = request.form.get('exp', DEFAULT_EXP)
    is_fisheye = request.form.get('is_fisheye', '0') == '1'
    dist = (np.asarray(json.loads(request.form['dist']), dtype=np.float64)
            if 'dist' in request.form else None)
    huber_k   = float(request.form.get('huber_k', 0.0))
    n_iter    = int(request.form.get('n_iter', 1))
    sigma_max = float(request.form.get('sigma_max', 0.0))
    model = _get_model(exp)

    # 1) Shared LiDAR→cam→project pipeline (kamikado/woven/CaaaS identical).
    H, W = img.shape[:2]
    keep, pts_cam, uv_full, z_cam, intensity = project_lidar_into_image(
        pts4, K, T_cl, W, H,
        is_fisheye=is_fisheye, dist=dist, z_min=0.5)
    if uv_full.shape[0] < 16:
        return jsonify(error=f'only {uv_full.shape[0]} pts landed in image; '
                              'check K / T_cam_lidar'), 400

    # 2) Sliding-tile inference. (per-sensor intensity normalisation is the
    # caller's job; cache build does it offline. Default /128 is a safe
    # mid-band for kamikado/woven; waymo would want /1.0.)
    intens_norm = np.clip(intensity / 128.0, 0.0, 1.0).astype(np.float32)
    ba_cfg = dict(tile_size=512, model_input_size=128,
                  max_pts_per_tile=256, min_pts_per_tile=8,
                  tile_stride=384)
    res = infer_tiles(model, img, uv_full, z_cam, K, ba_cfg,
                       torch.device(_DEVICE), intensity=intens_norm)
    if res is None:
        return jsonify(error='no tiles passed min_pts threshold'), 500
    uv_pool, par_pool, z_pool = res
    n_pool = int(uv_pool.shape[0])

    # 3) σ pre-filter (in image-px since infer_tiles already exp+scaled par[:,2:4]).
    sigma_pt = np.sqrt(par_pool[:, 2] * par_pool[:, 3])
    if sigma_max > 0.0:
        keep_pre = sigma_pt <= sigma_max
        uv_pool = uv_pool[keep_pre]; par_pool = par_pool[keep_pre]
        z_pool = z_pool[keep_pre]; sigma_pt = sigma_pt[keep_pre]
    n_after = int(uv_pool.shape[0])
    if n_after < 6:
        return jsonify(error=f'too few pts after sigma_max filter ({n_after})'), 400

    # 4) BA: 1-step closed-form (huber_k=0) or IRLS Huber otherwise.
    delta_6 = solve_dofs(uv_pool, par_pool, z_pool, K,
                          _DOF_PRESETS['6dof_ext'], damping=1e-3,
                          huber_k=(huber_k if huber_k > 0 else None),
                          n_iter=n_iter)
    cov   = solve_dofs._last_cov
    sigma = np.sqrt(np.diag(cov))
    irls_w = getattr(solve_dofs, '_last_weights', None)
    irls_w = irls_w.astype(float).tolist() if irls_w is not None else None

    # 5) BA-after re-projection of the FULL set of input points: rotate +
    # translate the cam-frame points by δ and project (pinhole only here;
    # fisheye demo paths use the un-distorted parent K). For a fisheye
    # frame we still report uv_after via pinhole-with-parent-K — that's
    # what training optimised against.
    omega = delta_6[:3]; t_delta = delta_6[3:6]
    th = float(np.linalg.norm(omega))
    if th < 1e-9:
        R_d = np.eye(3)
    else:
        k = omega / th
        K_x = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R_d = (np.eye(3) + np.sin(th) * K_x
               + (1 - np.cos(th)) * (K_x @ K_x))
    pts_after = (R_d @ pts_cam.T).T + t_delta[None, :]
    if is_fisheye and dist is not None:
        uv_after = project_kannala(pts_after,
                                     np.asarray(K, dtype=np.float64),
                                     np.asarray(dist, dtype=np.float64))
    else:
        uv_after = project_pinhole(pts_after,
                                    np.asarray(K, dtype=np.float64))

    return jsonify(
        ok=True,
        n_input_pts=int(pts4.shape[0]),
        n_in_image=int(uv_full.shape[0]),
        n_pool=n_pool,
        n_pool_after_filter=n_after,
        dof_names=_DOF_PRESETS['6dof_ext'],
        delta_pred=delta_6.tolist(),
        sigma_pred=sigma.tolist(),
        cov=cov.tolist(),
        elapsed_ms=int((time.time() - t0) * 1000),
        exp=exp,
        is_fisheye=is_fisheye,
        # per-pt dump for the front-end overlay (BA-pre vs BA-post).
        uv_before=uv_full.astype(float).tolist(),
        uv_after=uv_after.astype(float).tolist(),
        z_cam=z_cam.astype(float).tolist(),
        # subsampled pool used by BA (with per-pt model output).
        pool=dict(
            uv=uv_pool.astype(float).tolist(),
            du=par_pool[:, 0].astype(float).tolist(),
            dv=par_pool[:, 1].astype(float).tolist(),
            sx=par_pool[:, 2].astype(float).tolist(),
            sy=par_pool[:, 3].astype(float).tolist(),
            rho=par_pool[:, 4].astype(float).tolist(),
        ),
        irls=dict(huber_k=huber_k, n_iter=n_iter, sigma_max=sigma_max,
                   weights=irls_w),
    )


@app.route('/api/calibrate_sequence', methods=['POST'])
def calibrate_sequence():
    """N-frame BA-fused calibration. Returns single δ_orig (6-DoF)."""
    return jsonify(error='not implemented; '
                          'wire scripts/ba/vcam_aggregator.py here'), 501


# ── Demo endpoints (PandaSet/Waymo cache, deterministic seed) ────────────

_DEMO_DS = {}


def _get_demo_ds(cache_key: str):
    if cache_key not in _DEMO_DS:
        from scripts.inference.infer_pipeline import make_ds
        cache_path = {
            'kamikado': '/cache/kamikado_v3_tiled',
            'woven': '/cache/woven_v3_tile',
            'waymo': '/cache/waymo_v3_tiled_i',
        }.get(cache_key)
        if cache_path is None:
            raise ValueError(f'unknown demo cache_key {cache_key}')
        ds, c = make_ds(DEFAULT_EXP, cache_path, split='val', oversample=1)
        _DEMO_DS[cache_key] = (ds, c)
    return _DEMO_DS[cache_key]


@app.route('/api/calibrate_demo')
def calibrate_demo():
    """GET ?cache=kamikado&idx=0&seed=42

    Picks the val sample idx, applies the dataset's deterministic
    perturbation under the given seed (so δ_GT is known), runs
    CaaaS BA on the full frame, returns δ_GT, δ_pred, Cov diag, plus
    a base64-encoded JPEG of the tile.
    """
    import base64, io as _io
    from PIL import Image as _PIL
    from scripts.ba.ba_multicam_corr import infer_tiles, solve_dofs, _DOF_PRESETS

    cache_key = request.args.get('cache', 'kamikado')
    idx = int(request.args.get('idx', 0))
    seed = int(request.args.get('seed', 42))
    # IRLS knobs (Huber M-estimator). huber_k=0 disables outlier reweighting.
    huber_k = float(request.args.get('huber_k', 0.0))
    n_iter  = int(request.args.get('n_iter', 1))
    # Pre-filter: drop pts with σ_scalar above sigma_max (model-px). 0 = keep all.
    sigma_max = float(request.args.get('sigma_max', 0.0))

    np.random.seed(seed)
    ds, c = _get_demo_ds(cache_key)
    model = _get_model(DEFAULT_EXP)

    # Pull the perturbed sample so δ_GT (pert_vec) is set on the dataset
    sample = ds[idx]
    pert_vec = sample[6].numpy() if len(sample) >= 7 else None

    inst = ds._load_inst(idx)
    full_jpg = bytes(inst['jpg_bytes'])
    img = np.asarray(_PIL.open(_io.BytesIO(full_jpg)).convert('RGB'))
    uv = inst['uv_full'].numpy().astype(np.float32)
    z  = inst['z_cam'].numpy().astype(np.float32)
    intensity = (inst['intensity'].numpy().astype(np.float32)
                 if 'intensity' in inst else None)
    K = inst['K_full'].numpy().astype(np.float32)
    is_fisheye_inst = bool(inst.get('is_fisheye', False))
    tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
    if tu0 or tv0:
        uv = uv - np.array([tu0, tv0], dtype=np.float32)
        K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
    H, W = img.shape[:2]
    keep = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0))
    uv = uv[keep]; z = z[keep]
    if intensity is not None:
        intensity = np.clip(intensity[keep] / 128.0, 0.0, 1.0).astype(np.float32)
    # GT projection is exactly `inst['uv_full']` (cache build wrote uv via
    # the same K+dist that produced the image), already filtered by `keep`.
    uv_gt = uv.copy()
    z_gt_keep = z.copy()

    ba_cfg = dict(tile_size=512, model_input_size=c['img_size'],
                  max_pts_per_tile=256, min_pts_per_tile=8,
                  tile_stride=384)
    res = infer_tiles(model, img, uv, z, K, ba_cfg, torch.device(_DEVICE),
                      intensity=intensity)
    if res is None:
        return jsonify(error='no tiles passed min_pts'), 500
    uv_full, par, z_full = res
    # par columns 2,3 are already linear σx,σy in original-image px (infer_tiles
    # applies np.exp + tile-scale). geom-mean as the per-pt σ scalar.
    sigma_pt = np.sqrt(par[:, 2] * par[:, 3])
    n_before = uv_full.shape[0]
    if sigma_max > 0.0:
        keep_pre = sigma_pt <= sigma_max
        uv_full = uv_full[keep_pre]; par = par[keep_pre]; z_full = z_full[keep_pre]
        sigma_pt = sigma_pt[keep_pre]
    n_after = uv_full.shape[0]
    if n_after < 6:
        return jsonify(error=f'too few pts after sigma_max filter '
                              f'({n_after} <= 6); raise sigma_max'), 400
    delta_6 = solve_dofs(uv_full, par, z_full, K,
                          _DOF_PRESETS['6dof_ext'], damping=1e-3,
                          huber_k=(huber_k if huber_k > 0 else None),
                          n_iter=n_iter)
    cov = solve_dofs._last_cov
    sigma = np.sqrt(np.diag(cov))
    irls_w = getattr(solve_dofs, '_last_weights', None)
    irls_w = irls_w.astype(float).tolist() if irls_w is not None else None

    # Encode the (perturbed) tile image for the demo UI.
    buf = _io.BytesIO()
    _PIL.fromarray(img).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    # Per-pt (uv, du, dv, sx, sy, rho) — for the front-end overlay.
    pts_json = dict(
        uv=uv_full.astype(float).tolist(),
        du=par[:, 0].astype(float).tolist(),
        dv=par[:, 1].astype(float).tolist(),
        sx=par[:, 2].astype(float).tolist(),
        sy=par[:, 3].astype(float).tolist(),
        rho=par[:, 4].astype(float).tolist(),
    )

    return jsonify(
        ok=True,
        cache=cache_key,
        idx=idx,
        seed=seed,
        n_pool=int(n_after),
        n_pool_before_filter=int(n_before),
        dof_names=_DOF_PRESETS['6dof_ext'],
        delta_pred=delta_6.tolist(),
        sigma_pred=sigma.tolist(),
        cov=cov.tolist(),
        delta_gt=(pert_vec.tolist() if pert_vec is not None else None),
        img_b64=img_b64,
        img_w=int(img.shape[1]), img_h=int(img.shape[0]),
        is_fisheye=is_fisheye_inst,
        pts=pts_json,
        # GT-pose re-projection (stored cam-frame pts → image via K + dist).
        # Subsampled to keep the JSON small.
        gt_proj=dict(
            uv=uv_gt[::max(1, len(uv_gt) // 4000)].astype(float).tolist(),
            z=z_gt_keep[::max(1, len(z_gt_keep) // 4000)].astype(float).tolist(),
            n_total=int(len(uv_gt)),
        ),
        irls=dict(huber_k=huber_k, n_iter=n_iter, weights=irls_w,
                   sigma_max=sigma_max),
    )


@app.route('/demo')
def demo_page():
    return DEMO_HTML


DEMO_HTML = '''<!doctype html>
<html><head><meta charset="utf-8"><title>CaaaS demo</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;padding:24px}
h1{margin:0 0 8px} .row{display:flex;gap:24px;align-items:flex-start}
button{font-size:18px;padding:8px 16px;border-radius:6px;
  background:#3a7;color:#fff;border:0;cursor:pointer}
button:hover{background:#4c8}
table{border-collapse:collapse;margin-top:12px}
td,th{border:1px solid #444;padding:4px 10px;text-align:right;font-family:monospace}
th{background:#222} td.label{text-align:left;background:#1a1a1a}
img{max-width:640px;border:2px solid #555;border-radius:6px}
small{color:#888}
.tabs{margin:8px 0;border-bottom:1px solid #444}
.tabs button{background:#222;color:#aaa;border-radius:6px 6px 0 0;
  margin-right:4px;font-size:14px;padding:6px 14px}
.tabs button.active{background:#3a7;color:#fff}
.tab{display:none}.tab.active{display:block}
</style></head><body>
<h1>CaaaS — Calibration As A Service <small>(Triple-A)</small></h1>
<p>cache:
  <select id="cache"><option>kamikado</option><option>woven</option><option>waymo</option></select>
  idx <input id="idx" type="number" value="0" style="width:60px">
  seed <input id="seed" type="number" value="42" style="width:60px">
  <button onclick="run()">Random shift &amp; predict</button>
  <button onclick="random_seed()">Re-roll seed</button>
</p>
<p>
  σ_max <input id="sigma_max" type="number" value="0" step="0.5" min="0" style="width:55px">
  <small>(image-px; 0 = keep all)</small>
  &nbsp;IRLS Huber k <input id="huber_k" type="number" value="0" step="0.5" min="0" style="width:55px">
  <small>(0 = off)</small>
  &nbsp;n_iter <input id="n_iter" type="number" value="3" min="1" max="10" style="width:45px">
  <span id="irls_stats" style="color:#9c9;margin-left:12px"></span>
</p>

<div class="tabs">
  <button id="tab_btn_full" class="active" onclick="show_tab('full')">Full image (GT / BA-before / BA-after)</button>
  <button id="tab_btn_tile" onclick="show_tab('tile')">Tile (per-pt prediction + σ)</button>
</div>

<div id="tab_full" class="tab active">
  <div class="row">
    <div>
      <div style="position:relative;display:inline-block">
        <img id="img_full" alt="">
        <canvas id="ov_full" style="position:absolute;left:0;top:0;pointer-events:none"></canvas>
      </div>
      <p><small id="meta_full"></small></p>
      <p>
        <label><input type="checkbox" id="show_gt" checked> GT projection (yellow)</label>
        <label><input type="checkbox" id="show_pre" checked> BA-before (red)</label>
        <label><input type="checkbox" id="show_post" checked> BA-after (green)</label>
      </p>
    </div>
    <div>
      <h3>6-DoF δ (predicted vs GT)</h3>
      <table id="tbl">
        <tr><th>DoF</th><th>δ_pred</th><th>σ_pred</th><th>δ_GT</th><th>err</th></tr>
      </table>
    </div>
  </div>
</div>

<div id="tab_tile" class="tab">
  <div class="row">
    <div>
      <div style="position:relative;display:inline-block">
        <img id="img_tile" alt="">
        <canvas id="ov_tile" style="position:absolute;left:0;top:0;pointer-events:none"></canvas>
      </div>
      <p><small id="meta_tile"></small></p>
      <p>
        <label><input type="checkbox" id="show_pts" checked> input pts (perturbed UV, red)</label>
        <label><input type="checkbox" id="show_pred" checked> predicted (UV+δ, green)</label>
        <label><input type="checkbox" id="show_sig" checked> 1.5σ ellipses</label>
      </p>
    </div>
  </div>
</div>
<script>
async function run() {
  const c = document.getElementById('cache').value;
  const i = document.getElementById('idx').value;
  const s = document.getElementById('seed').value;
  const hk = document.getElementById('huber_k').value;
  const ni = document.getElementById('n_iter').value;
  const sm = document.getElementById('sigma_max').value;
  const base = location.pathname.replace(/\/demo$/, '');
  const r = await fetch(`${base}/api/calibrate_demo?cache=${c}&idx=${i}&seed=${s}&huber_k=${hk}&n_iter=${ni}&sigma_max=${sm}`);
  const d = await r.json();
  if (!d.ok) { document.getElementById('meta').innerText = 'ERROR: ' + d.error; return; }
  const im = document.getElementById('img');
  im.onload = () => draw_overlay(d);
  im.src = 'data:image/jpeg;base64,' + d.img_b64;
  document.getElementById('meta').innerText =
    `cache=${d.cache} idx=${d.idx} seed=${d.seed}  pool_N=${d.n_pool}` +
    (d.n_pool_before_filter && d.n_pool_before_filter !== d.n_pool
       ? ` (filtered from ${d.n_pool_before_filter} by σ_max=${d.irls.sigma_max})`
       : '');
  // IRLS stats: how many points got down-weighted by Huber
  const stats = document.getElementById('irls_stats');
  if (d.irls && d.irls.weights) {
    const w = d.irls.weights;
    const n_full = w.filter(x => x >= 0.999).length;
    const n_down = w.length - n_full;
    const w_min = Math.min(...w);
    stats.innerText = `IRLS: huber_k=${d.irls.huber_k} iter=${d.irls.n_iter}  full-weight=${n_full}/${w.length}  outliers=${n_down}  min_w=${w_min.toFixed(2)}`;
  } else {
    stats.innerText = 'IRLS: off (1-step closed-form)';
  }
  window._last = d;
  const tb = document.getElementById('tbl');
  tb.innerHTML = '<tr><th>DoF</th><th>δ_pred</th><th>σ_pred</th><th>δ_GT</th><th>err</th></tr>';
  const gt = d.delta_gt || new Array(6).fill(null);
  d.dof_names.forEach((nm, k) => {
    const dp = d.delta_pred[k], sp = d.sigma_pred[k];
    const dg = gt[k];
    const err = (dg !== null && dg !== undefined) ? (dp - dg) : null;
    tb.innerHTML += `<tr><td class=label>${nm}</td><td>${dp.toFixed(4)}</td><td>${sp.toFixed(4)}</td><td>${dg!==null?dg.toFixed(4):'—'}</td><td>${err!==null?err.toFixed(4):'—'}</td></tr>`;
  });
}
function random_seed() {
  document.getElementById('seed').value = Math.floor(Math.random() * 100000);
  run();
}
window.onload = run;
</script>
</body></html>'''


if __name__ == '__main__':
    port = int(os.environ.get('CAAAS_PORT', 5005))
    app.run(host='0.0.0.0', port=port, threaded=True)
