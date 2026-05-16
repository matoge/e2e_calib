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


def _project(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pts_cam (N, 3) → uv (N, 2) and depth (N,) ; assumes z>0 already filtered."""
    z = pts_cam[:, 2]
    u = (pts_cam[:, 0] / z) * K[0, 0] + K[0, 2]
    v = (pts_cam[:, 1] / z) * K[1, 1] + K[1, 2]
    return np.stack([u, v], axis=-1), z


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
    """Single-frame calibration. multipart: image, pts, K, [T_cam_lidar], [exp]."""
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
    model = _get_model(exp)

    # LiDAR → cam frame, drop behind-camera, project
    Pl = np.concatenate([pts4[:, :3], np.ones((pts4.shape[0], 1))], axis=1)
    Pc = (T_cl @ Pl.T).T[:, :3]
    front = Pc[:, 2] > 0.5
    Pc = Pc[front]
    intens = pts4[front, 3:4]
    uv, z = _project(Pc, K)
    H, W = img.shape[:2]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[in_img]; z = z[in_img]; intens = intens[in_img]

    # Build a single tile centered on image center (TODO: proper tile sweep
    # + BA aggregate for sequence_calibrate; this single-frame variant
    # treats the whole image as one tile).
    # Resize image to 128 (model input), uvd in local px.
    S = 128
    uv_loc = uv * (S / max(H, W))
    d_norm = (z / 100.0).astype(np.float32)
    pts_uvd = np.concatenate([uv_loc.astype(np.float32),
                               d_norm[:, None],
                               np.zeros_like(d_norm)[:, None],  # is_obj
                               (intens / 128.0).clip(0, 1).astype(np.float32),
                               ], axis=1)
    # Subsample to grid (model expects ~256 query points)
    if pts_uvd.shape[0] > 256:
        idx = np.random.RandomState(0).permutation(pts_uvd.shape[0])[:256]
        pts_uvd = pts_uvd[idx]

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_t = torch.nn.functional.interpolate(img_t, size=(S, S),
                                              mode='bilinear', align_corners=False)
    img_t = img_t.to(_DEVICE)
    pts_t = torch.from_numpy(pts_uvd).unsqueeze(0).to(_DEVICE)
    pad = torch.zeros(1, pts_t.shape[1], dtype=torch.bool, device=_DEVICE)
    vfp = torch.tensor([[float(K[0, 0]) / S]], device=_DEVICE)

    use_intensity = bool(getattr(model, 'use_intensity', False))
    if use_intensity:
        point_in = torch.cat([pts_t[..., :3], pts_t[..., 4:5]], dim=-1)
    else:
        point_in = pts_t[..., :3]

    # bucket_uvd: (1, G^2, K, C); 0-fill — single-frame demo, frustum encoder
    # is asked to look at the same point cloud as Q.
    G, Kpc, C = 16, 8, 4 if use_intensity else 3
    bucket = torch.zeros(1, G * G, Kpc, C, device=_DEVICE)
    bucket_valid = torch.zeros(1, G * G, Kpc, dtype=torch.bool, device=_DEVICE)

    with torch.no_grad(), torch.autocast(device_type=_DEVICE, dtype=torch.float16):
        params = model(img_t, point_in, key_padding_mask=pad, vfp=vfp,
                       bucket_uvd=bucket, bucket_valid=bucket_valid)[0].float().cpu().numpy()
    # params: (N, 5) [du, dv, log_sx, log_sy, rho]
    pred_uv = uv_loc + params[:, :2]
    log_sx, log_sy, rho = params[:, 2], params[:, 3], np.tanh(params[:, 4])
    sx, sy = np.exp(log_sx), np.exp(log_sy)

    return jsonify(
        ok=True,
        n_input_pts=int(pts4.shape[0]),
        n_used=int(pts_uvd.shape[0]),
        pred_uv_local=pred_uv.tolist(),
        sigma_uv=np.stack([sx, sy], axis=-1).tolist(),
        rho=rho.tolist(),
        elapsed_ms=int((time.time() - t0) * 1000),
        exp=exp,
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

    ba_cfg = dict(tile_size=512, model_input_size=c['img_size'],
                  max_pts_per_tile=256, min_pts_per_tile=8,
                  tile_stride=384)
    res = infer_tiles(model, img, uv, z, K, ba_cfg, torch.device(_DEVICE),
                      intensity=intensity)
    if res is None:
        return jsonify(error='no tiles passed min_pts'), 500
    uv_full, par, z_full = res
    delta_6 = solve_dofs(uv_full, par, z_full, K,
                          _DOF_PRESETS['6dof_ext'], damping=1e-3)
    cov = solve_dofs._last_cov
    sigma = np.sqrt(np.diag(cov))

    # Encode the (perturbed) tile image for the demo UI.
    buf = _io.BytesIO()
    _PIL.fromarray(img).save(buf, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    return jsonify(
        ok=True,
        cache=cache_key,
        idx=idx,
        seed=seed,
        n_pool=int(uv_full.shape[0]),
        dof_names=_DOF_PRESETS['6dof_ext'],
        delta_pred=delta_6.tolist(),
        sigma_pred=sigma.tolist(),
        cov=cov.tolist(),
        delta_gt=(pert_vec.tolist() if pert_vec is not None else None),
        img_b64=img_b64,
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
img{max-width:512px;border:2px solid #555;border-radius:6px}
small{color:#888}
</style></head><body>
<h1>CaaaS — Calibration As A Service <small>(Triple-A)</small></h1>
<div class="row">
  <div>
    <p>cache:
      <select id="cache"><option>kamikado</option><option>woven</option><option>waymo</option></select>
      idx <input id="idx" type="number" value="0" style="width:60px">
      seed <input id="seed" type="number" value="42" style="width:60px">
      <button onclick="run()">Random shift &amp; predict</button>
      <button onclick="random_seed()">Re-roll seed</button>
    </p>
    <img id="img" alt="">
    <p><small id="meta"></small></p>
  </div>
  <div>
    <h3>6-DoF δ (predicted vs GT)</h3>
    <table id="tbl">
      <tr><th>DoF</th><th>δ_pred</th><th>σ_pred</th><th>δ_GT</th><th>err</th></tr>
    </table>
  </div>
</div>
<script>
async function run() {
  const c = document.getElementById('cache').value;
  const i = document.getElementById('idx').value;
  const s = document.getElementById('seed').value;
  const base = location.pathname.replace(/\/demo$/, '');
  const r = await fetch(`${base}/api/calibrate_demo?cache=${c}&idx=${i}&seed=${s}`);
  const d = await r.json();
  if (!d.ok) { document.getElementById('meta').innerText = 'ERROR: ' + d.error; return; }
  document.getElementById('img').src = 'data:image/jpeg;base64,' + d.img_b64;
  document.getElementById('meta').innerText =
    `cache=${d.cache} idx=${d.idx} seed=${d.seed} pool_N=${d.n_pool}`;
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
