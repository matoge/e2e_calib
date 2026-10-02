"""Generic V3 cache LiDAR→image projection viewer. Port 5007.

Shows the pre-baked `uv_full` overlaid on the cache's `jpg_bytes`, grouped
by scene / frame. Works on any build_*_v3 cache (kamikado, woven, tss4,
pandaset) since they all share the same inst schema.

Run:
    DEMO_CACHE=/raid/.../cache_v5/woven_v3_cal_ok \\
    python scripts/webapp/cache_viz.py
    → http://localhost:5007
"""
from __future__ import annotations
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from datasets.pandaset_full import PandaSetCalibDatasetFull, decode_inst_img

CACHE = Path(os.environ.get(
    'DEMO_CACHE',
    '/raid/home/hfunaya/cache_v5/woven_v3_cal_ok'))

app = Flask(__name__)
_DS_TRAIN = None
_DS_VAL   = None
_SCENE_IDX: dict[str, list[tuple[int, str]]] = {}


def _load():
    """Build (scene -> list[(idx, split)]) index once."""
    global _DS_TRAIN, _DS_VAL, _SCENE_IDX
    if _DS_TRAIN is not None: return
    _DS_TRAIN = PandaSetCalibDatasetFull(
        str(CACHE), split='train', img_size=256,
        min_crop_px=512, max_crop_px=512,
        max_rot_deg=0.0, max_offset_m=0.0,
        grid_n=16, oversample=1, crop_grid=False, share_pert=False)
    _DS_VAL   = PandaSetCalibDatasetFull(
        str(CACHE), split='val', img_size=256,
        min_crop_px=512, max_crop_px=512,
        max_rot_deg=0.0, max_offset_m=0.0,
        grid_n=16, oversample=1, crop_grid=False, share_pert=False)
    for ds, split in ((_DS_TRAIN, 'train'), (_DS_VAL, 'val')):
        for i, fn in enumerate(ds.fnames):
            # fname → scene via inst load, but that's slow. Use fname prefix instead.
            # build_woven_sequence_v3 uses scene_short md5-prefix; just group by
            # the chars BEFORE the frame number for a reasonable approximation.
            key = fn.rsplit('_t', 1)[0]           # strip tile suffix
            _SCENE_IDX.setdefault(key, []).append((i, split))
    print(f'[cache-viz] cache={CACHE}  train={len(_DS_TRAIN)}  val={len(_DS_VAL)}  '
          f'groups={len(_SCENE_IDX)}', flush=True)


@app.route('/api/scenes')
def api_scenes():
    _load()
    out = sorted(_SCENE_IDX.keys())
    return jsonify(scenes=out, cache=str(CACHE),
                    n_train=len(_DS_TRAIN), n_val=len(_DS_VAL))


@app.route('/api/scene_frames')
def api_scene_frames():
    _load()
    scene = request.args['scene']
    items = _SCENE_IDX.get(scene, [])
    return jsonify(scene=scene, n=len(items), items=items[:200])


@app.route('/api/overlay.jpg')
def api_overlay():
    _load()
    idx = int(request.args['idx'])
    split = request.args.get('split', 'val')
    max_d = float(request.args.get('max_d', 60.0))
    pt_sz = int(request.args.get('pt_sz', 2))
    show_box = int(request.args.get('box', 1))
    ds = _DS_VAL if split == 'val' else _DS_TRAIN
    inst = ds._load_inst(idx)
    img = decode_inst_img(inst).permute(1, 2, 0).numpy().astype(np.uint8)   # H,W,3 RGB
    # uv_full is in PARENT image coords; shift into tile-local coords.
    tile_u0 = int(inst.get('tile_u0', 0))
    tile_v0 = int(inst.get('tile_v0', 0))
    uv  = inst['uv_full'].numpy() - np.array([tile_u0, tile_v0], dtype=np.float32)
    z   = inst['z_cam'].numpy()
    IH, IW = img.shape[:2]
    is_obj = inst.get('is_obj', None)
    is_obj = is_obj.numpy().astype(bool) if is_obj is not None else np.zeros(len(uv), dtype=bool)
    m = ((z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < IW)
          & (uv[:, 1] >= 0) & (uv[:, 1] < IH))
    uv_m = uv[m].astype(np.int32)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if m.sum() > 0:
        rng  = np.linalg.norm(inst['pts'].numpy()[m] if 'pts' in inst else np.ones((m.sum(), 3)), axis=1)
        z8 = (np.clip(rng / max_d, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(z8, cv2.COLORMAP_TURBO).reshape(-1, 3)  # BGR
        for (u, v), c in zip(uv_m, colors):
            cv2.circle(bgr, (int(u), int(v)), pt_sz, c.tolist(), -1)
    if show_box and 'cuboids' in inst:
        for cub in inst['cuboids']:
            # Draw cuboid in cam-frame, projected via K_full (and KB4 if fisheye).
            # Simple AABB only — read-only sanity, not full annotation render.
            pos = np.asarray(cub.get('pos', [0,0,0]), dtype=np.float32)
            dims = np.asarray(cub.get('dims', [0,0,0]), dtype=np.float32)
            # Skip — cuboid viz has 2 projection modes (kb vs pinhole); keep
            # scope tight and just annotate presence.
    cv2.putText(bgr, f'{split}#{idx}  scene={inst.get("scene","?")[:60]}  '
                      f'frame={int(inst.get("frame",-1))}  N={len(uv_m)}',
                 (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg')


_INDEX = """<!DOCTYPE html>
<meta charset=utf-8>
<title>cache viz</title>
<style>
    body {font-family:sans-serif;margin:12px;background:#111;color:#ddd}
    select,input,button {font-size:14px;background:#222;color:#ddd;border:1px solid #555;padding:4px}
    .row {margin:6px 0}
    img {max-width:100%;height:auto;border:1px solid #555}
</style>
<h2>V3 cache LiDAR→image overlay</h2>
<div class=row>
    <label>scene group:</label>
    <select id=scene></select>
    <label>sample:</label>
    <select id=idx></select>
    <label>max_d (m):</label><input id=max_d type=number value=60 step=1 style="width:60px">
    <label>pt size:</label><input id=pt_sz type=number value=2 step=1 style="width:40px">
    <button onclick="nextSample()">next</button>
</div>
<div class=row id=stats></div>
<img id=overlay>
<script>
let items = [];
async function fetchScenes() {
    const r = await fetch('/api/scenes');
    const d = await r.json();
    document.getElementById('stats').textContent =
        `cache ${d.cache}   train=${d.n_train} val=${d.n_val} groups=${d.scenes.length}`;
    const sel = document.getElementById('scene');
    sel.innerHTML = d.scenes.map(s => `<option>${s}</option>`).join('');
    sel.onchange = loadScene;
    await loadScene();
}
async function loadScene() {
    const scene = document.getElementById('scene').value;
    const r = await fetch(`/api/scene_frames?scene=${encodeURIComponent(scene)}`);
    const d = await r.json();
    items = d.items;
    const sel = document.getElementById('idx');
    sel.innerHTML = items.map(([i,s],k) => `<option value=${k}>${s}#${i}</option>`).join('');
    sel.onchange = refresh;
    refresh();
}
function nextSample() {
    const sel = document.getElementById('idx');
    sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
    refresh();
}
function refresh() {
    const k = +document.getElementById('idx').value;
    if (!items[k]) return;
    const [idx, split] = items[k];
    const url = `/api/overlay.jpg?idx=${idx}&split=${split}`
        + `&max_d=${document.getElementById('max_d').value}`
        + `&pt_sz=${document.getElementById('pt_sz').value}&t=${Date.now()}`;
    document.getElementById('overlay').src = url;
}
document.getElementById('max_d').onchange = refresh;
document.getElementById('pt_sz').onchange = refresh;
fetchScenes();
</script>
"""


@app.route('/')
def index():
    return Response(_INDEX, mimetype='text/html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5007))
    print(f'[cache-viz] cache: {CACHE}')
    print(f'[cache-viz] open http://localhost:{port}/')
    app.run(host='0.0.0.0', port=port, debug=False)
