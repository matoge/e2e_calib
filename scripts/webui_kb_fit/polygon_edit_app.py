"""Polygon editor for TSS4 dashboard / bonnet / kerare masks.

Loads `_outputs/_dashboard_polygons.json` (list of polygons, each a list
of [x, y] in original-image pixels) and serves a Konva-based canvas
where each vertex is a draggable circle. Click an edge to insert a new
vertex; right-click a vertex to delete it. Save round-trips back to the
same JSON.

Run:
    /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/webui_kb_fit/polygon_edit_app.py
    → http://0.0.0.0:5008/

Background image is one tss4_fcm jpg (raw fisheye, 3840×2160). Polygon
coordinates stay in original-image pixels; the canvas just scales them
for display.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / '_outputs' / '_dashboard_polygons.json'

# First available raw fcm jpg under the local woven canary dir.
BG_CANDIDATES = sorted(Path(
    '/raid/home/hfunaya/woven_canary_local/canary_unilab/test01').glob(
    'sequence=*/tss4_fcm/0000_*.jpg'))
BG_JPG = BG_CANDIDATES[0] if BG_CANDIDATES else None
if BG_JPG is None:
    raise SystemExit('no tss4_fcm jpg under /raid/home/hfunaya/woven_canary_local')


app = Flask(__name__, static_folder=str(ROOT / '_outputs'))


def _default_polys(W: int, H: int) -> list[list[list[float]]]:
    """4 starter polygons: bottom (bonnet), left-top, right-top, top
    (vignetting / kerare). Each is a rectangle the user can refine.
    """
    return [
        # bottom (= bonnet)
        [[0, H * 0.75], [W, H * 0.75], [W, H], [0, H]],
        # left-top
        [[0, 0], [W * 0.15, 0], [W * 0.15, H * 0.25], [0, H * 0.25]],
        # right-top
        [[W * 0.85, 0], [W, 0], [W, H * 0.25], [W * 0.85, H * 0.25]],
        # top centre (= vignetting strip)
        [[W * 0.15, 0], [W * 0.85, 0], [W * 0.85, H * 0.10], [W * 0.15, H * 0.10]],
    ]


def _load_polys() -> list[list[list[float]]]:
    if not JSON_PATH.is_file():
        # Auto-create with 4 starter rectangles (bottom + left-top + right-top + top).
        img = cv2.imread(str(BG_JPG))
        H, W = img.shape[:2]
        return _default_polys(W, H)
    return json.loads(JSON_PATH.read_text()).get('polygons', [])


def _save_polys(polygons: list[list[list[float]]]) -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps({'polygons': polygons}, indent=2))


def _bg_png_b64(max_w: int = 1600) -> tuple[str, int, int]:
    img = cv2.imread(str(BG_JPG))
    H, W = img.shape[:2]
    scale = min(1.0, max_w / W)
    nw = int(round(W * scale))
    nh = int(round(H * scale))
    img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_s, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf.tobytes()).decode('ascii'), W, H


@app.route('/')
def index() -> str:
    bg, W, H = _bg_png_b64()
    polys = _load_polys()
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>polygon mask editor</title>
<style>
  body {{ margin:0; background:#222; color:#eee; font-family:sans-serif }}
  #toolbar {{ padding:6px 10px; background:#333; display:flex; gap:8px;
              align-items:center }}
  #toolbar button {{ background:#444; color:#eee; border:1px solid #666;
                     padding:4px 12px; cursor:pointer }}
  #toolbar button:hover {{ background:#555 }}
  #info {{ margin-left:auto; color:#aaa; font-size:12px }}
  #stage {{ background:#000 }}
  .hint {{ font-size:12px; color:#aaa }}
</style></head>
<body>
<div id="toolbar">
  <button onclick="addPoly()">+ Polygon</button>
  <button onclick="delPoly()">- Polygon</button>
  <button onclick="save()" style="background:#2a6">Save</button>
  <span class="hint">left-drag vertex | dbl-click edge=insert vertex |
        right-click vertex=delete | click polygon=select</span>
  <span id="info"></span>
</div>
<div id="stage"></div>
<script src="https://cdn.jsdelivr.net/npm/konva@9/konva.min.js"></script>
<script>
const IMG_W = {W}, IMG_H = {H};
const VIEW_W = Math.min(IMG_W, 1600);
const SCALE = VIEW_W / IMG_W;
const VIEW_H = Math.round(IMG_H * SCALE);
let POLYS = {json.dumps(polys)};
let activeIdx = POLYS.length ? 0 : -1;
const COLORS = ['#e74c3c','#3498db','#f1c40f','#2ecc71','#9b59b6','#e67e22'];

const stage = new Konva.Stage({{container:'stage', width:VIEW_W, height:VIEW_H}});
const layerBG = new Konva.Layer(); stage.add(layerBG);
const layer   = new Konva.Layer(); stage.add(layer);
const img = new Image();
img.onload = () => {{
  layerBG.add(new Konva.Image({{image:img, x:0,y:0,width:VIEW_W,height:VIEW_H,
                                  listening:false}}));
  layerBG.draw(); redraw();
}};
img.src = 'data:image/jpeg;base64,{bg}';

function toView(pt) {{ return [pt[0]*SCALE, pt[1]*SCALE]; }}
function toOrig(x,y) {{ return [x/SCALE, y/SCALE]; }}
function colorOf(i) {{ return COLORS[i%COLORS.length]; }}

function redraw() {{
  layer.destroyChildren();
  POLYS.forEach((poly, pi) => {{
    const c = colorOf(pi);
    const fill = pi===activeIdx ? c+'66' : c+'22';
    const flat = poly.flatMap(toView);
    const line = new Konva.Line({{
      points: flat, closed:true,
      fill, stroke:c,
      strokeWidth: pi===activeIdx?2:1, opacity:0.95}});
    line.on('mousedown', (e) => {{
      if (e.evt.button===0) {{ activeIdx = pi; e.cancelBubble = true; redraw(); }}
    }});
    line.on('dblclick', (e) => {{
      if (pi!==activeIdx) return;
      const p = stage.getPointerPosition();
      // insert vertex on the closest edge
      let bestI = 0, bestD = 1e9;
      for (let i=0;i<poly.length;i++) {{
        const a = toView(poly[i]);
        const b = toView(poly[(i+1)%poly.length]);
        const d = pointToSeg(p.x,p.y,a[0],a[1],b[0],b[1]);
        if (d<bestD) {{ bestD=d; bestI=i; }}
      }}
      poly.splice(bestI+1, 0, toOrig(p.x, p.y));
      redraw();
    }});
    layer.add(line);
    poly.forEach((pt, vi) => {{
      const v = toView(pt);
      const handle = new Konva.Circle({{
        x:v[0], y:v[1], radius:6, fill:'#fff', stroke:c, strokeWidth:2,
        draggable: pi===activeIdx}});
      handle.on('dragmove', () => {{
        poly[vi] = toOrig(handle.x(), handle.y());
        line.points(poly.flatMap(toView));
      }});
      handle.on('mousedown contextmenu', (e) => {{
        if (e.evt.button===2 && pi===activeIdx) {{
          e.evt.preventDefault();
          if (poly.length>3) {{ poly.splice(vi,1); redraw(); }}
        }}
      }});
      layer.add(handle);
    }});
  }});
  layer.draw();
  document.getElementById('info').innerText =
    `polygons=${{POLYS.length}}  active=${{activeIdx}}  ` +
    `verts=${{POLYS[activeIdx]?.length||0}}`;
}}

function pointToSeg(px,py,ax,ay,bx,by) {{
  const dx=bx-ax, dy=by-ay;
  const t = Math.max(0, Math.min(1, ((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy)));
  const cx = ax+t*dx, cy = ay+t*dy;
  return Math.hypot(px-cx, py-cy);
}}

window.addEventListener('contextmenu', e => e.preventDefault());

function addPoly() {{
  // start in the middle as a small triangle
  const cx = IMG_W/2, cy = IMG_H/2;
  POLYS.push([[cx-100,cy-100],[cx+100,cy-100],[cx,cy+100]]);
  activeIdx = POLYS.length - 1;
  redraw();
}}
function delPoly() {{
  if (activeIdx<0) return;
  POLYS.splice(activeIdx,1);
  activeIdx = Math.min(activeIdx, POLYS.length-1);
  redraw();
}}
function save() {{
  fetch('/api/save', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                       body: JSON.stringify({{polygons: POLYS}})}})
    .then(r => r.json())
    .then(r => alert('saved: ' + JSON.stringify(r)));
}}
</script>
</body></html>
"""


@app.route('/api/save', methods=['POST'])
def api_save():
    polys = request.get_json(silent=True) or {}
    polys = polys.get('polygons', [])
    _save_polys(polys)
    return jsonify({'n': len(polys), 'path': str(JSON_PATH)})


@app.route('/api/reload')
def api_reload():
    return jsonify({'polygons': _load_polys()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=False, threaded=True)
