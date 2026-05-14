"""Browser slider demo — every input/output goes through the SAME
`PandaSetCalibDatasetFull.build_window` that `__getitem__` uses at training
time, so projection / bucket / rep-selection / image-crop logic stays in one
place. At training the perturbation is sampled randomly; here it comes from
the UI sliders.

  GET  /            HTML demo
  GET  /sample      JSON {png_b64, metrics}
  GET  /sample.png  Just the PNG (for direct linking)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.pandaset_full import PandaSetCalibDatasetFull
from models.model_depth import CalibNetDepth
from scripts.ba.ba_multicam_corr import resolve_dof_list, solve_dofs, delta_to_dict


CACHE = Path(os.environ.get(
    'DEMO_CACHE', '/mnt/nvme6t/e2e_calib_cache/tss4_v3_tiled'))
CKPT = Path(os.environ.get(
    'DEMO_CKPT', 'experiments/tss4_20260514_intensity_4ch_100ep_framesplit/best_model.pt'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CFG = dict(
    img_size=128, in_channels=3, n_layers=4,
    use_convnext=True, use_frustum=True, deform_mode='sl',
    use_intensity=True,
)
# Match the training-time dataset config; only oversample dropped to 1 (we
# never iterate, just call ds[idx]). `pose_frame='orig'` matches the TSS4
# 100ep cfg (no vcam reparam at training).
DS_KW = dict(
    img_size=128, min_crop_px=128, max_crop_px=384,
    max_rot_deg=1.5, max_offset_m=0.6,
    oversample=1, pose_frame='orig',
)
# Crop chosen for the demo: centered, 384 px (= training max_crop_px) so the
# model gets an in-distribution input.
DEMO_CS = 384

MODEL: CalibNetDepth | None = None
DS: PandaSetCalibDatasetFull | None = None
KEYS: list[str] = []
RENDER_LOCK = threading.Lock()


def _ensure_loaded():
    global MODEL, DS, KEYS
    if MODEL is None:
        m = CalibNetDepth(**MODEL_CFG).to(DEVICE).eval()
        sd = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd, strict=False)
        MODEL = m
    if DS is None:
        # TSS4 frame-split puts ALL val frames in one short recording segment;
        # use 'train' so the seed→scene mapping cycles across multiple scenes.
        DS = PandaSetCalibDatasetFull(str(CACHE), split='train', **DS_KW)
    if not KEYS:
        # Use the dataset's own fnames so seed indexing matches val-split sizing.
        # Deterministic shuffle ensures consecutive seeds cover different scenes.
        import random as _r
        keys = list(DS.fnames)
        _r.Random(20260514).shuffle(keys)
        KEYS.extend(keys)


def _build_sample(seed: int, ox: float, oy: float, tx: float, ty: float):
    """Use the dataset's exact build_window helper. We bypass __getitem__'s
    random pivot+crop+perturb sampling and supply the user's perturbation +
    a centered DEMO_CS crop instead. Skips empty / tiny-pts tiles so a single
    seed always lands on something with usable LiDAR."""
    # Retry up to 30 sequential seeds — empty tiles cluster at fisheye edges.
    for offset in range(30):
        fname = KEYS[(seed + offset) % len(KEYS)]
        idx = DS.fnames.index(fname)
        inst = DS._load_inst(idx)
        n_pts = (inst['pts'].shape[0] if hasattr(inst['pts'], 'shape')
                  else len(inst['pts']))
        if n_pts >= DS.min_pts:
            break
    else:
        return None

    # Inst contents — match __getitem__'s naming.
    if 'jpg_bytes' in inst:
        IH, IW = int(inst['IH']), int(inst['IW'])
        img_full_tensor = None
    else:
        IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
        img_full_tensor = inst['img']
    K = inst['K_full'].numpy()
    pts = inst['pts'].numpy()
    cp = inst['cam_pos'].numpy()
    R_gt = inst['R_gt'].numpy()
    intensity = inst['intensity'].numpy() if hasattr(inst['intensity'], 'numpy') \
                 else np.asarray(inst['intensity'])
    if 'uv_full' in inst and 'z_cam' in inst:
        uv_full = inst['uv_full'].numpy()
    else:
        T_gt = inst['T_gt'].numpy()
        homo = np.column_stack([pts, np.ones(len(pts))])
        pts_cam_gt = (T_gt @ homo.T)[:3].T
        uv_full = ((K @ pts_cam_gt.T)[:2] / np.maximum(pts_cam_gt[:, 2:].T, 1e-6)).T.astype(np.float32)
    is_obj_full = (inst['is_obj'].numpy().astype(bool) if 'is_obj' in inst
                    else np.zeros(len(pts), dtype=bool))
    tile_u0 = int(inst.get('tile_u0', 0))
    tile_v0 = int(inst.get('tile_v0', 0))
    # uv_full from cache is in PARENT coords; subtract tile origin for tile-local.
    if tile_u0 or tile_v0:
        uv_full = uv_full - np.array([tile_u0, tile_v0], dtype=np.float32)

    # Centered crop, clamped to tile.
    cs = min(DEMO_CS, IW, IH)
    u0 = max(0, (IW - cs) // 2)
    v0 = max(0, (IH - cs) // 2)

    # User perturbation → R_off, cp_off, K_pert, pert_vec (orig frame).
    ypr = np.array([0.0, float(oy), float(ox)], dtype=np.float64)  # [yaw, pitch, roll]
    R_pert = Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    R_off = R_gt @ R_pert
    t_delta_world = R_gt @ np.array([float(tx), float(ty), 0.0], dtype=np.float64)
    cp_off = cp + t_delta_world
    K_pert = K.copy()                                    # no fx/fy perturbation
    pert_vec = np.array([t_delta_world[0], t_delta_world[1], t_delta_world[2],
                          ypr[0], ypr[1], ypr[2], 0.0, 0.0], dtype=np.float32)

    # Use ALL valid candidate pts — same as training's wide-pad filter would.
    cand_idx = np.arange(len(pts), dtype=np.int64)
    pts_c = pts[cand_idx]
    intens_c = intensity[cand_idx]
    uv_gt_c = uv_full[cand_idx]

    built = DS.build_window(
        inst, pts_c, intens_c, uv_gt_c, cand_idx, is_obj_full,
        u0, v0, cs, K, R_off, cp_off, K_pert, cp, pert_vec,
        tile_u0, tile_v0, img_full_tensor, IW, IH,
    )
    if built is None:
        return None
    img_t, true_uvd_t, dist_uvd_t, vfp_t, buc_t, buc_v_t, pert_t = built
    return dict(
        inst=inst, IH=IH, IW=IW, K=K, tile_u0=tile_u0, tile_v0=tile_v0,
        u0=u0, v0=v0, cs=cs, scale=MODEL_CFG['img_size'] / cs,
        img=img_t, true_uvd=true_uvd_t, dist_uvd=dist_uvd_t,
        vfp=vfp_t, bucket_uvd=buc_t, bucket_valid=buc_v_t,
        pert_vec=pert_t,
    )


def _run_model(pkt: dict) -> np.ndarray:
    img = pkt['img'].unsqueeze(0).to(DEVICE).float().div_(255.0)
    dist = pkt['dist_uvd'].unsqueeze(0).to(DEVICE)
    if MODEL_CFG['use_intensity']:
        dist_in = torch.stack(
            [dist[..., 0], dist[..., 1], dist[..., 2], dist[..., 4]], dim=-1)
    else:
        dist_in = dist[..., :3]
    pad = torch.zeros(1, dist.shape[1], dtype=torch.bool, device=DEVICE)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = MODEL(img, dist_in, key_padding_mask=pad,
                    vfp=pkt['vfp'].view(1).to(DEVICE),
                    bucket_uvd=pkt['bucket_uvd'].unsqueeze(0).to(DEVICE),
                    bucket_valid=pkt['bucket_valid'].unsqueeze(0).to(DEVICE))
    params = (out[0] if isinstance(out, tuple) else out)[0].float().cpu().numpy()
    return params


def _render(seed: int, ox: float, oy: float, tx: float, ty: float):
    """Return (png_bytes, metrics_dict)."""
    _ensure_loaded()
    with RENDER_LOCK:
        pkt = _build_sample(seed, ox, oy, tx, ty)
        if pkt is None:
            buf = io.BytesIO()
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.text(0.5, 0.5, 'no in-crop points — try smaller perturbation',
                    ha='center', va='center', fontsize=14, color='red')
            ax.axis('off'); fig.patch.set_facecolor('#222')
            plt.savefig(buf, format='png', facecolor='#222'); plt.close(fig)
            return buf.getvalue(), dict(seed=seed, error='no in-crop points')

        # Model forward — Δuv predictions per representative pt.
        import time
        _t = time.time()
        params = _run_model(pkt)
        elapsed_ms = (time.time() - _t) * 1e3

        # Tile-pixel coords for the rendering canvas.
        scale = pkt['scale']; u0, v0 = pkt['u0'], pkt['v0']
        dist_uv_local = pkt['dist_uvd'][:, :2].numpy()
        true_uv_local = pkt['true_uvd'][:, :2].numpy()
        pred_uv_local = dist_uv_local + params[:, :2]
        # local-128 → tile-px = local / scale + (u0, v0)
        to_tile = lambda u: u / scale + np.array([u0, v0], dtype=np.float32)
        dist_uv_tile = to_tile(dist_uv_local)
        true_uv_tile = to_tile(true_uv_local)
        pred_uv_tile = to_tile(pred_uv_local)

        err_pre  = np.linalg.norm(dist_uv_tile - true_uv_tile, axis=1)
        err_post = np.linalg.norm(pred_uv_tile - true_uv_tile, axis=1)

        # BA solve on per-pt predictions in MODEL-input frame.
        K = pkt['K'].astype(np.float32)
        K_loc = K.copy()
        K_loc[0, 0] *= scale; K_loc[1, 1] *= scale
        K_loc[0, 2] = (K[0, 2] - pkt['tile_u0'] - u0) * scale
        K_loc[1, 2] = (K[1, 2] - pkt['tile_v0'] - v0) * scale
        par_arr = np.column_stack([
            params[:, 0], params[:, 1],
            np.exp(params[:, 2]), np.exp(params[:, 3]),
            np.tanh(params[:, 4]) * 0.99,
        ]).astype(np.float32)
        z_arr = (pkt['dist_uvd'][:, 2].numpy() * 100.0).astype(np.float32)
        ba_cfg = dict(dof=['omega_x', 'omega_y', 'tx', 'ty'], damping=1e-3)
        dof_names = resolve_dof_list(ba_cfg)
        try:
            delta = solve_dofs(dist_uv_local, par_arr, z_arr,
                                K_loc, dof_names, damping=ba_cfg['damping'])
            dof_vals = delta_to_dict(delta, dof_names)
            ba_corr = dict(
                omega_x_deg=float(dof_vals.get('omega_x', 0.0)),
                omega_y_deg=float(dof_vals.get('omega_y', 0.0)),
                tx_m=float(dof_vals.get('tx', 0.0)),
                ty_m=float(dof_vals.get('ty', 0.0)),
            )
        except Exception as e:
            ba_corr = dict(error=str(e))

        inst = pkt['inst']
        metrics = dict(
            seed=int(seed),
            scene=inst.get('scene'),
            frame=int(inst.get('frame', -1)),
            n_pts=int(len(err_post)),
            user_perturbation=dict(
                omega_x_deg=float(ox), omega_y_deg=float(oy),
                tx_m=float(tx), ty_m=float(ty),
            ),
            err_pre_px=dict(mean=float(err_pre.mean()),
                             median=float(np.median(err_pre)),
                             p95=float(np.percentile(err_pre, 95))),
            err_post_px=dict(mean=float(err_post.mean()),
                              median=float(np.median(err_post)),
                              p95=float(np.percentile(err_post, 95))),
            ba_correction=ba_corr,
            elapsed_ms=round(float(elapsed_ms), 1),
            crop=dict(u0=int(u0), v0=int(v0), cs=int(pkt['cs'])),
        )

        # Render full tile + overlays.
        img = np.asarray(Image.open(io.BytesIO(inst['jpg_bytes'] if 'jpg_bytes' in inst else inst['jpg'])).convert('RGB'))
        IH, IW = pkt['IH'], pkt['IW']
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=96)
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((u0, v0), pkt['cs'], pkt['cs'],
                                     fill=False, ec='cyan', lw=1.5))
        ax.scatter(true_uv_tile[:, 0], true_uv_tile[:, 1],
                   c='lime', s=18, marker='x', linewidths=1.0,
                   label=f'GT ({len(true_uv_tile)})', zorder=3)
        ax.scatter(dist_uv_tile[:, 0], dist_uv_tile[:, 1],
                   c='red', s=18, marker='x', linewidths=1.0,
                   label='perturbed', zorder=3)
        ax.quiver(dist_uv_tile[:, 0], dist_uv_tile[:, 1],
                   pred_uv_tile[:, 0] - dist_uv_tile[:, 0],
                   pred_uv_tile[:, 1] - dist_uv_tile[:, 1],
                   angles='xy', scale_units='xy', scale=1,
                   color='deepskyblue', width=0.003, headwidth=3.5,
                   headlength=4, alpha=0.8, zorder=4)
        ax.scatter(pred_uv_tile[:, 0], pred_uv_tile[:, 1],
                   c='deepskyblue', s=22, marker='o',
                   edgecolors='white', linewidths=0.4,
                   label='model pred', zorder=5)
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
        ax.set_title(f'{inst.get("scene")}  f={inst.get("frame", -1)}  '
                     f'crop={pkt["cs"]}px  model={elapsed_ms:.0f} ms',
                     fontsize=9, color='white')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.6)
        fig.patch.set_facecolor('#222')
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#222', dpi=96)
        plt.close(fig)
        return buf.getvalue(), metrics


# ─── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(title='e2e_calib slider demo')


@app.on_event('startup')
def _startup():
    _ensure_loaded()


@app.get('/sample.png')
def sample_png(seed: int = Query(0), ox: float = Query(0.0), oy: float = Query(0.0),
               tx: float = Query(0.0), ty: float = Query(0.0)):
    png, _ = _render(seed, ox, oy, tx, ty)
    return Response(content=png, media_type='image/png')


@app.get('/sample')
def sample(seed: int = Query(0), ox: float = Query(0.0), oy: float = Query(0.0),
            tx: float = Query(0.0), ty: float = Query(0.0)):
    png, metrics = _render(seed, ox, oy, tx, ty)
    return dict(png_b64=base64.b64encode(png).decode(), metrics=metrics)


DEMO_HTML = """<!doctype html>
<html><head><meta charset='utf-8'><title>e2e_calib slider demo</title>
<style>
  body { font-family: monospace; margin: 16px; background:#111; color:#ddd; }
  .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  img { background:#000; max-width: 640px; border:1px solid #444; }
  .panel { min-width: 380px; }
  label { display:block; margin: 8px 0 2px; }
  input[type=range] { width: 320px; }
  .val { color:#7af; }
  button { background:#36c; color:#fff; border:0; padding:8px 16px;
           cursor:pointer; margin-right: 8px; }
  button:hover { background:#48d; }
  .legend span { display:inline-block; width:12px; height:12px; vertical-align:middle;
                 margin-right:6px; border:1px solid #555; }
  .metrics { font-size: 16px; line-height: 1.6; }
  .metrics h3 { margin: 12px 0 4px; color: #8df; font-size: 18px;
                border-bottom: 1px solid #345; padding-bottom: 2px; }
  .metric-row { display: grid; grid-template-columns: 110px 1fr 1fr 1fr; gap: 8px;
                font-variant-numeric: tabular-nums; padding: 3px 0; }
  .metric-row.head { color:#8af; font-weight:bold; border-bottom:1px solid #345; }
  .num-pre  { color:#fc6; }
  .num-post { color:#6df; font-weight: bold; }
  .ba-row { display: grid; grid-template-columns: 80px 1fr 1fr 1fr; gap: 8px;
            font-variant-numeric: tabular-nums; font-size: 16px; padding: 3px 0; }
  .ba-row.head { color:#8af; font-weight:bold; border-bottom:1px solid #345; }
  .ba-pred { color:#6df; font-weight: bold; }
  .ba-gt   { color:#fc6; }
  .ba-diff { color:#aaa; font-size: 13px; }
  .scene { font-size: 12px; color:#888; margin-top: 8px; }
</style></head>
<body>
<h2>e2e_calib — sliders perturb, model predicts the correction (shared dataset code)</h2>
<div class="row">
  <img id="vis" src="" alt="loading…">
  <div class="panel">
    <div class="legend">
      <span style="background:#0f0"></span>GT&nbsp;
      <span style="background:#f44"></span>perturbed&nbsp;
      <span style="background:#0bf"></span>model pred
    </div>
    <label>ωx (roll about cam X): <span class="val" id="oxv">0.00</span> deg</label>
    <input type="range" id="ox" min="-2" max="2" step="0.05" value="0">
    <label>ωy (pitch about cam Y): <span class="val" id="oyv">0.00</span> deg</label>
    <input type="range" id="oy" min="-2" max="2" step="0.05" value="0">
    <label>tx: <span class="val" id="txv">0.00</span> m</label>
    <input type="range" id="tx" min="-1" max="1" step="0.01" value="0">
    <label>ty: <span class="val" id="tyv">0.00</span> m</label>
    <input type="range" id="ty" min="-1" max="1" step="0.01" value="0">
    <p>
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <button id="reset">Reset sliders</button>
    </p>

    <div class="metrics">
      <h3>Per-point projection error vs GT</h3>
      <div class="metric-row head"><div></div><div>mean</div><div>median</div><div>p95</div></div>
      <div class="metric-row">
        <div>before fix</div>
        <div class="num-pre"  id="pre_mean">–</div>
        <div class="num-pre"  id="pre_med">–</div>
        <div class="num-pre"  id="pre_p95">–</div>
      </div>
      <div class="metric-row">
        <div>after model</div>
        <div class="num-post" id="post_mean">–</div>
        <div class="num-post" id="post_med">–</div>
        <div class="num-post" id="post_p95">–</div>
      </div>

      <h3>BA 4-DoF correction (predicted vs your input)</h3>
      <div class="ba-row head"><div></div><div>BA pred</div><div>your input</div><div>diff</div></div>
      <div class="ba-row">
        <div>ωx</div>
        <div class="ba-pred" id="ba_ox">–</div>
        <div class="ba-gt"   id="gt_ox">–</div>
        <div class="ba-diff" id="d_ox">–</div>
      </div>
      <div class="ba-row">
        <div>ωy</div>
        <div class="ba-pred" id="ba_oy">–</div>
        <div class="ba-gt"   id="gt_oy">–</div>
        <div class="ba-diff" id="d_oy">–</div>
      </div>
      <div class="ba-row">
        <div>tx</div>
        <div class="ba-pred" id="ba_tx">–</div>
        <div class="ba-gt"   id="gt_tx">–</div>
        <div class="ba-diff" id="d_tx">–</div>
      </div>
      <div class="ba-row">
        <div>ty</div>
        <div class="ba-pred" id="ba_ty">–</div>
        <div class="ba-gt"   id="gt_ty">–</div>
        <div class="ba-diff" id="d_ty">–</div>
      </div>
      <div class="scene" id="scene">–</div>
    </div>
  </div>
</div>
<script>
let SEED = 0;
let busy = false, pending = false;
const ids = ['ox','oy','tx','ty'];
const imgEl = document.getElementById('vis');
const $ = id => document.getElementById(id);
function vals() {
  return Object.fromEntries(ids.map(id => [id, parseFloat($(id).value)]));
}
function fmtNum(x, d) { return (x==null) ? '–' : Number(x).toFixed(d); }
function setMetrics(m) {
  if (!m) return;
  const pre = m.err_pre_px || {}, post = m.err_post_px || {};
  $('pre_mean').textContent = fmtNum(pre.mean, 2) + ' px';
  $('pre_med').textContent  = fmtNum(pre.median, 2) + ' px';
  $('pre_p95').textContent  = fmtNum(pre.p95, 2) + ' px';
  $('post_mean').textContent = fmtNum(post.mean, 2) + ' px';
  $('post_med').textContent  = fmtNum(post.median, 2) + ' px';
  $('post_p95').textContent  = fmtNum(post.p95, 2) + ' px';
  const ba = m.ba_correction || {}, gt = m.user_perturbation || {};
  function row(k, suffix, unit, decimals) {
    const p = ba[k], g = gt[k];
    $('ba_' + suffix).textContent = (p==null) ? '–' : (p>=0?'+':'') + p.toFixed(decimals) + ' ' + unit;
    $('gt_' + suffix).textContent = (g==null) ? '–' : (g>=0?'+':'') + g.toFixed(decimals) + ' ' + unit;
    if (p!=null && g!=null) {
      // BA returns "amount of perturbation" — same sign as user input when
      // model is correct. So |BA − user| is the recovery error.
      const d = p - g;
      $('d_' + suffix).textContent = '|err|=' + Math.abs(d).toFixed(decimals);
    } else $('d_' + suffix).textContent = '–';
  }
  row('omega_x_deg', 'ox', '°', 3);
  row('omega_y_deg', 'oy', '°', 3);
  row('tx_m',        'tx', 'm', 3);
  row('ty_m',        'ty', 'm', 3);
  $('scene').textContent =
    `seed=${m.seed}  scene=${m.scene}  frame=${m.frame}  ` +
    `crop=${m.crop?.cs}px  n_pts=${m.n_pts}  inference=${m.elapsed_ms}ms`;
}
async function render() {
  if (busy) { pending = true; return; }
  busy = true; pending = false;
  const v = vals();
  const qs = `seed=${SEED}&ox=${v.ox}&oy=${v.oy}&tx=${v.tx}&ty=${v.ty}`;
  try {
    const r = await fetch('/sample?' + qs);
    if (!r.ok) { busy = false; if (pending) render(); return; }
    const o = await r.json();
    imgEl.src = 'data:image/png;base64,' + o.png_b64;
    setMetrics(o.metrics);
  } catch (e) { console.error(e); }
  busy = false; if (pending) render();
}
ids.forEach(id => {
  const el = $(id);
  el.addEventListener('input', () => {
    $(id+'v').textContent = parseFloat(el.value).toFixed(id.startsWith('o') ? 2 : 3);
    render();
  });
});
$('next').addEventListener('click', () => { SEED++; render(); });
$('prev').addEventListener('click', () => { SEED = Math.max(0, SEED-1); render(); });
$('reset').addEventListener('click', () => {
  ids.forEach(id => { $(id).value = 0; $(id+'v').textContent = '0.00'; });
  render();
});
render();
</script>
</body></html>
"""


@app.get('/', response_class=HTMLResponse)
def index():
    return DEMO_HTML


def main():
    import uvicorn
    uvicorn.run('scripts.serving.demo_app:app',
                host=os.environ.get('DEMO_HOST', '0.0.0.0'),
                port=int(os.environ.get('DEMO_PORT', 8765)),
                workers=1, log_level='info')


if __name__ == '__main__':
    main()
