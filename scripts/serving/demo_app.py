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
from matplotlib.patches import Ellipse
import numpy as np
import torch
from fastapi import APIRouter, FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.pandaset_full import PandaSetCalibDatasetFull
from models.model_depth import CalibNetDepth
from scripts.ba.ba_multicam_corr import resolve_dof_list, solve_dofs, delta_to_dict


CACHE = Path(os.environ.get(
    'DEMO_CACHE', '/cache/kamikado_v3_tiled'))
DEMO_EXP = os.environ.get('DEMO_EXP', 'km_wv_8gpu_200ep_os4')
CKPT = Path(os.environ.get(
    'DEMO_CKPT', f'experiments/{DEMO_EXP}/best_model.pt'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read MODEL_CFG from the same experiments/{exp}/config.py that the trainer
# wrote, so this stays in sync with whichever ckpt is loaded.
def _read_exp_cfg() -> dict:
    cfg_path = Path('experiments') / DEMO_EXP / 'config.py'
    if not cfg_path.is_file():
        return dict(img_size=128, in_channels=3, n_layers=2,
                    use_convnext=False, use_frustum=True, deform_mode='sl',
                    use_intensity=True)
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location('_demo_cfg', cfg_path)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    c = mod.CFG
    return dict(
        img_size=c.get('img_size', 128),
        in_channels=c.get('in_channels', 3),
        n_layers=c.get('n_layers', 2),
        use_convnext=c.get('use_convnext', False),
        use_frustum=c.get('use_frustum', True),
        deform_mode=c.get('deform_mode', 'sl'),
        use_intensity=c.get('use_intensity', True),
    )
MODEL_CFG = _read_exp_cfg()
# Read DS_KW from the same exp config (img_size / crop / pose_frame).
def _read_exp_ds_kw() -> dict:
    cfg_path = Path('experiments') / DEMO_EXP / 'config.py'
    if not cfg_path.is_file():
        return dict(img_size=128, min_crop_px=128, max_crop_px=384,
                    max_rot_deg=1.5, max_offset_m=0.6,
                    oversample=1, pose_frame='orig')
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location('_demo_dskw', cfg_path)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    c = mod.CFG
    return dict(
        img_size=c.get('img_size', 128),
        min_crop_px=c.get('min_crop_px', 128),
        max_crop_px=c.get('max_crop_px', 384),
        max_rot_deg=c.get('max_rot_deg', 1.5),
        max_offset_m=c.get('max_offset_m', 0.6),
        oversample=1,
        pose_frame=c.get('pose_frame', 'orig'),
    )
DS_KW = _read_exp_ds_kw()
# UI crop slider goes up to 384 (extrapolating beyond max_crop_px is fine for
# inspection — model handles it via vfp rescaling, just out-of-distribution).
DEMO_CS = int(os.environ.get('DEMO_CS_MAX', 384))

MODEL: CalibNetDepth | None = None
DS: PandaSetCalibDatasetFull | None = None
KEYS: list[str] = []
RENDER_LOCK = threading.Lock()


def _ensure_loaded():
    global MODEL, DS, KEYS
    if MODEL is None:
        # Use the official loader so MODEL_CFG matches the ckpt's training cfg.
        from scripts.inference.infer_calib import load_calib_model
        MODEL = load_calib_model(DEMO_EXP, device=str(DEVICE)).eval()
    if DS is None:
        # Use the same DS_KW the trainer logged in experiments/{exp}/config.py.
        DS = PandaSetCalibDatasetFull(str(CACHE), split='train', **DS_KW)
    if not KEYS:
        # Use the dataset's own fnames so seed indexing matches val-split sizing.
        # Deterministic shuffle ensures consecutive seeds cover different scenes.
        import random as _r
        keys = list(DS.fnames)
        _r.Random(20260514).shuffle(keys)
        KEYS.extend(keys)


def _build_sample(seed: int, ox: float, oy: float, tx: float, ty: float,
                  cs_override: int | None = None):
    """Use the dataset's exact build_window helper. We bypass __getitem__'s
    random pivot+crop+perturb sampling and supply the user's perturbation +
    a centered DEMO_CS crop instead. Skips empty / tiny-pts tiles so a single
    seed always lands on something with usable LiDAR. cs_override (px) lets
    the UI pick a non-default crop size."""
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

    # Centered crop, clamped to tile. UI may override via cs_override.
    cs_target = int(cs_override) if cs_override else DEMO_CS
    cs = min(cs_target, IW, IH)
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


def _render(seed: int, ox: float, oy: float, tx: float, ty: float,
            cs_override: int | None = None):
    """Return (png_bytes, metrics_dict)."""
    _ensure_loaded()
    with RENDER_LOCK:
        pkt = _build_sample(seed, ox, oy, tx, ty, cs_override=cs_override)
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
        # Same error in MODEL-input (128 px) coords — directly comparable to
        # val_mse printed in train.log (which is in 128-frame).
        err_pre_local  = err_pre  * scale
        err_post_local = err_post * scale

        # Per-pt sigma (model output channels 2,3 are log_sigma_uv; ch 4 is raw
        # rho through tanh*0.99). In MODEL-input pixels — multiply by 1/scale
        # to get tile-pixel sigmas for display + Mahalanobis error.
        sigma_u_local = np.exp(params[:, 2])
        sigma_v_local = np.exp(params[:, 3])
        rho           = np.tanh(params[:, 4]) * 0.99
        sigma_u_tile = sigma_u_local / scale
        sigma_v_tile = sigma_v_local / scale
        # Mahalanobis residual (pred - GT)^T Σ^-1 (pred - GT). In a perfectly
        # calibrated model with proper σ, the mean Mahalanobis² should be ≈ 2.0
        # (Chi² 2-DoF mean). Higher → overconfident; lower → underconfident.
        du = pred_uv_local[:, 0] - true_uv_local[:, 0]
        dv = pred_uv_local[:, 1] - true_uv_local[:, 1]
        det_sigma = sigma_u_local ** 2 * sigma_v_local ** 2 * (1 - rho ** 2)
        det_sigma = np.maximum(det_sigma, 1e-12)
        m2 = ((du / sigma_u_local) ** 2 + (dv / sigma_v_local) ** 2
              - 2 * rho * du * dv / (sigma_u_local * sigma_v_local)) / np.maximum(1 - rho ** 2, 1e-6)

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
            err_pre_local_px=dict(mean=float(err_pre_local.mean()),
                                   median=float(np.median(err_pre_local)),
                                   p95=float(np.percentile(err_pre_local, 95))),
            err_post_local_px=dict(mean=float(err_post_local.mean()),
                                    median=float(np.median(err_post_local)),
                                    p95=float(np.percentile(err_post_local, 95))),
            sigma_px=dict(
                u_mean=float(sigma_u_tile.mean()),
                u_median=float(np.median(sigma_u_tile)),
                v_mean=float(sigma_v_tile.mean()),
                v_median=float(np.median(sigma_v_tile)),
                rho_mean=float(rho.mean()),
            ),
            mahalanobis2=dict(
                mean=float(m2.mean()),
                median=float(np.median(m2)),
                p95=float(np.percentile(m2, 95)),
            ),
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
        # 1-σ covariance ellipses around each predicted point (per-pt model
        # confidence) — eigen-decomp of [[σu², ρσuσv], [ρσuσv, σv²]].
        a = sigma_u_tile ** 2
        c = sigma_v_tile ** 2
        b = rho * sigma_u_tile * sigma_v_tile
        tr = a + c
        disc = np.maximum((tr / 2) ** 2 - (a * c - b * b), 0.0)
        s = np.sqrt(disc)
        lam1 = tr / 2 + s
        lam2 = np.maximum(tr / 2 - s, 0.0)
        # eigenvector angle of λ1; arctan2(λ1 - a, b) is stable when b ≠ 0
        ang = np.where(np.abs(b) > 1e-9,
                        np.degrees(np.arctan2(lam1 - a, b)),
                        np.where(a >= c, 0.0, 90.0))
        for i in range(len(pred_uv_tile)):
            ell = Ellipse(xy=(pred_uv_tile[i, 0], pred_uv_tile[i, 1]),
                          width=2.0 * np.sqrt(lam1[i]),
                          height=2.0 * np.sqrt(lam2[i]),
                          angle=ang[i], fill=False,
                          edgecolor='deepskyblue', linewidth=0.7,
                          alpha=0.55, zorder=4.5)
            ax.add_patch(ell)
        ax.scatter(pred_uv_tile[:, 0], pred_uv_tile[:, 1],
                   c='deepskyblue', s=22, marker='o',
                   edgecolors='white', linewidths=0.4,
                   label='model pred (±1σ)', zorder=5)
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
# Routes register on a router so the app can be served at either root '/' or
# behind a path prefix (e.g. `clearml.budda.site/demo` via Cloudflare tunnel
# without path-strip). Mount under both so direct LAN access at root still
# works and the public CF route at /demo* lands on the same handlers.
app = FastAPI(title='e2e_calib slider demo')
router = APIRouter()


@app.on_event('startup')
def _startup():
    _ensure_loaded()


@router.get('/sample.png')
def sample_png(seed: int = Query(0), ox: float = Query(0.0), oy: float = Query(0.0),
               tx: float = Query(0.0), ty: float = Query(0.0),
               cs: int = Query(0)):
    png, _ = _render(seed, ox, oy, tx, ty, cs_override=cs or None)
    return Response(content=png, media_type='image/png')


@router.get('/sample')
def sample(seed: int = Query(0), ox: float = Query(0.0), oy: float = Query(0.0),
            tx: float = Query(0.0), ty: float = Query(0.0),
            cs: int = Query(0)):
    png, metrics = _render(seed, ox, oy, tx, ty, cs_override=cs or None)
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
    <label>ωx — pitch (rotation about cam X axis, image up/down): <span class="val" id="oxv">0.00</span> deg</label>
    <input type="range" id="ox" min="-2" max="2" step="0.05" value="0">
    <label>ωy — yaw (rotation about cam Y axis, image left/right): <span class="val" id="oyv">0.00</span> deg</label>
    <input type="range" id="oy" min="-2" max="2" step="0.05" value="0">
    <label>tx: <span class="val" id="txv">0.00</span> m</label>
    <input type="range" id="tx" min="-1" max="1" step="0.01" value="0">
    <label>ty: <span class="val" id="tyv">0.00</span> m</label>
    <input type="range" id="ty" min="-1" max="1" step="0.01" value="0">
    <label>crop size (px): <span class="val" id="csv">__DEMO_CS__</span></label>
    <input type="range" id="cs" min="64" max="__DEMO_CS_MAX__" step="16" value="__DEMO_CS__">
    <p>
      <button id="prev">Prev</button>
      <button id="next">Next</button>
      <button id="reset">Reset sliders</button>
    </p>

    <div class="metrics">
      <h3>Per-point projection error vs GT</h3>
      <div class="metric-row head"><div></div><div>mean</div><div>median</div><div>p95</div></div>
      <div class="metric-row">
        <div>tile-px before</div>
        <div class="num-pre"  id="pre_mean">–</div>
        <div class="num-pre"  id="pre_med">–</div>
        <div class="num-pre"  id="pre_p95">–</div>
      </div>
      <div class="metric-row">
        <div>tile-px after</div>
        <div class="num-post" id="post_mean">–</div>
        <div class="num-post" id="post_med">–</div>
        <div class="num-post" id="post_p95">–</div>
      </div>
      <div class="metric-row">
        <div>128-px before</div>
        <div class="num-pre"  id="pre_l_mean">–</div>
        <div class="num-pre"  id="pre_l_med">–</div>
        <div class="num-pre"  id="pre_l_p95">–</div>
      </div>
      <div class="metric-row">
        <div>128-px after (= val_mse)</div>
        <div class="num-post" id="post_l_mean">–</div>
        <div class="num-post" id="post_l_med">–</div>
        <div class="num-post" id="post_l_p95">–</div>
      </div>

      <h3>Per-point σ (model confidence)</h3>
      <div class="metric-row head"><div></div><div>mean</div><div>median</div><div>ρ̄</div></div>
      <div class="metric-row">
        <div>σ_u (px)</div>
        <div class="num-post" id="su_mean">–</div>
        <div class="num-post" id="su_med">–</div>
        <div class="num-post" id="rho_mean">–</div>
      </div>
      <div class="metric-row">
        <div>σ_v (px)</div>
        <div class="num-post" id="sv_mean">–</div>
        <div class="num-post" id="sv_med">–</div>
        <div></div>
      </div>
      <div class="metric-row">
        <div>Mahalanobis²</div>
        <div class="num-post" id="m2_mean">–</div>
        <div class="num-post" id="m2_med">–</div>
        <div class="num-post" id="m2_p95">–</div>
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

      <details style="margin-top:18px;font-size:13px;line-height:1.5">
      <summary style="cursor:pointer;color:#9cf">How is the BA correction computed?</summary>
      <div style="margin-top:8px;padding:10px 14px;background:#1a1a1a;border:1px solid #444;border-radius:6px">
      <p><b>Pipeline</b>: per-tile model output is a 2-D residual
      <code>(Δu, Δv)</code> with a 2×2 covariance <code>Σ_uv</code> per
      LiDAR point (parametrised as <code>σx, σy, ρ</code> in the network's
      head). BA lifts that 2-D residual into a 4-DoF (or 6-DoF) extrinsic
      <code>δ</code> by Gauss–Newton on the linearised re-projection.</p>

      <p><b>Per-point Jacobian</b>: for a 3-D LiDAR point at depth
      <code>Z</code> with image projection <code>(u, v)</code>, the
      derivatives <code>J_p = [∂u/∂δ, ∂v/∂δ]</code> are closed-form
      (see <code>scripts/ba/ba_multicam_corr.py</code>: <code>_jac_omega_x
      / omega_y / omega_z / tx / ty / tz</code>). For example
      <code>∂u/∂tx = fx/Z</code>, <code>∂u/∂ωy = fx + fx·X²/Z²</code>.</p>

      <p><b>Information-matrix accumulation</b>: each point contributes
      <code>H_p = J_pᵀ Σ_uv⁻¹ J_p</code> and
      <code>b_p = J_pᵀ Σ_uv⁻¹ r_p</code>, where
      <code>r_p = (Δu, Δv)</code> is the predicted residual. We sum
      across <i>all valid points in the tile</i>:
      <code>H = Σ H_p, b = Σ b_p</code>.</p>

      <p><b>Linear solve</b>: <code>δ = (H + λI)⁻¹ b</code>
      (Levenberg damping <code>λ = 1e-3</code>). The solver returns
      <code>(ωx, ωy, tx, ty)</code> in degrees / metres.
      <code>Cov(δ) = (H + λI)⁻¹</code> is also exposed so each DoF gets a
      one-sigma uncertainty.</p>

      <p><b>Why 2-D → 4-D works</b>: a horizontal pixel shift can be
      explained by either yaw or tx, but the two have different per-pixel
      Jacobians (yaw scales with <code>fx + fx·X²/Z²</code>, tx with
      <code>fx/Z</code>). Points at varying depth break the degeneracy;
      bg points near the edge constrain rotations, foreground close-by
      objects constrain translations. The per-point Σ from the model lets
      BA down-weight uncertain points automatically.</p>

      <p><b>Currently shown above</b>: BA pred is the GN solution
      <code>δ</code>; "your input" is the perturbation you added with the
      sliders (<code>−</code>your_input is the GT correction sign).
      "diff" is the absolute residual between BA's recovered correction
      and what would exactly cancel your input.</p>
      </div>
      </details>
    </div>
  </div>
</div>
<script>
let SEED = 0;
let busy = false, pending = false;
const ids = ['ox','oy','tx','ty','cs'];
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
  const preL = m.err_pre_local_px || {}, postL = m.err_post_local_px || {};
  $('pre_l_mean').textContent  = fmtNum(preL.mean, 2) + ' px';
  $('pre_l_med').textContent   = fmtNum(preL.median, 2) + ' px';
  $('pre_l_p95').textContent   = fmtNum(preL.p95, 2) + ' px';
  $('post_l_mean').textContent = fmtNum(postL.mean, 2) + ' px';
  $('post_l_med').textContent  = fmtNum(postL.median, 2) + ' px';
  $('post_l_p95').textContent  = fmtNum(postL.p95, 2) + ' px';
  const sig = m.sigma_px || {}, m2 = m.mahalanobis2 || {};
  $('su_mean').textContent  = fmtNum(sig.u_mean, 2) + ' px';
  $('su_med').textContent   = fmtNum(sig.u_median, 2) + ' px';
  $('sv_mean').textContent  = fmtNum(sig.v_mean, 2) + ' px';
  $('sv_med').textContent   = fmtNum(sig.v_median, 2) + ' px';
  $('rho_mean').textContent = fmtNum(sig.rho_mean, 3);
  $('m2_mean').textContent  = fmtNum(m2.mean, 2);
  $('m2_med').textContent   = fmtNum(m2.median, 2);
  $('m2_p95').textContent   = fmtNum(m2.p95, 2);
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
  const qs = `seed=${SEED}&ox=${v.ox}&oy=${v.oy}&tx=${v.tx}&ty=${v.ty}&cs=${v.cs}`;
  try {
    const r = await fetch('sample?' + qs);
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


@router.get('/', response_class=HTMLResponse)
def index():
    return (DEMO_HTML
            .replace('__DEMO_CS__', str(DEMO_CS))
            .replace('__DEMO_CS_MAX__', str(DEMO_CS)))


# Mount the same router at both root and /demo so direct LAN access
# (http://yokohama1:8765/) and the Cloudflare-tunneled path-based route
# (https://clearml.budda.site/demo) both resolve. CF tunnel doesn't strip
# path prefixes, so the prefix has to be a real, registered mount.
app.include_router(router)
app.include_router(router, prefix='/demo')


# Convenience: /demo without trailing slash → /demo/ so relative `sample` URLs
# in the served HTML resolve under the demo prefix.
@app.get('/demo', include_in_schema=False)
def _demo_redirect():
    return RedirectResponse(url='/demo/', status_code=307)


def main():
    import uvicorn
    uvicorn.run('scripts.serving.demo_app:app',
                host=os.environ.get('DEMO_HOST', '0.0.0.0'),
                port=int(os.environ.get('DEMO_PORT', 8765)),
                workers=1, log_level='info')


if __name__ == '__main__':
    main()
