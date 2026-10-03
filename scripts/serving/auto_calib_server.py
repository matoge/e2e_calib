"""FastAPI server for LiDAR–camera auto-calibration (CalibNet2 S3 cam6).

Runs on port 8501 by default. LOOM's tasks.html pings /health for the
'Auto Calib' indicator; the LOOM apply flow POSTs to /calibrate/sequence.

Endpoints
    GET  /                       — HTML landing page (browsable status)
    GET  /health                 — {status:"ok", model:..., device:...}
    POST /calibrate/sequence     — seq_dir + stride → joint δ + H + k
    POST /calibrate/frame        — single-frame raw arrays → δ + H
    POST /reload                 — hot-swap to a different ckpt

Start:
    CKPT=experiments/kmwv_s3_ba40_512r256_0901_1344 \\
    /home/hfunaya/.pyenv/versions/3.10.4/bin/python \\
        scripts/serving/auto_calib_server.py --port 8501

The `--host 0.0.0.0` binding lets LOOM (same host) and remote LOOM clients
(browsing tasks.html) both reach /health directly.
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'inference'))

from calib_lib import (
    load_frame_data, build_model, calibrate_frame, pool_frames,
    cam_delta_to_setting, cam_sd_to_setting, CalibResult,
)


DEFAULT_CKPT = REPO / 'experiments' / 'kmwv_s3_ba40_512r256_0901_1344'
DEFAULT_HOOD_MASK_ROOT = Path('/home/hfunaya/git/loom/backend/assets/hood_masks')

app = FastAPI(
    title='CalibNet2 Auto-Calibration API',
    version='0.1',
    description=(
        'Return the residual 6-DoF LiDAR–camera extrinsic error of a '
        'WovenSequence, plus a properly calibrated information matrix. '
        'Design: model outputs per-point (μ, σ, W); a zero-parameter '
        'outer Gauss-Newton solves 6-DoF per frame; frames pool by summing '
        'per-frame information matrices (equivalent to joint solve when the '
        'unknown is the same rig extrinsic).\n\n'
        '**Auto-docs:** /docs (Swagger UI), /redoc, /openapi.json — all served '
        'over the same origin via LOOM at http://172.16.200.185:8082'
        '/api/calibration-server/…'),
    root_path=os.environ.get('ROOT_PATH', ''),
)
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                    allow_methods=['*'], allow_headers=['*'])

STATE = dict(model=None, ckpt=None, device='cuda', hood_mask_root=None,
             loaded_at=None, request_count=0, lock=threading.Lock())


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _load_model(ckpt_dir: Path):
    print(f'[auto_calib] loading {ckpt_dir}...', flush=True)
    t0 = time.time()
    model = build_model(ckpt_dir, device=STATE['device'])
    STATE['model'] = model
    STATE['ckpt'] = str(ckpt_dir)
    STATE['loaded_at'] = time.time()
    print(f'[auto_calib] loaded in {time.time()-t0:.1f}s '
          f'(cuda={torch.cuda.is_available()})', flush=True)


# ---------------------------------------------------------------------------
# Health / landing
# ---------------------------------------------------------------------------
@app.get('/health', tags=['status'])
def health():
    """Liveness + which ckpt is loaded.

    Used by LOOM tasks.html for the 'Auto Calib' indicator dot.
    Returns 200 with `{status:"ok", model:..., device:...}` when the model
    has finished loading, otherwise `{status:"loading"}`.
    """
    return {
        'status': 'ok' if STATE['model'] is not None else 'loading',
        'model': STATE['ckpt'],
        'device': STATE['device'],
        'loaded_at': STATE['loaded_at'],
        'request_count': STATE['request_count'],
    }


TEST_ASSETS_DIR = (REPO / 'docs' / 'assets' / '2026-09-30_unilab001_apply').resolve()
if TEST_ASSETS_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(TEST_ASSETS_DIR)),
              name='static')


@app.get('/test-page', response_class=HTMLResponse, tags=['status'],
         include_in_schema=False)
def test_page():
    """Human-readable API doc + a live example run on unilab_001/test01."""
    ready = STATE['model'] is not None
    return f"""<!doctype html><html><head>
<title>CalibNet2 Auto-Calibration API — Test Page</title>
<meta charset="utf-8">
<style>
body {{font-family: system-ui,-apple-system,sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em;
       color: #222; line-height: 1.5;}}
h1 {{border-bottom: 2px solid #333; padding-bottom: .3em}}
h2 {{border-bottom: 1px solid #ccc; padding-bottom: .2em; margin-top: 1.8em}}
code, pre {{background:#f4f4f4; padding: 2px 6px; border-radius: 3px;
            font-family: SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 13px}}
pre {{padding: 10px 14px; overflow-x: auto}}
table {{border-collapse: collapse; margin: 1em 0}}
th, td {{border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top}}
th {{background:#eee}}
.badge {{display: inline-block; padding: 2px 8px; border-radius: 12px;
          font-size: 12px; font-weight: 600}}
.ok {{background:#d4edda; color:#155724}}
.no {{background:#f8d7da; color:#721c24}}
.mono {{font-family: SFMono-Regular, Menlo, Consolas, monospace;}}
.numeric {{font-family: SFMono-Regular, Menlo, Consolas, monospace; text-align: right}}
img {{max-width: 100%; border: 1px solid #ddd; border-radius: 4px}}
figcaption {{color:#555; font-size:13px; margin-top:.5em}}
</style></head>
<body>
<h1>CalibNet2 Auto-Calibration API</h1>
<p>
Status: <span class="badge {'ok' if ready else 'no'}">{'READY' if ready else 'LOADING'}</span> ·
device: <code>{STATE['device']}</code> ·
model: <code>{STATE['ckpt']}</code> ·
requests served: <b>{STATE['request_count']}</b>
</p>

<p>
Takes a WovenSequence directory (LiDAR points + fisheye images + the
current calibration written in <code>setting-&lt;ip&gt;.json</code>) and returns
the residual 6-DoF LiDAR↔camera extrinsic error, with a per-DoF standard
deviation. The joint solve pools information matrices from every
40-tile fused BA run on each sampled frame — no re-training needed to
add more frames, just add them to the pool.
</p>

<h2>Endpoints</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Purpose</th></tr>
<tr><td>GET</td><td><code>/health</code></td>
    <td>Liveness + which ckpt is loaded (used by LOOM tasks.html indicator)</td></tr>
<tr><td>POST</td><td><code>/calibrate/sequence</code></td>
    <td>Sample <code>stride</code>/<code>max_frames</code> frames from a seq, run joint pool</td></tr>
<tr><td>POST</td><td><code>/calibrate/frame</code></td>
    <td>Single-frame result; optional per-tile HAT/μ for visualization</td></tr>
<tr><td>POST</td><td><code>/reload</code></td>
    <td>Hot-swap to a different ckpt directory</td></tr>
</table>

<h2><code>POST /calibrate/sequence</code></h2>
<h3>Request</h3>
<pre>{{
  "seq_dir":      "/home/.../sequence=&lt;id&gt;",   # absolute path
  "stride":       5,        # sample every Nth frame; default 5
  "max_frames":   10,       # cap on # frames used; default 10
  "use_hood_mask": true,    # drop LiDAR pts inside ego-hood mask
  "use_gate3":    true,     # χ² gate 3 on multi-frame pool
  "return_per_frame": false # include per-frame δ in the response
}}</pre>
<h3>Response</h3>
<pre>{{
  "status": "success",
  "ckpt": "&lt;path to checkpoint&gt;",
  "n_frames_requested": 10,
  "n_frames_used":      10,   # after gate3 rejection
  "n_tiles_total":     265,   # across all frames
  "n_points_total":  42253,
  "elapsed_s":         2.86,
  "k_dispersion":      0.34,  # χ²/6 — >1 means H-sum is over-confident
  # ADDITIVE delta for LOOM setting.json  (rad + m, sign fixed):
  "deltas": {{ "rot_x": .., "rot_y": .., "rot_z": ..,
              "mp_x": .., "mp_y": .., "mp_z": .. }},
  "sd":     {{ ... }},        # per-DoF 1σ, k-corrected
  "sd_raw": {{ ... }},        # per-DoF 1σ, raw H⁻¹
  # Raw camera-frame δ  (deg + m):
  "delta_cam_deg": [ω_x pitch, ω_y yaw, ω_z roll],
  "delta_cam_m":   [t_x, t_y, t_z],
  "H_cam": <b>6×6</b> nested list
}}</pre>

<h2><code>POST /calibrate/frame</code></h2>
<p>Same shape as sequence but for one frame. Additional flags:</p>
<pre>{{
  "seq_dir": "...", "frame_idx": 0,
  "use_hood_mask": true,
  "return_per_point":   true,   # per-tile [uv_hat, μ, σ] arrays  (for viz)
  "return_hood_polygon": true   # polygon vertices from the hood mask JSON
}}</pre>

<h2>Coordinate conventions</h2>
<table>
<tr><th></th><th>Vehicle (setting.json)</th><th>Camera (solver output)</th></tr>
<tr><td>Frame</td><td>FLU — x=fwd, y=left, z=up</td><td>FRD — x=right, y=down, z=fwd</td></tr>
<tr><td>Rot units</td><td>rad</td><td>deg</td></tr>
<tr><td>Rot axes</td><td>rot[0]=roll (X), rot[1]=pitch (Y), rot[2]=yaw (Z)</td>
    <td>ω_x=pitch, ω_y=yaw, ω_z=roll</td></tr>
<tr><td>Sign</td><td colspan="2">Solver δ is the current bias. Additive fix (LOOM: new = old + Δ) → <code>Δ = −δ_veh</code>. The API applies this sign flip in <code>deltas</code> already.</td></tr>
</table>

<h2>Time-discrepancy handling</h2>
<p>WovenSequence records the shutter/LiDAR-sweep offset per sequence
(sometimes per frame) as <code>camera_delay_ms</code>. The loader
compensates by interpolating the pose from LiDAR time to camera shutter
time (<code>_pose_at_camera_time</code>) before projection — so the model
sees geometrically synchronised inputs. For unilab_001/test01 this
sequence declares <code>50.55 ms</code>; the compensation shifts pose by
~0.45 m at 9 m/s ego motion.</p>

<h2>Ego-hood mask</h2>
<p>Auto-resolved by <code>metadata.car_id</code>:
<code>backend/assets/hood_masks/car&lt;car_id&gt;.png</code>. Any LiDAR point that
projects inside a nonzero pixel is dropped <b>before</b> tiling — the
network never sees the ego bonnet, and BA never weights it.</p>

<h2>Example run — unilab_001 / test01, frame 0, F=10</h2>
<figure>
<img src="/static/api_test_overlay.png" alt="whole frame overlay">
<figcaption>Whole-frame tile-based visualization at frame 0. Red × =
LiDAR projected with the current calibration (HAT). Cyan ○ = model-
corrected projection (HAT + μ). Yellow lines = per-point Δuv. Green tile
boxes = fused, red tile boxes = skipped (fewer than 8 points).
Magenta polygon + area = ego-hood mask. Yellow zoom boxes marked L / C /
R match the panels below. Title reports the joint 6-DoF from the F=10
sequence pool.</figcaption>
</figure>
<figure style="margin-top:1.5em">
<img src="/static/api_test_overlay_zoom_LCR.png" alt="LCR zoom">
<figcaption>Zoom into left / center / right regions of the frame. The
~0.26° pitch bias is most visible on distant building rooflines
(centre / right panels); side-of-image trees (L) show a smaller,
mostly-vertical shift consistent with a pitch-dominated correction.</figcaption>
</figure>

<h2>Reproducing this example</h2>
<pre># From e2e_calib repo root:
python scripts/serving/client_apply_and_viz.py \\
  --seq /home/hfunaya/git/loom/backend/assets/woven_sequence/unilab_001/test01/sequence=ip654-lidar0-1432519511400022000-1432519516399877000 \\
  --api http://127.0.0.1:8501 \\
  --stride 5 --max-frames 10 --viz-frame-idx 0 \\
  --out docs/assets/2026-09-30_unilab001_apply/api_test_overlay.png

# Numerical pass/fail suite:
python tests/test_auto_calib_api.py</pre>

<h2>Bootstrapping the server</h2>
<pre>CKPT=experiments/kmwv_s3_ba40_512r256_0901_1344 \\
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \\
    scripts/serving/auto_calib_server.py --port 8501

# For persistence without sudo:
# ~/crontab entries (idempotent launcher checks port before starting):
*/2 * * * *  /home/hfunaya/git/e2e_calib/scripts/serving/start_auto_calib.sh
@reboot      /home/hfunaya/git/e2e_calib/scripts/serving/start_auto_calib.sh</pre>
</body></html>
"""


@app.get('/', response_class=HTMLResponse, tags=['status'], include_in_schema=False)
def landing():
    ready = STATE['model'] is not None
    return f"""<!doctype html><html><head><title>CalibNet2 Auto-Calib</title>
<style>body{{font-family:sans-serif;max-width:720px;margin:2em auto;padding:0 1em}}
code{{background:#f4f4f4;padding:2px 6px;border-radius:3px}}
.ok{{color:#28a745}} .no{{color:#dc3545}}</style></head>
<body>
<h1>CalibNet2 Auto-Calibration API</h1>
<p>Status: <b class="{'ok' if ready else 'no'}">{'READY' if ready else 'LOADING'}</b></p>
<ul>
<li>ckpt: <code>{STATE['ckpt']}</code></li>
<li>device: <code>{STATE['device']}</code></li>
<li>loaded_at: <code>{STATE['loaded_at']}</code></li>
<li>requests served: <b>{STATE['request_count']}</b></li>
</ul>
<h3>Endpoints</h3>
<ul>
<li><code>GET /health</code> — JSON status (used by LOOM tasks.html indicator)</li>
<li><code>POST /calibrate/sequence</code> — <code>{{seq_dir, stride, max_frames, use_hood_mask, use_gate3}}</code></li>
<li><code>POST /calibrate/frame</code> — for embedded / raw-array use (advanced)</li>
<li><code>POST /reload</code> — <code>{{ckpt_dir}}</code>, hot-swap</li>
</ul>
</body></html>
"""


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------
class ReloadReq(BaseModel):
    ckpt_dir: str


@app.post('/reload', tags=['admin'],
          description=('Hot-swap the loaded model to a different checkpoint '
                       'without restarting the server. Blocks in-flight '
                       'requests via `STATE.lock`.'))
def reload_ckpt(req: ReloadReq):
    p = Path(req.ckpt_dir)
    if not (p / 'best_model.pt').exists():
        raise HTTPException(400, f'best_model.pt not found in {p}')
    with STATE['lock']:
        _load_model(p)
    return {'status': 'ok', 'ckpt': STATE['ckpt']}


# ---------------------------------------------------------------------------
# Sequence-level calibration
# ---------------------------------------------------------------------------
DEFAULT_SEQ = (
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/unilab_001/test01/'
    'sequence=ip654-lidar0-1432519511400022000-1432519516399877000'
)


class SeqReq(BaseModel):
    """POST body for /calibrate/sequence — a sequence-level auto-calibration."""
    seq_dir: str = Field(
        ..., example=DEFAULT_SEQ,
        description=('Absolute path on the LOOM host to a WovenSequence '
                     'directory. Must contain `tss4_fcm/*.jpg`, '
                     '`vls128_rear_axle/*.npz`, `metadata.json`, and '
                     '`setting-<ip>.json`.'))
    stride: int = Field(
        5, ge=1, le=50, example=5,
        description='Sample every Nth frame from the sequence. '
                    '5 means frame 0, 5, 10, ...')
    max_frames: int = Field(
        10, ge=1, le=200, example=10,
        description='Cap on the number of frames actually run through the '
                    'network. Increases σ⁻¹ roughly √F but adds runtime.')
    use_hood_mask: bool = Field(
        True, description='Drop LiDAR points that project inside the ego '
                          'hood mask (auto-resolved by `metadata.car_id`).')
    use_gate3: bool = Field(
        True, description='Apply the χ² gate 3 outlier rejection on the '
                          'multi-frame pool. Recommended.')
    return_per_frame: bool = Field(
        False, description='Include the per-frame 6-DoF δ + tile counts in '
                           'the response `per_frame` field. Payload gets larger.')

    model_config = ConfigDict(json_schema_extra={
        'example': {
            'seq_dir': DEFAULT_SEQ,
            'stride': 5, 'max_frames': 10,
            'use_hood_mask': True, 'use_gate3': True,
            'return_per_frame': False,
        }
    })


class DeltaDict(BaseModel):
    """6-DoF delta (or per-axis σ) in the LOOM setting.json convention:
    `rot_*` in RADIANS (vehicle-FLU Euler [roll, pitch, yaw]),
    `mp_*` in METRES (camera translation in vehicle frame).
    Signed so that `setting.rot += delta.rot_*` is the correct fix
    (i.e. this delta already applies the sign flip vs the solver output)."""
    rot_x: float = Field(..., description='Δroll  (vehicle-X axis rotation) [rad]')
    rot_y: float = Field(..., description='Δpitch (vehicle-Y axis rotation) [rad]')
    rot_z: float = Field(..., description='Δyaw   (vehicle-Z axis rotation) [rad]')
    mp_x:  float = Field(..., description='Δt_x [m], vehicle forward')
    mp_y:  float = Field(..., description='Δt_y [m], vehicle left')
    mp_z:  float = Field(..., description='Δt_z [m], vehicle up')


class SeqResp(BaseModel):
    """/calibrate/sequence — joint-pooled calibration result."""
    status: str = Field(..., example='success')
    ckpt: str = Field(..., description='Absolute path to the checkpoint '
                                        'used for this response.')
    n_frames_requested: int
    n_frames_used: int = Field(..., description='After optional gate3 rejection.')
    n_tiles_total: int
    n_points_total: int
    elapsed_s: float
    stride: int
    k_dispersion: float = Field(
        ..., description=('χ²/6 measured on the pooled residuals. Values > 1 '
                           'mean `Σ H_f` is over-confident (frames not fully '
                           'independent); divide covariance by k for the '
                           'calibrated sd. Values < 1 mean the per-frame H '
                           'was conservative.'))
    deltas: DeltaDict = Field(..., description='Additive delta for setting.json. '
                                                'ALREADY sign-flipped so `new = old + Δ`.')
    sd:     DeltaDict = Field(..., description='Per-DoF 1σ after k-correction '
                                                '(same units as `deltas`).')
    sd_raw: DeltaDict = Field(..., description='Per-DoF 1σ from raw H⁻¹ '
                                                '(before k-correction).')
    delta_cam_deg: list[float] = Field(..., description='[ω_x pitch, ω_y yaw, '
                                                          'ω_z roll] in CAMERA FRD, degrees.')
    delta_cam_m:   list[float] = Field(..., description='[t_x, t_y, t_z] '
                                                          'in CAMERA FRD, metres.')
    H_cam: list[list[float]] = Field(..., description='6×6 information matrix '
                                                        '(camera frame). Sum-across-'
                                                        'frames = joint-solve H.')
    per_frame: list | None = Field(None, description='Only present when '
                                                       '`return_per_frame=true`.')


@app.post('/calibrate/sequence', response_model=SeqResp, tags=['calibrate'],
          responses={
              200: {'description': 'Joint 6-DoF calibration for the sampled '
                                    'frames of the sequence.'},
              400: {'description': 'seq_dir not found.'},
              500: {'description': 'no frames produced usable results '
                                    '(usually 0 tiles per frame — cache '
                                    'schema mismatch or empty LiDAR).'},
              503: {'description': 'model not loaded yet — retry after a '
                                    'few seconds.'},
          },
          description=(
"""Auto-calibrate one WovenSequence.

Samples `stride` × `max_frames` frames, runs 40-tile fused Gauss-Newton per frame,
and pools the per-frame information matrices into one joint 6-DoF estimate.

**What you send**  →  a `seq_dir` on the LOOM host + optional sampling knobs.

**What you get back**
* `deltas` — additive delta ready to add to `setting-<ip>.json` (rad + m, sign fixed)
* `sd` / `sd_raw` — k-corrected / raw per-DoF 1σ
* `k_dispersion` — χ²/6 of the joint fit; interpret `deltas ± sd` as the calibrated 1σ
* `delta_cam_deg` / `delta_cam_m` — raw camera-frame 6-DoF (for logging / debugging)
* `H_cam` — 6×6 information matrix for further downstream fusion

---

### Live example — unilab_001 / test01, F=10

The "Try it out" panel below is pre-filled with the exact request that produced these figures.
Click **Execute** to reproduce (~3 s round-trip).

**Whole frame** (frame 0, 40-tile overlay + hood-mask polygon + L / C / R zoom regions):

![overlay](/api/calibration-server/static/api_test_overlay.png)

**Zoom L / C / R** (yellow boxes above → per-region red-X→cyan-○ correction arrows):

![zoom LCR](/api/calibration-server/static/api_test_overlay_zoom_LCR.png)

Manually-tuned LOOM calibration on this sequence agrees on pitch ≈ +0.27°, matching
the API result. The 6-DoF numbers, σ, and k for this exact run are in the header of the
top figure. Reproduce locally with:

```bash
python scripts/serving/client_apply_and_viz.py \\
    --seq <seq_dir> --api http://127.0.0.1:8501 \\
    --stride 5 --max-frames 10 --viz-frame-idx 0 \\
    --out out.png
```
"""))
def calibrate_sequence(req: SeqReq):
    """Auto-calibrate a WovenSequence, F-frame joint pool.

    (See the endpoint description above for the full explanation.)
    """
    if STATE['model'] is None:
        raise HTTPException(503, 'model not loaded yet')
    seq = Path(req.seq_dir)
    if not seq.exists():
        raise HTTPException(400, f'sequence dir not found: {seq}')

    hood_root = STATE['hood_mask_root'] if req.use_hood_mask else None
    STATE['request_count'] += 1

    with STATE['lock']:
        # Enumerate frame indices — we peek via load_frame_data(idx=0) to get
        # frame count via reading metadata; short version: sample every stride
        # up to max_frames, error out if that overruns.
        # Cheap: reuse the loader's metadata call by loading one frame first.
        from build_woven_sequence_v3 import _load_metadata, _get_poses
        fids, _ = _get_poses(_load_metadata(seq))
        idxs = list(range(0, len(fids), req.stride))[:req.max_frames]

        t0 = time.time()
        per_frame_results = []
        per_frame_meta = []
        for i in idxs:
            try:
                fd = load_frame_data(seq, i, hood_mask_root=hood_root)
            except Exception as e:
                per_frame_meta.append(dict(frame_idx=i, error=str(e)))
                continue
            res = calibrate_frame(STATE['model'], fd, device=STATE['device'])
            if res.n_tiles_fused == 0:
                per_frame_meta.append(dict(frame_idx=i, fid=fd.fid, error='no valid tiles'))
                continue
            per_frame_results.append(res)
            per_frame_meta.append(dict(
                frame_idx=i, fid=fd.fid, delay_ms=fd.delay_ms,
                n_tiles=res.n_tiles_fused, n_points=res.n_points,
                n_hood_dropped=fd.n_hood_dropped,
                delta_cam=res.delta_cam.tolist(),
            ))
        elapsed = time.time() - t0

    if not per_frame_results:
        raise HTTPException(500, 'no frames produced usable results')

    joint = pool_frames(per_frame_results,
                        mode='gate3' if req.use_gate3 else 'sum')

    # Frame data is heterogeneous across frames but the setting rot/mp is the
    # same, and R_cam_from_veh is fixed → convert from cam to setting frame
    # using the last-loaded frame's frame_data.
    fd_last = load_frame_data(seq, idxs[0], hood_mask_root=hood_root)
    adj    = cam_delta_to_setting(joint.delta_cam, fd_last.R_cam_from_veh)
    sd_raw = cam_sd_to_setting(joint.sd(k_correct=False), fd_last.R_cam_from_veh)
    sd_kc  = cam_sd_to_setting(joint.sd(k_correct=True),  fd_last.R_cam_from_veh)

    resp = dict(
        status='success',
        ckpt=STATE['ckpt'],
        n_frames_requested=len(idxs),
        n_frames_used=joint.n_frames,
        n_tiles_total=joint.n_tiles_fused,
        n_points_total=joint.n_points,
        elapsed_s=round(elapsed, 2),
        stride=req.stride,
        k_dispersion=round(joint.k, 3),
        # Additive delta for LOOM setting.json (rad for rot, m for mp).
        deltas=adj,
        sd=sd_kc,
        sd_raw=sd_raw,
        # Raw camera-frame δ for downstream analysis / logging.
        delta_cam_deg=joint.delta_cam[:3].tolist(),
        delta_cam_m  =joint.delta_cam[3:].tolist(),
        H_cam=joint.H_cam.tolist(),
    )
    if req.return_per_frame:
        resp['per_frame'] = per_frame_meta
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# Single-frame raw calibration (Layer 1, expert use)
# ---------------------------------------------------------------------------
class FrameReq(BaseModel):
    """POST body for /calibrate/frame — single-frame calibration."""
    seq_dir: str = Field(..., example=DEFAULT_SEQ)
    frame_idx: int = Field(0, ge=0, description='0-based frame index within '
                                                  'the sequence (0 = first).')
    use_hood_mask: bool = True
    return_per_point:     bool = Field(False, description='Include per-tile '
                                                            '[uv_hat, μ, σ] arrays '
                                                            '(needed by the viz client).')
    return_hood_polygon:  bool = Field(False, description='Include hood mask '
                                                            'polygon vertices.')

    model_config = ConfigDict(json_schema_extra={
        'example': {
            'seq_dir': DEFAULT_SEQ, 'frame_idx': 0,
            'use_hood_mask': True, 'return_per_point': True,
            'return_hood_polygon': True,
        }
    })


@app.post('/calibrate/frame', tags=['calibrate'],
          description=(
"""Single-frame calibration.

Same tile-fused BA as /calibrate/sequence but for one frame — useful when
you want the per-tile HAT/μ/σ arrays for a visualization overlay
(`return_per_point=true`). No frame-to-frame pooling.

If you just want the numerical answer, use `/calibrate/sequence` — it's F× more
information for the same wall-clock (all frames run in parallel-batched forward).
"""))
def calibrate_frame_endpoint(req: FrameReq):
    if STATE['model'] is None:
        raise HTTPException(503, 'model not loaded yet')
    seq = Path(req.seq_dir)
    if not seq.exists():
        raise HTTPException(400, f'seq not found: {seq}')
    hood_root = STATE['hood_mask_root'] if req.use_hood_mask else None
    STATE['request_count'] += 1
    with STATE['lock']:
        fd = load_frame_data(seq, req.frame_idx, hood_mask_root=hood_root)
        # Re-do the model forward here (mirrors calib_lib.calibrate_frame)
        # so we can extract per-tile (uv_hat, μ, σ) for visualization.
        from calib_lib import (_tile_grid, _build_batch, CS as _CS, S as _S,
                                 GRID_N as _GRID_N)
        import torch as _torch
        cells = _tile_grid(fd.IW, fd.IH, _CS)
        imgs, q_in, vfp, buck, bvalid, kpm, per_tile = _build_batch(
            fd.image_rgb, fd.uv_hat, fd.z, fd.intensity, fd.pts_cam, fd.K,
            cells, STATE['device'])
        with _torch.no_grad():
            out = STATE['model'](imgs, q_in, vfp=vfp, bucket_uvd=buck,
                                  bucket_valid=bvalid, key_padding_mask=kpm,
                                  mode='calib')
        per_pt, W_head = (out if isinstance(out, tuple) else (out, None))

        # Aggregate to 6-DoF (same as calib_lib but inline so per-point survives).
        from scripts.ba.ba_torch import solve_pinhole_xyz
        all_pts, all_duv, all_W = [], [], []
        for b, td in enumerate(per_tile):
            if td is None: continue
            Nq = len(td['idx'])
            mu_orig = per_pt[b, :Nq, :2].cpu().numpy() * (_CS / _S)
            all_pts.append(td['pts_cam']); all_duv.append(mu_orig)
            if W_head is not None:
                all_W.append(W_head[b, :Nq].cpu().numpy())
            else:
                sx = _torch.exp(per_pt[b, :Nq, 2]).cpu().numpy() * (_CS/_S)
                sy = _torch.exp(per_pt[b, :Nq, 3]).cpu().numpy() * (_CS/_S)
                Wf = np.zeros((Nq, 2, 2), dtype=np.float32)
                Wf[:, 0, 0] = 1.0/(sx**2); Wf[:, 1, 1] = 1.0/(sy**2)
                all_W.append(Wf)
        pts_cam_all = np.concatenate(all_pts, 0)
        duv_all     = np.concatenate(all_duv, 0)
        W_all       = np.concatenate(all_W,   0)
        pts_t = _torch.from_numpy(pts_cam_all).double().unsqueeze(0).to(STATE['device'])
        duv_t = _torch.from_numpy(duv_all).double().unsqueeze(0).to(STATE['device'])
        W_t   = _torch.from_numpy(W_all).double().unsqueeze(0).to(STATE['device'])
        K_t   = _torch.from_numpy(fd.K.astype(np.float64)).unsqueeze(0).to(STATE['device'])
        v_t   = _torch.from_numpy(np.ones(len(pts_cam_all), dtype=bool)).unsqueeze(0).to(STATE['device'])
        from calib_lib import DOF as _DOF
        delta, H = solve_pinhole_xyz(pts_t, duv_t, W_t, K_t, _DOF,
                                      valid=v_t, n_iter=10, damping=1e-3,
                                      robust='huber', huber_k=1.5)
        delta = delta[0].cpu().numpy(); H = H[0].cpu().numpy()

    from calib_lib import CalibResult as _CR
    res = _CR(delta_cam=delta, H_cam=H, n_frames=1,
              n_tiles_fused=sum(1 for t in per_tile if t is not None),
              n_points=len(pts_cam_all))
    adj    = cam_delta_to_setting(res.delta_cam, fd.R_cam_from_veh)
    sd_raw = cam_sd_to_setting(res.sd(k_correct=False), fd.R_cam_from_veh)

    resp = dict(
        status='success',
        fid=fd.fid, frame_idx=req.frame_idx, delay_ms=fd.delay_ms,
        n_tiles=res.n_tiles_fused, n_points=res.n_points,
        n_hood_dropped=fd.n_hood_dropped,
        deltas=adj, sd=sd_raw,
        delta_cam_deg=res.delta_cam[:3].tolist(),
        delta_cam_m  =res.delta_cam[3:].tolist(),
        H_cam=res.H_cam.tolist(),
    )
    if req.return_per_point:
        # Per-tile HAT/μ/σ in original px so the client just plots.
        tiles_out = []
        for b, td in enumerate(per_tile):
            if td is None:
                tiles_out.append({'tile_idx': b, 'u0': int(cells[b][0]),
                                    'v0': int(cells[b][1]), 'skipped': True})
                continue
            Nq = len(td['idx'])
            mu_model = per_pt[b, :Nq, :2].cpu().numpy()
            log_sx = per_pt[b, :Nq, 2].cpu().numpy()
            log_sy = per_pt[b, :Nq, 3].cpu().numpy()
            mu_orig = (mu_model * (_CS / _S)).tolist()
            sx_orig = (np.exp(log_sx) * (_CS / _S)).tolist()
            sy_orig = (np.exp(log_sy) * (_CS / _S)).tolist()
            tiles_out.append(dict(
                tile_idx=b, u0=int(cells[b][0]), v0=int(cells[b][1]),
                skipped=False,
                uv_hat=td['uv_orig'].tolist(),      # in ORIGINAL image px
                mu_orig=mu_orig,
                sigma_x_orig=sx_orig, sigma_y_orig=sy_orig,
            ))
        resp['tiles'] = tiles_out
        resp['image_size'] = [int(fd.IW), int(fd.IH)]
    if req.return_hood_polygon and hood_root is not None:
        car_id_ = getattr(fd, 'car_id', None)
        # metadata car_id via re-load — cheap
        from build_woven_sequence_v3 import _load_metadata as _lm
        car_id_ = _lm(seq).get('car_id')
        poly_json = hood_root / f'car{car_id_}.polygon.json'
        if poly_json.exists():
            resp['hood_polygon'] = json.load(open(poly_json)).get('polygon', [])
    return resp


# ---------------------------------------------------------------------------
# Generic payload endpoints — no filesystem, no dataset assumption.
#
# The client sends image bytes + LiDAR points + intrinsics + LiDAR→cam rigid
# transform (already time-compensated for camera shutter delay if needed).
# The server just runs inference. This is the API surface any downstream tool
# (LOOM, kamikado, external repos, CI smoke tests) should target — the
# existing /calibrate/{frame,sequence} routes are convenience wrappers that
# happen to know how to read a WovenSequence directory.
# ---------------------------------------------------------------------------
import base64 as _b64


def _decode_image(image_b64: Optional[str], image_url: Optional[str]) -> np.ndarray:
    """base64 JPEG/PNG → (H, W, 3) uint8 RGB."""
    if image_b64:
        buf = _b64.b64decode(image_b64)
    elif image_url:
        raise HTTPException(400, 'image_url not supported yet — use image_b64')
    else:
        raise HTTPException(400, 'either image_b64 or image_url is required')
    import cv2
    img_arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, 'image decode failed (not a valid JPEG/PNG?)')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_points(points_xyzi_b64: Optional[str],
                    points_xyzi_list: Optional[list]) -> np.ndarray:
    """base64 float32 (N,4) OR list of [x,y,z,i] → (N, 4) float32 numpy."""
    if points_xyzi_b64:
        buf = _b64.b64decode(points_xyzi_b64)
        arr = np.frombuffer(buf, dtype=np.float32)
        if arr.size % 4 != 0:
            raise HTTPException(400, f'points_b64 length {arr.size} not divisible by 4')
        return arr.reshape(-1, 4)
    if points_xyzi_list:
        arr = np.asarray(points_xyzi_list, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise HTTPException(400, f'points_xyzi_list must be (N,4), got {arr.shape}')
        return arr
    raise HTTPException(400, 'either points_xyzi_b64 or points_xyzi_list is required')


class GenericCamera(BaseModel):
    """Camera intrinsics + LiDAR→cam extrinsics. Everything the server needs
    to project LiDAR into the image plane. All units are metres / pixels /
    radians as documented per field.
    """
    resolution: list = Field(..., min_length=2, max_length=2,
                              description='[IW, IH] in pixels',
                              example=[3840, 2160])
    fc: list = Field(..., min_length=2, max_length=2,
                      description='[fx, fy] in pixels',
                      example=[1024.0, 1024.0])
    cc: list = Field(..., min_length=2, max_length=2,
                      description='[cx, cy] principal point in pixels',
                      example=[1920.0, 1080.0])
    dist_kb4: list = Field(default=[0.0, 0.0, 0.0, 0.0], min_length=4, max_length=4,
                            description='Kannala-Brandt 4 fisheye coefficients '
                                        '[k1, k2, k3, k4]. All zeros = pinhole.',
                            example=[0.0, 0.0, 0.0, 0.0])
    T_cam_lidar: list = Field(..., description='4×4 LiDAR→camera rigid transform '
                                                'AT CAMERA SHUTTER TIME. Client is '
                                                'responsible for any pose time '
                                                'compensation between LiDAR sweep '
                                                'and camera exposure.')
    setting_rot_rad: Optional[list] = Field(
        default=None, min_length=3, max_length=3,
        description='Current LOOM setting.json rotation [roll, pitch, yaw] '
                    '(rad, ZYX Euler, camera-to-vehicle FLU). Optional — '
                    'needed only if you want the response in "additive to '
                    'setting.json" form (`deltas` field). If null the '
                    'response only carries the raw camera-FRD δ.')
    setting_mp_m: Optional[list] = Field(
        default=None, min_length=3, max_length=3,
        description='Current LOOM setting.json translation [x, y, z] (m, '
                    'camera-in-vehicle FLU). Same optionality as setting_rot_rad.')
    R_cam_from_veh: Optional[list] = Field(
        default=None,
        description='Optional 3×3 override. If null, computed from '
                    'setting_rot_rad + R_TO_RDF. Needed to convert the model\'s '
                    'camera-FRD δ back to vehicle-FLU / setting.json space.')


class GenericFrameReq(BaseModel):
    """One frame of raw data — image + LiDAR + calibration."""
    image_b64: Optional[str] = Field(
        None, description='JPEG or PNG bytes, base64-encoded. Required unless '
                          'image_url is provided (not yet supported).')
    image_url: Optional[str] = None
    points_xyzi_b64: Optional[str] = Field(
        None, description='LiDAR points as base64 float32 (N,4) '
                          '[x, y, z, intensity] in LiDAR frame.')
    points_xyzi_list: Optional[list] = Field(
        None, description='Alternative: JSON list of [x,y,z,intensity]. '
                          'Slower — prefer points_xyzi_b64 for >10k points.')
    camera: GenericCamera
    hood_polygon_uv: Optional[list] = Field(
        default=None, description='Optional list of [u, v] polygon vertices '
                                    '(image pixel space) — points projecting '
                                    'inside this polygon are dropped before '
                                    'the model sees them. Same semantic as '
                                    'the built-in hood mask.')
    frame_id: Optional[str] = Field(default='gen_0',
                                      description='Bookkeeping id, echoed back.')


class GenericSeqReq(BaseModel):
    """Multi-frame joint auto-calibration. `frames` is 1..F frames; the
    server runs `calibrate_frame` on each and pools via inverse-variance
    (χ² gate + k dispersion, same as /calibrate/sequence)."""
    frames: list  # list of GenericFrameReq-shaped dicts
    use_gate3: bool = True
    k_correct: bool = True


def _frame_data_from_payload(payload: GenericFrameReq,
                              force_idx: int = 0):
    """Payload → FrameData (identical dataclass to what load_frame_data
    returns). The rest of the pipeline (calibrate_frame + pool_frames)
    doesn't care where the FrameData came from."""
    from calib_lib import FrameData as _FD
    from scripts.util.projection import project_lidar_into_image as _proj
    img_rgb = _decode_image(payload.image_b64, payload.image_url)
    IH, IW = int(img_rgb.shape[0]), int(img_rgb.shape[1])
    if [IW, IH] != list(payload.camera.resolution):
        raise HTTPException(400, f'image {IW}x{IH} != camera.resolution '
                                    f'{payload.camera.resolution}')
    pts_xyzi = _decode_points(payload.points_xyzi_b64, payload.points_xyzi_list)
    K = np.array([[payload.camera.fc[0], 0.0, payload.camera.cc[0]],
                   [0.0, payload.camera.fc[1], payload.camera.cc[1]],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(payload.camera.dist_kb4, dtype=np.float64)
    T_cl = np.array(payload.camera.T_cam_lidar, dtype=np.float64)
    if T_cl.shape != (4, 4):
        raise HTTPException(400, f'T_cam_lidar must be 4x4, got {T_cl.shape}')
    is_fisheye = bool(np.any(np.abs(dist) > 1e-12))
    _, pts_cam, uv_full, z_full, int_full = _proj(
        pts_xyzi, K, T_cl, IW, IH,
        is_fisheye=is_fisheye, dist=dist if is_fisheye else None, z_min=0.5)

    # Hood polygon: drop points inside.
    n_dropped = 0
    if payload.hood_polygon_uv:
        import cv2
        poly = np.array(payload.hood_polygon_uv, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise HTTPException(400, 'hood_polygon_uv must be ≥3 (u,v) pairs')
        keep = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)),
                                                False) < 0
                          for (u, v) in uv_full], dtype=bool)
        n_dropped = int((~keep).sum())
        pts_cam  = pts_cam[keep]
        uv_full  = uv_full[keep]
        z_full   = z_full[keep]
        int_full = int_full[keep]

    # Optional pose bits. If missing, unit conversion downstream produces
    # camera-FRD numbers only (no setting.json-space additive delta).
    if payload.camera.R_cam_from_veh is not None:
        R_cv = np.asarray(payload.camera.R_cam_from_veh, dtype=np.float64)
    elif payload.camera.setting_rot_rad is not None:
        from scipy.spatial.transform import Rotation as _Rot
        roll, pitch, yaw = payload.camera.setting_rot_rad
        R_c2v = _Rot.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
        from calib_lib import R_TO_RDF as _R2RDF
        R_cv = _R2RDF @ R_c2v.T
    else:
        R_cv = np.eye(3)

    setting_rot = np.array(payload.camera.setting_rot_rad or [0, 0, 0],
                             dtype=np.float64)
    setting_mp  = np.array(payload.camera.setting_mp_m or [0, 0, 0],
                             dtype=np.float64)

    return _FD(fid=str(payload.frame_id or f'gen_{force_idx}'),
                frame_idx=force_idx,
                image_rgb=img_rgb,
                pts_cam=pts_cam, uv_hat=uv_full, z=z_full,
                intensity=int_full, K=K, dist=dist, T_cl=T_cl,
                IW=IW, IH=IH, delay_ms=0.0,
                R_cam_from_veh=R_cv,
                setting_rot=setting_rot, setting_mp=setting_mp,
                n_hood_dropped=n_dropped)


@app.post('/calibrate/generic/frame', tags=['calibrate'],
          description=(
"""Single-frame auto-calibration from a raw payload — no filesystem access.

Send image bytes + LiDAR points + camera intrinsics + LiDAR→cam extrinsic.
Server runs CalibNet2 + 40-tile fused BA on the frame and returns 6-DoF δ
(camera FRD, degrees + metres) plus the 6×6 information matrix H.

If you also pass `setting_rot_rad` + `setting_mp_m` (LOOM setting.json
[roll,pitch,yaw] / [x,y,z] for the vehicle-to-camera transform), the
response additionally carries `deltas` — the additive correction to write
back to setting.json — in the same [rot_x, rot_y, rot_z, mp_x, mp_y, mp_z]
form the `/calibrate/frame` endpoint uses.

Sub-pixel `theta` fisheye (Kannala-Brandt 4) via `camera.dist_kb4` (all
zeros = plain pinhole)."""))
def calibrate_generic_frame(req: GenericFrameReq):
    from scripts.util.projection import project_lidar_into_image  # noqa: F401
    from scipy.spatial.transform import Rotation  # noqa: F401
    if STATE['model'] is None:
        raise HTTPException(503, 'model not loaded yet')
    STATE['request_count'] += 1
    from calib_lib import calibrate_frame as _cf
    with STATE['lock']:
        fd = _frame_data_from_payload(req, force_idx=0)
        res = _cf(STATE['model'], fd, device=STATE['device'])
    resp = dict(
        status='success',
        fid=fd.fid,
        n_tiles=res.n_tiles_fused,
        n_points=res.n_points,
        n_hood_dropped=fd.n_hood_dropped,
        delta_cam_deg=res.delta_cam[:3].tolist(),
        delta_cam_m  =res.delta_cam[3:].tolist(),
        H_cam=res.H_cam.tolist(),
        image_size=[int(fd.IW), int(fd.IH)],
    )
    # setting.json-space additive delta only if pose was provided.
    if (req.camera.setting_rot_rad is not None
            and req.camera.setting_mp_m is not None):
        adj    = cam_delta_to_setting(res.delta_cam, fd.R_cam_from_veh)
        sd_raw = cam_sd_to_setting(res.sd(k_correct=False), fd.R_cam_from_veh)
        resp['deltas'] = adj
        resp['sd']     = sd_raw
    return resp


@app.post('/calibrate/generic/sequence', tags=['calibrate'],
          description=(
"""Multi-frame joint auto-calibration from raw payloads.

Runs /calibrate/generic/frame internally on each frame in `frames` (list of
1..F GenericFrameReq bodies) then pools per-frame information matrices
with inverse-variance weighting, χ²-gate (drop-3σ) and over-dispersion
correction (k · Σ scaling). Same math as /calibrate/sequence.

Use this when you have a short multi-frame clip; each additional frame
tightens σ by roughly √F, capped by k."""))
def calibrate_generic_sequence(req: GenericSeqReq):
    if STATE['model'] is None:
        raise HTTPException(503, 'model not loaded yet')
    if not req.frames:
        raise HTTPException(400, 'frames must contain at least one entry')
    STATE['request_count'] += 1
    from calib_lib import calibrate_frame as _cf, pool_frames as _pool
    t0 = time.time()
    per_frame = []
    with STATE['lock']:
        for i, f in enumerate(req.frames):
            payload = GenericFrameReq.model_validate(f)
            fd = _frame_data_from_payload(payload, force_idx=i)
            per_frame.append((fd, _cf(STATE['model'], fd, device=STATE['device'])))
    results = [r for (_fd, r) in per_frame]
    pooled = _pool(results, mode='gate3' if req.use_gate3 else 'sum',
                     gate_c=3.0, gate_iters=2)

    resp = dict(
        status='success',
        n_frames_input=len(req.frames),
        n_frames_used=pooled.n_frames,
        n_tiles_fused=pooled.n_tiles_fused,
        n_points=pooled.n_points,
        k_dispersion=float(pooled.k),
        delta_cam_deg=pooled.delta_cam[:3].tolist(),
        delta_cam_m  =pooled.delta_cam[3:].tolist(),
        H_cam=pooled.H_cam.tolist(),
        elapsed_s=round(time.time() - t0, 2),
    )
    # setting.json-space additive delta (uses first frame's pose bundle;
    # all frames share the same LOOM setting.json in normal use).
    first_fd = per_frame[0][0]
    if (req.frames[0].get('camera', {}).get('setting_rot_rad') is not None
            and req.frames[0].get('camera', {}).get('setting_mp_m') is not None):
        adj    = cam_delta_to_setting(pooled.delta_cam, first_fd.R_cam_from_veh)
        sd_raw = cam_sd_to_setting(pooled.sd(k_correct=req.k_correct),
                                    first_fd.R_cam_from_veh)
        resp['deltas'] = adj
        resp['sd']     = sd_raw
    return resp


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path,
                    default=Path(os.environ.get('CKPT', DEFAULT_CKPT)))
    ap.add_argument('--hood-mask-root', type=Path,
                    default=Path(os.environ.get('HOOD_MASK_ROOT',
                                                  DEFAULT_HOOD_MASK_ROOT)))
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8501)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    STATE['device'] = args.device
    STATE['hood_mask_root'] = args.hood_mask_root
    _load_model(args.ckpt)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
