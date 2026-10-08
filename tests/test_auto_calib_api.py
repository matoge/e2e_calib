"""Independent end-to-end test for the auto-calibration API.

Prereq: auto_calib_server running (default: http://127.0.0.1:8501).

Runs three phases:
  1. GET  /health                          — server is up and model loaded
  2. POST /calibrate/frame  (frame 0)      — single-frame numerical sanity
  3. POST /calibrate/sequence (F=10 pool)  — joint result within expected bands

Each check prints PASS / FAIL. Exit code 0 iff every check passes.

    python tests/test_auto_calib_api.py \\
        --seq /home/hfunaya/git/loom/backend/assets/woven_sequence/unilab_001/test01/sequence=ip654-lidar0-1432519511400022000-1432519516399877000

Ranges are calibrated to the current S3 cam6 checkpoint on the
unilab_001/test01 sequence. Manual LOOM calibration on this sequence
agreed with pitch ≈ +0.27°, which is the anchor of the "pass" band.
"""
from __future__ import annotations
import argparse, sys, json, math
import requests
import numpy as np

DEG = 180.0 / math.pi


def _check(name: str, ok: bool, detail: str = '') -> bool:
    marker = 'PASS' if ok else 'FAIL'
    print(f'  [{marker}] {name}' + (f'   {detail}' if detail else ''), flush=True)
    return ok


def _in_range(x: float, lo: float, hi: float) -> bool:
    return lo <= x <= hi


def _abs_le(x: float, thr: float) -> bool:
    return abs(x) <= thr


def test_health(api: str) -> bool:
    print('\n[1/3] GET /health', flush=True)
    r = requests.get(f'{api}/health', timeout=5)
    all_ok = True
    all_ok &= _check('status_code == 200', r.status_code == 200,
                     f'got {r.status_code}')
    j = r.json()
    all_ok &= _check('status == "ok"', j.get('status') == 'ok',
                     f'got {j.get("status")}')
    all_ok &= _check('model loaded', j.get('model') is not None,
                     f'{j.get("model")}')
    all_ok &= _check('device == "cuda"', j.get('device') == 'cuda',
                     f'got {j.get("device")}')
    return all_ok


def test_frame(api: str, seq: str) -> bool:
    print('\n[2/3] POST /calibrate/frame (frame_idx=0)', flush=True)
    r = requests.post(f'{api}/calibrate/frame',
                      json={'seq_dir': seq, 'frame_idx': 0,
                            'use_hood_mask': True},
                      timeout=60)
    all_ok = True
    all_ok &= _check('status_code == 200', r.status_code == 200,
                     f'got {r.status_code}: {r.text[:200]}')
    if r.status_code != 200:
        return False
    j = r.json()
    all_ok &= _check('status == "success"', j.get('status') == 'success')
    all_ok &= _check('n_tiles >= 20 (out of 40)', j.get('n_tiles', 0) >= 20,
                     f'{j.get("n_tiles")}/40')
    all_ok &= _check('n_points >= 2000', j.get('n_points', 0) >= 2000,
                     f'{j.get("n_points")}')
    all_ok &= _check('n_hood_dropped > 0 (mask applied)',
                     j.get('n_hood_dropped', 0) > 0,
                     f'{j.get("n_hood_dropped")} dropped')
    all_ok &= _check('delay_ms > 30 (compensation applied)',
                     j.get('delay_ms', 0) > 30,
                     f'{j.get("delay_ms")}ms')

    # Camera-frame δ: pitch (deg) is the primary axis for unilab_001
    dcam = j.get('delta_cam_deg', [0, 0, 0])
    pitch_cam = dcam[0]           # ω_x in cam FRD = camera pitch
    all_ok &= _check('camera pitch in [0.10°, 0.40°]',
                     _in_range(pitch_cam, 0.10, 0.40),
                     f'{pitch_cam:+.4f}°')

    # Setting-frame delta (rad) — sanity: within capture range (~0.5°)
    d = j['deltas']
    all_ok &= _check('|setting Δrot| < 0.02 rad (~1.1°)',
                     all(abs(d[k]) < 0.02 for k in ['rot_x','rot_y','rot_z']),
                     f'rot=({d["rot_x"]:+.4f},{d["rot_y"]:+.4f},{d["rot_z"]:+.4f}) rad')
    all_ok &= _check('|setting Δmp| < 0.10 m',
                     all(abs(d[k]) < 0.10 for k in ['mp_x','mp_y','mp_z']),
                     f'mp=({d["mp_x"]:+.4f},{d["mp_y"]:+.4f},{d["mp_z"]:+.4f}) m')
    return all_ok


def test_sequence(api: str, seq: str) -> bool:
    print('\n[3/3] POST /calibrate/sequence (F=10, stride=5)', flush=True)
    r = requests.post(f'{api}/calibrate/sequence',
                      json={'seq_dir': seq, 'stride': 5, 'max_frames': 10,
                            'use_hood_mask': True, 'use_gate3': True,
                            'return_per_frame': True},
                      timeout=120)
    all_ok = True
    all_ok &= _check('status_code == 200', r.status_code == 200,
                     f'got {r.status_code}')
    if r.status_code != 200:
        return False
    j = r.json()
    all_ok &= _check('status == "success"', j.get('status') == 'success')
    all_ok &= _check('elapsed < 30 s', j.get('elapsed_s', 999) < 30,
                     f'{j.get("elapsed_s")}s')
    all_ok &= _check('n_frames_used == 10', j.get('n_frames_used') == 10,
                     f'{j.get("n_frames_used")}/10')
    all_ok &= _check('n_tiles_total >= 200', j.get('n_tiles_total', 0) >= 200,
                     f'{j.get("n_tiles_total")}')
    all_ok &= _check('n_points_total >= 30000',
                     j.get('n_points_total', 0) >= 30000,
                     f'{j.get("n_points_total")}')

    # k dispersion sanity
    k = j.get('k_dispersion', 0)
    all_ok &= _check('0.1 ≤ k ≤ 5.0', _in_range(k, 0.1, 5.0), f'k={k}')

    # camera-frame pitch (the anchor: manual calib on this seq agrees ≈ 0.27°)
    dcam = j['delta_cam_deg']
    pitch_cam = dcam[0]
    all_ok &= _check('camera pitch in [0.20°, 0.35°]',
                     _in_range(pitch_cam, 0.20, 0.35),
                     f'{pitch_cam:+.4f}°')
    all_ok &= _check('|camera yaw| < 0.20°',
                     _abs_le(dcam[1], 0.20),
                     f'{dcam[1]:+.4f}°')
    all_ok &= _check('|camera roll| < 0.15°',
                     _abs_le(dcam[2], 0.15),
                     f'{dcam[2]:+.4f}°')

    # Post-k σ tightens by ≥ 4× vs single-frame (√10 ideal = 3.16, but F=10 with
    # k≈0.4 gives even tighter). Absolute bounds:
    sd = j['sd']
    all_ok &= _check('σ pitch after k-corr < 0.010 rad (~0.6°)',
                     sd['rot_y'] < 0.010, f'{sd["rot_y"]:.5f} rad')
    all_ok &= _check('σ pitch after k-corr > 0 (finite)',
                     sd['rot_y'] > 0, f'{sd["rot_y"]:.5f}')

    # Per-frame consistency: pitch across 10 frames within 0.15° range
    if 'per_frame' in j:
        pitches = [fr['delta_cam'][0] for fr in j['per_frame']
                   if 'delta_cam' in fr]
        if pitches:
            span = max(pitches) - min(pitches)
            all_ok &= _check(
                'per-frame pitch spread < 0.15°',
                span < 0.15,
                f'{min(pitches):+.4f} … {max(pitches):+.4f}° (Δ={span:.4f}°)')

    # Setting-frame Δ (rad + m): additive-safe bounds
    d = j['deltas']
    all_ok &= _check('|setting Δrot_y| < 0.01 rad (~0.6°)',
                     abs(d['rot_y']) < 0.01,
                     f'{d["rot_y"]:+.5f} rad')
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://127.0.0.1:8501')
    ap.add_argument('--seq', default=(
        '/home/hfunaya/git/loom/backend/assets/woven_sequence/'
        'unilab_001/test01/sequence=ip654-lidar0-1432519511400022000-1432519516399877000'
    ))
    args = ap.parse_args()

    print(f'auto_calib_server test  ·  api={args.api}  ·  seq={args.seq.split("/")[-1][:60]}…')
    ok1 = test_health(args.api)
    ok2 = test_frame(args.api, args.seq)
    ok3 = test_sequence(args.api, args.seq)

    all_ok = ok1 and ok2 and ok3
    print()
    print(f'RESULT: {"ALL PASS" if all_ok else "FAIL"}   '
          f'(health={"PASS" if ok1 else "FAIL"} '
          f'frame={"PASS" if ok2 else "FAIL"} '
          f'sequence={"PASS" if ok3 else "FAIL"})', flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
