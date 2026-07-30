# webui_kb_fit — TSS4 / Woven KB4 fisheye tools + crop designer

Small Flask webapp + companion CLIs to author pinhole crops from KB4
fisheye sequences so they can be trained with SplatAD / gsplat.

Two apps live here:

- **`app.py`** (port 5007) — TSS4 KB4 fit + crop designer (main).
- **`polygon_edit_app.py`** (port 5008) — polygon editor for
  dashboard / bonnet / kerare masks.

## Quick start

```bash
# Python must be the pyenv 3.10.4 (system 3.8 breaks on PEP 604)
/home/hfunaya/.pyenv/versions/3.10.4/bin/python \
    scripts/webui_kb_fit/app.py   # 0.0.0.0:5007
```

If port 5007 is bound already (e.g. a stale instance):

```bash
fuser -n tcp 5007          # show PID
kill -9 <PID>              # or: pkill -9 -f 'webui_kb_fit/app.py'
```

Access from a laptop over SSH:

```bash
ssh -N -L 15007:localhost:5007 dgx2
# browser → http://localhost:15007/crop_designer
```

## Crop designer (`/crop_designer`)

Rectifies one frame to a balance=1.0 stretched pinhole canvas
(default 10000×5083; auto-scales H to the input aspect for canary
sequences, e.g. 10000×5625 for 3840×2160) and lets you drag a
symmetric rectangle around the target region.

- **vehicle** selector at the top of the toolbar. Populated from
  `/api/crop_options`:
  - `tss4 / 248` — TSS4 raw sequence 20230612_001946 (uses
    `loom/backend/assets/woven_sequence/llinking_26/recalibration.json`).
  - `canary / ipXXX` — every Woven canary sequence under
    `CANARY_SEQ_ROOTS` in `app.py` that has both
    `setting-<ipXXX>.json` and `tss4_fcm/*.jpg`. The vehicle's own
    `setting-<vehicle>.json` fcm block is used for KB4 params.
- **Reload image** re-fetches with the current selection.
- **Save → JSON** writes rect-canvas coords to
  `_outputs/crop_boxes.json` (overwrites; rename per-vehicle after saving,
  see [`_outputs/crop_boxes_ip607.json`](_outputs/crop_boxes_ip607.json)
  for an example).

Adding a new canary root: edit `CANARY_SEQ_ROOTS` in `app.py`.

## Bake the designed crop into a pinhole dataset

```bash
python scripts/webui_kb_fit/undistort_and_crop.py \
    --seq-dir /raid/.../sequence=ip607-lidar0-.../ \
    --vehicle ip607 \
    --out-dir /raid/.../ip607_designed \
    --rect-W 10000 --rect-H 5625 \
    --crop-x0 456 --crop-y0 1890 --crop-x1 9544 --crop-y1 4457
```

- `--seq-dir` + `--vehicle ipXXX` reads `<seq>/setting-<vehicle>.json`
  (canary layout, this is the ip607/ip708 path).
- `--recalib <path>` + `--vehicle 247/248/249` reads a TSS4-style
  `recalibration.json` (the original path).
- Output: per-frame `<stem>.jpg` (undistorted+cropped pinhole) plus
  `K_rect.json` with the crop-adjusted `fx/fy/cx/cy/width/height` and
  the source KB4 for reference.

## Hand off to SplatAD

The SplatAD kick script (`scripts/splatad_kb/_kick_splatad_woven.sh`)
via `woven_dataparser.py` expects a Woven-canary-shaped sequence with
the pinhole rectify under `_pinhole/`:

```bash
# after undistort_and_crop.py:
BAKED=/raid/.../ip607_designed
PINHOLE=$SEQ_DIR/_pinhole/camera/front_camera
mkdir -p $PINHOLE
ln -f $BAKED/*.jpg $PINHOLE/                                  # or cp
python3 -c "
import json
d = json.load(open('$BAKED/K_rect.json'))
json.dump({k:d[k] for k in ('fx','fy','cx','cy','width','height')},
          open('$PINHOLE/intrinsics.json','w'), indent=2)
"
# then kick SplatAD:
scripts/splatad_kb/_kick_splatad_woven.sh $SEQ_DIR ip607
```

Full end-to-end (SAM3 mask → GICP → projection → KB→pinhole → GS)
lives in `loom/tools/woven_sequence_gs/`; this webapp is the manual
crop step for that pipeline.

## Files

| file | purpose |
|---|---|
| `app.py` | Flask webapp (TSS4 KB4 fit + crop_designer + lidar3d, port 5007) |
| `templates/crop_designer.html` | rect canvas + symmetric crop rectangle |
| `templates/index.html` | TSS4 KB4 manual fit / GN sweep UI |
| `templates/lidar3d.html` | LiDAR viewer |
| `undistort_and_crop.py` | KB → balance=1.0 pinhole → rect crop → K_rect.json |
| `undistort_fcm_recalib.py` | bare KB → pinhole (no crop), for previewing recalib |
| `polygon_edit_app.py` | polygon editor for dashboard/bonnet/kerare masks (port 5008) |
| `_outputs/_dashboard_polygons.json` | polygon set authored by the editor |
| `_outputs/crop_boxes_ip607.json` | crop for the ip607 canary pinhole dataset |
