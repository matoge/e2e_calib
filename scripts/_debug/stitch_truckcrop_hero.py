"""Stitch 5-scene truck-area crops (3 rows × 5 cols) into a single hero
image for the 1-frame BA blog post.

Reads each scene's ba_reproj_overlay_truckcrop.png (3 vertical stack)
and concatenates them horizontally.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from PIL import Image

EXP = 'km_wv_wm_n4_img128_cs256_512_200ep_dgx1_16gpu_resume'
SCENES = [
    'points_ip664_D_20260226_224648_d005_3000_3020',
    'points_ip664_D_20260301_222527_d006_800_820',
    'points_ip664_D_20260304_231950_d007-mdc_IWATESAN_inside_2',
    'points_ip664_D_20260404_041811_d005_510_530',
    'points_ip664_D_20260405_232105_d002_350_370',
]
SRC_ROOT = REPO / 'experiments' / EXP / '_eval_vis' / 'multi_scene'
OUT = REPO / 'docs' / 'assets' / '2026-05-18_one_frame_ba' / 'ba_truckcrop_hero.png'

panels = []
for s in SCENES:
    p = SRC_ROOT / s / 'ba_reproj_overlay_truckcrop.png'
    if not p.is_file():
        print(f'  missing: {p}')
        continue
    panels.append(Image.open(p).convert('RGB'))

if not panels:
    print('no panels found')
    sys.exit(1)

# All panels have the same height (3 × 1024 stacked). Resize widths to
# match the smallest if there's any mismatch.
h = min(p.height for p in panels)
w_total = sum(p.width for p in panels)
GAP = 8
canvas = Image.new('RGB', (w_total + GAP * (len(panels) - 1), h), 'black')
x = 0
for p in panels:
    canvas.paste(p, (x, 0))
    x += p.width + GAP

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)
print(f'wrote → {OUT}  size={canvas.size}')
