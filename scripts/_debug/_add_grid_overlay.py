"""Overlay a faint 32-px grid on an existing INIT/FIT overlay jpg.

The overlay jpg is a vertical stack of N rows (each a full-frame 3840×1944
image) with ~16 px of caption between rows.  We detect dark "non-image" gaps
to find row boundaries, then draw a 32-px grid only inside the image rows.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def find_image_rows(arr: np.ndarray, ih: int = 1944) -> list[tuple[int, int]]:
    """Return [(y0,y1), ...] of image rows by scanning row-mean brightness."""
    H, W = arr.shape[:2]
    gray = arr.mean(axis=(1, 2)) if arr.ndim == 3 else arr.mean(axis=1)
    rows = []
    y = 0
    while y < H:
        # find next non-blank row
        while y < H and gray[y] < 5:
            y += 1
        if y >= H:
            break
        y0 = y
        # scan ih pixels (assume each row is exactly ih tall)
        y1 = min(H, y0 + ih)
        rows.append((y0, y1))
        y = y1
    # filter rows with reasonable height
    rows = [(a, b) for a, b in rows if (b - a) >= ih - 4]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--iw', type=int, default=3840)
    ap.add_argument('--ih', type=int, default=1944)
    ap.add_argument('--color', type=str, default='cyan',
                    help='grid color (PIL named or "#rrggbb")')
    ap.add_argument('--alpha', type=int, default=70,
                    help='0-255, line alpha')
    ap.add_argument('--major-every', type=int, default=128,
                    help='draw a brighter line every this many image-px '
                         '(0 to disable)')
    args = ap.parse_args()

    out = args.out or args.src.with_name(args.src.stem + f'_grid{args.grid}.jpg')

    im = Image.open(args.src).convert('RGBA')
    arr = np.array(im)
    H, W = arr.shape[:2]
    rows = find_image_rows(arr, ih=args.ih)
    print(f'[grid] src={args.src.name}  HxW={H}x{W}')
    print(f'[grid] detected {len(rows)} image rows: {rows}')

    overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # convert color name → rgb
    from PIL import ImageColor
    rgb = ImageColor.getrgb(args.color)
    minor_rgba = (*rgb, args.alpha)
    major_rgba = (*rgb, min(255, args.alpha * 2))

    for (y0, y1) in rows:
        # image is letterboxed to fit panel width W; if W != iw, scale grid
        sx = W / args.iw
        sy = (y1 - y0) / args.ih
        # vertical lines
        for gx in range(0, args.iw + 1, args.grid):
            x = int(round(gx * sx))
            major = (args.major_every > 0 and gx % args.major_every == 0)
            draw.line([(x, y0), (x, y1 - 1)],
                      fill=major_rgba if major else minor_rgba,
                      width=1)
        # horizontal lines
        for gy in range(0, args.ih + 1, args.grid):
            y = int(round(y0 + gy * sy))
            major = (args.major_every > 0 and gy % args.major_every == 0)
            draw.line([(0, y), (W - 1, y)],
                      fill=major_rgba if major else minor_rgba,
                      width=1)

    blended = Image.alpha_composite(im, overlay).convert('RGB')
    blended.save(out, quality=92)
    print(f'[grid] wrote {out}')


if __name__ == '__main__':
    main()
