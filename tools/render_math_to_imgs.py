"""Pre-process a markdown file: render every $$...$$ block and every $...$
inline equation to a PNG via codecogs.com (LaTeX → PNG renderer) and
replace the equation in-place with an image reference.

Why: Confluence wiki markup chokes on `{r}` etc. inside LaTeX source
because `{...}` is its macro syntax. Replacing equations with images
side-steps the issue and renders cleanly. We use codecogs because:
  - matplotlib mathtext doesn't support `\\begin{bmatrix}`, `\\boxed`,
    `\\boldsymbol`, etc.
  - Installing TeX Live locally is heavy.
codecogs accepts URL-encoded LaTeX and returns a PNG.

Usage:
  python tools/render_math_to_imgs.py docs/blog/foo.md \\
      --assets-dir docs/assets/foo/eqs

Output:
  - <md>.expanded.md
  - <assets_dir>/eq_NNN.png
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://latex.codecogs.com/png.image"
DPI_DISPLAY = 200
DPI_INLINE = 160


def render_math(tex_src: str, out_png: Path, *, display: bool):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sub = re.sub(r'\s+', ' ', tex_src).strip()
    if display:
        sub = r'\displaystyle ' + sub
    dpi = DPI_DISPLAY if display else DPI_INLINE
    encoded = urllib.parse.quote(sub, safe='')
    url = f'{ENDPOINT}?\\dpi{{{dpi}}}\\bg{{white}}{encoded}'
    # codecogs occasionally throttles; small retry.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if data[:8] == b'\x89PNG\r\n\x1a\n':
                out_png.write_bytes(data)
                return
            raise ValueError(f'response not PNG (head={data[:16]!r})')
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(0.4 * (attempt + 1))


def process(md_path: Path, assets_dir: Path) -> Path:
    text = md_path.read_text(encoding='utf-8')
    counter = 0

    def repl_block(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        out_png = assets_dir / f'eq_{counter:03d}.png'
        try:
            render_math(m.group(1), out_png, display=True)
        except Exception as e:
            print(f'  [warn] block #{counter} render failed: {e}', file=sys.stderr)
            return m.group(0)
        return f'\n\n![equation]({_relpath(out_png, md_path.parent)})\n\n'

    text = re.sub(r'\$\$([\s\S]+?)\$\$', repl_block, text)

    def repl_inline(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        out_png = assets_dir / f'eq_{counter:03d}.png'
        try:
            render_math(m.group(1), out_png, display=False)
        except Exception as e:
            print(f'  [warn] inline #{counter} render failed: {e}', file=sys.stderr)
            return m.group(0)
        return f'![eq]({_relpath(out_png, md_path.parent)})'

    text = re.sub(r'(?<!\$)\$([^$\n]+?)\$(?!\$)', repl_inline, text)

    out_md = md_path.with_suffix('.expanded.md')
    out_md.write_text(text, encoding='utf-8')
    print(f'rendered {counter} equation(s) → {assets_dir}')
    print(f'wrote rewritten md → {out_md}')
    return out_md


def _relpath(target: Path, start: Path) -> str:
    target = target.resolve()
    start = start.resolve()
    common = Path(*[a for a, b in zip(target.parts, start.parts) if a == b])
    up = len(start.parts) - len(common.parts)
    down = target.parts[len(common.parts):]
    return '/'.join(['..'] * up + list(down))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('md_file', type=Path)
    ap.add_argument('--assets-dir', type=Path, required=True)
    args = ap.parse_args()
    process(args.md_file, args.assets_dir)


if __name__ == '__main__':
    main()
