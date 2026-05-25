#!/usr/bin/env python3
"""Render a blog markdown file to a self-contained HTML page that uses
docs/assets/report.css for styling — same look as the rest of the e2e_calib
report hub. No Confluence, no nl2br, no per-image alt-text caption noise.

Usage:
    tools/render_blog_html.py docs/blog/2026-05-22_subpixel_calib.md
        -> docs/blog/2026-05-22_subpixel_calib.html
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import markdown


HEAD = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="../assets/report.css">
<style>
  /* blog-local: keep prose narrow, let figures breathe */
  body {{ background: var(--bg); }}
  main.r-blog {{
    max-width: var(--col);
    margin: 0 auto;
    padding: 72px 36px 120px;
  }}
  main.r-blog > p > img,
  main.r-blog > figure > img {{
    width: 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
  }}
  /* image-only table cell -> figure */
  main.r-blog table td > img,
  main.r-blog table td > p > img {{
    width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }}
  /* hide alt-text "captions" markdown emits as <em> after <img> */
  main.r-blog table td > p > em:only-child {{ display: none; }}
  main.r-blog > p > em:only-child {{
    display: block;
    text-align: center;
    color: var(--ink-dim);
    font-size: 13.5px;
    margin-top: 8px;
  }}
  /* tables that are really just image grids */
  main.r-blog table {{
    table-layout: fixed;
  }}
  main.r-blog table th:empty {{ display: none; }}
  main.r-blog table thead tr:has(th:empty:only-child),
  main.r-blog table thead:has(tr > th:empty) {{ display: none; }}
  /* eyebrow above title */
  .r-eyebrow {{ margin-bottom: 22px; }}
  /* code blocks */
  main.r-blog pre {{ font-size: 12.5px; }}
  /* blockquote (used as headline callout) */
  main.r-blog > blockquote {{
    border-left: 4px solid var(--accent);
    background: var(--panel);
    padding: 14px 22px;
    margin: 0 0 36px;
    color: var(--ink);
    font-style: normal;
    font-size: 15.5px;
    line-height: 1.7;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
  }}
</style>
</head>
<body>
<main class="r-blog">
"""

FOOT = """\
</main>
</body>
</html>
"""


def first_h1(md_text: str) -> str:
    m = re.search(r"^#\s+(.+)$", md_text, flags=re.MULTILINE)
    return m.group(1).strip() if m else "blog"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_file", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    md_text = args.md_file.read_text(encoding="utf-8")
    title = first_h1(md_text)

    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )

    # Markdown emits <img alt="x" src=... /> followed by alt-text shown via
    # itself; if the alt text is the slug (e.g. "hero17"), suppress it.
    # Simpler: drop alt="..." entirely so no slug ever shows.
    body_html = re.sub(r'(<img[^>]*?)\salt="[^"]*"', r"\1", body_html)

    out = args.out or args.md_file.with_suffix(".html")
    out.write_text(
        HEAD.format(title=title) + body_html + FOOT, encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
