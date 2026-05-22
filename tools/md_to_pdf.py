"""Render a blog markdown file to a single PDF for Confluence embedding.

  md → HTML (+ inline math rendered via matplotlib mathtext as SVG/PNG)
     → weasyprint → PDF

Why:
  Confluence's wiki / storage pipeline mangles LaTeX source, and rendering
  every equation through codecogs depends on a flaky external service.
  matplotlib's mathtext is local, supports \\mathbf, \\Sigma_\\sigma,
  \\text{...} via \\mathrm, \\frac, \\tfrac, \\le, \\partial, \\arg\\min,
  \\mathbb, etc. — enough for our blogs.

Usage:
  python tools/md_to_pdf.py docs/blog/2026-05-20_principled_ml_calib.md \\
      --out docs/blog/2026-05-20_principled_ml_calib.pdf

Notes:
  - Local image references (![alt](relpath.png)) are resolved relative to
    the markdown file and embedded as data URIs so the PDF is portable.
  - Math rendering falls back to verbatim source if mathtext rejects the
    expression; we log a warning.
"""
from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['svg.fonttype'] = 'path'
import matplotlib.pyplot as plt

import markdown


CSS = """
@page {
  size: A4;
  margin: 16mm 14mm 16mm 14mm;
  @bottom-right {
    content: counter(page) " / " counter(pages);
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 8.5pt;
    color: #666;
  }
}
body {
  font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
  font-size: 9pt;
  line-height: 1.38;
  color: #222;
  text-align: justify;
  hyphens: auto;
}
.title-block {
  column-span: all;
  margin-bottom: 10pt;
  border-bottom: 1px solid #ccc;
  padding-bottom: 6pt;
}
.title-block h1 { text-align: center; }
.title-meta { color: #666; font-size: 9pt; margin-top: 2pt; text-align: center; }
.body-cols {
  column-count: 2;
  column-gap: 6mm;
  column-rule: 0px solid #eee;
}
h1 { font-size: 16pt; margin: 0 0 4pt; line-height: 1.18; }
h2 {
  font-size: 11.5pt;
  margin: 12pt 0 4pt;
  border-bottom: 1px solid #ddd;
  padding-bottom: 1pt;
  break-after: avoid;
}
h3 { font-size: 10pt; margin: 9pt 0 3pt; break-after: avoid; }
h4 { font-size: 9.5pt; margin: 7pt 0 2pt; break-after: avoid; font-style: italic; }
p, li { margin: 3pt 0; }
ul, ol { margin: 3pt 0 3pt 13pt; padding: 0; }
img.math-inline { vertical-align: -0.22em; height: 1.0em; }
img.math-block {
  display: block;
  margin: 5pt auto;
  max-width: 100%;
}
img:not(.math-inline):not(.math-block) {
  display: block;
  max-width: 100%;
  margin: 5pt auto;
  page-break-inside: avoid;
}
.fullwidth {
  column-span: all;
  margin: 8pt 0;
}
.fullwidth img { max-width: 100%; }
pre, code {
  font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 7.5pt;
}
pre {
  background: #f6f6f6;
  border: 1px solid #e1e1e1;
  border-radius: 3px;
  padding: 4pt 6pt;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
  line-height: 1.25;
}
code { background: #f0f0f0; padding: 0 2px; border-radius: 2px; font-size: 8pt; }
pre code { background: transparent; padding: 0; }
table {
  border-collapse: collapse;
  margin: 5pt 0;
  width: 100%;
  font-size: 7.8pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #ccc;
  padding: 2pt 4pt;
  text-align: left;
  vertical-align: top;
  line-height: 1.25;
}
th { background: #f0f0f0; }
blockquote {
  border-left: 2px solid #bbb;
  margin: 5pt 0;
  padding: 1pt 8pt;
  color: #444;
  background: #f9f9f9;
  font-size: 8.6pt;
}
hr { border: none; border-top: 1px solid #ccc; margin: 8pt 0; }
"""


def render_math_to_svg_b64(tex: str, *, display: bool) -> str | None:
    """Render LaTeX → vector SVG (base64 data URI) using matplotlib mathtext
    with Computer Modern fonts. Returns None on failure.
    """
    fontsize = 13 if display else 11
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)
    text = f"${tex}$"
    try:
        t = fig.text(0, 0, text, fontsize=fontsize)
        fig.canvas.draw()
        bbox = t.get_window_extent()
        w_in = bbox.width / fig.dpi + 0.03
        h_in = bbox.height / fig.dpi + 0.03
        plt.close(fig)
        fig = plt.figure(figsize=(w_in, h_in))
        fig.patch.set_alpha(0.0)
        fig.text(0.005, 0.5, text, fontsize=fontsize, va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight',
                    pad_inches=0.0, transparent=True)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/svg+xml;base64,{b64}'
    except Exception as e:
        plt.close(fig)
        sys.stderr.write(f'  [warn] math render failed for {tex!r}: {e}\n')
        return None


def replace_math(md_text: str) -> str:
    def block_repl(m: re.Match) -> str:
        src = m.group(1).strip()
        uri = render_math_to_svg_b64(src, display=True)
        if uri is None:
            return m.group(0)
        return f'<p style="text-align:center;"><img class="math-block" src="{uri}" alt="{_html_attr(src)}" /></p>'

    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', block_repl, md_text)

    def inline_repl(m: re.Match) -> str:
        src = m.group(1).strip()
        uri = render_math_to_svg_b64(src, display=False)
        if uri is None:
            return m.group(0)
        return f'<img class="math-inline" src="{uri}" alt="{_html_attr(src)}" />'

    # Inline math: allow the body to contain backslash-escapes etc., but no
    # newline (otherwise we eat across paragraphs). Use a hand-rolled scan
    # that respects `\$` escapes and treats fenced `$...$` greedily-but-
    # newline-bounded — this catches rows in markdown tables where the
    # previous regex's `[^$\n]+?` was too restrictive on edge cases.
    out = []
    i = 0
    n = len(md_text)
    while i < n:
        ch = md_text[i]
        if ch == '$':
            # find matching $ on the same line, skipping \$
            j = i + 1
            while j < n and md_text[j] != '\n':
                if md_text[j] == '\\':
                    j += 2
                    continue
                if md_text[j] == '$':
                    break
                j += 1
            if j < n and md_text[j] == '$' and j > i + 1:
                src = md_text[i + 1:j].strip()
                uri = render_math_to_svg_b64(src, display=False)
                if uri is not None:
                    out.append(
                        f'<img class="math-inline" src="{uri}" '
                        f'alt="{_html_attr(src)}" />')
                    i = j + 1
                    continue
            out.append(ch)
            i += 1
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def _html_attr(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;'))


def embed_local_images(html: str, md_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        full = m.group(0)
        src = m.group(1)
        if src.startswith(('http://', 'https://', 'data:')):
            return full
        img_path = (md_dir / src).resolve()
        if not img_path.is_file():
            sys.stderr.write(f'  [warn] image not found: {img_path}\n')
            return full
        ext = img_path.suffix.lstrip('.').lower()
        mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'gif': 'image/gif', 'svg': 'image/svg+xml'}.get(ext, 'image/png')
        b64 = base64.b64encode(img_path.read_bytes()).decode('ascii')
        return full.replace(f'src="{src}"', f'src="data:{mime};base64,{b64}"')

    return re.sub(r'<img[^>]+src="([^"]+)"[^>]*/?>', repl, html)


FRONTMATTER_RE = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)


def split_frontmatter(md_text: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(md_text)
    if not m:
        return {}, md_text
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, md_text[m.end():]


def md_to_html_doc(md_path: Path) -> str:
    raw = md_path.read_text(encoding='utf-8')
    fm, body = split_frontmatter(raw)

    # Strip an in-body H1 if it duplicates the title — but here we want it
    # in the PDF since there's no Confluence-rendered title above.
    body = replace_math(body)

    html_body = markdown.markdown(
        body,
        extensions=['fenced_code', 'tables', 'footnotes', 'attr_list',
                    'sane_lists', 'codehilite'],
        extension_configs={
            'codehilite': {'guess_lang': False, 'noclasses': True},
        },
        output_format='html5',
    )
    html_body = embed_local_images(html_body, md_path.parent)

    title = fm.get('title') or md_path.stem
    meta = []
    if 'date' in fm: meta.append(fm['date'])
    if 'author' in fm: meta.append(fm['author'])
    meta_html = ' &middot; '.join(meta)

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8" />'
        f'<title>{_html_attr(title)}</title>'
        f'<style>{CSS}</style></head><body>'
        '<div class="title-block">'
        f'<div class="title-meta">{meta_html}</div>'
        '</div>'
        f'{html_body}'
        '</body></html>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('md_file', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--html-out', type=Path, default=None,
                    help='also write the intermediate HTML for debugging')
    args = ap.parse_args()

    html = md_to_html_doc(args.md_file)
    if args.html_out:
        args.html_out.write_text(html, encoding='utf-8')
        print(f'[md2pdf] wrote html → {args.html_out}')

    from weasyprint import HTML
    args.out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(args.md_file.parent.resolve())).write_pdf(
        str(args.out))
    print(f'[md2pdf] wrote pdf  → {args.out}')


if __name__ == '__main__':
    main()
