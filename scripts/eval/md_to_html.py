"""Convert one or more .md files to standalone .html via python-markdown.
Static output — embeds the existing report.css for visual parity with other docs.
Usage:
    python scripts/eval/md_to_html.py docs/unified_progression.md docs/dynamic_object_sigma.md
"""
import argparse, sys, pathlib
import markdown

TEMPLATE = """<!doctype html>
<html lang="ja"><head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="assets/report.css">
<style>
body {{ max-width: 980px; margin: 24px auto; padding: 0 24px;
       font-family: -apple-system, BlinkMacSystemFont, 'Hiragino Sans', sans-serif;
       line-height: 1.7; color: #222; }}
img {{ max-width: 100%; height: auto; border-radius: 6px;
       box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 14px 0; }}
table {{ border-collapse: collapse; margin: 14px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 12px; }}
th {{ background: #f6f6f6; }}
pre, code {{ background: #f6f6f6; padding: 2px 6px; border-radius: 3px;
             font-family: 'JetBrains Mono', Menlo, monospace; font-size: 0.9em; }}
pre {{ padding: 12px; overflow-x: auto; line-height: 1.4; }}
h1 {{ border-bottom: 2px solid #444; padding-bottom: 8px; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 36px; }}
</style>
</head><body>
{body}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='.md files')
    args = ap.parse_args()

    md = markdown.Markdown(
        extensions=['fenced_code', 'tables', 'codehilite', 'toc',
                    'sane_lists', 'extra'])

    for p in args.paths:
        src = pathlib.Path(p)
        if not src.exists():
            print(f'  skip (missing): {src}')
            continue
        text = src.read_text()
        html_body = md.convert(text)
        title = src.stem.replace('_', ' ')
        out = src.with_suffix('.html')
        out.write_text(TEMPLATE.format(title=title, body=html_body))
        print(f'  wrote {out}')
        md.reset()


if __name__ == '__main__':
    main()
