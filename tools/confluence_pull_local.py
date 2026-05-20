"""Pull a Confluence page (body + attachments) to a local directory and
write a self-contained HTML you can open with VS Code.

Usage:
  python tools/confluence_pull_local.py <page_id> [--out DIR]

Env: CONFLUENCE_TOKEN must be set.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'https://confluence.tri-ad.tech'


def api_get(path: str, token: str) -> bytes:
    req = urllib.request.Request(BASE + path,
        headers={'Authorization': f'Bearer {token}',
                  'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def download(url: str, token: str, dst: Path):
    req = urllib.request.Request(url,
        headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        dst.write_bytes(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('page_id')
    ap.add_argument('--out', default='/tmp/confluence_local')
    args = ap.parse_args()
    token = os.environ['CONFLUENCE_TOKEN']

    out = Path(args.out) / args.page_id
    (out / 'attachments').mkdir(parents=True, exist_ok=True)

    # 1. body.view = Confluence's server-side rendered HTML
    page = json.loads(api_get(
        f'/rest/api/content/{args.page_id}?expand=body.view,version,space',
        token))
    title = page['title']
    html_body = page['body']['view']['value']
    print(f'pulled page "{title}" v{page["version"]["number"]}')

    # 2. attachments — paginate
    atts = []
    start, limit = 0, 200
    while True:
        page_at = json.loads(api_get(
            f'/rest/api/content/{args.page_id}/child/attachment'
            f'?start={start}&limit={limit}&expand=version', token))
        atts.extend(page_at['results'])
        if len(page_at['results']) < limit:
            break
        start += limit
    print(f'  {len(atts)} attachment(s)')

    # Map attachment title (filename) → local path. Confluence escapes
    # spaces / special chars in URLs; rewrite both the url-encoded and
    # raw forms.
    local_for: dict[str, str] = {}
    for a in atts:
        name = a['title']
        dl = a['_links']['download']  # /download/attachments/<id>/<name>?...
        local = out / 'attachments' / name
        try:
            download(BASE + dl, token, local)
            local_for[name] = f'attachments/{name}'
            print(f'  ↓ {name}  ({local.stat().st_size} bytes)')
        except Exception as e:
            print(f'  [warn] failed to fetch {name}: {e}')

    # 3. rewrite all <img src=...> in body.view to local file refs.
    # body.view's <img> usually points to /download/attachments/.../<name>?...
    def repl_img_src(m: re.Match) -> str:
        url = m.group(1)
        # try every attachment name in the URL
        for name, local in local_for.items():
            if name in url or urllib.parse.quote(name) in url:
                return f'src="{local}"'
        return m.group(0)
    html_body = re.sub(r'src="([^"]+)"', repl_img_src, html_body)

    # 4. wrap in a minimal HTML so VS Code preview is clean
    html = f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 920px;
         margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #222; }}
  h1, h2, h3 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.2em;
                margin-top: 1.5em; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
  pre, code {{ background: #f6f6f6; padding: 0.2em 0.4em; border-radius: 3px; }}
  pre {{ padding: 0.7em; overflow-x: auto; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
</style>
</head><body>
<h1>{title}</h1>
{html_body}
</body></html>
'''
    out_html = out / 'index.html'
    out_html.write_text(html, encoding='utf-8')
    print(f'wrote → {out_html}')
    print()
    print(f'  open with VS Code:  code {out_html}')
    print(f'  or in browser:      file://{out_html}')


if __name__ == '__main__':
    main()
