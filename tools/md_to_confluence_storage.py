"""md → Confluence storage XHTML, then PUT to update an existing page.

Why: representation="wiki" silently mis-renders standard Markdown
(lists, headings, images all break). Going through storage XHTML keeps
the structure intact and lets us reference already-uploaded
attachments via <ac:image><ri:attachment .../></ac:image>.

Usage:
  python tools/md_to_confluence_storage.py update <page_id> <md>
  python tools/md_to_confluence_storage.py preview <md> [--out PATH]

Env: CONFLUENCE_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import markdown

BASE = 'https://confluence.tri-ad.tech'


def md_to_html(md_text: str) -> str:
    # Confluence shows the page title at the top of the rendered page,
    # so an in-body H1 with the same text would duplicate it. Strip a
    # leading "# title" line if present.
    lines = md_text.lstrip().splitlines()
    if lines and lines[0].startswith('# '):
        md_text = '\n'.join(lines[1:]).lstrip('\n')
    return markdown.markdown(
        md_text,
        extensions=['fenced_code', 'tables', 'footnotes', 'attr_list',
                    'sane_lists'],
        output_format='xhtml',
    )


def rewrite_imgs(html: str) -> str:
    """Replace <img src="..."> with Confluence storage's <ac:image>.

    All our images are uploaded as page attachments, so we only need
    the basename to reference them. We also drop any alt/title since
    Confluence storage doesn't take them on <ac:image> (they'd cause
    a different validation error).
    """
    def repl(m: re.Match) -> str:
        src = m.group(1)
        # ignore http(s) absolutes — leave as <img>
        if src.startswith(('http://', 'https://', 'data:')):
            return m.group(0)
        name = os.path.basename(src)
        return (f'<ac:image ac:align="center" ac:layout="center">'
                f'<ri:attachment ri:filename="{name}" /></ac:image>')

    html = re.sub(r'<img[^>]+src="([^"]+)"[^>]*/?>', repl, html)
    return html


def html_to_storage(html: str) -> str:
    # Confluence storage requires a single <ac:* xmlns:*> root in some
    # cases; in practice plain XHTML body works. Make sure self-closing
    # ac/ri tags are well-formed (markdown lib already produces XHTML).
    return html


def get_page(page_id: str, token: str) -> dict:
    req = urllib.request.Request(
        f'{BASE}/rest/api/content/{page_id}?expand=version,space,body.storage',
        headers={'Authorization': f'Bearer {token}',
                  'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def put_page(page_id: str, token: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f'{BASE}/rest/api/content/{page_id}',
        method='PUT',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {token}',
                  'Content-Type': 'application/json',
                  'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_up = sub.add_parser('update')
    p_up.add_argument('page_id')
    p_up.add_argument('md', type=Path)
    p_pv = sub.add_parser('preview')
    p_pv.add_argument('md', type=Path)
    p_pv.add_argument('--out', type=Path, default=Path('/tmp/_storage.xhtml'))
    args = ap.parse_args()

    md_text = args.md.read_text(encoding='utf-8')
    html = md_to_html(md_text)
    storage = rewrite_imgs(html)

    if args.cmd == 'preview':
        args.out.write_text(storage, encoding='utf-8')
        print(f'wrote → {args.out}  ({len(storage)} chars)')
        return

    token = os.environ['CONFLUENCE_TOKEN']
    cur = get_page(args.page_id, token)
    new_ver = cur['version']['number'] + 1
    payload = {
        'version': {'number': new_ver},
        'title': cur['title'],
        'type': cur['type'],
        'body': {'storage': {'value': storage,
                              'representation': 'storage'}},
    }
    print(f'updating {cur["title"]} (id={args.page_id}, v{cur["version"]["number"]} → v{new_ver})')
    print(f'storage size: {len(storage)} chars')
    res = put_page(args.page_id, token, payload)
    print(f'OK  v{res["version"]["number"]}  {BASE}{res["_links"]["webui"]}')


if __name__ == '__main__':
    main()
