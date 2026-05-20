"""Headless screenshot a Confluence page using a Bearer token.

Usage:
  python tools/screenshot_confluence.py <page_id> [--out PATH]

Env: CONFLUENCE_TOKEN
"""
import argparse
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('page_id')
    ap.add_argument('--out', type=Path,
                    default=Path('/home/hfunaya/git/e2e_calib/_tmp/conf.png'))
    ap.add_argument('--base-url', default='https://confluence.tri-ad.tech')
    args = ap.parse_args()
    token = os.environ.get('CONFLUENCE_TOKEN')
    if not token:
        print('CONFLUENCE_TOKEN not set', file=sys.stderr); sys.exit(2)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    url = f'{args.base_url}/pages/viewpage.action?pageId={args.page_id}'
    print(f'opening {url}')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            extra_http_headers={'Authorization': f'Bearer {token}'},
            viewport={'width': 1280, 'height': 900},
        )
        page = ctx.new_page()
        page.goto(url, wait_until='networkidle', timeout=60000)
        # Wait for Confluence's main render container to be visible.
        try:
            page.wait_for_selector('#main-content, #main, .wiki-content',
                                    timeout=20000)
        except Exception as e:
            print(f'  [warn] main selector not seen: {e}')
        page.screenshot(path=str(args.out), full_page=True)
        print(f'wrote → {args.out}  ({args.out.stat().st_size} bytes)')
        browser.close()


if __name__ == '__main__':
    main()
