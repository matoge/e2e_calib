#!/usr/bin/env python3
"""Publish a markdown blog post to Confluence (LOOM space by default).

- Markdown -> HTML via python-markdown (handles tables, fences, lists properly).
- Local images referenced as `![](path/to.png)` are uploaded as attachments
  and rewritten to `<ac:image>` storage-format tags.
- Supports both create and update of blog posts (`type=blogpost`), unlike
  the loom helper which hard-codes `type=page` on update.

Auth: $CONFLUENCE_TOKEN (PAT). Title: defaults to first H1 in the markdown.

Usage:
    tools/publish_blog.py docs/blog/<file>.md                 # create blog
    tools/publish_blog.py docs/blog/<file>.md --page-id <id>  # update existing
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import markdown
import requests


BASE_URL = "https://confluence.tri-ad.tech"


def md_to_html(md_text: str) -> str:
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="xhtml",
    )


def collect_images(md_text: str, md_dir: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md_text):
        url = m.group(1).strip()
        if url.startswith(("http://", "https://")):
            continue
        p = (md_dir / url).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        out.append(p)
    return out


def rewrite_images_in_html(html: str) -> str:
    # python-markdown emits <img alt="..." src="path" /> for ![](path).
    def repl(m: re.Match) -> str:
        attrs = m.group(0)
        src = re.search(r'src="([^"]+)"', attrs)
        alt = re.search(r'alt="([^"]*)"', attrs)
        if not src:
            return attrs
        url = src.group(1)
        if url.startswith(("http://", "https://")):
            return attrs
        filename = Path(url).name
        caption = (
            f"<p><em>{alt.group(1)}</em></p>"
            if alt and alt.group(1)
            else ""
        )
        return (
            f'<ac:image><ri:attachment ri:filename="{filename}" /></ac:image>'
            f"{caption}"
        )

    return re.sub(r"<img[^>]+/?>", repl, html)


def first_h1(md_text: str) -> Optional[str]:
    m = re.search(r"^#\s+(.+)$", md_text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def get_content(session: requests.Session, page_id: str) -> dict:
    url = f"{BASE_URL}/rest/api/content/{page_id}?expand=version,space"
    r = session.get(url)
    r.raise_for_status()
    return r.json()


def create_blog(
    session: requests.Session, space: str, title: str, html: str
) -> dict:
    data = {
        "type": "blogpost",
        "title": title,
        "space": {"key": space},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    r = session.post(f"{BASE_URL}/rest/api/content", json=data)
    r.raise_for_status()
    return r.json()


def update_content(
    session: requests.Session, page_id: str, title: str, html: str
) -> dict:
    info = get_content(session, page_id)
    content_type = info.get("type", "page")
    version = info["version"]["number"]
    data = {
        "id": page_id,
        "type": content_type,
        "title": title,
        "version": {"number": version + 1},
        "body": {"storage": {"value": html, "representation": "storage"}},
    }
    r = session.put(f"{BASE_URL}/rest/api/content/{page_id}", json=data)
    r.raise_for_status()
    return r.json()


def upload_attachment(
    session: requests.Session, page_id: str, path: Path
) -> None:
    url = (
        f"{BASE_URL}/rest/api/content/{page_id}/child/attachment"
        "?allowDuplicated=true"
    )
    h = {
        "Authorization": session.headers["Authorization"],
        "X-Atlassian-Token": "no-check",
    }
    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, "application/octet-stream")}
        r = requests.post(url, headers=h, files=files)
    if r.status_code == 400 and "already exists" in r.text.lower():
        # Fallback: update existing attachment by name.
        existing = session.get(
            f"{BASE_URL}/rest/api/content/{page_id}/child/attachment"
            f"?filename={path.name}"
        ).json()
        results = existing.get("results", [])
        if results:
            att_id = results[0]["id"]
            with open(path, "rb") as fh:
                files = {"file": (path.name, fh, "application/octet-stream")}
                r = requests.post(
                    f"{BASE_URL}/rest/api/content/{att_id}/data",
                    headers=h,
                    files=files,
                )
    r.raise_for_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_file", type=Path)
    ap.add_argument("--space", default="LOOM")
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--page-id",
        default=None,
        help="If given, update this content (works for both blog and page).",
    )
    args = ap.parse_args()

    token = os.environ.get("CONFLUENCE_TOKEN") or os.environ.get(
        "CONFLUENCE_API_TOKEN"
    )
    if not token:
        print("CONFLUENCE_TOKEN not set", file=sys.stderr)
        return 2

    md_text = args.md_file.read_text(encoding="utf-8")
    md_dir = args.md_file.parent.resolve()
    title = args.title or first_h1(md_text) or args.md_file.stem
    # Confluence already renders the title above the body — strip the
    # leading H1 so it doesn't show twice.
    md_body = re.sub(r"\A\s*#\s+.+\n+", "", md_text, count=1)
    images = collect_images(md_body, md_dir)
    html = rewrite_images_in_html(md_to_html(md_body))

    session = requests.Session()
    session.headers.update(headers(token))

    if args.page_id:
        print(f"Updating content {args.page_id} (title={title!r})")
        result = update_content(session, args.page_id, title, html)
        page_id = args.page_id
    else:
        print(f"Creating blog in {args.space!r} (title={title!r})")
        result = create_blog(session, args.space, title, html)
        page_id = result["id"]

    for img in images:
        print(f"  attaching {img.name}")
        upload_attachment(session, page_id, img)

    webui = result.get("_links", {}).get("webui", "")
    print(f"OK: id={page_id} url={BASE_URL}{webui}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
