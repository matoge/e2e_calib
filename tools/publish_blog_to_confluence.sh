#!/bin/bash
# publish_blog_to_confluence.sh — e2e_calib 側から Confluence に blog 投稿する薄ラッパー
#
# 使い方:
#   # 新規作成（添付つき）
#   tools/publish_blog_to_confluence.sh create docs/blog/2026-05-03_frustum_qual_v2_retraction.md
#
#   # 既存ページ更新
#   tools/publish_blog_to_confluence.sh update <page_id> docs/blog/2026-05-03_frustum_qual_v2_retraction.md
#
# 依存:
#   - loom リポの tools/publish_wiki_to_confluence.py
#   - 環境変数 CONFLUENCE_TOKEN （= CONFLUENCE_API_TOKEN でも可）
#
# 仕様:
#   - 第二引数 (.md パス) と同じディレクトリ構造の相対パスで参照されている画像を
#     勝手にぜんぶ --attachments に束ねて渡す
#   - blog の場合は --blog フラグ、 --auto-title で本文先頭の # 見出しをタイトルに

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
LOOM_PUB="${LOOM_PUB:-$HOME/git/loom/tools/publish_wiki_to_confluence.py}"

if [[ ! -f "$LOOM_PUB" ]]; then
  echo "[err] loom publisher not found: $LOOM_PUB" >&2
  echo "      set LOOM_PUB=/path/to/publish_wiki_to_confluence.py" >&2
  exit 2
fi

if [[ -z "${CONFLUENCE_TOKEN:-}" && -n "${CONFLUENCE_API_TOKEN:-}" ]]; then
  export CONFLUENCE_TOKEN="$CONFLUENCE_API_TOKEN"
fi
if [[ -z "${CONFLUENCE_TOKEN:-}" ]]; then
  echo "[err] CONFLUENCE_TOKEN not set" >&2
  exit 2
fi

cmd="${1:-}"
case "$cmd" in
  create)
    md="${2:-}"
    shift 2 || true
    ;;
  update)
    page_id="${2:-}"
    md="${3:-}"
    shift 3 || true
    ;;
  *)
    echo "usage: $0 {create|update <page_id>} <md_file> [--extra args to loom publisher]" >&2
    exit 2
    ;;
esac

if [[ -z "${md:-}" || ! -f "$md" ]]; then
  echo "[err] md file not found: ${md:-<unset>}" >&2
  exit 2
fi

md_abs="$(cd "$(dirname "$md")" && pwd)/$(basename "$md")"
md_dir="$(dirname "$md_abs")"

# md 本文から ![](relative/path.png) を拾って絶対パスに直す
mapfile -t attachments < <(
  python3 - "$md_abs" "$md_dir" <<'PY'
import re, sys, os
md_path, md_dir = sys.argv[1], sys.argv[2]
with open(md_path, "r", encoding="utf-8") as f:
    txt = f.read()
seen, out = set(), []
for m in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', txt):
    url = m.group(1).strip()
    if url.startswith(("http://", "https://")):
        continue
    abs_p = os.path.normpath(os.path.join(md_dir, url))
    if abs_p in seen or not os.path.isfile(abs_p):
        continue
    seen.add(abs_p)
    out.append(abs_p)
print("\n".join(out))
PY
)

echo "[info] md         : $md_abs"
echo "[info] attachments: ${#attachments[@]} file(s)"
for a in "${attachments[@]}"; do echo "         - $a"; done

case "$cmd" in
  create)
    # Pull title from the first H1 of the md (publisher requires --title).
    auto_title="$(awk '/^# /{sub(/^# +/, ""); print; exit}' "$md_abs")"
    if [[ -z "${auto_title:-}" ]]; then
      auto_title="$(basename "${md_abs%.md}")"
    fi
    # Strip any caller-supplied --title so the auto one wins; otherwise
    # forward extras as-is.
    extra_args=()
    skip_next=0
    user_title=""
    for arg in "$@"; do
      if [[ $skip_next -eq 1 ]]; then user_title="$arg"; skip_next=0; continue; fi
      if [[ "$arg" == "--title" ]]; then skip_next=1; continue; fi
      extra_args+=("$arg")
    done
    title="${user_title:-$auto_title}"
    exec python3 "$LOOM_PUB" create "$md_abs" \
      --blog --title "$title" \
      --attachments "${attachments[@]}" \
      "${extra_args[@]}"
    ;;
  update)
    exec python3 "$LOOM_PUB" update "$page_id" "$md_abs" \
      --attachments "${attachments[@]}" \
      "$@"
    ;;
esac
