"""Tiny ZIP-upload sidecar.

Runs inside the clearml_backend docker network so the clearml-webserver
nginx can proxy_pass to it. Drops files straight into a bind-mounted
target dir.

Routes:
    GET  /upload              → static HTML (drag-drop UI)
    POST /api/upload          → multipart form, field 'file', .zip only,
                                overwrites by name
    GET  /api/upload_list     → {dir, files: [{name, size, mtime}]}

ENV:
    UPLOAD_DIR        target dir (default /data)
    MAX_BYTES         body size cap (default 1.5 GB)
    UI_TITLE          window title shown in HTML (default 'kamikado upload')
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/data"))
MAX_BYTES  = int(os.environ.get("MAX_BYTES", str(1024 * 1024 * 1536)))  # 1.5 GB
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def _safe_zip_path(filename: str) -> Path:
    if not filename:
        abort(400, description="missing filename")
    safe = secure_filename(filename)
    if not safe.lower().endswith(".zip"):
        abort(400, description="only .zip files accepted")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR / safe


@app.route("/upload")
def upload_page():
    return send_from_directory(STATIC_DIR, "upload.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if f is None:
        abort(400, description="no 'file' field")
    dst = _safe_zip_path(f.filename or "")
    f.save(dst)
    st = dst.stat()
    return jsonify(name=dst.name, size=st.st_size, mtime=st.st_mtime)


@app.route("/api/upload_list")
def api_upload_list():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(UPLOAD_DIR.glob("*.zip")):
        st = p.stat()
        rows.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return jsonify(dir=str(UPLOAD_DIR), files=rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
