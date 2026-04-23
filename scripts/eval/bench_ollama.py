"""Benchmark ollama server throughput at N concurrent requests.

RTX 3090 (yokohama0) で測った基準値は YOKOHAMA0_INFERENCE.md に記載。
同じ GPU (3090) なら似たスコアが出るはず。

使い方:
    # localhost 向け
    python3 bench_ollama.py

    # リモート (Tailscale) 向け
    HOST=http://100.100.9.3:11434 python3 bench_ollama.py

    # モデル切り替え
    MODEL=gemma4:31b CTX=8192 python3 bench_ollama.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, os, time, threading, urllib.request

URL    = os.environ.get("HOST", "http://localhost:11434") + "/api/generate"
MODEL  = os.environ.get("MODEL", "gemma3n:e4b")
CTX    = int(os.environ.get("CTX", "32768"))
NPRED  = int(os.environ.get("NPRED", "256"))
PROMPT = "Pythonで二分探索を実装して、コメント付きで詳しく説明して。関数名は binary_search。"


def one(req_id, out):
    body = json.dumps({
        "model":   MODEL,
        "prompt":  PROMPT,
        "stream":  False,
        "options": {"num_ctx": CTX, "num_predict": NPRED},
    }).encode()
    t0 = time.time()
    r  = urllib.request.urlopen(URL, body, timeout=600)
    d  = json.loads(r.read())
    dt = time.time() - t0
    out[req_id] = (d["eval_count"], d["eval_duration"] / 1e9, dt)


def run(n):
    out = {}
    threads = [threading.Thread(target=one, args=(i, out)) for i in range(n)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total_tok = sum(out[i][0] for i in range(n))
    print(f"N={n}  wall={wall:.1f}s  total={total_tok} tok  agg={total_tok/wall:.1f} tok/s")
    for i in range(n):
        ec, ed, _ = out[i]
        print(f"  req{i}: {ec} tok / {ed:.1f}s = {ec/ed:.1f} tok/s")


if __name__ == "__main__":
    print(f"target={URL}  model={MODEL}  ctx={CTX}  num_predict={NPRED}")
    one(0, {})  # warmup (model load)
    for n in [1, 2, 4]:
        run(n)
        time.sleep(1)
