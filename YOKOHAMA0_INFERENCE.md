# yokohama0 推論サーバー

Tailscale 経由で他ノードから叩く用のリファレンス。

## エンドポイント

```
Tailscale IP:  100.100.9.3
Hostname:      yokohama0   (Tailscale MagicDNS で解決可)
Port:          11434
Base URL:      http://100.100.9.3:11434
```

LAN (同セグメント) なら `192.168.1.12` でも可。

## 現在ロード済みモデル

```
gemma3n:e4b    Q4_K_M    6.9B    context_max 32768    VRAM 12.7 GB
```

常駐 (`OLLAMA_KEEP_ALIVE=-1`)、`gemma4:31b` も pull 済だがアンロード中。

## 性能 (RTX 3090 実測)

### gemma3n:e4b  (Q4_K_M, 6.9B, ctx=32768, NUM_PARALLEL=4)

| 並列 | per-req tok/s | aggregate tok/s | latency (256 tok) |
|-----:|--------------:|----------------:|------------------:|
|   1  |          127  |           108   |           2.4 s   |
|   2  |           93  |           154   |           3.3 s   |
|   4  |           72  |           230   |           4.5 s   |

同時 4 リクエストまで並列可。

### gemma4:31b  (Q4_K_M, 31.3B, ctx=4096, NUM_PARALLEL=2) 参考

| 並列 | per-req tok/s | aggregate tok/s |
|-----:|--------------:|----------------:|
|   1  |           9.4 |           9.4   |
|   2  |           8.9 |          17.3   |
|   4  |           9.0 |          17.6   |

(NUM_PARALLEL=2 なので N=4 はキューイング。per-req=8.9 で線形スケール)

### ベンチ再現

`bench_ollama.py` を実行すればどの 3090 でも同等スコアが出るはず。

```bash
# ローカル
python3 bench_ollama.py

# リモート (Tailscale 経由)
HOST=http://100.100.9.3:11434 python3 bench_ollama.py

# モデル切り替え
MODEL=gemma4:31b CTX=4096 python3 bench_ollama.py
```

## 叩き方

### シェル (native API)

```bash
curl -s http://100.100.9.3:11434/api/generate -d '{
  "model": "gemma3n:e4b",
  "prompt": "質問",
  "stream": false,
  "options": {"num_ctx": 32768, "num_predict": 512}
}' | jq -r .response
```

### シェル (ストリーム)

```bash
curl -N http://100.100.9.3:11434/api/generate -d '{...,"stream":true}'
```

### シェル (OpenAI 互換)

```bash
curl http://100.100.9.3:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma3n:e4b","messages":[{"role":"user","content":"hi"}]}'
```

### Python (httpx)

```python
import httpx
r = httpx.post("http://100.100.9.3:11434/api/generate",
    json={"model": "gemma3n:e4b", "prompt": "hi", "stream": False,
          "options": {"num_ctx": 32768}}, timeout=60)
print(r.json()["response"])
```

### Python (OpenAI SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://100.100.9.3:11434/v1", api_key="ollama")
r = client.chat.completions.create(model="gemma3n:e4b",
    messages=[{"role": "user", "content": "hi"}])
print(r.choices[0].message.content)
```

## サーバー設定 (参考)

- Service: `ollama.service` (systemd, enabled, auto-start)
- Drop-in: `/etc/systemd/system/ollama.service.d/parallel.conf`
  ```
  Environment="OLLAMA_HOST=0.0.0.0:11434"
  Environment="OLLAMA_NUM_PARALLEL=4"
  Environment="OLLAMA_KEEP_ALIVE=-1"
  ```
- Tailscale tailnet 内から誰でも到達可 (認証なし、LAN/外部からは Tailscale 必須)

## 運用コマンド (yokohama0 上)

```bash
# モデル一覧
ollama list

# 別モデルを pull
ollama pull gemma3:12b

# 今ロードされてるモデル
curl -s http://localhost:11434/api/ps | jq

# ログ
sudo journalctl -u ollama -f

# 再起動
sudo systemctl restart ollama

# ベンチ再実行
python3 /tmp/bench_ollama.py
```

## 注意

- `OLLAMA_NUM_PARALLEL` は KV cache を並列数ぶん事前確保するので、大きすぎると VRAM オーバーで CPU オフロード (劇的に遅くなる)。`gemma4:31b` を動かすときは `NUM_PARALLEL=2` に下げる。
- `context_max` はモデル依存 (`gemma3n:e4b` は 32K まで、`gemma4:31b` は 262K まで宣言だが VRAM 的に 32K 止まり)
