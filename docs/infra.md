# Infra / ClearML DDP 投げ方メモ

2026-05-03 時点。 `infra/*` と dgx2 の `~/mcp_hub/` に docker 化スタックを置いた。

目的:
- `nohup python3 slack_listener.py &` / `docker run vllm ...` / `accelerate launch ...` を
  全部 ssh セッション非依存な `docker compose` と ClearML queue 経由に置き換える。
- dgx2 だけでなく dgx1/3/4 に同じセットアップを撒ける状態にする。

## 全体構成

```
+-------------------------------------------+          +--------------------------------+
|  dev laptop / e2e_calib repo              |          |  ClearML server                |
|  ( /home/hfunaya/git/e2e_calib )          |          |  https://clearml.budda.site    |
|                                           |          |                                |
|  $ ./infra/submit_clearml_task.sh ...     |  --->    |  queue: dgx2-gpu               |
|                                           |          |         dgx1-gpu (予定)        |
+-------------------------------------------+          +----------------|---------------+
                                                                        |
                                                                        v
                                               +---------------------------------------+
                                               |  dgx2 (同じことを dgx1/3/4 にも展開)    |
                                               |  docker compose stacks:               |
                                               |                                       |
                                               |   1. mcp-hub             (listener/   |
                                               |      (~/mcp_hub/         digest/...)  |
                                               |       docker-compose.yml)             |
                                               |                                       |
                                               |   2. gemma4-31b          (vLLM)       |
                                               |      (docker-compose.gemma.yml)       |
                                               |                                       |
                                               |   3. clearml-agent-dgx2 (worker)      |
                                               |      (docker-compose.clearml-         |
                                               |       agent.yml)                      |
                                               |          |                            |
                                               |          v  docker-in-docker          |
                                               |      nvcr.io/nvidia/pytorch:24.02     |
                                               |      + accelerate + clearml           |
                                               |      = e2e-calib-train:local          |
                                               +---------------------------------------+
```

## dgx2 側の常駐サービス

全部 `docker compose` で `restart: unless-stopped`。 ssh セッションが切れても死なない。

| サービス | compose file | 中身 |
|---|---|---|
| `mcp-slack-listener` | `~/mcp_hub/docker-compose.yml` | persona 付き slack bot (mcp_hub 共通 image) |
| `mcp-context-refresher` | 同上 | context auto refresh |
| `digest-scheduler` (ofelia) | 同上 | 平日 08:00 JST に daily_digest を exec |
| `gemma4-31b` | `~/mcp_hub/docker-compose.gemma.yml` | vLLM, TP=8, tool-calling 有効, :8000 |
| `clearml-agent-dgx2` | `~/mcp_hub/docker-compose.clearml-agent.yml` | ClearML agent (dgx2-gpu queue を listen, docker-in-docker) |

### 共通 image: `mcp-hub:local`

`~/mcp_hub/Dockerfile`, `requirements.txt` で 1 枚にまとめた。
中身は python:3.11-slim + slack_bolt/slack_sdk/websocket-client のみ。
listener / refresher / digest は全部同じ image を `command:` で使い分け。

### ClearML agent の queue 指定に関する注意

`allegroai/clearml-agent:latest` の entrypoint は末尾で
```
python3 -m clearml_agent daemon --docker "${CLEARML_AGENT_DEFAULT_BASE_DOCKER}" --force-current-version ${CLEARML_AGENT_EXTRA_ARGS}
```
を固定で呼ぶ。 `command:` で `clearml-agent daemon --queue ...` を上書きしても **効かない**。
queue 指定は必ず `CLEARML_AGENT_EXTRA_ARGS="--queue dgx2-gpu --gpus all --foreground"` の env で渡すこと。
`docker-compose.clearml-agent.yml` はその形にしてある。

他 DGX に配布するときは:
```
CLEARML_WORKER_ID: dgx1-gpu
CLEARML_AGENT_EXTRA_ARGS: "--queue dgx1-gpu --gpus all --foreground"
```
のように上書きするだけでよい。

## e2e_calib 側の新ファイル

### `infra/Dockerfile.train`

base: `nvcr.io/nvidia/pytorch:24.02-py3`
追加: `accelerate==0.34.2`, `clearml==1.16.4`, `einops`, `pyyaml`, `tini`, `rsync`

まず agent ホストで一度ビルドしておく:
```
cd ~/git/e2e_calib
docker build -f infra/Dockerfile.train -t e2e-calib-train:local .
```

将来的に docker hub か内部 registry に push するなら、 `submit_clearml_task.sh` の
`--image` をそれに差し替える。

### `infra/submit_clearml_task.sh`

```
./infra/submit_clearml_task.sh \
    --name wm_ddp_v721_frustum \
    --script scripts/training/train_ps_v3_ddp.py \
    --queue dgx2-gpu \
    --args "--config configs/waymo/v721_frustum.yaml --epochs 200 --bs 64"
```

- queue を切り替えれば dgx2 / dgx1 / 別 GPU ノードにそのまま流せる
- `--docker-args` で `--shm-size=64g --gpus all -v /mnt/fsx:/mnt/fsx -v /dev/shm:/dev/shm --ipc=host`
  を必ず渡している。 DDP で NCCL が落ちるのは 99% ここ
- `accelerate launch --num_processes=8 --mixed_precision=bf16` で包むので、
  学習 script 側は `Accelerator()` 対応である必要がある (`train_ps_v3_ddp.py` は OK)

## 既存 `nohup` / 手動 `docker run` からの移行状況

| 旧 | 新 | 状態 |
|---|---|---|
| `nohup python3 slack_listener.py &` | `docker compose -f ~/mcp_hub/docker-compose.yml up -d` | **移行済み** (2026-05-03) |
| `docker run vllm/vllm-openai:latest ...` (start_gemma.sh) | `docker compose -f docker-compose.gemma.yml up -d` | compose 書いた。 live container はまだ旧手動起動 (Up 9d)。次回再起動時に切替 |
| crontab `0 8 * * 1-5 ~/mcp_hub/run_daily_digest.sh` | ofelia の `ofelia.job-exec.daily-digest` label | 移行済み (host crontab も一応生かしたまま) |
| `nohup accelerate launch ... &` | `./infra/submit_clearml_task.sh ...` → `dgx2-gpu` queue | infra ファイルは commit 済み。 現在走ってる v720/v720b は旧方式。 v721 以降で ClearML 運用に移行 |

## 他 DGX に展開する手順 (dgx1 を例に)

```
# 1. compose スタックを rsync
rsync -av ~/mcp_hub/ dgx1:~/mcp_hub/

# 2. queue 名を書き換え
ssh dgx1 "sed -i 's/dgx2-gpu/dgx1-gpu/g' ~/mcp_hub/docker-compose.clearml-agent.yml"

# 3. ClearML 側に queue を作成 (サーバの web UI か API)
curl -s -X POST https://clearml.budda.site/api/v2.23/queues.create \
  -u "$ACCESS:$SECRET" -H "Content-Type: application/json" \
  -d '{"name": "dgx1-gpu"}'

# 4. agent を起動
ssh dgx1 "cd ~/mcp_hub && docker compose -f docker-compose.clearml-agent.yml up -d"

# 5. e2e_calib 学習 image をビルド (初回のみ)
ssh dgx1 "cd ~/git/e2e_calib && docker build -f infra/Dockerfile.train -t e2e-calib-train:local ."
```

あとは `./infra/submit_clearml_task.sh --queue dgx1-gpu ...` で流せる。

## トラブルシュート

### agent は up してるのに queue が反応しない
```
curl -s https://clearml.budda.site/api/v2.23/workers.get_all -u "$ACCESS:$SECRET" \
  | jq '.data.workers[] | {id, queues: [.queues[].name]}'
```
`queues` が想定 queue と一致していれば OK。 違っていれば `CLEARML_AGENT_EXTRA_ARGS` 要確認。

### gemma が 502 返す
```
ssh dgx2 docker logs gemma4-31b --tail 200
ssh dgx2 docker restart gemma4-31b
```

### slack bot が落ちた
```
ssh dgx2 "cd ~/mcp_hub && docker compose logs --tail 200 slack-listener"
ssh dgx2 "cd ~/mcp_hub && docker compose restart slack-listener"
```
