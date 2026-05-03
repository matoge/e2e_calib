# 共有 FSX (`/mnt/fsx/tmp/hfunaya`) 運用ルール

**最終更新**: 2026-05-03
**マスター**: このファイル (`docs/infra_shared_fsx.md`) が Source of Truth。
`/mnt/fsx/tmp/hfunaya/README.md` はここのコピー。更新したら両方反映する。

## 1. なぜこのルールが要るのか

過去、キャッシュ／前処理データが以下の場所にバラバラに置かれて「どこにあるか分からない」状態になった:

- `/mnt/fsx/tmp/hfunaya/cache/{nuscenes_v3_tiled, waymo_v3_tiled, ...}`
- `/mnt/fsx/tmp/hfunaya/e2e_calib_cache/{waymo_v3_tiled, pandaset_mc_s64_lazy, ...}`
- ホスト個別 `/home/hfunaya/cache/...`, `/mnt/nvme6t/...`

同じ名前 (`waymo_v3_tiled`) で中身・パラメータ・生成日の違うキャッシュが 2 箇所に並存するなど、再現性が崩壊するパターンが発生。

## 2. 大原則

1. **データ実体は `/mnt/fsx/tmp/hfunaya/` 以下に置く**。Lustre は 4 DGX 全台から同じ絶対パスで見える。
2. **すべてのデータセットは ClearML Dataset に登録する** (external_files 方式、物理コピー無し)。これが「どこに何があるか」の単一の真実。
3. **命名規則**: `{dataset}_{fmt}_{variant}` (ex: `waymo_v3_tiled`, `nuscenes_v3_tiled`, `pandaset_mc_s64_lazy`)。ClearML project は `e2e_calib/datasets`。
4. **host-local cache (`/home/hfunaya/cache`) は廃止**。既存があれば symlink を fsx に貼って段階的に移行する。

## 3. ディレクトリレイアウト

```
/mnt/fsx/tmp/hfunaya/
├── README.md                  <- このファイルのコピー (cat で見れる)
├── raw/                       <- 入力データ (書き換えない想定)
│   ├── waymo_v2/ ───────────┐  (Waymo OD v2 parquet、7 component)
│   ├── waymo_lcp/              (= waymo_v2/lidar_camera_projection へのシンボリックリンク集)
│   ├── pandaset/                (Pandaset raw)
│   ├── ddad/                    (DDAD raw)
│   ├── zod/                     (ZOD raw)
│   └── nuscenes/  (要配置)
├── cache/                     <- 前処理キャッシュ (生成物、再生成可能)
│   ├── waymo_v3_tiled/         (512×512 tile × 5cam × 798seg、~125GB)
│   ├── waymo_v3_full/          (full-frame、~222GB)
│   ├── nuscenes_v3_tiled/      (~83GB)
│   ├── pandaset_v3_full/       (~854MB)
│   ├── pandaset_mc_s64_lazy/   (multi-cam s=64 lazy)
│   └── {dataset}_{fmt}_{variant}/ の形で追加
├── runs/                      <- ClearML task artifact cache (自動管理、手で触らない)
├── experiments/               <- 手動 DDP (ClearML 未使用時の逃げ道)
├── logs/                      <- 一時ジョブログ  (永続化したいものは repo か ClearML)
├── code/                      <- 共有 helper (使うなら)
└── envs/
    └── miniconda3/            <- 共有 conda env (全ホストから同じパスで見える)
        └── envs/e2e/          <- 現行 e2e_calib 用 python env
```

**移行方針 (非破壊)**:
- 既存の `cache/` と `e2e_calib_cache/` は壊さない。
- 新ルール下のパスは上記 `cache/`、`raw/` 配下。
- `cache/` 配下に既存のものは今あるので OK。`e2e_calib_cache/` 配下のデータには `cache/` 側から symlink を張って **どちらからも見える** 状態にする (詳細は §6)。
- `raw/` は今 `pandaset/`, `waymo_v2/`, `ddad/`, `zod/`, `waymo_lcp/` として root 直下にあるので、そこに symlink を張るだけ。

## 4. データ追加／更新の手順

新しい前処理キャッシュを作るときは:

```bash
# 1) cache/ 配下に書き出す
python scripts/preprocessing/build_waymo_v3.py \
    --tile --cams 1,2,3,4,5 --stride 10 \
    --out /mnt/fsx/tmp/hfunaya/cache/waymo_v3_tiled

# 2) ClearML Dataset に登録 (物理コピー無し、external_files)
python scripts/data_preparation/register_dataset.py \
    --path /mnt/fsx/tmp/hfunaya/cache/waymo_v3_tiled \
    --name waymo_v3_tiled \
    --tags waymo tile v3 5cam stride10 \
    --description "512x512 / stride=384 / y_start=200 / 5cam / 798seg / stride 10"
```

**バッチ登録**:

```bash
python scripts/data_preparation/register_dataset.py \
    --config scripts/data_preparation/datasets.yaml
```

`datasets.yaml` には既存キャッシュ全部が記述済み。**キャッシュを増やしたらこの yaml を更新してコミット**すること。

## 5. ClearML クラスタ構成

```
ClearML server (dgx1):   http://172.16.200.217:{18008/18080/18081}
                         https://clearml.budda.site (外部ドメイン経由も可)

queue:
  dgx1-gpu  → dgx1 agent (予定。Dify 同居)
  dgx2-gpu  → dgx2 agent (稼働中、docker mode、/mnt/fsx mount 済み)
  dgx3-gpu  → dgx3 agent (予定、アイドル 16×V100)
  dgx4-gpu  → dgx4 agent (予定、アイドル 16×V100)
  default   → dgx3 + dgx4 (自由振り分け用)
```

### agent 各ノードに撒く手順

```bash
# dgx3 を例に:

# 1. mcp_hub rsync (compose ファイル群)
rsync -av dgx2:~/mcp_hub/ dgx3:~/mcp_hub/

# 2. queue 名書き換え
ssh dgx3 "sed -i 's/dgx2-gpu/dgx3-gpu/g; s/dgx2-gpu/dgx3-gpu/g' ~/mcp_hub/docker-compose.clearml-agent.yml"

# 3. ClearML server 側に queue を作成 (web UI または API)
curl -s -X POST http://172.16.200.217:18008/queues.create \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"name": "dgx3-gpu"}'

# 4. agent 起動
ssh dgx3 "cd ~/mcp_hub && docker compose -f docker-compose.clearml-agent.yml up -d"

# 5. 学習 image を fsx 経由で配布 (初回のみ)
#    各ノードで build し直すと 4× の pull 時間がかかるので、
#    1 ノード (dgx2 想定) で build → `docker save` → Lustre 上の tar を
#    dgx{1,3,4} で `docker load` する。 詳細は §5.1。
./infra/distribute_docker_image.sh     # dgx2 で実行
```

agent container には **必ず `/mnt/fsx:/mnt/fsx` を bind mount** する。dgx2 の compose は `CLEARML_AGENT_DOCKER_HOST_MOUNT=/mnt/fsx:/mnt/fsx` を env で指定している。

### 5.1 Docker image の統一配布 (fsx 経由)

4 DGX 全台で同じ `e2e-calib-train:local` を使いたい。 registry を立てるのは大げさなので、
Lustre を transport にする:

```
dgx2 (builder)                            dgx1 / dgx3 / dgx4 (consumers)
├─ docker build -f infra/Dockerfile.train   ├─ docker load -i /mnt/fsx/tmp/hfunaya/images/...tar
├─ docker save → /mnt/fsx/.../images/*.tar  │
└─ (tar は Lustre 上に残す、再利用可)      └─ 以降は build 不要
```

配布ヘルパー: `infra/distribute_docker_image.sh`

```bash
# 1) dgx2 で一度だけ build
ssh dgx2 "cd ~/git/e2e_calib && docker build -f infra/Dockerfile.train -t e2e-calib-train:local ."

# 2) dgx2 で save → 他 3 台に load
ssh dgx2 "cd ~/git/e2e_calib && ./infra/distribute_docker_image.sh"

# 3) 確認
for h in dgx1 dgx2 dgx3 dgx4; do ssh $h "docker images e2e-calib-train:local"; done
```

- tar の置き場: `/mnt/fsx/tmp/hfunaya/images/e2e-calib-train_local.tar` (`:` は `_` に置換)
- サイズ目安: nvcr.io/nvidia/pytorch:24.02-py3 ベースで **~8-10 GB** (save 後、非圧縮)
- Dockerfile を変更したら再 build + 再配布。 tar は **git-ignore / 生成物**、 同名 overwrite で更新。
- dgx1 には Lustre が繋がっているので tar は Lustre に置いたままで `docker load` で OK。

**ヒント**: image 更新は数ヶ月に 1 回程度の頻度で十分。 python pkg を増やしたい時は
通常 `requirements.train.txt` への追記 → rebuild。
`CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1` により agent は venv 再生成しないので、
**image 側 site-packages が効く**。

## 6. HEATRUN (開発機) からの submit

HEATRUN には **`/mnt/fsx` はマウントされていない**。submit は可能だが、ローカルでデータを読む処理 (register_dataset 等) は DGX 上で実行する必要がある。

### 学習タスクの submit

```bash
cd ~/git/e2e_calib
./infra/submit_clearml_task.sh \
    --name  wm_ddp_v721_frustum \
    --script scripts/training/train_ps_v3_ddp.py \
    --queue  dgx3-gpu \
    --args   "--config configs/waymo/v721_frustum.yaml --epochs 200 --bs 64"
```

queue 名を差し替えるだけで任意のノードに投げられる (`dgx2-gpu`/`dgx3-gpu`/`dgx4-gpu`/`default`)。

### 前処理 / dataset 登録タスクの submit

前処理は CPU ワークなので、別途 CPU queue を用意するか、GPU queue に投げて GPU は idle で回す。現行は GPU queue に投げる運用:

```bash
# Waymo tile 再生成 (dgx3 に投げる例)
./infra/submit_clearml_task.sh \
    --name  waymo_v3_tiled_rebuild \
    --script scripts/preprocessing/build_waymo_v3.py \
    --queue  dgx3-gpu \
    --args   "--tile --cams 1,2,3,4,5 --stride 10 --workers 32 --out /mnt/fsx/tmp/hfunaya/cache/waymo_v3_tiled"
```

## 7. 学習スクリプト側でのデータ取得

### パターン A: 直接パス参照 (現行)

```python
dataset = PandaSetCalibDatasetFull(
    cache_dir="/mnt/fsx/tmp/hfunaya/cache/waymo_v3_tiled",
    ...
)
```

fsx は全ノードから同じパスなので OK。ただし **docs/infra_shared_fsx.md に記載されていないパスは使わない**こと。

### パターン B: ClearML Dataset 経由 (推奨、将来)

```python
from clearml import Dataset
ds = Dataset.get(dataset_name="waymo_v3_tiled", dataset_project="e2e_calib/datasets")
cache_dir = ds.get_local_copy()  # external_files なので /mnt/fsx/... が返る
```

メリット: バージョン固定可能 (`dataset_version` や `dataset_id` で指定)、lineage が UI に残る。

## 8. よくある失敗 (FAQ)

### Q. 同じ名前のキャッシュが複数ある
- `cache/waymo_v3_tiled` (古い、root 所有、2026-05-03 13:57 生成) と
  `e2e_calib_cache/waymo_v3_tiled` (新、2026-05-03 18:18 生成) のように並存。
- → ClearML Dataset には **それぞれ別 name** で登録 (`waymo_v3_tiled_old`, `waymo_v3_tiled`)。
- → 新スクリプトでは常に `cache/waymo_v3_tiled` を参照。古い方は一定期間後に削除。

### Q. HEATRUN で `register_dataset.py --dry-run` したら「does not exist」ばかり
- HEATRUN には `/mnt/fsx` がマウントされていない。DGX のいずれか (dgx2/3/4) で実行する。
- これは意図した挙動。HEATRUN は submit 専用、データ操作は DGX 上で。

### Q. ClearML server に繋がらない
- 内部 IP: `http://172.16.200.217:18008` (DGX 内から)
- 外部 DNS: `https://clearml.budda.site` (HEATRUN / ラップトップから)
- agent と client で URL が違っても同じ server を指していれば OK (task project は共有)。

### Q. dgx2 の vLLM と同居して大丈夫？
- vLLM は GPU0-7 占有 (32GB 常駐、utilization 0% のアイドル推論)。
- ClearML agent は docker mode で GPU 指定可能。dgx2-gpu queue に投げる task は GPU8-15 を使うようにする (tobd: 実装)。
- 重い学習は **dgx3-gpu / dgx4-gpu** に投げる方が無難。

## 9. 今日 (2026-05-03) 時点の既存 cache 一覧

| ClearML name | path | size | 生成スクリプト | 備考 |
|---|---|---|---|---|
| `waymo_v3_tiled` | `/mnt/fsx/tmp/hfunaya/e2e_calib_cache/waymo_v3_tiled` | 125 GB | `build_waymo_v3.py --tile` (commit 8e6d4b8) | 5cam / stride 10 / 637K inst |
| `waymo_v3_tiled_old` | `/mnt/fsx/tmp/hfunaya/cache/waymo_v3_tiled` | 284 GB | 不明 (root 所有) | legacy |
| `waymo_v3_full` | `/mnt/fsx/tmp/hfunaya/cache/waymo_v3_full` | 222 GB | `build_waymo_v3.py` (no --tile) | full-frame |
| `waymo_v3_bench` | `/mnt/fsx/tmp/hfunaya/cache/waymo_v3_bench` | 9.9 GB | bench subset | |
| `nuscenes_v3_tiled` | `/mnt/fsx/tmp/hfunaya/cache/nuscenes_v3_tiled/nuscenes_v3_tiled` | 83 GB | `build_nuscenes_v3.py` (推定) | 二重ネスト注意 |
| `pandaset_v3_full` | `/mnt/fsx/tmp/hfunaya/cache/pandaset_v3_full` | 854 MB | `build_pandaset_full_v3.py` | |
| `pandaset_mc_s64_lazy` | `/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_mc_s64_lazy` | 66 GB | 不明 | multi-cam s=64 |
| `pandaset_mc_s64_cache.pt` | `/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_mc_s64_cache.pt` | 65 GB | 同上、eager 版 | 単一ファイル |
| `waymo_v2_raw` | `/mnt/fsx/tmp/hfunaya/waymo_v2/training` | 453 GB | 公式 | Waymo OD v2 7 component |
| `pandaset_raw` | `/mnt/fsx/tmp/hfunaya/pandaset` | 81 GB | 公式 | |
| `ddad_raw` | `/mnt/fsx/tmp/hfunaya/ddad` | 257 GB | 公式 | |
| `zod_raw` | `/mnt/fsx/tmp/hfunaya/zod` | 2.1 GB | 公式 | |

Lustre 使用: 51/60 TB (88%)、残 7.6 TB。

## 10. TODO (2026-05-03)

- [ ] `cache/waymo_v3_tiled` (old) vs `e2e_calib_cache/waymo_v3_tiled` (new) の正式選別 → 片方削除
- [ ] `e2e_calib_cache/` 配下 → `cache/` に symlink で統合
- [ ] dgx3 / dgx4 に ClearML agent デプロイ
- [ ] dgx3 / dgx4 の `~/git/e2e_calib` repo をセットアップ
- [ ] 学習 script を `Dataset.get()` ベースに段階移行

## 11. 家 (HEATRUN 以外) から submit するには

「家のラップトップから ClearML に task を投げ、実行は DGX2/3/4 でやらせる」パターン。
**家はエージェントを持つ必要はない** (=ローカル GPU で回す必要はない)。
家は「submit client」として振る舞い、code は git 経由、データは Lustre 経由
(agent が `file:///mnt/fsx/...` を container に bind mount 済み) で到達する。

### 11.1 家の初期セットアップ (一度だけ)

```bash
# 1) repo を clone
git clone git@github.com:matoge/e2e_calib.git ~/git/e2e_calib
cd ~/git/e2e_calib

# 2) clearml をインストール (submit だけなので軽い)
pip install clearml   # server への REST API のみ使用

# 3) clearml.conf を DGX2 からコピー
scp dgx2:~/clearml.conf ~/clearml.conf
#    もしくは https://clearml.budda.site で自分の API credentials を発行して
#    `clearml-init` で対話的に作る
```

これで終わり。データセット本体は **家には落ちてこない** (そもそも存在しない)。
Lustre 上のパス (`/mnt/fsx/tmp/hfunaya/...`) は agent 側で見える前提で、
script 内で参照されるだけ。

### 11.2 何を同期する必要があるか

結論: **コード (git) と ClearML credentials (clearml.conf) だけ**。

| レイヤ | 何処で何が起きるか | 家がすべきこと |
|---|---|---|
| task 定義 | `clearml-task` が REST API で task を作成 | `~/clearml.conf` があれば OK |
| code | agent が指定 commit を `git clone` | push した commit を agent が pull |
| dataset | agent container 内で `Dataset.get()` or 直接 `file:///mnt/fsx/...` 参照 | **何もしない** (Lustre 共有) |
| image | agent が `e2e-calib-train:local` を docker pull / reuse | DGX 側で build 済み |

家で code 変更 → `git push origin <branch>` → submit 時に `--branch <branch>`
を渡すだけで、 agent 側 (dgx2/3/4) が clone して実行する。
`git pull` は agent が勝手にやる。家では DGX に ssh する必要はない。

### 11.3 家からの submit レシピ

リポジトリに `ALLOW_REMOTE_ONLY=1` という環境変数を立てれば、
script / launcher の「ローカル存在チェック」を skip して投げられる。

```bash
cd ~/git/e2e_calib
git push origin main   # or feature branch

# 2026-05 更新:
#   submit_clearml_task.sh はローカル repo が無い (= agent だけで回す) ケースも
#   サポートする。script が手元に無い場合は ALLOW_REMOTE_ONLY=1 を立てる。
ALLOW_REMOTE_ONLY=1 ./infra/submit_clearml_task.sh \
    --name   home_pandaset_smoke \
    --script scripts/training/train_ps_v3_ddp.py \
    --queue  dgx2-gpu \
    --num-gpus 4 \
    --args   "--config configs/pandaset/v3_full.yaml --epochs 2 --smoke"
```

queue は `dgx{1,2,3,4}-gpu` から選ぶ。 `--cache` 未指定なら
`submit_clearml_task.sh` が自動で `/mnt/fsx/tmp/hfunaya/cache/pandaset_v3_full`
(Lustre 共有) を注入する。

> **注**: 2026-05-03 初版では「dgx1 は Lustre 無し」という誤った前提で
> queue 別の cache path 出し分けを入れていたが、実際は dgx1 も Lustre を
> マウント済み (`192.168.1.1@o2ib:...:/lustre on /mnt/fsx type lustre`)
> なので廃止した。host-local な `/home/hfunaya/cache/pandaset_v3_full`
> (dgx1 に 1.2 GB 残っている) は過去 rsync の残骸で、当面削除もしないが
> 参照もしない。

### 11.4 どんなデータが使えるか知りたい (家から)

ClearML Dataset に登録されていれば、家からカタログを引ける:

```python
from clearml import Dataset
ds_list = Dataset.list_datasets(
    dataset_project="e2e_calib/datasets",
    only_completed=True,
)
for d in ds_list:
    print(d["name"], d["tags"], d["description"][:60])
```

`Dataset.get(name="waymo_v3_tiled", dataset_project="e2e_calib/datasets")` で
メタデータ (サイズ / ファイル数 / URI) を確認可能。 実体 (`file:///mnt/fsx/...`)
は家からは見えないが、 **agent 側で `get_local_copy()` すればそのまま
`/mnt/fsx/...` が返る** (external_files 登録済みなのでダウンロードは走らない)。

### 11.5 家で submit → 失敗時のデバッグ

1. ClearML Web (`https://clearml.budda.site`) で該当 task を開く
2. `Console` タブで agent のログ (`git clone`, `docker run`, script stderr) を読む
3. 典型的な失敗:
   - `git clone` が 403: branch を push してない / private repo の場合は
     agent 側に `~/.ssh` or `CLEARML_AGENT_GIT_USER/PASS` が要る
   - `bind mount fails`: docker_args の `-v` にホスト側に存在しないパスが入っている。
     2026-05-03 版以降の `submit_clearml_task.sh` なら `/mnt/fsx` と `/dev/shm` しか
     mount しない (全 DGX で valid)。古い版を使っていたら `git pull`。
   - `Local file not found: aiohttp @ file:///rapids/...`: container で agent が
     pip を再インストールしようとしている。 docker_args に
     `-e CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1` が入っているか確認
4. 家 → `ssh dgx2` は**しなくてよい**。 全情報は ClearML Web に出る。
