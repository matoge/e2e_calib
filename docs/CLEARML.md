# ClearML setup (self-hosted)

ローカルで ClearML server を立てて、`--clearml` フラグ付きの training run を
ダッシュボード <http://localhost:18080> で見れるようにする手順。

## 1. システム前提

- Docker 28+ / docker-compose v2
- 8080/8081/8008 が他サービスで埋まってたので **18080/18008/18081** に remap
  済み (`~/clearml-server/docker-compose.yml`)
- snap docker は `/opt` を bind-mount できないので、データは
  `~/clearml-data/` に置いてる

## 2. サーバ起動

```bash
# 1回だけ (既に実行済み):
sudo mkdir -p /opt/clearml/...   # not used (snap restriction)
mkdir -p ~/clearml-data/{data/{elastic_7,mongo_4/db,redis,fileserver},logs,config}
sudo sysctl -w vm.max_map_count=262144
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf

# 起動:
cd ~/clearml-server
docker compose up -d

# 状態:
docker compose ps
# 全コンテナ Up なら OK
# - clearml-webserver  → :18080  (UI)
# - clearml-apiserver  → :18008  (REST API)
# - clearml-fileserver → :18081  (artifact storage)
# - clearml-elastic, mongo, redis  (内部のみ)
```

## 3. クライアント設定 (~/.clearml.conf)

ブラウザで <http://localhost:18080> を開き、

1. ユーザ作成 (名前だけ、メール不要)
2. 右上アイコン → Settings → Workspace → "Create new credentials"
3. JSON が表示される

`~/clearml.conf` に以下を書く (port が 18xxx になってる点が重要):

```hocon
api {
    web_server: http://localhost:18080
    api_server: http://localhost:18008
    files_server: http://localhost:18081
    credentials {
        "access_key" = "..."
        "secret_key" = "..."
    }
}
sdk {}
```

接続テスト:

```bash
python -c "
from clearml import Task
t = Task.init(project_name='test', task_name='ping', auto_connect_frameworks={'pytorch': False})
t.get_logger().report_scalar('m','x',value=1,iteration=0)
print(t.get_output_log_web_page())
t.close()
"
```

URL が出ればOK。

## 4. 学習スクリプトに統合

`train_cross_frame.py` は `--clearml` フラグで自動アップロード:

```bash
python train_cross_frame.py --name v64 --multi-frame ... --clearml
```

ダッシュボードで見られる scalar:

- `train/loss`, `train/err_AB`, `train/err_BA`, `train/base`
- `val/err`, `val/nll`
- `overfit/val/train_err` ← 1.0 超えると過学習傾向
- `overfit/val_nll - train_loss` ← σ 崩壊の「剥がれ具合」

## 5. 既に走らせた run を後から取り込む

`--clearml` 付けずに走らせた実験 (v55, v62, v63 等) は train.log を
パースして手動アップロードできる:

```bash
python scripts/eval/import_train_log.py \
  --log experiments/cross_frame_v63_aug_sigma2_lidardrop/train.log \
  --name v63_aug_sigma2_lidardrop
```

## 6. リモートマシン上で走らせる (サクライ2)

サクライ2 にも `pip install clearml-agent` で agent 入れて、

```bash
# サクライ2 上で 1 回だけ:
clearml-init   # ローカルと同じ creds + http://<this-machine-ip>:18008 を入力
clearml-agent daemon --queue gpu --detached
```

ローカルから:

```bash
clearml-task --queue gpu --project e2e_calib/cross-frame \
  --name v65_remote --script train_cross_frame.py \
  --args full=True multi_frame=True ...
```

agent が pick up → サクライ2 で実行 → ログ自動アップロード。

## トラブル

- **`mkdir /opt/clearml: read-only file system`**: snap docker は /opt を
  bind-mount 不可。`~/clearml-data/` に変更済み (compose の path)
- **port 8080 / 8081 in use**: 18080 / 18081 / 18008 に remap 済み
- **`vm.max_map_count`**: elasticsearch 必須、262144 以上
