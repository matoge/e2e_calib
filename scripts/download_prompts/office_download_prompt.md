# 運転データセット一括ダウンロード指示（社内マシン用）

このファイルを丸ごと貼って Claude Code に渡す想定。目的・前提・各データセットのコマンドまで自己完結。

---

## 目的

カメラ・LiDAR キャリブレーション残差ネット (`e2e_calib` の cross-frame residual net) 学習用に、公開運転データセットをできるだけ多く集める。**3D bbox ラベルは不要**。必要なのは:

- カメラ intrinsic（歪み含む）
- カメラ↔LiDAR extrinsic
- ego-pose（world フレーム）
- カメラ↔LiDAR 時刻同期（or 補正可能なオフセット既知）

ラベルが無い / 少ない split も OK。むしろ非ラベル split のほうが量が多くて嬉しい（ONCE unlabeled 1M frames など）。

## 前提マシン環境（チェック & 足りないものは入れる）

```bash
# 事前に確認
df -h                                        # 保存先に 5 TB 以上の空き
gcloud auth list                             # Waymo 用 (GCS)
aws --version                                # AWS CLI
which s5cmd                                  # Argoverse 2 用（なければ下で install）
which aria2c                                 # parallel HTTP DL（KITTI 速度改善に）
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
python -c "import gdown; print(gdown.__version__)" 2>/dev/null || pip install gdown

# s5cmd がなければ
curl -sL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz | tar xzf - -C /tmp s5cmd && sudo mv /tmp/s5cmd /usr/local/bin/

# aria2c がなければ
sudo apt-get install -y aria2
```

## 保存先

全部ここの下に集約する（ユーザーが実行時に書き換え）:

```bash
DATAROOT=/path/to/datasets   # ← 埋める。1 マウントで 5 TB 以上推奨
LOGDIR=$DATAROOT/_logs
mkdir -p "$DATAROOT" "$LOGDIR"
```

## 合計目安（全部落とした場合）

| 項目 | サイズ |
|---|---|
| PandaSet (HF georghess) | 45 GB |
| Waymo v2 training | ~400 GB |
| Waymo v2 validation | ~75 GB |
| Waymo v2 testing | ~6 GB |
| KITTI raw | ~200 GB |
| KITTI odometry | ~170 GB |
| Argoverse 2 Sensor (train+val+test) | ~1.2 TB |
| Argoverse 2 Lidar (無ラベルだが LiDAR-only 大量) | ~3.5 TB（選択） |
| ONCE labeled (train+val+test) | ~500 GB |
| ONCE unlabeled_small | ~30 GB |
| ONCE unlabeled_medium | ~250 GB |
| ONCE unlabeled_large | ~500 GB |
| nuScenes v1.0 trainval | ~640 GB |
| A2D2 (Audi) | ~2 TB |
| ZOD (Zenseact) Frames + Sequences | ~600 GB |
| **合計**（重い候補全部） | **~10 TB** |

回線が太くても 10 TB は 1 日で落ちないので、**優先度順**で順次。バンドを食い合わせないよう 4 本並列までを目安。

---

## 実行順（推奨優先度）

優先度は「キャリブ品質の高さ × サイズ効率」で並べた。**上から順に走らせる**。

### 1. PandaSet（最優先、45 GB、小さく速い）

HuggingFace ミラー（`georghess/pandaset`）に zip 1 本で置いてある。

```bash
cd "$DATAROOT"
mkdir -p pandaset_hf pandaset
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='georghess/pandaset', filename='pandaset.zip',
    repo_type='dataset', local_dir='./pandaset_hf',
)
print('DL:', p)
PY
unzip -q pandaset_hf/pandaset.zip -d pandaset
# 展開後、103 シーンが pandaset/001, pandaset/002, ... の形で並ぶ
ls pandaset | wc -l   # 103 になる
```

### 2. Waymo Open Dataset v2 (gcloud、~480 GB 全部で)

`gcloud auth login` が済んでいる前提。training が一番大きい:

```bash
# training (798 segments × 6 components)
for c in camera_image camera_calibration lidar lidar_box lidar_calibration vehicle_pose; do
  mkdir -p "$DATAROOT/waymo/training/$c"
  nohup gcloud storage cp -r "gs://waymo_open_dataset_v_2_0_0/training/$c/*" \
    "$DATAROOT/waymo/training/$c/" > "$LOGDIR/waymo_train_$c.log" 2>&1 &
done
wait

# validation (202 segments)
for c in camera_image camera_calibration lidar lidar_box lidar_calibration vehicle_pose; do
  mkdir -p "$DATAROOT/waymo/validation/$c"
  gcloud storage cp -r "gs://waymo_open_dataset_v_2_0_0/validation/$c/*" \
    "$DATAROOT/waymo/validation/$c/" 2>&1 | tee -a "$LOGDIR/waymo_val.log"
done

# testing (16 segments、小さい、一瞬)
for c in camera_image camera_calibration lidar lidar_box lidar_calibration vehicle_pose; do
  mkdir -p "$DATAROOT/waymo/testing/$c"
  gcloud storage cp -r "gs://waymo_open_dataset_v_2_0_0/testing/$c/*" \
    "$DATAROOT/waymo/testing/$c/" 2>&1 | tee -a "$LOGDIR/waymo_test.log"
done
```

**注意:** `calibration`/`vehicle_pose` が無いと pose GT が取れないので全コンポーネント必須。`lidar_box` は 3D bbox で本来不要だが小さいので落としておく。

### 3. Argoverse 2 Sensor (1.2 TB、AWS 公開バケット)

リージョン明示が要る（バケットは `us-east-1`）:

```bash
mkdir -p "$DATAROOT/argoverse2/sensor"
AWS_REGION=us-east-1 nohup s5cmd --no-sign-request --numworkers 32 \
  cp 's3://argoverse/datasets/av2/sensor/*' "$DATAROOT/argoverse2/sensor/" \
  > "$LOGDIR/av2_sensor.log" 2>&1 &
```

### 4. KITTI odometry（~170 GB、直 S3、eu-central-1）

日本からは遅いので `aria2c -x 16 -s 16` で並列化:

```bash
mkdir -p "$DATAROOT/kitti_odometry"
cd "$DATAROOT/kitti_odometry"
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti
for f in data_odometry_calib.zip data_odometry_poses.zip data_odometry_velodyne.zip data_odometry_gray.zip data_odometry_color.zip; do
  aria2c -x 16 -s 16 -c "$BASE/$f"
done
# 全部展開
for z in *.zip; do unzip -q "$z"; done
```

### 5. KITTI raw（~200 GB、直 S3）

61 drive × sync.zip。drive list は KITTI devkit の `raw_data_downloader.sh` と同じもの:

```bash
mkdir -p "$DATAROOT/kitti_raw"
cd "$DATAROOT/kitti_raw"

# 全 drive ID（KITTI raw_data_downloader 由来、61 drive）
DRIVES=(
  2011_09_26_drive_0001 2011_09_26_drive_0002 2011_09_26_drive_0005 2011_09_26_drive_0009
  2011_09_26_drive_0011 2011_09_26_drive_0013 2011_09_26_drive_0014 2011_09_26_drive_0015
  2011_09_26_drive_0017 2011_09_26_drive_0018 2011_09_26_drive_0019 2011_09_26_drive_0020
  2011_09_26_drive_0022 2011_09_26_drive_0023 2011_09_26_drive_0027 2011_09_26_drive_0028
  2011_09_26_drive_0029 2011_09_26_drive_0032 2011_09_26_drive_0035 2011_09_26_drive_0036
  2011_09_26_drive_0039 2011_09_26_drive_0046 2011_09_26_drive_0048 2011_09_26_drive_0051
  2011_09_26_drive_0052 2011_09_26_drive_0056 2011_09_26_drive_0057 2011_09_26_drive_0059
  2011_09_26_drive_0060 2011_09_26_drive_0061 2011_09_26_drive_0064 2011_09_26_drive_0070
  2011_09_26_drive_0079 2011_09_26_drive_0084 2011_09_26_drive_0086 2011_09_26_drive_0087
  2011_09_26_drive_0091 2011_09_26_drive_0093 2011_09_26_drive_0095 2011_09_26_drive_0096
  2011_09_26_drive_0101 2011_09_26_drive_0104 2011_09_26_drive_0106 2011_09_26_drive_0113
  2011_09_26_drive_0117 2011_09_28_drive_0001 2011_09_28_drive_0002 2011_09_28_drive_0016
  2011_09_28_drive_0021 2011_09_28_drive_0034 2011_09_28_drive_0035 2011_09_28_drive_0037
  2011_09_28_drive_0038 2011_09_28_drive_0039 2011_09_28_drive_0043 2011_09_28_drive_0045
  2011_09_28_drive_0047 2011_09_29_drive_0004 2011_09_29_drive_0026 2011_09_29_drive_0071
  2011_09_30_drive_0016 2011_09_30_drive_0018 2011_09_30_drive_0020 2011_09_30_drive_0027
  2011_09_30_drive_0028 2011_09_30_drive_0033 2011_09_30_drive_0034 2011_10_03_drive_0027
  2011_10_03_drive_0034 2011_10_03_drive_0042 2011_10_03_drive_0047 2011_10_03_drive_0058
)
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
for d in "${DRIVES[@]}"; do
  aria2c -x 8 -s 8 -c "$BASE/${d}/${d}_sync.zip"
done
# 日付ごとのカリブレーション zip（5 日分）
for date in 2011_09_26 2011_09_28 2011_09_29 2011_09_30 2011_10_03; do
  aria2c -x 8 -s 8 -c "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${date}_calib.zip"
done
# 展開
for z in *.zip; do unzip -qo "$z"; done
```

### 6. ONCE（Huawei、~1.3 TB、Google Drive）

**Google Drive 匿名クォータは ~15 GB/日**。自分のアカウントで OAuth を通せば突破できる。1 度 `gdown --fuzzy` で認証を済ませれば残りは一気に落ちる。

```bash
mkdir -p "$DATAROOT/once"
cd "$DATAROOT/once"

# gdown は folder 再帰 DL 対応
# 優先度: unlabeled_large > unlabeled_medium > unlabeled_small > train > val > test
# (ラベルは要らないので unlabeled 系を先に)

# 全 folder ID（source: https://once-for-auto-driving.github.io/download.html）
declare -A ONCE
# train
ONCE[train_anno]=14cI2vleBokHEtSLjZAgzmjL1cFOkoYwD
ONCE[train_lidar]=1gUCYhCFIEuRePMIRzzRQGTY1L0XmbsUq
ONCE[train_camera]=1E85-kPxCatAUGx-EvnNJJpnJ7pfxqmPH
# val
ONCE[val_anno]=1RtrDhhI6-zLEexW5fVWm3mJLwaSe56en
ONCE[val_lidar]=1MgSsa5lHsVFLM6qICf5kvmg-cxIqrRif
ONCE[val_camera]=1zYgxnU5NBoAWz9TvMgfapKkZ2YhTtyl7
# test
ONCE[test_anno]=1JetPb_IWD8IJ1-YK0XZ_7KGngwYco-EM
ONCE[test_lidar]=1th2igCde-3apihpxILCZkhCfWgqgh6FT
ONCE[test_camera]=1hd8L36qNuh_7hI0yb_xjt9yYqBVXlMyx
# unlabeled_small
ONCE[unl_small_anno]=136kzKQG3lUwR3MyUUWEswPVaY56avWgh
ONCE[unl_small_lidar]=15phWhQy5QvhECHjzdM881RBfRBLYqvNb
ONCE[unl_small_camera]=1gxxkM-K7lA2unT5cVQnZ1UxSOOshatZQ
# unlabeled_medium (lidar p1-p4, camera p1-p4)
ONCE[unl_med_anno]=1pi3fyHJJbkxLk7ONfvKH4ptUlW32o5vM
ONCE[unl_med_lidar_p1]=1uvBIGFhBgSoQOoVOt_wa2v1R0wSzmJZ0
ONCE[unl_med_lidar_p2]=1JjOggKQO5zrEapBTx13cE79A_PzSeQzN
ONCE[unl_med_lidar_p3]=1dEqJ_suuYbB2vbSwO8Kw__rNBBKfJ5qJ
ONCE[unl_med_lidar_p4]=1qgiTZwV9XIa_UUodSuL17QWkpFfgn9-l
ONCE[unl_med_camera_p1]=1-V4UN3WyahQZ7K12i0SWTlYlv1Xh5eEB
ONCE[unl_med_camera_p2]=1Bta7R4dex-wI4iWoLZlE4M2faV7ZskTV
ONCE[unl_med_camera_p3]=1l-Jjk3J1ogE8_BWbpxnDc9KFFcbLEmXS
ONCE[unl_med_camera_p4]=1t_Ea2IGSwIS3HZ3UWcCz_9PLUgdw3_Y6
# unlabeled_large (lidar p5-p9, camera p5-p9)
ONCE[unl_lrg_anno]=1HphPhWZt9OKUdiTgd-G1Wzo8S8TZm7nm
ONCE[unl_lrg_lidar_p5]=19hCy7N7n2YIM_P-K0e5hKCv9-L0G8BRJ
ONCE[unl_lrg_lidar_p6]=1jCXmeddekH3ynF7IRLIe2DYXVZUcUhbw
ONCE[unl_lrg_lidar_p7]=1pD7ayCIkPLpNaK2SyEHUNxp6026z6a0k
ONCE[unl_lrg_lidar_p8]=1zKTUWJ6n8BA9wvLneDmO-Q14vw6xWMCj
ONCE[unl_lrg_lidar_p9]=14FoWbvDmwHeCKfYMNtOlIbNdoBogIrSZ
ONCE[unl_lrg_camera_p5]=14LKOfIcW-pKfypU1Cjz8If9L7K7tPYoY
ONCE[unl_lrg_camera_p6]=1xt3i5zFJvTGqhPYcQgndav2WNO-qYmZC
ONCE[unl_lrg_camera_p7]=15nxLHdAuOgyYkh21MtYAuyYYAI3d5xRu
ONCE[unl_lrg_camera_p8]=1eVO5YynrxCBptqARb6xfh-IUOVIfR7Hb
ONCE[unl_lrg_camera_p9]=1Ygr8O3MRIBtHSorIqXK-JSuBQtQVxonw

# 例: unlabeled_large/lidar p5 を落とす
gdown --folder "https://drive.google.com/drive/folders/${ONCE[unl_lrg_lidar_p5]}" -O unl_lrg_lidar_p5/

# 一括ループ（クォータに引っかかったら翌日再開、-c で resume）
for key in "${!ONCE[@]}"; do
  echo "=== $key ==="
  gdown --folder "https://drive.google.com/drive/folders/${ONCE[$key]}" -O "$key/" || echo "STUCK $key, retry later"
done
```

**詰まったら:** Google Drive 認証を個人アカウントで通す:
```bash
# gdown は内部で oauth token 使わないので、代わりに rclone を使うのが確実
rclone config                                # "gdrive" を新設、web auth 通す
rclone copy --drive-shared-with-me gdrive: "$DATAROOT/once" -P --transfers 8
```

### 7. nuScenes (~640 GB、要登録)

登録必須（学術 or 商用 license 選択）。`https://www.nuscenes.org/sign-up` でアカウント作って token 取る。token が取れれば AWS S3 から直 DL 可能:

```bash
# token を https://www.nuscenes.org/nuscenes/download.php で取得
# v1.0-trainval_meta.tgz, v1.0-trainval{01..10}_blobs.tgz, v1.0-test_meta.tgz, v1.0-test_blobs.tgz
# URL は token 付きで一回きりなので、ブラウザで right-click → "URL コピー" して下記で使う
cd "$DATAROOT/nuscenes"
for u in "TOKEN付きURL1" "TOKEN付きURL2" ...; do
  aria2c -x 16 -s 16 -c "$u"
done
for f in *.tgz; do tar xzf "$f"; done
```

### 8. A2D2（Audi、~2 TB、直 S3、登録不要だが使用許諾同意）

URL は `https://www.a2d2.audi/a2d2/en/download.html` からコピペ、aev-autonomous-driving-dataset-audi バケット:

```bash
mkdir -p "$DATAROOT/a2d2"
cd "$DATAROOT/a2d2"
# 全部落とすなら camera_lidar_semantic + camera_lidar + 3d_bbox の 3 split
# calibration 用途なら camera_lidar (no labels) だけで OK、~1 TB
BASE=https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com
aria2c -x 16 -s 16 "$BASE/camera_lidar-20180810150607_bus_signals.tar"
aria2c -x 16 -s 16 "$BASE/camera_lidar-20180810150607_camera_frontcenter.tar"
# ... (ファイル一覧はサイトから取る; 15 個ぐらいある)
```

### 9. Zenseact Open Dataset (ZOD、~600 GB、北欧)

HF ミラーあり（`Zenseact/ZOD`）or 公式 SDK:

```bash
pip install zod
# CLI で download (デフォで S3 fallback)
zod download --output-dir "$DATAROOT/zod" --subset-full --dataset-type frames --version full
# Sequences も欲しい場合:
zod download --output-dir "$DATAROOT/zod" --subset-full --dataset-type sequences --version full
```

### 10. KITTI-360（~180 GB、要 email 登録）

`https://www.cvlibs.net/datasets/kitti-360/download.php` でメアド登録して zip URL を受け取る。届いた URL を aria2c で落とすだけ:

```bash
mkdir -p "$DATAROOT/kitti360"
cd "$DATAROOT/kitti360"
# メール届いた URL を順に
aria2c -x 16 -s 16 -c "URL1"
aria2c -x 16 -s 16 -c "URL2"
# ...
```

---

## 並列化戦略

- **同時 4 本まで**（1 Gbps マシンなら各 30 MB/s × 4 = ~1 Gbps 飽和）
- 10 Gbps 回線なら 8〜16 本 OK
- KITTI は **eu-central-1** なので日本/米国から遅い → aria2c の `-x 16 -s 16` で補う
- Argoverse 2 は s5cmd の `--numworkers` を回線に合わせて 32〜64
- Google Drive (ONCE) は rate limit 個別なので他と並列で OK

## 完了チェック

各 DL 後に件数・サイズで sanity check:

```bash
# Waymo training 各コンポーネント 798 ファイル（+ 1 の _metadata があっても OK）
for c in camera_image camera_calibration lidar lidar_box lidar_calibration vehicle_pose; do
  n=$(ls "$DATAROOT/waymo/training/$c" | grep parquet | wc -l)
  echo "$c: $n (expect 798)"
done

# Waymo validation 202, testing 16
# Argoverse 2 train 750, val 150, test 150 (sensor)
ls "$DATAROOT/argoverse2/sensor/train" | wc -l
ls "$DATAROOT/argoverse2/sensor/val" | wc -l
ls "$DATAROOT/argoverse2/sensor/test" | wc -l

# PandaSet 103 scenes
ls "$DATAROOT/pandaset" | wc -l
```

## 既知の落とし穴

- **Argoverse 2**: `BucketRegionError` → `AWS_REGION=us-east-1` を明示
- **KITTI S3**: 日本から ~1 MB/s しか出ない時がある。aria2c で並列数上げれば 10〜30 MB/s に改善
- **ONCE GDrive**: 匿名 15 GB/日クォータ。rclone + 個人アカウントでバイパス
- **HF download**: `hf` CLI (v1.x) が typer のバージョン問題で動かないことがある。`python -c "from huggingface_hub import hf_hub_download"` 経由のほうが安定
- **Waymo GCS**: `gcloud storage` は `gsutil` より 2〜3 倍速い、必ず前者を使う

## 完了後

全部 `$DATAROOT/` 下に落ちる想定。後段の前処理（`waymo_to_pandaset.py` 相当）はこのプロンプトの範囲外。個別に走らせる。
