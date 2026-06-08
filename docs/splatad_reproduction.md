# SplatAD on PandaSet — 再現手順 (Blackwell RTX 5080)

`neurad-studio` フレームワーク内で `splatad` メソッドを使い、PandaSet 各シーンに対して SO3xR3 pose 補正付き SplatAD 学習 + render を回す。

## 環境

- GPU: RTX 5080 (Blackwell, sm_120, 16 GB VRAM) ×2
- Driver: `nvidia-driver-580-open` ← **Blackwell は必ず `-open`**
  ([memory: feedback_blackwell_open_driver](../.claude/projects/-home-hiro-git-e2e-calib/memory/feedback_blackwell_open_driver.md))
- Docker runtime: `nvidia` (NVIDIA Container Toolkit)
- Data: `/mnt/nvme6t/pandaset/` (= PandaSet 103 scene、`.pkl.gz` 形式)
- Output: `/home/hiro/git/gs_drive_demo/splatad_outputs_y1/`

## 1. Docker image build

```bash
cd /home/hiro/git/gs_drive_demo/neurad-studio-blackwell
docker build -t neurad-studio:blackwell-ngc .
# 所要 30-40 分 (= tinycudann + neurad-studio + splatad + pip deps コンパイル)
```

**Dockerfile の重要ポイント** (= 自前 fork でなく nuance あり):

| パラメータ | 値 | 意味 |
|---|---|---|
| `FROM` | `nvcr.io/nvidia/pytorch:25.04-py3` | NGC PyTorch 2.7 + CUDA 12.8 (= sm_120 cudafe++ 対応) |
| `TORCH_CUDA_ARCH_LIST` | `"12.0"` | sm_120 ターゲット (= Blackwell) |
| `TCNN_CUDA_ARCHITECTURES` | `120` | tinycudann も sm_120 ビルド |
| `MAX_JOBS=2` | | tinycudann cicc OOM 回避 (= 16GB host RAM 想定) |
| neurad-studio fork | `georghess/neurad-studio` | nerfstudio base + AD ベンチマーク |
| splatad | `carlinds/splatad` | nerfstudio method `splatad` 登録 (= ns-train で見える) |
| pyyaml pin loose | `>=6.0` | NGC has 6.0.2、`==6.0` だと build 失敗 |
| `BUILD_NO_CUDA=1` | | splatad init 時 CUDA compile 走らせない (= 後で必要時のみ) |
| `PIP_CONSTRAINT=` | unset | NGC の setuptools==78 pin を外す (= neurad-studio は <70 要求) |
| `universal_pathlib` reinstall | | Python 3.12 bytecode mismatch 回避 |

**異 arch (= V100 / Hopper / etc) 用に再ビルド**:
```bash
# Dockerfile の以下 3 行を書き換え
ENV TORCH_CUDA_ARCH_LIST="7.0"     # V100 = sm_70
ENV TCNN_CUDA_ARCHITECTURES=70
# tag を 区別
docker build -t neurad-studio:v100 .
```

multi-arch ビルド (= 1 image でどこでも動かす場合):
```bash
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0;12.0"
# compile time 3-4× (= 各 arch ごと cubin 生成)
```

## 2. 単一シーン: train + render

```bash
# scene 番号 (PandaSet 001-122)
SCENE=001
# 出力先ホスト dir
OUT_HOST=/home/hiro/git/gs_drive_demo/splatad_outputs_y1
# データホスト dir
PS_HOST=/mnt/nvme6t/pandaset

# 走らせる
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${PS_HOST}:/workspace/pandaset \
  -v ${OUT_HOST}:/workspace/outputs \
  --name ps${SCENE}_so3xr3 \
  neurad-studio:blackwell-ngc bash -c "
    set -e
    # TRAIN: 30k iter、SO3xR3 で per-frame 6DoF pose 補正学習
    ns-train splatad \
      --viewer.quit-on-train-completion True \
      --max-num-iterations 30000 \
      --output-dir /workspace/outputs \
      --experiment-name ps${SCENE}_so3xr3_front \
      --pipeline.model.mcmc-cap-max 10000000 \
      --pipeline.model.camera-optimizer.mode SO3xR3 \
      pandaset-data --data /workspace/pandaset --sequence ${SCENE} \
      --add-missing-points False --cameras front
    # RENDER: train 終了後の config.yml を auto 検出して dataset render
    CFG=\$(ls -1d /workspace/outputs/ps${SCENE}_so3xr3_front/splatad/*/config.yml | head -1)
    ns-render dataset \
      --load-config \$CFG \
      --output-path /workspace/outputs/ps${SCENE}_so3xr3_front/render_train \
      --pose-source train --rendered-output-names rgb
  "
```

### 重要フラグの意味

| フラグ | 効果 | 必須? |
|---|---|---|
| `--viewer.quit-on-train-completion True` | 学習終わったら viewer も exit (= orchestrator hang 回避) | ✓ orchestrator では必須 |
| `--max-num-iterations 30000` | 30k step (= 短縮版、本来 100k) | smoke 用 |
| `--pipeline.model.mcmc-cap-max 10000000` | gaussians 上限 10M (= 5080 16GB 用) | ✓ |
| `--pipeline.model.camera-optimizer.mode SO3xR3` | **per-frame 6DoF pose 補正 ON** (= デフォルト 'off') | **pose 検証時 必須** |
| `--cameras front` | 前カメラのみ (= 6 cam 全部の 1/6 メモリ) | smoke 用 |
| `--add-missing-points False` | LiDAR 投影外の点を補完しない | ✓ 安定 |

### `--pose-source train` の罠

`ns-render dataset --pose-source train` で render する時、`camera_optimizer` の `use_camopt_in_eval` のデフォルト = False → **学習した pose_adjustment が render に適用されない**。検証時は明示的に ON:

```yaml
# config.yml を edit、または splatad 側のデフォルト変更
pipeline:
  model:
    camera_optimizer:
      use_camopt_in_eval: true
```

(= 過去ハマった、[memory: reference_splatad_calib_modes](../.claude/projects/-home-hiro-git-e2e-calib/memory/reference_splatad_calib_modes.md) 参照)

## 3. 全シーン orchestrator (= 103 scene 2 GPU 並列)

```bash
cd /home/hiro/git/gs_drive_demo/splatad_outputs_y1
nohup bash run_all_ps_so3xr3.sh > orchestrator_$(date +%m%d_%H%M).log 2>&1 &
```

スクリプトの key 部分:
```bash
SCENES=$(ls /mnt/nvme6t/pandaset/ | grep -E '^[0-9]{3}$' | sort)
SKIP="001 004 002 028 003"   # 既に done なシーン
# 2 GPU に交互割り当て、各 scene が前 scene 終了まで wait
```

進捗ログ: `orchestrator_<date>.log` + per-scene `ps<NNN>_so3xr3_front_<date>.log`

完了見積:
- per scene: train 30k iter ≈ 30-45 分 + render 5-10 分 = 1 scene 40-55 分
- 103 scene / 2 GPU = 52 scene per GPU = **35-50 時間 (= 1.5-2 日)**
- 全 6 cam にすると VRAM 圧迫 + 時間 4-6×

## 4. 結果検証 (= 学習が成立してるか)

### 4.1 PSNR 確認
```bash
# train log に PSNR が記録される
grep -E "PSNR|psnr" /home/hiro/git/gs_drive_demo/splatad_outputs_y1/ps001_so3xr3_front_*_train.log | tail -5
```

### 4.2 Pose 補正 effective か (= gauge ambiguity 検出)
```bash
# SO3xR3 ON render と pose adj OFF render を md5 比較
md5sum render_train/train/rgb/40.jpg render_train_NO_pose_adj/train/rgb/40.jpg
# 同じ = gauge ambiguity (= ego 静止シーン、補正効いてない)
# 異なる = 補正が pixel に効いてる (= ego motion ありシーン)
```

(= [splatad_ps_results_summary.md](splatad_ps_results_summary.md) で詳細)

### 4.3 ego motion magnitude
```bash
# scene 別の ego 移動量 (= motion ある scene を弾く前段階)
# vehicle pose の連続 frame 差分を計算、平均速度 < 1 m/s なら静止 (= gauge ambiguity)
```

## 5. トラブルシュート

| 症状 | 原因 | 対処 |
|---|---|---|
| `nvidia-smi` で GPU 見えない | Blackwell に closed driver 入れた | `apt install nvidia-driver-580-open` → reboot |
| docker `Unable to find image` | image を別ストレージに持って行かなかった | bind mount 戻す or 再 build |
| `ns-train` 起動するが OOM | gaussians 10M 超 + 6 cam | `--mcmc-cap-max 5000000` or `--cameras front` |
| train 終わっても viewer 残る | `quit-on-train-completion` 忘れ | フラグ追加で再 run |
| render が学習前と pixel 同じ | `use_camopt_in_eval=False` の罠 | config.yml で True に or 静止シーン |
| tinycudann build OOM | host RAM <32GB で `MAX_JOBS` 過大 | `MAX_JOBS=1` に下げる |

## 6. 関連 doc

- [splatad_ps_results_summary.md](splatad_ps_results_summary.md) — シーン別 pose 補正結果まとめ (= 何を見れば学習成立判定可能か)
- [splatad_ps001_pose_verification.md](splatad_ps001_pose_verification.md) — PS001 の +3.65 dB PSNR 改善 詳細
- [memory: reference_splatad_calib_modes](../.claude/projects/-home-hiro-git-e2e-calib/memory/reference_splatad_calib_modes.md) — SplatAD の 2 系 optimizer (static / velocity) の使い分け
- [memory: feedback_blackwell_open_driver](../.claude/projects/-home-hiro-git-e2e-calib/memory/feedback_blackwell_open_driver.md) — Blackwell GPU は必ず `-open` driver
