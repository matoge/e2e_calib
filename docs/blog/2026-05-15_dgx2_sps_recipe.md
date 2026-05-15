# DGX2の8GPUを埋める — DataLoader↔GPUの帯域パイプを設計する

## TL;DR

- **LMDBに集約** すると `inst/*.pt` の open(2) 大量発行から解放され、`np.frombuffer` zero-copy で worker サンプル生成が **1サンプル ~8 ms** に。
- **DGX2のメモリ帯域はすごい**：DDR4 ×6ch ×2sock = ~256 GB/s、page cacheに **1.3 TiB の LMDB全載せ** で disk read = 0、すべてRAMから供給。CPU 49 core が常時アクティブ、bond NIC 200 Gbps、まだ余裕。
- **GPU も割とまだまだ使える**：V100 32GB のうち bs=384/rank で 6-9 GB しか食ってない (27%)。bs=768 で sps 1.3-1.5x まだ伸びる。**8 GPU で 5500-5800 sps** に到達、目標3000の **約1.9倍**。CPU 1サンプル 8ms × 64worker / 1000ms = 理論上限8000のうち 70% 引き出し。

![SPS / val_nll](../assets/dgx2_sps/sps_curve.png)

## 主役は dataloader と GPU の "帯域パイプ"

CalibrationNet は1.09M paramsしか無い。**モデル自体は軽い**。  
重いのは**データ生成**：

1. **LiDAR点ごとのランダムサンプル → 3D逆投影 を毎iter繰り返す**  
   `dist_uvd` (256点 / sample, 5ch) と `bucket_uvd` (G²×K = 256×8 = 2048点 / sample, 4ch) を画像cropに対し再投影。乱数crop / 乱数extrinsic perturbation / oversample で同じframeを何度も舐める。**1サンプルあたりCPU 7-8 ms**。

2. **TransformerでLiDAR-画像の全パターンを学習**  
   cuboid内だけじゃなく、画像全画素 × 全lidar点（道路, ポール, 標識, 背景全域）の関係を学ぶ設計。教師信号は obj cuboid + bg 全域の二層NLL。**収束が極端に遅い**、50ep 走らせて val_nll が 4.07 → 3.45 程度。たくさん回さないとダメ。

3. **frustum encoderの3×3 cell近傍 gather**  
   `(B, Nq, 9, K, 4)` の gather + topk + mini cross-attn を Q ごと。BS=384/rank, Nq=256で **Q=98,304点**, KV候補=8,847,360。fp16でもbackward重い。

つまり **GPU側の演算量** と **CPU側のサンプル生成** がほぼ釣り合う設計。8GPU埋めるためには「**dataloader が GPU に追いつき続ける帯域**」が全てになる。

## 帯域パイプの全体図

```
[disk: LMDB on /home (md1) ]
       │  (page cache mmap, 800-1200 GB/s 級)
       ▼
[ host RAM 1.5 TiB DDR4 ]              ← Active(file)+Inactive 1.3 TiB
       │  numpy.frombuffer (zero-copy) + augmentation
       ▼
[ 64 worker process / 49 active CPU core ] ← bottleneck-1
       │  pickle → IPC → main rank
       ▼
[ pin_memory CPU buffer ]
       │  PCIe Gen3 x16 × 8GPU ≈ 128 GB/s
       ▼
[ V100 HBM ×8 ]  ← bottleneck-2 (frustum gather + attn)
       │  forward / backward
       ▼
[ NCCL allreduce 8GPU ]                ← bottleneck-3 (idle時間生む)
```

各層がパイプの太さで律速する。**どこか1個でも細いと8GPU埋められない**。

## 観測値（kamikado + woven joint, V100×8 / bs=384 fp16）

| 設定 | sps(global) |
|---|---:|
| ep1 (warmup含む, os=1) | 1697 |
| ep5 安定 (os=1) | 3779 |
| ep11 (os=1) | 3933 |
| ep1 (oversample=4) | 3666 |
| ep5 (oversample=4) | 5517 |
| **ep7 (oversample=4)** | **5750** |

`oversample=1 → 4` でsps落ちないのは**dataloaderがGPU側に追いつき続けている**証拠。  
1サンプル CPU 8ms × 64 worker / 1000ms = **理論上限 8000 sps**。実測 3900 は半分弱で、残りはGPU側律速＋NCCL barrier。

## DGX2のリソースと、各層の使用状況

| リソース | 設備値 | 使用 | 余裕 |
|---|---|---|---|
| CPU コア | 96 (Xeon 8168 ×2) | 49コア active (top 4905%) | **約半分余り** |
| RAM | 1.5 TiB DDR4 | anon 50 GB + **page cache 1.3 TiB (LMDB mmap)** = 88% | 量はぎりぎり |
| RAM 帯域 | 約 256 GB/s (DDR4-2666 ×6ch ×2sock) | 推定 50-100 GB/s 流れている | **20-50%余り** |
| /dev/shm | 756 GB | 223 GB | OK |
| /home (md1) | 28 TB | キャッシュ展開先 | OK |
| /mnt/fsx | 60 TB Lustre | 元データ置場（random read 遅め、学習には使わない） | — |
| GPU 0-15 HBM | 32 GB ×16 | 6-9 GB / 32 GB / GPU | **3-4倍余り** |
| GPU util | 8GPU使用中 | 86-93% | OK |
| H2D PCIe | 16 GB/s ×8 | imgs(B=384,3ch,64×64,uint8)=4.5 MB/iter ≪ 16 GB/s | 圧倒的余り |

→ **CPUと NCCL barrier が律速**。RAM帯域・GPU メモリ・PCIeはまだ伸ばせる。

## レシピ — DataLoader↔GPU パイプを太くする

### 1. **LMDB tile cache** + class-level env cache

`inst/00000000.pt` を1.5M個並べると、open(2) のメタデータ走査だけで遅い。LMDB に packed して `np.frombuffer` で **zero-copy** に。さらに：

```python
class PandaSetCalibDatasetFull(Dataset):
    _LMDB_ENV_CACHE: dict = {}     # (pid, path) → env
    def _open_lmdb(self):
        key = (os.getpid(), str(self.lmdb_path))
        env = self._LMDB_ENV_CACHE.get(key)
        if env is None:
            env = lmdb.open(self.path, readonly=True, lock=False, ...)
            self._LMDB_ENV_CACHE[key] = env
        self._lmdb_env = env
```

これで train + val splits が**同じpathを同じworker内で開いても衝突しない**。

### 2. **`forkserver` でなく `spawn`** + `persistent_workers=True`

`forkserver` は親プロセスのスナップショットを継承する。**LMDB env の registry衝突**を踏むので spawn 一択：

```python
DataLoader(..., num_workers=8, persistent_workers=True,
           multiprocessing_context='spawn')
```

`persistent_workers=True` で**worker起動コストをepoch毎に払わない**。これがepoch境界のidleを大きく削る。

### 3. JPEG → 64×64 を `TurboJPEG.crop_decode` でMCU境界に

PIL.Image.open は lazy decode で 16ms/sample。TurboJPEG の crop+decode で **4.5ms** に縮む（**3.5×**）。Dockerfileで `libturbojpeg0-dev` + `PyTurboJPEG` 必須。

### 4. **`num_workers=8/rank × 8rank = 64`** が最大点

DGX2 96コアで実測：
- `num_workers=8` 以上は forkserver fd 上限 + persistent_workers のメモリ重複で逆効果
- **64 worker で 49 コア active = 0.75 core/worker**（IO待ち含む）
- `num_workers=20` を試すと逆に sps が下がる（worker起動律速 + 共有メモリの fd枯渇）

### 5. **page cache = 帯域パイプの第一段**

DGX2の RAM 1.5 TiB のうち **1.3 TiB が LMDB mmap の page cache**。  
disk read は最初の epoch だけ（Lustre/md1 から）、以降は **メモリ→numpy→torch の純粋メモリ帯域勝負**になる。  
推定 50-100 GB/s 流れている（pcm-memoryで実測中）。

→ 帯域を活用するキー：**LMDB を /home (md1) か /dev/shm に置く**、**`/mnt/fsx` (Lustre) からの直読みは避ける**（Lustreは random small read が遅く、page cacheにも乗りにくい）。

### 6. fp16 + intensity 4ch で **PCIe → HBM** を半分に

H2D は dist_uvd と bucket_uvd で各サンプル数十KB、`pin_memory=True` で DMA。  
さらに**fp16 mixed precision**でモデルパラメータと中間 tensor が半分。bs=384 でも HBM 9 GB / 32 GB しか食わない → **bs=768 まで上げて GPU側パイプを太くする余地がある**。

### 7. NCCL barrier を最小化する

`accel.gather` で stats集計 / `accel.wait_for_everyone` でvis後同期 / val phase の barrier が SPS の "見えない 10-20%" を食う。  
- val を毎epoch でなく **`--val-every 5`** で薄める
- vis_pretrain / midtrain_vis を rank-0 のみで叩いて全rank待たせない
- `find_unused_parameters=False`（デフォ） — `True` だと AllReduce が hang して 22× slowdown を踏む

## 失敗から学んだ落とし穴

- **NGC PyTorch 派生 image (numpy<2 ABI lock)** → woven dataset (numpy 2.x で pickle されてる) が `numpy._core` ImportError。`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` に切り替え。
- **torch 2.11+cu130 wheel に sm_70 が含まれない** → V100 で動かない。**torch 2.5.1+cu121 公式 wheel** を使うと sm_50..sm_90 全部入り。
- **`lmdb.Error: already open in this process`** は5通りの方法で踏んだ。class-level env cache が唯一動く解。
- **forkserver の fd 上限 (256, hardcoded in CPython 3.10)** — `set_sharing_strategy('file_system')` で `/tmp` 経由になり緩和。
- **`max_tasks_per_child` は Python 3.11+** — 3.10 docker は `sys.version_info` ガードでフォールバック。

## 結論

V100 × 8 + 96 CPU の DGX2 で、CalibNet (1M params) の SPS を**3900 sps**に乗せるレシピ：

```bash
docker run --gpus '"device=8-15"' --shm-size=128g \
  --ulimit nofile=1048576 \
  -v ~/cache/<ds>_v3_tiled:/cache/<ds>:ro \
  e2e-calib-train:np2 \
  accelerate launch --num_processes=8 --mixed_precision=fp16 \
    train_ps_v3_ddp.py --cache /cache/a,/cache/b \
    --batch-size 384 --workers 8 --oversample 4 \
    --min-crop-px 128 --max-crop-px 256
```

**LMDB on /home + page cache 1.3 TiB → numpy zero-copy → 64 worker (49 core, 8ms/sample) → pinmem DMA → V100×8 fp16 → NCCL allreduce**  
これでパイプの上から下まで詰まらない。

伸ばせる余地：
- **bs=768** で GPU HBM 9→18 GB、PCIe帯域は依然余裕、+30-50% sps期待
- **val を 5epoch毎** に間引いて barrier を削減
- **Stage1 (image-only) で frustum_enc OFF** → 6000+ sps の warmup → Stage2 で frustum を有効化

まだ DGX2 はパワー余ってる。
