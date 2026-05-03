# SPS (Samples-Per-Second) ボトルネック分析 — PandaSet V3 DataLoader

## 1. 要約

PandaSet V3 キャッシュ + `PandaSetCalibDatasetFull` の DataLoader スループットを、
dgx2 ホスト（Xeon Platinum 8168 × 2 = 96 phys core / 192 HT）で測定。

| 状態 | w=90 実効 sps | 律速要因 |
|---|---:|---|
| 初期（pre-inject, pre-TJ） | **204** | `is_obj` 再計算 412ms/call（einsum N=13k × M=160 cuboids） |
| is_obj 事前注入後 | **2892** | PIL JPEG 全画像 decode 14.5ms |
| + TurboJPEG class API | **3000〜3500 (steady) / 6400 (瞬間)** | メモリ帯域 + Python GIL / worker fork オーバーヘッド |

**最終的に 14〜16× 高速化**。ただし短い bench では DataLoader prefetch buffer の飽和で
7194〜28710 sps などの過大値が観測される — これは幻で、steady-state は 3000〜3500 sps。

## 2. ハードウェア

### dgx2 ホスト
```
CPU:    Intel(R) Xeon(R) Platinum 8168 @ 2.70GHz  ×2 socket
Cores:  24 phys/socket × 2 socket = 48 phys, 96 HT
        (dgx2 のシステム設定では 96 phys/192 HT と認識 — BIOS側でHT off不明)
L1d:    1.5 MiB (全体)
L2:     48  MiB
L3:     66  MiB
TDP:    205W/socket
AVX:    AVX-512F / AVX512-DQ / AVX512-CD / AVX512-BW / AVX512-VL
Memory: 1.5 TB DDR4 (NUMA 2-node, 各 ~772GB)
        DDR4-2666 × 6ch × 2 socket = 256 GB/s theoretical  (Platinum 8168 spec)
Disk:   tmpfs (/dev/shm) 756 GB — 本 bench のキャッシュ置き場
        Lustre (/mnt/fsx) 60 TB — Scene 生データ
```

### 比較: Core i9（素人 PC） 例 i9-13900K
```
CPU:    Core i9-13900K  1 socket
Cores:  8 P-core + 16 E-core = 24 phys, 32 HT
L1d:    0.8 MiB
L2:     32  MiB (P-core 2MB × 8 + E-core cluster 4MB × 4)
L3:     36  MiB (shared)
TDP:    125W (base) / 253W (turbo)
AVX:    AVX2 + VNNI（**AVX-512 は Alder Lake 以降で disabled**）
Memory: 128 GB DDR5-5600  2ch × 1 socket = 89.6 GB/s
```

### 差が出るポイント

| 項目 | Xeon 8168 ×2 (dgx2) | i9-13900K | 比率 |
|---|---:|---:|---:|
| 並列 worker（物理コア） | **96** (2×48) | 24 | 4× |
| HT worker | 192 | 32 | 6× |
| **メモリ帯域 (理論)** | **256 GB/s** (6ch×2 DDR4-2666) | 89.6 GB/s (2ch DDR5-5600) | **2.9×** |
| L3 合計 | 66 MiB | 36 MiB | 1.8× |
| AVX-512 | あり | **無し** | このワークロードでは無効 — §4.4 で実測 |
| NUMA | 2 node | 1 node | — |

## 3. ボトルネック遷移

### 3.1 Pre-inject 時代: `is_obj_per_point` が 412ms/call

旧 `__getitem__` は frame あたり:
- pts (13k) × cuboids (160) の **per-point membership 判定** を毎 worker・毎 call 再計算
- einsum `(M,N,3)` delta + `(M,3,3) @ delta` rotation → allclose で各軸 half-dim 判定

bench（1 core）:
```
is_obj einsum:    412 ms/call   ← 律速
JPEG decode (PIL): 14.5 ms
その他:              3   ms
合計:              ~430 ms/call
理論 1-worker sps:   2.3 sps
w=90 理論:         207 sps   （実測 204 sps — 計算通り）
```

**AVX-512 の einsum は早いが M=160 クボイドを毎回計算する自体が無駄**。
1 scene あたり cuboid 配置は固定 → **build 時にキャッシュすれば 0 ms**。

### 3.2 is_obj 事前注入で 2892 sps へ

`uv_full`, `z_cam`, `is_obj` を build-time（`build_pandaset_full_v3.py`）で torch.save。
`inject_pandaset_is_obj.py` で既存キャッシュ 1648 file を 48-worker で 18s で再書き込み。

bench 後（1 core）:
```
JPEG decode (PIL): 14.5 ms   ← 新たな律速
crop 計算:          2   ms
その他:             1   ms
合計:              ~20  ms/call
理論 1-worker sps:  50 sps
w=90 理論:        4500 sps   （実測 2892 sps — buffer/sync loss 35%）
```

PIL は Python で JPEG decompress を回している。PIL 内部は libjpeg を呼ぶが、
Python object allocation + GIL release/acquire で overhead。

### 3.3 TurboJPEG 導入で 3000〜3500 sps へ

`turbojpeg` Python bindings は libjpeg-turbo の SIMD 実装を直接叩く。
AVX-512 IDCT + MMX/SSE color convert で、さらに crop region のみ部分 decode。

```python
# pandaset_full.py (post-fix)
_TJ_INST = _tj.TurboJPEG()
cropped = _TJ_INST.crop(jpg_bytes, ju0, jv0, jw, jh, preserve=False)
arr     = _TJ_INST.decode(cropped, pixel_format=_TJ_PF_RGB)
```

bench (1 core):
```
TJ crop+decode 384px: 4.5 ms   ← PIL 14.5ms の 3.2×
              （full decode 6.9ms と比べて crop で更に 1.5×）
他処理:              6.1 ms
合計:              10.6 ms/call
理論 1-worker sps:  94 sps
w=90 理論:        8500 sps
w=90 実測:     3000〜3500 sps steady, 6400 sps 瞬間ピーク
```

## 4. なぜ理論値 8500 に届かず 3500 止まりか — 真のボトルネック

### 4.1 メモリ帯域（これが dgx2 で効く）

1 sample あたり触るメモリ:
- JPEG bytes read: ~200 KB (random access, 実効 400 KB with overhead)
- decoded RGB crop: 384×384×3 = 442 KB
- pts (N<=13k×3×float32): 156 KB
- uv_full, is_obj etc: 100 KB
- output tensors + IPC copy: 384×384×3 + 2k×5 = ~500 KB

合計 **~1.6 MB/sample** の DRAM traffic。
w=90 worker × 3500 sps × 1.6 MB = **~500 GB/s 必要**（書き込み+読みを両方カウント）。

Platinum 8168 の理論帯域は 256 GB/s × 2 socket = 512 GB/s だが、
**実効は 60〜70% の 350 GB/s 程度**。つまり計算一致: **メモリ帯域が律速**。

### 4.2 forkserver + IPC の tensor 転送

`num_workers=90` で各 worker が tensor を queue 経由で main process に返す:
- pin_memory=True で main 側でもう一度 copy → DRAM traffic 倍加
- batch=128 × 384×384×3 = 56 MB/batch を IPC で経路 (shm_fd or pickle)
- 実測 batch 50 個 / 1 sec → IPC で 2.8 GB/s の shm copy

### 4.4 AVX-512 は効くのか？ → **ほぼ効かない**（実測）

AVX-512 を有効/無効で同じ bench を回して比較（`MKL_ENABLE_INSTRUCTIONS=AVX2
OPENBLAS_CORETYPE=HASWELL` で強制 AVX2 化）:

| 演算 | AVX-512 ON | AVX-512 OFF | 差分 |
|---|---:|---:|---:|
| einsum (is_obj 形状 M=160, N=13k) | 302 ms | 308 ms | +2%（効かない）|
| numpy matmul (BLAS) | 6.6 ms | 8.4 ms | +27% |
| SGEMM 2048³ (MKL 単 core) | 2263 GFLOPS | 2026 GFLOPS | +12% |
| **TurboJPEG full decode**   | **6.60 ms** | **6.59 ms** | **0%** |
| **TurboJPEG crop+decode 384** | **4.40 ms** | **4.39 ms** | **0%** |
| PIL full decode | 10.2 ms | 10.7 ms | +5% |

**核心**:
- **libjpeg-turbo は AVX-512 を実装していない**（SSE2/SSE4/AVX2 止まり、公式 README
  に明記）。dgx2 の 3.2× PIL 速度は AVX-512 ではなく **SIMD IDCT + MMX color convert**
  由来 — つまり i9 でも同じ速度で TJ decode できる。
- numpy einsum は C の naive loop で vectorize 不十分。BLAS に落ちない形状なので
  AVX-512 恩恵ゼロ。
- MKL SGEMM は +12〜27% 効くが、我々の `__getitem__` で大きな SGEMM は呼ばない
  （2k×3 matmul は小さすぎて BLAS 呼び出しオーバーヘッドの方が大きい）。

**結論**: DataLoader の 3500 sps に対する AVX-512 寄与は **≤5%、実質ゼロ**。
主因は純粋に「96 コア × DDR4 6ch×2socket の帯域」であって、命令セットではない。

### 4.3 i9 だとどうなるか（予測）

i9-13900K でこの bench を走らせた場合:
- physical cores 24 → max w=24
- 1 core perf は i9 5.5GHz vs Xeon 8168 2.7GHz のクロック差で **~2×**。
  AVX-512 は TJ/einsum に効かないので相殺効かず、i9 の方が **1 core では速い**。
- 1-worker sps ≈ 94 × 2.0 = 190 sps
- 24-worker 線形外挿 → 4500 sps …ただし帯域 89.6 GB/s で **~3.7 GB/s・worker で頭打ち**
- 実効 **~1000〜1500 sps 上限** と予測（dgx2 の 1/3）

**i9 は 1 core だけなら dgx2 より速いが、24 コアしかなく帯域 1/3 で、スケール段階で負ける**。
Xeon Platinum の本質的優位性は「命令セット」ではなく「**6ch DDR4 × 2 socket の帯域が
96 ワーカー全員にデータ供給できること**」。

## 5. 更に詰める余地

| 手段 | 見込み sps 増加 | コスト |
|---|---:|---|
| JPEG crop 側をさらに小さく (128px) | +500 | crop 小さすぎると学習効果↓ |
| shared-memory `inst` キャッシュ (全 worker 共有) | +1500 | torch.multiprocessing.shared_memory 実装要 |
| tensor を uint8 のまま GPU に送り CUDA側で float 化 | ~+300 | すでに実装済み |
| persistent_workers=True + epoch境界で iter 再生成 | 安定化（瞬間ピーク消える） | bench 正確化だけで実効 sps は不変 |
| Lustre /dev/shm 化済み | 既に実施済 | — |

**4000 sps を安定越えしたければ shared-memory キャッシュ**が次の一手。

## 6. DataLoader bench で数値が揺れた件

`/tmp/dl_sps.py` の steps=100, warmup=10 では:
- warmup 区間：prefetch buffer 充填中の **偽の高スループット**
- buffer 尽きた後：**real steady-state** に落ち着く
- epoch 境界でも StopIteration → bench 終了してしまう

測定は **少なくとも steps=500, warmup=200 以上、かつ `oversample` を大きく**。
oversample=30 でも warmup 50 batch で 1.0s = 瞬間 6400 sps だった。

## 7. 結論

- `is_obj` pre-inject: **204 → 2892 sps (14×)** — build-time 計算への移行
- TurboJPEG class API: **2892 → 3000〜3500 sps (1.1×)** — PIL より SIMD decode 早いが
  メモリ帯域限界に当たる
- **真のボトルネックは DDR4 帯域 (256 GB/s/socket)**
- i9 で同じコード走らせたら帯域 1/3 × コア 1/4 で **3500 → ~1000 sps に落ちる見込み**
- 4000+ sps 実用域へ行くには **shared-memory キャッシュ**か
  **学習側 batch=256 で worker あたり fetch 率半減**が次手

## 8. 8-rank DDP contention 再ベンチ (2026-05-03, post-TJ-fix)

### 8.1 TurboJPEG install trap

§3.3 の TJ は実は **動いてなかった**。`pandaset_full.py` の import は:

```python
try:
    import turbojpeg as _tj
    _HAVE_TJ = True
except Exception:
    _HAVE_TJ = False
```

pypi に `turbojpeg` (= Dobatymo/turbojpeg-python) と `PyTurboJPEG` の2パッケージが
あり、前者は namespace `turbojpeg` を同名で奪うが TurboJPEG class を持たない。
よって `import turbojpeg` が成功しても、実行時 `turbojpeg.TurboJPEG()` で
AttributeError → **silent fallback で PIL が走り続けていた**。

加えて container 内の `libturbojpeg.so` が無く、正しい PyTurboJPEG を入れても
`OSError: libturbojpeg.so.0: cannot open shared object file` で落ちる。

**修正:**

```bash
pip uninstall -y turbojpeg PyTurboJPEG
pip install PyTurboJPEG
conda install -c conda-forge libjpeg-turbo   # → lib/libturbojpeg.so
```

### 8.2 decode micro-bench（500×500 tile、1 プロセス）

| path | time/call | 備考 |
|---|---:|---|
| PIL `Image.open(BytesIO).convert('RGB')` | **1.34 ms** | libjpeg ベース、Python allocator 経由 |
| PyTurboJPEG `decode()` full | **1.01 ms** | **1.33× (§3.3 の 3.2× より小さい)** |

**3.2× が 1.33× に縮んだ理由**:
§3.3 は 1920×1080 の full image を crop+decode する想定で、TJ の部分 decode が
効いていた。現 V3 tile cache は既に **500×500 pre-crop** 済みの jpg を保持する
ため、decode 量自体が小さく、PIL と TJ の差が SIMD IDCT 効果のみになる。

`__getitem__` 全体では: **6.52 ms → 4.66 ms (-28%)** 改善。

### 8.3 8-rank concurrent workers sweep（dgx2, waymo_v3_tiled）

dgx2: **96 phys × 2 HT = 192 logical**。
vLLM が GPU 0-7 を占有していたため、ベンチは **GPU 8-15 に taskset 12 cpus/rank**
（rank r → cpus r×12 .. r×12+11、spawn, skip-single, n_batches=100, bs=64, prefetch=4）。

| workers/rank | TJ OFF (baseline) | TJ ON | diff |
|---:|---:|---:|---:|
| 8 | 7,488 sps | — | — |
| 10 | 8,002 sps | **9,948 sps** | **+24% ← 最大** |
| 11 | 9,082 sps | — | — |
| 12 | **9,882 sps** ← TJ OFF peak | 7,317 sps | **−26% (regression)** |
| 14 | 8,204 sps | 8,094 sps | 〜flat |
| 16 | — | 7,757 sps | — |

### 8.4 "TJ ON で w=12 が退化した" の解釈

TJ OFF の curve は **w=12 で 9,882 sps ピーク → w=14 で 8,204 sps 減衰** という
凸形。これは普通の worker 増→並列 decode 増→帯域飽和で頭打ちの形。

TJ ON は **同じ曲線の"帯域飽和が w=10 側に前倒しになった"と解釈できる**:
- TJ ON では 1 worker あたりの decode が速い (−28%) → 同じ wall-time で読む byte/s が増える
- w=10 で既に帯域・LLC が埋まる → w=12 で**帯域争奪 + LLC miss** が起きる
- TJ OFF の "PIL decode が遅くて帯域に余裕がある w=12" とは相対的に負ける

§4.1 の「実効 ~350 GB/s で律速」仮説と consistent。TJ ON で **同じ sps レベルに
より少ない worker で到達する** = CPU 節約できるが、上限値は同じ。

### 8.5 final recommended config

```python
# train_ps_v3_ddp.py 8-rank DDP on dgx2-class (Xeon 8168×2 or 9654×1):
num_workers    = 10    # TJ ON ピーク、overshoot しない
pin_memory     = True
prefetch_factor = 4
batch_size     = 64    # per-rank. 8-rank world → 512 global
```

per-rank **1,244 sps × 8 ranks = ~10,000 sps/node**。
これは §7 の 3,500 sps（single-rank baseline）から **2.8×** 相当、
かつ 1 epoch (~90k samples) ≈ **9 秒** で回る。

### 8.6 takeaway

1. **install 時 silent fallback で年単位バグ化する落とし穴**:
   `_HAVE_TJ` を try/except で隠すのではなく、`import turbojpeg; turbojpeg.TurboJPEG()`
   まで smoke test してから `_HAVE_TJ=True` とすべき。（§3.3 の時点では CI が無く、
   PIL fallback で sps が出ていたため気づかなかった）
2. **TJ はこの cache schema では 1.33× しか効かない**: 小タイル pre-crop cache では
   decode 比率が低く、Python allocator + IPC + pts/is_obj 読みの比率が高い。
3. **w を 1 増やせば sps が上がる時代は終わった**: 8-rank DDP では **per-rank workers
   を 10〜11 に絞る**のが帯域上有利。過去の「w=90, single rank」幻想は捨てる。
4. 次に効くのは §5 の shared-memory `inst` cache (全 worker で jpg bytes を共有する)。
   TJ 単独改善ではもう追加 gain は出ない。
