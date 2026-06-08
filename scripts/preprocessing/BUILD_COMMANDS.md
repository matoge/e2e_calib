# FULL LMDB build commands per dataset

**目的**: CND2 学習で使う FULL LMDB cache を、各データセットでどう作るかの recipe。
全部 `data.lmdb/` + `meta.pt` (train/val list) + (optional) `inst/` の構成。
LMDB 直書きスクリプトがあるものはそちら、無いものは `build_*_v3.py` (FULL モード) → `convert_tile_cache_to_lmdb.py` の 2 step。

実行は **DGX-2** 想定。出力は全部 **`/raid/home/hfunaya/cache_v5/`** に統一。
Python は `/home/hfunaya/.pyenv/versions/3.10.4/bin/python`。

| dataset | out path | status | mode |
|---|---|---|---|
| pandaset (1cam) | `cache_v5/pandaset_v3_full` | ✅ symlink → fsx | 2-step |
| kamikado | `cache_v5/kamikado_v3_full` | ✅ direct-LMDB | 1-step |
| woven (TSS4 9 calibrated seqs) | `cache_v5/woven_v3_full` | 🟡 build 中 | 2-step |
| waymo (all cam) | `cache_v5/waymo_v3_full` | ❌ TODO | 2-step |
| nuscenes (all cam) | `cache_v5/ns_v3_full` | ❌ TODO | 2-step |

---

## PandaSet (1cam, FULL)

raw: `/mnt/fsx/tmp/hfunaya/pandaset/`
本体は fsx 上のまま、cache_v5 から symlink:

```bash
# 既存 (cache_v5 から fsx に向く symlink)
ln -s /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
      /raid/home/hfunaya/cache_v5/pandaset_v3_full
```

build を再実行する場合 (FULL mode、no `--tile`):
```bash
python -u scripts/preprocessing/build_pandaset_full_v3.py \
  --src /mnt/fsx/tmp/hfunaya/pandaset \
  --out /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
  --workers 16
python -u scripts/preprocessing/convert_tile_cache_to_lmdb.py \
  --cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
  --workers 16
```

---

## Kamikado (FCM fisheye, direct-LMDB, 1-step)

raw: `/home/hfunaya/raw/kamikado/scenes/` (8 scenes, 1154 frames)
out: `/raid/home/hfunaya/cache_v5/kamikado_v3_full/data.lmdb`

`build_kamikado_full_lmdb.py` は inst/ を経由せず ShardWriter 経由で直 LMDB。
truncated PNG (上流で切れた末尾フレーム 2 枚) は try/except で skip。

```bash
python -u scripts/preprocessing/build_kamikado_full_lmdb.py \
  --src /home/hfunaya/raw/kamikado/scenes \
  --out /raid/home/hfunaya/cache_v5/kamikado_v3_full \
  --workers 16 --val-frac 0.15
```

zip 9 個 → 8 scenes 展開 (d007-005 は重複 zip):
```bash
cd /home/hfunaya/raw/kamikado
for z in *.zip; do
  base="${z%.zip}"
  [ -d "scenes/$base" ] || unzip -q "$z" -d scenes/
done
```

---

## Woven Sequence (TSS4 FCM, llinking_27 calibrated 9 seqs, 2-step)

raw: `/raid/home/hfunaya/raw/woven_sequence/llinking_27/{tf_long,tf_long2,tf_longer}/sequence=*`
out: `/raid/home/hfunaya/cache_v5/woven_v3_full/data.lmdb`

calibrated 9 sequences のリスト → `mcp_hub/docs/silver_stones/_assets/woven_calibrated.md`。
9 seqs は 3 task に散らばってるので staging dir に symlink 集めて build:

```bash
# Step 0: rsync 元 → dgx2 (43GB)
ssh heatrun 'rsync -aH --info=progress2 \
  ~/git/loom/backend/assets/woven_sequence/llinking_27/ \
  dgx2:/raid/home/hfunaya/raw/woven_sequence/llinking_27/'

# Step 1: staging dir に 9 seqs の symlink
STAGE=/raid/home/hfunaya/raw/woven_sequence/_calibrated9_staging
ROOT=/raid/home/hfunaya/raw/woven_sequence/llinking_27
mkdir -p "$STAGE"
for pair in \
  "tf_long2|ip651_189184482543847979_9918142019096712818_1750227410081-1750227420081" \
  "tf_long2|ip654_1337941440921107425_16943630305775105398_1749030654176-1749030664176" \
  "tf_longer|ip654_10511537532319552321_14646038103307509472_1753882714399-1753882764499" \
  "tf_longer|ip654_10511537532319552321_14646038103307509472_1753882811199-1753882838799" \
  "tf_longer|ip654_11137701898941033422_2113867448976404804_1752799159099-1752799179099" \
  "tf_long|ip651_14938829560602802478_5752152086836244364_1751264780080-1751264800080" \
  "tf_long|ip651_18441102532766589267_18129178250481698576_1750144642075-1750144662075" \
  "tf_long|ip651_5284742168915424463_5271805908131813636_1750657496110-1750657516110" \
  "tf_long|ip654_2516331698292429311_18010983316757154787_1748944591101-1748944611101"; do
  task="${pair%%|*}"; seq="${pair##*|}"
  ln -sfn "$ROOT/$task/sequence=$seq" "$STAGE/${task}__${seq}"
done

# Step 2: FULL inst/ 出す (no --tile)
python -u scripts/preprocessing/build_woven_sequence_v3.py \
  --src "$STAGE" \
  --out /raid/home/hfunaya/cache_v5/woven_v3_full \
  --workers 16

# Step 3: inst/ → data.lmdb
python -u scripts/preprocessing/convert_tile_cache_to_lmdb.py \
  --cache /raid/home/hfunaya/cache_v5/woven_v3_full \
  --workers 16
```

---

## Waymo (all cam, FULL) — TODO

raw: `/mnt/fsx/tmp/hfunaya/waymo_v2/` (453GB Open Dataset)
out: `/raid/home/hfunaya/cache_v5/waymo_v3_full/`

```bash
python -u scripts/preprocessing/build_waymo_v3.py \
  --src /mnt/fsx/tmp/hfunaya/waymo_v2 \
  --out /raid/home/hfunaya/cache_v5/waymo_v3_full \
  --workers 16
# (no --tile)
python -u scripts/preprocessing/convert_tile_cache_to_lmdb.py \
  --cache /raid/home/hfunaya/cache_v5/waymo_v3_full \
  --workers 16
```

注: build_waymo_v3.py が all-cam を出すか single-cam (front) かは引数次第。
all-cam が要るなら `--cams FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT` を指定。

---

## NuScenes (all cam, FULL) — TODO

raw: `/mnt/fsx/data/nuscenes/` (or `/mnt/fsx/tmp/hfunaya/raw/nuscenes/`)
out: `/raid/home/hfunaya/cache_v5/ns_v3_full/`

```bash
python -u scripts/preprocessing/build_nuscenes_v3.py \
  --src /mnt/fsx/data/nuscenes \
  --out /raid/home/hfunaya/cache_v5/ns_v3_full \
  --workers 16
# (no --tile)
python -u scripts/preprocessing/convert_tile_cache_to_lmdb.py \
  --cache /raid/home/hfunaya/cache_v5/ns_v3_full \
  --workers 16
```

注: NuScenes 6cam (FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT)。
all-cam 出すなら `--cams ...` (script 仕様次第)。

---

## 学習側

`train_cnd2_ddp.py --cache <comma-separated paths>` で複数 cache を ConcatDataset 化:

```bash
python -u datasets/train_cnd2_ddp.py \
  --cache /raid/home/hfunaya/cache_v5/pandaset_v3_full,/raid/home/hfunaya/cache_v5/waymo_v3_full,/raid/home/hfunaya/cache_v5/kamikado_v3_full,/raid/home/hfunaya/cache_v5/woven_v3_full \
  --epochs 50 ...
```

`--u-band` / `--per-cache-oversample` で per-cache 設定差。
`--pair-mode` は **PS と Waymo (pose GT あり) のみ** 有効。Kamikado / Woven は calib-only。

---

## 出力検査

```bash
python -c "
import lmdb, torch
from pathlib import Path
ROOT = Path('/raid/home/hfunaya/cache_v5')
for c in sorted(ROOT.iterdir()):
    p = c.resolve() if c.is_symlink() else c
    if not (p/'data.lmdb').exists():
        print(f'{c.name}: NO data.lmdb'); continue
    try:
        m = torch.load(p/'meta.pt', weights_only=False)
        e = lmdb.open(str(p/'data.lmdb'), readonly=True, lock=False, subdir=True, max_dbs=0)
        with e.begin() as t:
            n = sum(1 for k,_ in t.cursor() if not k.startswith(b'__cubs__/'))
        e.close()
        print(f\"{c.name}: lmdb_keys={n} train={len(m.get('train', []))} val={len(m.get('val', []))} cam={m.get('cam')} fisheye={m.get('is_fisheye', False)}\")
    except Exception as ex:
        print(f'{c.name}: ERR {ex}')
"
```
