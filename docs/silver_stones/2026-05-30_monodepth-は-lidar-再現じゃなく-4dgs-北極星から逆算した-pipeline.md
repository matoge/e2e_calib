---
date: 2026-05-30T20:10+09:00
author: hfunaya
tags: ["monodepth", "4dgs", "north-star", "calibnet2", "geometry", "lidar-bias"]
streams: ["e2e_calib", "north-star"]
status: silver
---
# monodepth は LiDAR 再現じゃなく 4DGS — 北極星から逆算した pipeline

**Origin:** ~/git/e2e_calib @ 6df36eb (git@github.com:matoge/e2e_calib.git)

## 何の話か

CalibNet2 の Q=UV / KV=modality bank / head=task-specific 構造は monodepth に展開できる (Q=image grid uv, head=D)。が、monodepth には **trap** がある — 「LiDAR を再現する」と「4DGS で真の geometry を取る」は別物で、北極星 (1cm 3D map) からの逆算では後者しか意味が無い。

## LiDAR 再現 monodepth の問題

LiDAR 観測には bias が乗ってる:

- 反射率低い面で抜ける (黒い車、水たまり、ガラス)
- 雨/雪/砂塵で誤検出
- 遠距離で sparse になる (200m 先でほぼ点が無い)
- **観測 ≠ 真実** (反射特性に依存した biased measurement)

→ LiDAR 再現を目指すと **LiDAR の bias を継承する**。1cm 地図要件で「そこに本当に何があったか」を知りたいのに、LiDAR が見落とした表面はそのまま見落とされる。

## 4DGS が筋な理由

multi-view photometric consistency = 「複数視点で同じ色に見える点」 = **物理的にそこに表面がある証拠**。

- LiDAR の反射率や測定原理に依存しない
- 地図の最終形 (Gaussian Splatting patching) と同じ表現
- ただし pose が正確じゃないと崩壊する → **pose 正確 = 4DGS 成立** の条件

## 北極星からの逆算

```
1cm 3D map (最終)
  ↑
4DGS で真の geometry
  ↑
正確な pose (cross-frame で √N)
  ↑
正確な calib (= sub-pixel visual feature)
  ↑
self-sup YRP+zoom で詰める ← #228-#233
```

monodepth は **副産物**: pose 詰まった後、4DGS から得られる D を image-only で predict する head を後付けすれば自然にできる。LiDAR 再現 monodepth は **やる価値が小さい** (LiDAR の bias を継承するだけ)。

## CalibNet2 multi-task の位置づけ

CalibNet2 が calib / odometry / monodepth / fusion-D の統一基盤になるのは事実だが、これは **副産物**であって目的ではない。本筋は:

- pose を sub-pixel で取る → 4DGS に渡す → 真の geometry → 1cm 地図

multi-task に展開できる「綺麗さ」に引っ張られて monodepth を LiDAR-mimic で kick するのは tunnel vision。北極星から見て critical path に乗らない。

## 含意

- monodepth は今は触らない
- self-sup → cross-frame pose → 4DGS の順で進む ([[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]])
- monodepth head は 4DGS pipeline が動いてから後付け

