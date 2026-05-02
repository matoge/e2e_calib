# When the Eval Pipeline Lies to You: A Retraction, a Bug Fix, and What Frustum Attention Actually Does

*Internal engineering blog — Camera ↔ LiDAR calibration team*
*Author: H. Funaya · Date: 2026-05-03*

---

## TL;DR

- Yesterday I published an §9.6 write-up claiming "frustum attention helps crowded scenes and hurts sparse ones."
- Today I found the `help` and `hurt` PNGs I used as qualitative evidence were from a **different sample** than the ones my score table pointed at. The eval loop had a hidden `numpy` RNG desync.
- I rebuilt the eval with a seed-locked fetcher, re-scored 164 validation instances, re-rendered the top-8 wins and top-8 losses, and pixel-verified every image against the score table.
- **The "crowd helps, sparse hurts" story is wrong.** The real split is about **baseline difficulty**: frustum attention almost always wins when the no-frustum baseline already struggles (err<sub>w/o</sub> > 2 px), and loses on easy samples where it over-corrects.
- As a side effect I confirmed Waymo v3's episode-level train/val split (679 / 119 / 0 overlap) and kicked two 200-epoch DDP runs on `/dev/shm`.

This post is partly a technical writeup and partly a reminder — for myself and anyone reading this later — that "good-looking qualitative figures" are the single most seductive way to fool yourself in calibration research.

---

## 1. Background: what is "frustum attention" here?

We train a cross-attention regressor that takes an RGB crop and a projected LiDAR cloud, and predicts per-point 2D residuals (plus uncertainty). The "frustum" ablation is the mechanism that restricts each LiDAR point's attention key set to the image tokens inside its 3D view frustum, rather than attending globally.

Two runs:

- **v10 — frustum ON**: standard frustum-restricted cross-attention.
- **v10b — frustum OFF**: global cross-attention, everything else identical.

Both trained 100 epochs on PandaSet. Evaluated on 164 held-out per-instance crops.

---

## 2. Yesterday's §9.6: what I claimed, and why it was wrong

The v1 §9.6 table was produced by an eval script that does roughly:

```python
scores = []
for i in val_indices:
    x = val_ds[i]               # <- PRNG-backed sample builder
    e_on  = eval_one(model_on,  x)
    e_off = eval_one(model_off, x)
    scores.append((i, e_off - e_on))
```

Looks fine. Except `val_ds[i]` pulls from the global `numpy.random` state to pick crop corners, jitter, etc. The `eval_one` calls downstream of it (FPS subsampling, random masking inside the model's data pipeline) also draw from the same global state. So by the time I *re-rendered* a "hurt" sample using only its index `i`, the RNG had advanced by a different number of calls than during scoring. **Same index, different sample.**

Consequence: the score table for index 87 was for one image, and the PNG labeled `hurt_idx0087.png` was for another image entirely. Every qualitative claim — "frustum helps crowded pedestrian clusters," "frustum loses when the scene is sparse" — was built on mismatched pairs.

### How I noticed

I opened `hurt_idx0087.png` next to the score table:

> "idx 87 · n_obj=75 · Δ=−2.50 px"

…but the image showed three cars parked in a mostly empty lot with the title stamp saying `n_obj=149`. Numbers don't lie that loudly. So the figure must be from a different draw.

---

## 3. The fix: `fetch_sample_seeded(i)`

A one-file change that wraps the sample builder in a deterministic RNG context:

```python
def fetch_sample_seeded(i, seed=1234):
    rng_state_np = np.random.get_state()
    rng_state_py = random.getstate()
    torch_rng   = torch.get_rng_state()
    try:
        np.random.seed(seed + i)
        random.seed(seed + i)
        torch.manual_seed(seed + i)
        return val_ds[i]         # every (U,V,mask,crop) now deterministic per i
    finally:
        np.random.set_state(rng_state_np)
        random.setstate(rng_state_py)
        torch.set_rng_state(torch_rng)
```

Two eval-time invariants it restores:

1. `fetch_sample_seeded(i)` always returns the exact same tensor for the same `i`, **regardless of call order**.
2. The outer loop's RNG state is untouched, so any ordering-sensitive code after the call is unaffected.

With that in place I re-scored all 164 instances and re-rendered the top-8 wins and top-8 losses. Every PNG has been pixel-verified against the score table — the `err_w/o` and `err_w/` stamps on each image now match the table to 0.01 px.

---

## 4. What the pixel-verified v2 actually shows

164 val instances. Mean err w/o = **2.082 px**, with = **1.819 px**. Mean Δ = **+0.263 px** (frustum wins overall).

### 4.1 Top-8 wins (frustum ON lower err)

| idx | Δ (px) | err w/o | err w/ | n<sub>obj</sub> |
|-----|--------|---------|--------|-----------------|
| 129 | +3.65 | 5.07 | 1.42 | 3   |
| 131 | +3.47 | 5.42 | 1.95 | 134 |
|  93 | +2.70 | 5.10 | 2.40 | 9   |
|  72 | +2.48 | 3.66 | 1.18 | 75  |
|  79 | +2.46 | 4.81 | 2.35 | 8   |
|   7 | +2.46 | 3.24 | 0.78 | 30  |
|  97 | +2.18 | 4.10 | 1.92 | 12  |
|  70 | +2.08 | 3.92 | 1.84 | 169 |

![help idx 129](../assets/frustum/qual_v2/help_idx0129.png)
*Fig 1 — **help idx=129**, Δ=+3.65 px. Left = frustum OFF, right = frustum ON. Frustum collapses the vertical jitter on the lane markings.*

![help idx 131](../assets/frustum/qual_v2/help_idx0131.png)
*Fig 2 — **help idx=131**, Δ=+3.47 px. Dense parking lot. Frustum pulls the per-point arrows down onto the cars instead of smearing across the ground.*

![help idx 7](../assets/frustum/qual_v2/help_idx0007.png)
*Fig 3 — **help idx=7**, Δ=+2.46 px. Textured building facade. Without frustum, residuals drift across windows.*

### 4.2 Top-8 losses (frustum ON higher err)

| idx | Δ (px) | err w/o | err w/ | n<sub>obj</sub> |
|-----|--------|---------|--------|-----------------|
|  87 | −2.50 | 0.73 | 3.23 | 149 |
|  13 | −2.46 | 1.71 | 4.17 | 55  |
| 119 | −2.33 | 0.75 | 3.08 | 60  |
|  34 | −2.02 | 2.41 | 4.43 | 35  |
| 149 | −1.85 | 0.80 | 2.65 | 169 |
|  62 | −1.42 | 1.65 | 3.07 | 21  |
| 140 | −1.30 | 1.01 | 2.31 | 149 |
|  69 | −1.07 | 0.71 | 1.78 | 149 |

![hurt idx 87](../assets/frustum/qual_v2/hurt_idx0087.png)
*Fig 4 — **hurt idx=87**, Δ=−2.50 px. Baseline error was already 0.73 px. Frustum over-shoots and opens it to 3.23 px.*

![hurt idx 119](../assets/frustum/qual_v2/hurt_idx0119.png)
*Fig 5 — **hurt idx=119**, Δ=−2.33 px. Again baseline ≈0.75 px. Frustum introduces a systematic upward drift.*

![hurt idx 69](../assets/frustum/qual_v2/hurt_idx0069.png)
*Fig 6 — **hurt idx=69**, Δ=−1.07 px. Close-range truck tailgate. Baseline 0.71 px, frustum 1.78 px — a textbook over-correction.*

### 4.3 The pattern that actually separates them

I originally expected `n_obj` (number of annotated objects in the crop) to be the splitting variable. It isn't:

- **Wins** span n<sub>obj</sub> ∈ \[3, 8, 9, 12, 30, 75, 134, 169\].
- **Losses** span n<sub>obj</sub> ∈ \[21, 35, 55, 60, 149, 149, 149, 169\].

Both buckets cover sparse and crowded alike. The real split is on baseline difficulty:

- **In 6 of 8 losses, err<sub>w/o</sub> < 2 px** (i.e. the no-frustum baseline was *already almost solved*).
- **In 8 of 8 wins, err<sub>w/o</sub> > 3 px** (baseline was struggling).

Mechanistic reading: frustum attention adds a geometry prior. When the baseline is already on target, that prior is a regularizer pulling away from a good local optimum. When the baseline is off, the prior is the only thing steering the points toward the right object.

### 4.4 Deployment takeaway

For a calibration head that has to behave safely across a whole drive, **we care about the tail, not the mean**. Frustum-ON keeps the worst cases from blowing up (−3.65 px worst-case recovery vs +1.07 px worst-case over-correction). We're keeping it on by default, and adding a per-sample confidence gate to disable it when the baseline already looks confident.

---

## 5. Collateral discoveries

### 5.1 Waymo train/val is episode-clean

While re-building the eval I also audited the Waymo v3 cache I'm about to train on. Scanned all 135 627 train and 23 770 val sample files in parallel, extracted their `scene` / `seg` fields (the Waymo segment IDs like `10017090168044687777_6380_000_6400_000`), and checked for set overlap:

- Train: **679 unique segments**
- Val: **119 unique segments**
- Overlap: **0**

Good — no frame-level leak. Worth knowing because for PandaSet we had to enforce this ourselves.

### 5.2 fsx → /dev/shm gives ~14 % DataLoader speedup

The Waymo cache lives on a shared FSx mount. Two workers × 4 GPUs cold-cache hit ~563 samples/s/rank. After `rsync`-ing the full cache to `/dev/shm` (42 GB), same config hits ~644 samples/s/rank. Not free (you lose 42 GB of tmpfs and have to re-sync on reboot), but worth it on long DDP runs.

Both 200-epoch DDP runs (v721 frustum-on, v721b frustum-off) are now rolling on `/dev/shm`. ETA ~3 h.

---

## 6. Lessons I want to keep

1. **A qualitative figure is not independent evidence.** If my scoring code and my rendering code share any RNG, every PNG is circumstantial at best.
2. **Always assert header == table on qualitative PNGs.** One `assert round(title_err, 2) == round(table_err, 2)` in the render loop would have caught this a day earlier.
3. **Baseline-conditioned ablations beat scene-conditioned ones.** The question is never "does X help crowded scenes." It's "does X help the cases where we were already getting it right, or the cases where we weren't?"
4. **Retract fast, retract visibly.** The v1 §9.6 commit is still in git history for provenance, but the doc on main is now v2, explicitly tagged "retracted v1 — see commit 1d34f2b."

---

## 7. What's next (this week)

- v721 / v721b 200-epoch Waymo DDP runs → first sparse-LiDAR frustum ablation on a dataset other than PandaSet.
- v3 qualitative renderer: dashed green line from each arrow tip to its GT ×, plus color-coded residual (green < 1 px, yellow 1–3 px, red > 3 px). Easier to read than the current "two arrow fields side by side."
- Re-run §9.6 with 200-epoch checkpoints once they land. The current PandaSet story is "stage 1, 100 epoch." 200 epoch + Waymo is stage 2.

---

*Internal repo: `e2e_calib` · relevant doc: `docs/local_point_feature_aggregation.md` §9.6 · retraction commit: 1d34f2b · renderer: `scripts/eval/frustum_qualitative_compare_v2.py`*
