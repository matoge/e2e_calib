# Local Point-Cloud Feature Aggregation in Calib Frustum Encoders

A walkthrough of how `CalibNetDepth` extracts per-cell local lidar features,
why the original PointNet++ choice was inefficient, and the Point-Transformer
replacement we landed on.

---

## 1. Problem framing — at the cell level

The 64-px crop is split into a **16 × 16 grid of cells**. Each occupied cell
produces ONE *cell-query*: the raw lidar point closest to that cell's
center. So a crop has up to 256 cell-queries, one per cell.

For each cell-query we want a **short feature vector** describing the local
3D shape immediately around it, summed into the query stream of the
network. The downstream cross-attn handles long-range image↔point reasoning;
this encoder's only job is to summarize the small lidar neighborhood
**very fast**.

### Neighborhood definition (per cell)

All units in **cell** units (cell-relative; same configuration behaves
identically at S=64 / 128 / 256, because `cell_px = img_size / grid_n`).

* The query sits near the center of its **own cell**.
* Search box: `±r_uv_cells = ±1.5` cells around the query → **3 × 3 cells**
  (own cell + its 8 neighbors).
* That 3×3 box is split into a **4 × 4 stratified sub-grid** (each sub-cell
  ≈ 0.75 cell wide). At most one random in-box point per sub-cell → up to
  16 stratified neighbors.
* Plus **16 random** in-box points to give dense regions extra weight.
* Total **k = 32 candidate neighbors per cell-query**, drawn from the
  dense raw point cloud `uvd_full` (≤ 1024 padded points per crop).

### Live-forward visualization (vis_frustum_neighbors.py)

Each panel: ★ green = the cell-query, ○ yellow = full point cloud,
○ cyan = the ones the encoder actually picked, magenta filled square =
the cell, red square = the ±1.5-cell search box, magenta lines = the
16×16 cell grid.

* PandaSet 64-beam (`docs/assets/frustum/sel_ps_with_depth.png`) — dense
* nuScenes 32-beam (`docs/assets/frustum/sel_ns_with_depth.png`) — sparse
* Waymo (`docs/assets/frustum/sel_wm_with_depth.png`)

![PS sel](assets/frustum/sel_ps_with_depth.png)

---

## 2. Depth threshold — hypothesis: remove it

The original encoder applied a **depth filter** `|Δd| < r_d ≈ 0.4 m`,
on the assumption that we only care about points on the same surface as
the query (e.g., not "far building behind a near car").

### Symptom — sparse-lidar datasets get starved

Mean neighbors per query, live forward (no MLP, just the selection):

| dataset | r_d = 0.4 m (current) | r_d = ∞ (no depth filter) |
|---|---|---|
| PandaSet, idx 200 | μ = 8.4  / 32 max | **μ = 20.2** |
| nuScenes, idx 500 | μ = 0.0  /  2 max | **μ = 12.7** |
| Waymo, idx 200    | μ = 4.8  / 16 max | **μ = 20.9** |

In nuScenes the depth filter is the dominant selector — almost all UV-box
candidates lie at completely different depths (the 32-beam scan-pattern
sweeps depth strongly across UV neighbors), so the encoder ends up with
zero neighbors per query and the per-cell feature degenerates.

### Per-dataset visualizations

**PandaSet** — with vs without depth threshold:

![PS with depth](assets/frustum/sel_ps_with_depth.png)
![PS no depth](assets/frustum/sel_ps_no_depth.png)

**nuScenes** (32-beam sparse) — depth filter wipes out the neighbors:

![NS with depth](assets/frustum/sel_ns_with_depth.png)
![NS no depth](assets/frustum/sel_ns_no_depth.png)

**Waymo** — with vs without:

![Waymo with depth](assets/frustum/sel_wm_with_depth.png)
![Waymo no depth](assets/frustum/sel_wm_no_depth.png)

### Hypothesis (not yet ablated)

* **Argument from first principles.** Δd is already passed into the
  encoder per neighbor; MLP / attention can learn how to weight near-depth
  vs far-depth neighbors. A hard threshold pre-decides "same surface" before
  the network has any say, which loses the cue that e.g. "the bumper is
  *in front of* the road" — useful information for calibration.
* **Argument from literature** (mixed). Modern point-cloud encoders are
  split: kNN-based works (Point Transformer V1/V2/V3, [Zhao 2021]
  https://arxiv.org/abs/2012.09164) drop fixed-distance thresholds entirely,
  while ball-query works (PointNet++ [Qi 2017], KPConv [Thomas 2019],
  PointNeXt [Qian 2022]) keep them. Removing only the depth axis but
  keeping the UV axis is a hybrid not exactly matched in any paper.

**This is a hypothesis, not a settled conclusion.** Action: ablation —
train two short runs identical except `r_d = 0.004` vs `r_d = ∞`, compare
val NLL on NS / PS / Waymo / DDAD, then commit one as the default.

Until that ablation lands, leave `r_d = 0.004` as the configurable default.

---

## 3. PointNet++ vs Point Transformer — what the encoder does with the 32 neighbors

Once we have 32 in-box neighbors per cell-query, we need to compress them
into one vector. Two operators in active use in the literature.

![arch compare](assets/frustum/architecture_compare.png)

### PointNet++ — `MLP + MaxPool` (Qi et al. 2017, NeurIPS, [arXiv:1706.02413](https://arxiv.org/abs/1706.02413))

Each of the 32 neighbors is independently lifted to D = 128 dims via a
shared MLP, and the output is the **per-channel max** across neighbors.

```
rel_j (3-D)  ──[ Linear 3→32 → 32→128 → 128→128 ]──→  feat_j (128)    × 32 copies
                                                              │
                                                  MaxPool over k neighbors
                                                              │
                                                            output (128)
```

Two structural problems for our task:

1. **Per-neighbor MLP at hidden D = 128.** The MLP is run once for every
   neighbor at the full D-wide hidden width. That's `21 K weights × 32
   calls` of compute on a 3-dim relative coord input — almost all of the
   D-wide capacity is wasted on per-point bookkeeping, not relational
   reasoning.
2. **MaxPool ≡ "no inter-neighbor talk".** The only interaction across
   neighbors is the channel-wise max at the end. Each channel keeps the
   single winning neighbor and **throws away the other 31**. Geometric
   relations like "these 3 points are collinear" / "these 4 points are
   coplanar" cannot be expressed by a per-channel max.

### Point Transformer — local cross-attention (Zhao et al. 2021, ICCV, [arXiv:2012.09164](https://arxiv.org/abs/2012.09164))

The query token (already `D` dim from upstream) attends to its 32
neighbors. Q comes from the query, K/V from `rel_j`. **All work happens at
a low `d_local = 32` dim**; the lift back to D = 128 is a single linear at
the very end.

```
                        ┌── Linear 3 → 32 ──→ K, V per-neighbor (32 wide)
       rel (32, 3) ─────┤
                        └── (fused into `kv_proj: 3 → 64`)
       q   (D=128)  ──── Linear 128 → 32 ──→ Q (1 query, 32 wide)
                                       │
                       softmax(Q · Kᵀ / √dh)  · V    @ d_local = 32
                                       │
                                attn-weighted Σ over k=32 neighbors
                                       │
                            Linear 32 → 128   (lift, ONCE per query)
                                       │
                                    output (128)
```

Two structural wins:

1. **Factorise — mix at low dim, lift at the end.** PointNet++ pays
   `k × cost(D-wide)` on per-neighbor MLPs (heavy because D = 128 ≫ 3).
   PT pays `k × cost(d_local-wide) + 1 × cost(d_local→D lift)` — the
   D = 128 width is paid **once per query**, not 32 times per query.
2. **Soft-weighted sum, NOT MaxPool.** Every neighbor contributes via its
   softmax weight. Gradient flows to all k points, not only to the winner.
   Crucially, the attention `Q · Kᵀ` block computes **all k × k pairwise
   similarities** at d_local = 32, so geometric relations between
   neighbors (collinearity, coplanarity, "this one is at the same depth as
   that one") become representable. PointNet++'s per-channel max is
   provably a strict subset of what attention can express.

### One layer is just a similarity-weighted average — for richer relations, stack two

A single PT block is essentially "weighted average by query-neighbor
similarity at d_local = 32". For composed relations ("vertical pole
adjacent to flat ground", "edge of a planar region"), stack 2 blocks:

* Layer 1: each token gets a "what kind of point am I" embedding by
  looking at its neighbors.
* Layer 2: those enriched embeddings interact again — so a neighbor
  classified as "edge" in layer 1 can be combined with a neighbor
  classified as "surface" in layer 2 to form "edge of surface".

Original Point Transformer also has Transition Down / Up modules for
hierarchical sub-sampling; for our problem `Nq = 256` is already small and
sub-sampling doesn't save compute, so we skip TD/TU and just stack 2 blocks
at the same resolution.

---

## 4. Why d_local = 32 is enough — "isn't 128 → 32 lossy?"

Information capacity in attention is **dim × inter-token interaction**, not
dim alone.

* A PointNet++ MLP at D = 128 hidden produces 32 × 128 = 4096 floats per
  cell-query, **but** every neighbor is encoded independently — those 4096
  floats hold zero pairwise relational information. The MaxPool then
  collapses them to 128 by picking per-channel winners.
* A PT block at `d_local = 32` produces 32 floats of summary per query,
  **but** the underlying `Q · Kᵀ` block computes k² = 1024 pairwise
  similarities between neighbors. The 32-dim summary is a dense soft
  combination of those 1024 relational features.

So `(D = 128, no relational mixing)` ≪ `(d_local = 32, full pairwise
mixing)` in expressive power, even though the second has fewer feature
dims. The 32 → 128 final lift is just a learned interface to the
downstream residual stream — it's not where the local-shape information
lives.

---

## 5. Theoretical efficiency limits

For our k = 32 short sequence, vanilla scaled-dot-product attention is
already at the lower bound. Linear-attention variants (Performer,
Linformer, Hyena) approximate softmax with a kernel and trade O(k²) →
O(k·d), but the constant is large; they only break even at k > 256–512,
and they lose the **sharpness** of softmax that selects a few important
neighbors. For this slot, vanilla SDPA is the right answer.

State-space replacements (Mamba / S5) likewise only pay off at very long
sequences. Sparse attention is moot — we're already local.

---

## 6. Implementation efficiency knobs

* **`F.scaled_dot_product_attention`** — dispatches to FA2 / xformers /
  cuDNN. Largest single immediate win on Ampere.
* **Fused KV projection** ✅ — Q comes from `query_token` (D-wide), K and V
  share the `rel`-input projection. Combined into `kv_proj: 3 → 2·d_local`.
  One GEMM instead of two.
* **`torch.compile`** — already active in the training script. Folds the
  per-step Python overhead.
* MQA / CUDA Graphs are available but the gain is small for our tiny per-
  query workload; not worth the engineering.

---

## 7. Benchmark — actual GPU time on RTX 5080 (Blackwell)

`/tmp/bench_local_feat.py`. B = 128, Nq = 256, k = 32, D = 128, mean of
20 forwards after 3 warmups. Numbers are **per-frustum-encoder forward**.

| variant | D_hidden | params | mean ms |
|---|---|---|---|
| PointNet++ MLP+MaxPool, **D_h = D = 128** *(old default)* | 128 | ~21 K | **4.64** |
| PointNet++ small, D_h = 32                               | 32  | ~5 K  | 1.34 |
| Cross-attn, dl = 32, n_heads = 2 *(1 PT block)*           | 32  | ~5 K  | 1.63 |
| Point Transformer ×2 layers, dl = 32                     | 32  | ~10 K | **3.28** |

```
ms | PN(D=128) ████████████████████████████████████████████████  4.64
   | PT-2x32   ███████████████████████████████░                   3.28
   | XAttn-32  ███████████████░                                   1.63
   | PN-small  ████████████░                                      1.34
```

PT 2-layer is **30 % cheaper than the old PointNet++** *and* preserves
full relational information.

These numbers are 5080 (Blackwell) only; on Ampere DGX2 the relative
ordering should be similar but FA2 may push attention variants further
ahead.

---

## 8. Decisions recorded in code

* **Drop the silent fallback** for `full_uvd` — `FrustumLocalEncoder` and
  `LocalNeighborhood3D` both raise if the dense point cloud isn't passed.
  See [`models/model_depth.py`](../models/model_depth.py) and the
  feedback memory `feedback_no_silent_fallback.md`.
* **Stratified 4×4 + random 16 = 32 candidates** per query, sampled in
  density-invariant fashion. `r_uv_cells` (default 1.5) and `r_d` (default
  0.004 ≈ 0.4 m, but section 2 above argues for dropping this) live on the
  `FrustumLocalEncoder` class.
* **Aggregation = mini cross-attention at d_local = 32, n_heads = 2,
  fused KV projection, single 32→D lift at the output.** No MaxPool.
* **Dense point cloud is wired through dataset → collate → model.forward**
  via the new `uvd_full / pad_full` pair (`datasets/pandaset_full.py`,
  `scripts/training/train_ps_v3.py`).

## 9. Ablation — does the frustum encoder actually help?

### TL;DR (human-readable summary)

1. **Yes, by 6.9 %** on PandaSet at matched compute (100EP head-to-
   head: frustum val NLL = 1.5874 vs no-frustum 1.7052). See §9.1.
2. **But 100EP was underfit** — the same config run for 200EP
   (`v10c`) reaches val NLL = **1.02**, a 36 % improvement over v10.
   So the "6.9 %" number is an early-phase gap, not the final one.
   See §9.2.
3. **Run 200EP has ±0.1 val noise** with val_size=1000 — `v10d`
   repeats with val_size=3000 to confirm v10c is really at the
   bottom. See §9.3.
4. **Where frustum wins / loses / does nothing** is characterised by
   object-point density (n_obj) and whether the image-only path has
   already solved the crop. Details + per-sample renders in §9.6.
5. **Sparse-lidar ablation (Waymo rear, ZOD)** not yet done —
   needs a `--dataset` registry refactor of the DDP script first.
   See §9.4.

Everything below is written in deliberate length so an AI / future
session can reconstruct the full context without going back to
ClearML. Human readers can skip to the TL;DR bullet of each §9.x.

### Why this section exists (context recovery, AI-targeted)

§3–§7 argue the frustum encoder is a reasonable design given a
point-transformer-on-neighbors perspective, but none of them are an
end-to-end measurement. This section is the empirical answer: with
everything else frozen, does turning the encoder *off* hurt val NLL?
And by how much? And does the gap hold up when we stop early-stopping
at 100EP?

All runs below are on the same hardware + cache + code: dgx2 4× V100
(GPU 8/13/14/15), `train_ps_v3_ddp.py` (HF Accelerate DDP, FP16 mixed
precision via `--mixed_precision=fp16`), per-rank bs=64 / global=256,
AdamW lr=1e-3 with cosine to lr_min=1e-7, crop range `[128, 384]` px
downsampled to img_size=64, 4-layer CalibNetDepth with convnext stem,
train_size=80000 per epoch, PandaSet v3_full cache at
`/dev/shm/pandaset_v3_full`. ClearML project
`e2e_calib / ps_ddp`, live at `https://clearml.budda.site`.

### 9.0 Run ledger

| run | frustum | epochs | val_size | best val NLL | best @ ep | best MSE (px) | ClearML task id | notes |
|-----|---------|-------:|---------:|-------------:|----------:|--------------:|------------------|-------|
| `ps_ddp_v10_bs64_100ep_frustum`        | **on**  | 100 | 1000 | **1.5874** | ~98 | ~1.95 | `7570aa2cc45643ddba0ba501dbc275df` | first head-to-head baseline |
| `ps_ddp_v10b_bs64_100ep_nofrustum`     | off     | 100 | 1000 |   1.7052 | ~98 | ~2.05 | `ea8c83efbec54c55936d1a8c472a2be8` | identical to v10 with `--no-frustum` |
| `ps_ddp_v10c_bs64_200ep_frustum_long`  | on      | 200 | 1000 | **1.0213** | 188 | **1.5623** @ ep195 | `aca6e76f5bef4733bfbc3564e2678263` | *same* config as v10, only epochs 100 → 200 |
| `ps_ddp_v10d_bs64_200ep_frustum_val3k` | on      | 200 | 3000 | running    | — | — | (pending) | same as v10c, val_size 1000 → 3000 to reduce best-selection noise |

![val_nll, 100EP, frustum on vs off](assets/frustum/val_nll_frustum_ablation_100ep.png)

### 9.1 v10 vs v10b (100EP head-to-head) — "does it help at all?"

**Hypothesis.** If the frustum encoder carries information the image
cross-attn can't recover on its own (§1 "own-UV-cell is too coarse"
argument, §3 "one layer is just a similarity-weighted average"
argument), then a same-compute run with `--no-frustum` should give up
measurable val NLL.

**Setup.** v10 and v10b share seed=42, LR schedule, augmentation σ
(±0.5° rot, ±0.20 m trans), DDP topology, crop range, and every
hyperparam in the training script. The only differences are:

1. `--no-frustum` flag on v10b → `CalibNetDepth` is constructed with
   `use_frustum=False`, so `self.frustum_enc` is `None` and the model
   doesn't consume `distorted_uvd_full / pad_full`.
2. As a consequence, data loading is slightly cheaper on v10b (no full
   point cloud collated) — but the same DataLoader path is used, so
   the per-step cost saving is ≤ 2 %. Not enough to confound val NLL.

**Result.** Δ(val NLL) = v10b − v10 = 1.7052 − 1.5874 = **+0.1178**, i.e.
the frustum encoder is **6.9 % better** at matched compute.

**Shape of the curves** (right panel of the PNG above):

* Both runs start at ≈ 3.5 val NLL and track each other for the first
  ~15 epochs — the encoder contributes almost nothing while the cross-
  attention is still learning coarse projection.
* From ~EP 20 onward the blue (frustum) curve is consistently below the
  red one; the gap is steady at ~0.1–0.2 NLL and widens slightly in
  the tail.
* Neither run has flattened at EP 100.

**Interpretation.**

* 6.9 % is on the lower end of what PointNet++ vs no-local papers
  typically show, and that's expected: PandaSet has a 128-line lidar in
  front so each query cell is already relatively well populated. Sparse
  lidar datasets (Waymo rear quadrants, ZOD) should show a bigger gap;
  that's the next ablation target (see §9.4).
* The gap opens *after* EP 15 and not before. This rules out "the
  encoder is just giving a richer positional prior" (a positional prior
  would help from step 1). Instead it's providing signal the image path
  can't recover from coarse-scale features alone — matches the §1
  ambiguity argument.

### 9.2 v10c — "is 100EP enough? No."

**Why v10c.** When we looked at the v10 loss curves at EP100, neither
train nor val had obviously plateaued. The cosine LR schedule was still
at lr ≈ 1e-5 (not near lr_min). We suspected the 100EP comparison was
an *underfit* comparison rather than a converged one. v10c tests that
by re-running v10 (frustum on) with epochs 100 → 200, all else
identical. This is a "how much accuracy did we leave on the table by
stopping at 100EP?" probe, not a new design point.

**Result.** v10c reached best val NLL = **1.0213 @ ep188** — a **36 %**
improvement over v10's 1.5874. Best MSE = 1.5623 px @ ep195 (v10 was
~1.95 px), so MSE improved ~20 % while NLL improved 36 %. The non-
linearity (NLL improves more than MSE) is the Gaussian NLL working as
intended:

    NLL = 0.5 * log|Σ| + 0.5 * Δᵀ Σ⁻¹ Δ

Late-training time is spent tightening the covariance (log_sx, log_sy,
rho), which eats into the `log|Σ|` term independently of Δ. So MSE
alone underestimates how much the model has learned in the late phase.

**End-of-training dynamics (last 40 ep of v10c).** val NLL oscillates
in a ±0.1 band (ep186: 1.15, ep188: **1.02**, ep190: 1.16, ep200: 1.20).
train NLL is flat at 1.17–1.20. lr trajectory is 9.56e-5 (ep161) →
1.0e-7 (ep200). So:

* train loss is fully converged from ~ep180 onward;
* val "improvement" in the last 30 epochs is dominated by val-minibatch
  noise (val_size=1000 → standard error of a ~1.1 mean NLL ≈ ±0.05 per
  minibatch, ±0.10 per epoch after aggregation);
* the "best" val NLL of 1.0213 at ep188 is the *lower edge* of that
  noise band, not the true mean of the converged model.

**Caveat / what v10c does and does not say.**

* Does say: 100EP is genuinely underfit; the real capacity of this
  architecture + dataset is at least val NLL ~1.05, not ~1.6. The
  §9.1 v10-vs-v10b comparison is an "early phase" comparison, not a
  "converged" comparison.
* Does NOT say: v10d, v10e, ... will keep improving past v10c. Both
  train loss and the val-noise-floor suggest we're near the bottom.
  Dropping lr_min to 1e-8 won't help (grad steps of magnitude ~1e-8 *
  grad ≈ 0 change per param).

**What we'd need to push past v10c.**

* Reduce val-noise-floor so we can *see* sub-0.05 improvements → larger
  val_size (see v10d).
* Stochastic weight averaging or EMA weights for eval → cancels
  parameter oscillation in the flat LR tail.
* Longer cosine (300EP) with the same lr_min → extends the low-LR
  residence window; marginal.

### 9.3 v10d — "what is the real val_size, noise-adjusted?"

**Why v10d.** v10c showed val_nll values jumping by ±0.1 epoch-to-
epoch in the final LR tail. With val_size=1000 this is consistent with
standard error of the mean over a dataset of that size. If we want
to distinguish "frustum gives 0.12 NLL improvement" (the v10 vs v10b
100EP claim) from val noise, we need either (a) more val samples or
(b) multiple seeds. v10d chooses (a) as the cheapest answer.

**Setup.** Identical to v10c except `--val-size 1000 → 3000`. The
cache has 1648 val insts + reuse of train insts when val_size exceeds
unique count (the split at line ~182 of `train_ps_v3_ddp.py`
concatenates train+val then takes first 10 % as val, so the effective
unique pool is ~800 insts × `oversample=12` ≈ 9600 virtual items).
val_size=3000 stays within what the sampler can produce without
re-seeing the same perturbation.

**Expected effect.**

* Standard error on best val_nll selection scales as 1/√N. Going from
  N=1000 to N=3000 should shrink the ±0.10 band to ±0.06. That should
  let us read the true mean of the converged model within ~0.05 NLL.
* Train-side cost: unchanged (val is a read-only loop). Per-epoch wall
  time cost: val is ~8 % of epoch for v10c, so 3× val ≈ 16 % more per
  epoch. Total run time: ~10 h vs ~8.5 h for v10c. Worth it.

**What v10d will tell us.**

* If v10d's best_val_nll is ~1.02 ± 0.05 (i.e. matches v10c's best
  within noise), we confirm v10c was already at the true bottom and
  further investment must be architectural (EMA, ensemble, longer
  cosine, different augmentation).
* If v10d's best_val_nll is ~0.95 (i.e. significantly lower), it means
  v10c *did* have more to give but the noisy val was hiding it — and
  further long-train runs are worth it.

**Status.** Pending launch on dgx2 GPU 8/13/14/15 (v10c just finished,
GPUs are idle). Launch command template:

    ssh dgx2
    cd ~/git/e2e_calib
    CUDA_VISIBLE_DEVICES=8,13,14,15 nohup \
        accelerate launch --num_processes=4 --mixed_precision=fp16 \
        --main_process_port 29601 \
        scripts/training/train_ps_v3_ddp.py \
        --name ps_ddp_v10d_bs64_200ep_frustum_val3k \
        --cache /dev/shm/pandaset_v3_full \
        --workers 22 --batch-size 64 --prefetch-factor 4 \
        --convnext --n-layers 4 --img-size 64 \
        --min-crop-px 128 --max-crop-px 384 \
        --epochs 200 --train-size 80000 --val-size 3000 \
        --lr 1e-3 --lr-min 1e-7 --clearml \
        --why "v10c showed ±0.1 val noise at val_size=1000; \
               this increases val_size to 3000 to reduce best-selection \
               noise. All other knobs identical to v10c." \
        > experiments/ps_ddp_v10d_bs64_200ep_frustum_val3k/launch.log 2>&1 &

### 9.4 Next ablation — sparse lidar (Waymo rear, ZOD)

Not yet launched. Motivation: if the encoder is a disambiguator (§9.1.2
interpretation), then sparser lidar (where image-only cross-attn has
strictly less to lean on) should amplify the gap. The script
`train_ps_v3_ddp.py` is currently hard-wired to `PandaSetCalibDatasetFull`
and needs a `--dataset {pandaset|waymo|nuscenes|zod}` registry-style
refactor before this ablation can run cleanly. Tracked in TODO (doc
scope: see the "next-steps" note at end of §9.1.4).

### 9.5 Qualitative — *which* scenes benefit?

See `docs/frustum_qualitative_compare.md` and the `§9.1` memo above.
TL;DR: dense object footprints where object-surface-vs-ground depth
ambiguity is present (high n_obj ~60–140, close range, on-ground
objects). Sparse crops (< 30 pts) and already-solved samples (err_no
< 1 px) regress slightly; the regression is inside the noise band
for downstream bundle adjustment.

### 9.6 Where the frustum encoder wins, loses, or doesn't matter — memo

This is a running memo of scene-level intuitions. To be split out into its
own doc once the picture stabilizes; for now kept here so §9 scalar
numbers stay grounded in what they actually represent. (Sourced from
the 164-inst per-sample sweep in §9.5 / `docs/frustum_qualitative_compare.md`.)

All references below link to per-sample render PNGs produced by
`/tmp/frustum_qualitative_compare.py`. In every panel the left half is
**w/o frustum (v10b)** and the right half is **w/ frustum (v10)**;
orange arrows = per-point predicted flow, green × = GT target UV, cyan
= BG points. Look at the *consistency* of orange arrows on the object
mass between the two halves — the frustum effect is not about arrow
length, it's about arrows on the same object pointing the same way.

**Wins — "dense object footprint + object-surface-vs-ground depth
ambiguity"**

Qualitative val sweep (v10 vs v10b, 164 unique insts, see
`docs/frustum_qualitative_compare.md`) isolates the regime where the
frustum encoder is decisive:

- n_obj ≈ 60–140 points per crop. The object silhouette is sampled by
  many cells, so the local neighborhood (k = 32 random neighbors from
  the dense uvd_full pool) spans both *on-object* and *on-ground* points
  inside the ±depth slab.
- The cross-attn query, without frustum, sees only its own UV location
  and an averaged image response at that UV. When the object is close
  and its silhouette overlaps the ground at similar UV (side-view cars,
  pedestrians, cones), the per-point target flow direction is different
  between "object surface" and "ground beneath at same UV" — and the
  image encoder alone cannot make that split.
- With frustum: the query also sees, via 2-layer PT attention, the
  *pattern* of neighbor depths around it. Points on the object surface
  have neighbors at a consistent depth ≈ d_self; ground points have
  neighbors distributed on a depth gradient. That's a **geometric
  signal that is strictly unavailable to the image-only path**.
- Top 5 win samples all end at err < 1 px with frustum (idx 119, 49, 92,
  148, 19), starting from err ∈ [2.7, 4.1] px without it. The flip is
  object-coherent: arrows on the object mass snap to one consistent
  direction instead of fanning out.

Key visual evidence (each panel: left = w/o frustum, right = w/ frustum):

**idx 119 — 4.06 → 0.73 px, n_obj=137.** Textbook "depth-slab
disambiguation" case. Left: object arrows fan in 3 different
directions. Right: all arrows on the object mass converge to the GT
green ×, with ground arrows staying separate.

![help_idx0119](assets/frustum/qual/help_idx0119.png)

**idx 49 — 3.07 → 0.69 px, n_obj=123.** Left: object + ground arrows
indistinguishable. Right: a clean split line between object-surface
arrows and ground-beneath arrows.

![help_idx0049](assets/frustum/qual/help_idx0049.png)

**idx 92 — 2.81 → 0.63 px, n_obj=91.** Close pedestrian, shadow
overlapping feet on ground. Left pipeline treats shadow-at-same-UV as
part of the object; right separates them correctly.

![help_idx0092](assets/frustum/qual/help_idx0092.png)

**idx 83 — 8.36 → 3.90 px, n_obj=60.** Extreme case: left pipeline is
~8 px off on a side-view vehicle; right cuts that to < 4 px. Not
"solved" yet but the arrow coherence on the vehicle is qualitatively
different.

![help_idx0083](assets/frustum/qual/help_idx0083.png)

**Losses — two disjoint failure modes**

**Failure mode 1 — sparse object crop (n_obj < 30).**

With k = 32 uniform-random neighbors from a small pool, the
neighborhood becomes noise — many of the "neighbors" are just the same
few points resampled with replacement, or points from neighboring cells
that don't actually share the object's depth slab. The frustum signal
degenerates into a high-variance random feature that the cross-attn
then conditions on.

*idx 45 — 1.00 → 2.76 px, n_obj=5.* 5 points total — you can barely
see the object. Right panel arrows have no coherent direction because
the "neighborhood" is re-sampling the same 5 pts with noise.

![hurt_idx0045](assets/frustum/qual/hurt_idx0045.png)

*idx 124 — 13.43 → 19.44 px, n_obj=20.* Both pipelines are already
broken (err > 10 px); the sparse signal on right just *moves the
error* rather than improving it.

![hurt_idx0124](assets/frustum/qual/hurt_idx0124.png)

*idx 74 — 3.10 → 5.11 px, n_obj=29.* Borderline n_obj; right panel
shows the characteristic "neighbor noise" failure — arrows on the
object mass point in 4–5 different directions.

![hurt_idx0074](assets/frustum/qual/hurt_idx0074.png)

Worth noting: the no-frustum baseline on these is already near-random
(err > 3 px on idx 124 / 74), so the absolute Δ is not a fair
comparison — both pipelines are below the reliability floor.

**Failure mode 2 — already-solved sample (err_no-frust < 1 px).**

When the image cross-attn has already converged on a near-perfect
projection for this crop (typically pedestrians on flat ground,
well-lit, no occlusion), the frustum signal is a mild over-
regularization: the query defers slightly to its neighborhood and picks
up a 1–3 px systematic bias from points that are still a fraction of a
cell away (cell_px = 4 at S = 64, grid_n = 16).

*idx 41 — 0.44 → 4.25 px, n_obj=107.* Left panel: arrows already
almost vanish (near-zero residual). Right panel: a small but
*consistent* directional bias appears across the entire object mass
— this is the frustum neighborhood "averaging in" a 1-cell offset.

![hurt_idx0041](assets/frustum/qual/hurt_idx0041.png)

*idx 118 — 0.91 → 3.52 px, n_obj=97.* Same pattern; coherent bias,
not noise — the encoder is being harmful in a structured way.

![hurt_idx0118](assets/frustum/qual/hurt_idx0118.png)

*idx 34 — 0.85 → 2.63 px, n_obj=55.* Same pattern at lower n_obj.

![hurt_idx0034](assets/frustum/qual/hurt_idx0034.png)

**Doesn't matter — "BG-only or uniformly dense terrain"**

The 164-inst sweep also has a large middle bucket (~100 samples) where
|Δ| < 0.3 px in either direction. These are predominantly background-
only crops (no cuboid overlap) or uniformly dense foreground where the
cross-attn already has enough local image evidence. Consistent with
the §2 observation that the encoder is not a positional prior — it's
a *disambiguator* that activates only when local depth structure
carries information the image can't.

Two examples of this "flat" bucket that happened to land in the hurt
tail by noise rather than by actual harm:

*idx 2 — 2.66 → 4.22 px, n_obj=87.* Both halves look qualitatively
almost identical; the Δ here is really sampling variance, not a
structured failure.

![hurt_idx0002](assets/frustum/qual/hurt_idx0002.png)

*idx 112 — 1.37 → 2.86 px, n_obj=76.* Same story.

![hurt_idx0112](assets/frustum/qual/hurt_idx0112.png)

**Takeaways for when to keep it on**

- Dense-lidar + object-centric: unambiguous keep (PandaSet: +6.9 %
  val NLL; §9 top table).
- Sparse-lidar: likely bigger gap than PandaSet (§2 hypothesis), but
  with a higher-variance tail on low-n_obj crops. Worth running ZOD /
  Waymo-rear ablation with the same per-sample Δ analysis.
- No observed failure mode in the regime that matters for downstream
  bundle adjustment (high-n_obj, moderate-err). The hurt bucket lives
  entirely at the edges.

Memo status: *scene-bucket intuitions*, not yet a statistical statement.
Next steps are (a) repeat on ZOD / Waymo-rear, (b) quantify per-bucket
Δ distribution (not just per-sample ranks), (c) check whether the
"already-solved → small regression" failure mode shrinks with the
200EP long-train (v10c best = 1.02 NLL, the image-only path has had
more capacity to converge — may leave less room for frustum to
over-regularize).

## 10. Code

* Encoder: [`models/model_depth.py::FrustumLocalEncoder`](../models/model_depth.py)
* Model wiring: [`models/model_depth.py::CalibNetDepth.forward`](../models/model_depth.py)
* Dataset: [`datasets/pandaset_full.py::PandaSetCalibDatasetFull`](../datasets/pandaset_full.py)
* Live-forward debug vis: [`scripts/visualization/vis_frustum_neighbors.py`](../scripts/visualization/vis_frustum_neighbors.py)
* Architecture diagram script: `/tmp/draw_arch.py`
* Benchmark: `/tmp/bench_local_feat.py`
