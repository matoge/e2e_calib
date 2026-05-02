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

## 9. Code

* Encoder: [`models/model_depth.py::FrustumLocalEncoder`](../models/model_depth.py)
* Model wiring: [`models/model_depth.py::CalibNetDepth.forward`](../models/model_depth.py)
* Dataset: [`datasets/pandaset_full.py::PandaSetCalibDatasetFull`](../datasets/pandaset_full.py)
* Live-forward debug vis: [`scripts/visualization/vis_frustum_neighbors.py`](../scripts/visualization/vis_frustum_neighbors.py)
* Architecture diagram script: `/tmp/draw_arch.py`
* Benchmark: `/tmp/bench_local_feat.py`
