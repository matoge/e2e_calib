# Patent Claim Draft (English Version)

## Theme

**Deterministic Token Condensation Architecture Based on Predicted Residual Uncertainty Prior to Multimodal Fusion**

Before fusing LiDAR and camera modalities, predict — using LiDAR alone — the locations that will *remain* difficult even after fusion, and concentrate compute there. Neither Top-K nor random: a fully deterministic selection via Weighted Farthest Point Sampling (WFPS) driven by predicted uncertainty.

---

## Target

- Multimodal 3D perception models for autonomous driving / ADAS (LiDAR × Camera)
- Query-based detection / segmentation architectures (BEVFormer, DETR3D, Sparse4D, etc.)
- On-vehicle edge inference where determinism is mandatory (functional safety)
- Extensible to arbitrary heterogeneous sensor fusion (Radar, Thermal, IMU, etc.)

Any system that must remain real-time even as the query count scales from hundreds to thousands.

---

## Problem Statement

### 1. Latency explosion from brute-force computation
Existing methods (BEVFormer, DETR3D, etc.) push the full query set (e.g., 1000 queries) through heavy fusion (Cross-Attention), causing O(N²) cost blow-up.

### 2. Fundamental waste of post-fusion selection
DETR-style two-stage query selection prunes queries by class confidence *after* fusion. By the time pruning happens, the expensive fusion has already been paid for — the core cost is not reduced.

### 3. Information redundancy in Top-K sampling
Selecting purely by top variance/score causes queries to cluster around a single object, discarding "scene-skeleton" information such as the ground plane, distant regions, and occlusion boundaries.

### 4. Non-determinism of random sampling
Methods whose output varies across runs are unacceptable for functional-safety-constrained vehicle systems.

### 5. Fragility of mean-based prediction from a single modality
Forcing a single sensor (LiDAR) to predict the mean (class or position) directly leads to overconfident wrong answers due to missing context. Using the mean as a gate propagates this bias downstream.

---

## Proposed Approach

### Overall architecture (three stages)

```
[LiDAR point cloud] → [Backbone] → [Shared prediction head (μ, σ²)]
                                       ↓ (use σ² only, discard μ)
                                 [Residual uncertainty map]
                                       ↓
                                 [WFPS: variance-weighted farthest point sampling]
                                       ↓ (condense 1000 → 16 tokens)
                                 [Sparse Cross-Attention with image]
                                       ↓
                                 [Final prediction]
```

### Step 1: Speculative prediction of residual uncertainty
- A head over LiDAR-only local features predicts **the uncertainty σ² that will remain even after fusion**.
- The head is shared, but **the mean μ is discarded; only σ² gates sampling.**
- Trained with an Aleatoric Loss:

$$L_{aleatoric} = \sum_i \left( \frac{\|y_i - \hat{y}_i\|^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2 \right)$$

Regions that "remain hard even when combined with the image" surface as high σ².

### Step 2: WFPS (Variance-Weighted Farthest Point Sampling)
A novel deterministic sampler that simultaneously resolves the *clustering* failure of Top-K and the *non-determinism* failure of random sampling.

**Selection score:**

$$S_i = \sigma_i^2 \times \min_{j \in \text{Selected}} \| \text{pos}_i - \text{pos}_j \|$$

1. Deterministically seed with the single point of highest σ².
2. Iteratively select the point maximizing (uncertainty × distance from already-selected points).
3. Repeat until K points (e.g., 16) are chosen.

Effects:
- **Elite squad**: the hardest high-σ² locations are secured first.
- **Scene skeleton**: isolated ground/far-field points survive via the distance term.
- **Full determinism**: identical input → identical 16 tokens, every time.

### Step 3: Sparse fusion
Heavy Cross-Attention with image features is performed *only* on the 16 condensed tokens.
Cost: O(N²) → effectively constant.

### Step 4: Securing differentiability
The sampling op itself is non-differentiable, but learning proceeds through two parallel paths:
- **Deep Supervision**: σ² prediction loss is applied directly to all 1000 queries at the selector head, independent of sampling outcome.
- **Straight-Through Estimator (STE)**: gradients from the downstream task loss are routed back to the selector through the weights of the chosen 16 tokens.

---

## Existing Technical Documents / Prior Art Survey

### Category A: Confidence / Uncertainty-Based Query Selection

| Prior Art | Mechanism | Critical Difference from Our Invention |
|-----------|-----------|---------------------------------------|
| **RT-DETR "Uncertainty-Minimal Query Selection"** (Zhao et al., 2304.08069) | Selects encoder features with **minimal** uncertainty (highest localization confidence) as decoder queries (top 300) | **Opposite polarity**: they pick *low* uncertainty (= confident answers), we pick *high* uncertainty (= residual difficulty after fusion). They operate **single-modality** on image encoder features; we operate **pre-fusion** using LiDAR-only signal to gate the *upcoming* image-fusion step. |
| **DETR / Two-stage Deformable DETR** | Top-K by objectness score after encoder | Post-encoder, single-modality, "answer-near" selection. Does not use uncertainty at all. |
| **DQ-DETR** | Dynamically chooses number of queries based on predicted count | Adjusts *count*, not *which*. No pre-fusion gating. |
| **IoU-aware Query Selection** | Selects by predicted IoU | Confidence-based, not uncertainty-as-information-gain. |

### Category B: Multimodal Token Pruning

| Prior Art | Mechanism | Critical Difference |
|-----------|-----------|---------------------|
| **MADTP** (Cao et al., CVPR 2024) | Aligns features across modalities, then prunes by alignment score | Requires **both modalities present** to compute alignment scores. Our approach prunes using **only one modality** *before* the other is consulted — the entire point is to avoid invoking the expensive modality. |
| **Multimodal Token Fusion (MTF, CVPR 2022)** | Prunes single-modal transformers, then re-uses pruned units for cross-modal fusion | Operates per-modality independently; no notion of predicting where the *fusion outcome* will remain uncertain. |
| **HiRED (Attention-Guided Token Dropping)** | Drops tokens based on attention weights in VLMs | Requires attention to be computed first → not pre-fusion. |

### Category C: Point Cloud Sampling

| Prior Art | Mechanism | Critical Difference |
|-----------|-----------|---------------------|
| **Farthest Point Sampling (FPS)** | Geometric diversity only | No information weighting; treats all points equal in importance. |
| **Density-Aware FPS** (2509.13213, 2025) | Weights FPS by local point density | Density is a **geometric prior**, not a **learned signal about future fusion**. |
| **Curvature-Informed FPS** (2411.16995) | Weights FPS by surface curvature | Again, geometric prior — not predictive of cross-modal residual uncertainty. |
| **SampleNet / Gumbel Subset Sampling** | Learned, **stochastic** differentiable sampling | Non-deterministic → fails functional-safety constraints. We achieve learning via Deep Supervision + STE while keeping selection *fully deterministic*. |
| **APES (Attention-based Point Cloud Edge Sampling)** | Samples salient edges via attention | Single-modality, geometric saliency; not tied to a downstream fusion step. |
| **SAMBLE** (CVPR 2025) | Shape-specific sampling via sparse attention map | Single-modality shape analysis; no fusion gating. |
| **CP-Net (Critical Point Net)** | Deterministic sampling by contribution to global max-pool | Determinism is shared, but the *criterion* is reconstruction contribution, not fusion-residual uncertainty. |

### Category D: General Transformer Token Reduction

| Prior Art | Mechanism | Critical Difference |
|-----------|-----------|---------------------|
| **DynamicViT** | Drops tokens mid-network using attention weights | Requires attention to be computed → post-attention, not pre-fusion. |
| **US10740433B2 (Universal Transformers)** | Adaptive computation depth per token | Adjusts depth, not selection; no multimodal angle. |
| **US20230112862A1 (Reuse Transformers)** | Reuses attention scores across layers | Computation-reuse, not token selection. |
| **US20260050766 (SSM Token Compression)** | Compresses tokens via state-space model summarization | Aggregation/compression, not selection; not multimodal-gated. |
| **vAttention** | Sparse attention with (ε,δ) guarantees, mixes top-k + statistical sampling | Approximates the attention op itself; does not reduce the *number* of queries entering fusion. |

### Category E: LiDAR-Camera Fusion Architectures

| Prior Art | Mechanism | Critical Difference |
|-----------|-----------|---------------------|
| **Li-ViP3D++ (Query-Gated Deformable Fusion)** | Gates deformable cross-attention with query-derived weights | Gates *how much* each query attends, not *which* queries proceed. All queries still incur fusion cost. |
| **DeepFusion / VAD** | Uncertainty modules suppress noisy modality | Operates on **modality-level** signal quality; ours operates on **per-query** prediction of post-fusion residual difficulty. |
| **SAMFusion (Princeton, 2024)** | Sensor-adaptive fusion under weather variation | Modality weighting, not query-level resource allocation. |
| **Mobileye camera-LiDAR fusion family** | Calibration and online relative-transform inference | Different problem domain entirely. |
| **Sparse4D / SparseFusion** | Query-based sparse 4D sampling across time/modality | **Retains all queries**; reduces *attention scope per query*, not query count. |

---

## The Four Pillars of Differentiation (Defensive Logic)

### Pillar 1: Polarity Inversion — "We pick what's hard, not what's easy"
Every existing query-selection method (DETR family, RT-DETR, IoU-aware selection) selects queries that are *most confidently* assigned to objects. Our invention deliberately selects queries with *maximal residual uncertainty* — i.e., the locations the model is least confident about *even after fusion*. This is the opposite evaluation axis and reflects a fundamentally different design philosophy: **invest compute where it changes the answer, not where it confirms it.**

### Pillar 2: Pre-Fusion Gating — "We gate before the expensive modality is touched"
MADTP, HiRED, DynamicViT and similar methods prune *after* the expensive computation has at least begun (attention scores or cross-modal alignment must be computed first). Our invention gates **strictly before the second modality is consulted**, using only single-modality features. The expensive fusion path is invoked on K << N tokens. No prior multimodal token-pruning patent we found operates strictly pre-fusion using only one modality's features as the gating signal.

### Pillar 3: Variance-Only Head Output — "We discard μ, gate on σ²"
Standard practice with shared prediction heads is to *use* the mean output. We deliberately discard μ and use only σ² for gating. This prevents single-modality overconfidence from corrupting selection. No prior art surveyed structurally enforces "shared head, mean discarded, variance-only gating."

### Pillar 4: Deterministic Information-Weighted Diversity — "WFPS"
- **Plain FPS / DA-FPS / Curvature-FPS**: deterministic but weighted by *geometric priors only*.
- **Gumbel Sampling / SampleNet**: information-aware but *stochastic*.
- **Top-K**: information-aware but suffers from *clustering*.

Our **WFPS** uniquely combines all three desirable properties simultaneously:
1. Deterministic (functional-safety compatible)
2. Weighted by learned information signal (predicted residual uncertainty)
3. Spatially diverse (distance term recovers scene skeleton)

No surveyed prior art combines these three.

---

## Recommended Independent Claim 1 (Reformulated)

Given the prior art landscape, Claim 1 should explicitly bake in **all four pillars** to avoid being read on by any single prior art:

> **Claim 1.** A method for multimodal sensor fusion, comprising:
> (a) extracting local features from a *first* sensor modality;
> (b) computing, from said local features and via a prediction head that also outputs a mean estimate, a per-feature *predicted residual uncertainty* representing uncertainty expected to remain after fusion with a *second* sensor modality, **wherein said mean estimate is not used for the selection of step (c)**;
> (c) selecting a subset of features by a *deterministic* procedure that maximizes a joint criterion of (i) predicted residual uncertainty and (ii) minimum feature-space or spatial distance to already-selected features;
> (d) performing cross-modal fusion with the second sensor modality on said selected subset only, without performing said cross-modal fusion on the unselected features.

The conjunction of (b)-with-discarded-mean, (c)-deterministic-uncertainty-weighted-diversity, and (d)-pre-fusion-gating is — as far as the surveyed literature shows — novel.

---

## Tactical Note for Examiner Response

The most likely 102/103 rejection citation will be **RT-DETR (Zhao et al., 2304.08069)** for "uncertainty-based query selection." The single strongest counter-argument:

> RT-DETR selects queries with **minimum** uncertainty (high-confidence localizations) within a **single image modality** *after* the encoder has run. The present invention selects features with **maximum** uncertainty in a **single modality** *before* the second modality is invoked, for the explicit purpose of allocating cross-modal compute. The two methods optimize opposite signals at opposite points in the pipeline; combining them would not yield the present invention.

---

## Representative Embodiments (for Specification)

- Distant small objects (sparse LiDAR returns; still ambiguous after fusion)
- Occluded objects (partial-observation reasoning under shadow)
- Adverse weather boundaries (fog/rain degrade camera reliability)
- Transparent/reflective surfaces — failure modes for *both* modalities

The shared property: regions where *neither modality alone nor naive fusion suffices*. Concentrating compute here is shown experimentally to preserve accuracy while substantially increasing throughput.

---

## References

- [RT-DETR (Zhao et al., uncertainty-minimal query selection)](https://arxiv.org/abs/2304.08069)
- [MADTP — Multimodal Alignment-Guided Dynamic Token Pruning, CVPR 2024](https://arxiv.org/abs/2403.02991)
- [Multimodal Token Fusion for Vision Transformers, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.pdf)
- [Density-Aware Farthest Point Sampling (2025)](https://arxiv.org/abs/2509.13213)
- [Curvature-Informed Furthest Point Sampling](https://arxiv.org/html/2411.16995v1)
- [APES — Attention-based Point Cloud Edge Sampling, CVPR 2023](https://arxiv.org/abs/2302.14673)
- [SAMBLE — Shape-Specific Point Cloud Sampling, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_SAMBLE_Shape-Specific_Point_Cloud_Sampling_for_an_Optimal_Trade-Off_Between_CVPR_2025_paper.pdf)
- [Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling](https://ar5iv.labs.arxiv.org/html/1904.03375)
- [Li-ViP3D++ — Query-Gated Deformable Camera-LiDAR Fusion](https://arxiv.org/abs/2601.20720)
- [SAMFusion — Sensor-Adaptive Multimodal Fusion (Princeton, 2024)](https://light.princeton.edu/wp-content/uploads/2024/09/SAMFusion.pdf)
- [HiRED — Attention-Guided Token Dropping for VLMs](https://arxiv.org/abs/2408.10945)
- [DETR / End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- [USPTO 20260050766 — Efficient Attention via SSM Token Compression](https://patents.justia.com/patent/20260050766)
- [US20230112862A1 — Reuse Transformers](https://patents.google.com/patent/US20230112862A1/en)
- [US10740433B2 — Universal Transformers](https://patents.google.com/patent/US10740433B2/en)
