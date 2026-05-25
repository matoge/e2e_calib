# TSS4 fcm 22-DoF partial success — from zero-shot to sub-pixel on 90% of the frame

## TL;DR

- A model `km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2` trained **without any TSS4 data**
  (Kamikado + Woven + tmpoc only) was applied **zero-shot to TSS4 fcm**, a 3840×1659 fish-eye
  camera (FOV ≈ 130°).
- The model predicts physically correct per-tile duv on 70-80% of the frame. By aggregating
  per-tile duv into a closed-form GN, we iteratively raise the calibration:
  - **6-DoF (ω + Δt)**: 70% of the frame is good. Poles and buildings line up cleanly.
  - **13-DoF (+KB4 + p1p2)**: 80% of the frame is sub-pixel.
  - **22-DoF (+KB10 + Δt)**: 90% of the frame is sub-pixel, wrms 1.22 → 1.19 px.
- **The wall**: the outer 10% of the frame (u < 384 / u > 3456) carries non-linear distortion
  too strong for KB10 to absorb. Forcing edge cells via `edge_boost` flips roll ω_z from
  −0.19° to **−0.39°** (a 2× swing), and the KB polynomial diverges. End-band bias starts
  fighting center accuracy.
- **Counter-measures**:
  - **Plan A**: Take 1426 TSS4 frames, bake the iter2 fit into K/D/T_gt/p, slice 512×512
    tiles from the **central 70%** strip, and **re-train the model** on that data. This
    blog ends with the kick.
  - **Plan B** (future): inject ~20% UV-non-linear augmentation so a single 128×128 tile
    can carry a per-pixel-varying distortion (the left and right halves of one tile see
    different θ).

---

## 1. Setup

TSS4 fcm is a 3840×1659 fish-eye camera (FOV ≈ 130°, fx ≈ 2325 px). It is **not in the
training set**. The base model was trained on Kamikado + Woven + tmpoc for 200ep on
DGX2 (12 GPU).

**Question**: can a self-supervised calib model that has never seen this camera predict
duv well enough that, by aggregating across many frames and tiles, we can recover the
full intrinsics + extrinsics?

---

## 2. Zero-shot — 6-DoF GN

### 2.1 Per-tile duv prediction

For tile t19 (right-edge) we hard-bin the predicted duv across 1426 frames at cell=32px:

![t19 zeroshot quiver](../assets/2026-05-25_tss4_22dof_partial/00b_t19_zeroshot_quiver.png)

- (a) hard-bin mean duv per cell
- (b) info-weighted duv per cell, W⁻¹·Σ W·d (outlier-robust)
- (c) per-cell |duv| heatmap

Edges show **30 px-class** duv with a smooth, physical pattern (red as u increases, blue
along the pole direction). The model is producing **physically meaningful predictions**
on a camera it has never seen.

### 2.2 INIT vs 6-DoF fit overlay

![INIT vs 6DoF](../assets/2026-05-25_tss4_22dof_partial/00_zeroshot_init_vs_6dof.jpg)

Top row = INIT (recalibration.json as-is). Buildings, poles, road markings all snap
into place over the central 70%. The frame edges still drift.

---

## 3. 13-DoF (KB4 + tangential)

Per-tile 13-DoF GN (ω + Δt + KB4 + dfxy + dcxy + p1 + p2) on t19:

![t19 13DoF resid](../assets/2026-05-25_tss4_22dof_partial/01b_t19_kb13dof_resid.png)

- (a) observed duv
- (b) fit-predicted duv
- (c) residual = obs − pred

Central and mid-range cells converge cleanly, but the far edges retain duv. The
weighted RMS drops obs 4.74 px → resid 2.07 px.

---

## 4. 22-DoF (KB10 + Δt) iter1

Extend KB to K=10 and add a 3-DoF Δt (rear→cam translation perturbation):

![iter1 22DoF resid](../assets/2026-05-25_tss4_22dof_partial/01_iter1_22dof_resid.png)

- Full-frame wrms = 1.22 px (just barely sub-pixel)
- ω = (0.07°, 0.13°, **−0.19°**) — yaw/pitch near zero, the −0.19° roll is real
- Δt = (−32, −2, +51) mm — about 5 cm forward, real-world physical perturbation

### KB4 vs KB10 overlay

![KB4 vs KB10](../assets/2026-05-25_tss4_22dof_partial/03_kb4_vs_kb10.jpg)

KB4 still leaves 5-10 px on pole tips at the edge; KB10 drives them sub-pixel.
But the very outer 5% (u < 200 / u > 3640) is still red.

---

## 5. iter2 chain — 90% sub-pixel

Bake the iter1 fit into the inst's K/D/T_gt/p, re-forward the model, and run
22-DoF GN again, initialized from iter1:

![iter1 → iter2 resid](../assets/2026-05-25_tss4_22dof_partial/06_resid_iter1_to_iter2.png)

Central 70-80% is fully sub-pixel. The remaining red is at the **outer 10%**
of u and at the **upper v-band boundary** (v ≈ 700).

### Inner-80% only

![iter2 inner-80%](../assets/2026-05-25_tss4_22dof_partial/07_resid_iter2_inner80.png)

| u-band (inner-80%) | resid (du, dv) | wrms |
|---|---|---|
| left  | (−0.19, +0.25) | |
| mid   | (+0.06, −0.16) | 1.26 px |
| right | (−0.01, +0.83) | |

→ **The central 70% of the frame is fully sub-pixel**.

---

## 6. iter3 — the edge-boost trap

The outer 10% still shows ~50 px of residual drift. GN is info-weighted, so central
cells dominate and the edges are effectively ignored. Boost edge cells by ×10:

![iter3 edge10](../assets/2026-05-25_tss4_22dof_partial/08_resid_iter3_edge10.png)

| band | resid (du, dv) | wrms |
|---|---|---|
| left  | (−0.03, +0.01) | |
| mid   | (+0.07, −0.05) | 1.18 px |
| right | (−0.04, +0.04) | |

**Numerically every band is sub-pixel.** But look at the actual overlay:

### init / iter1 / iter2 / iter3 stacked overlay

![4-stage overlay](../assets/2026-05-25_tss4_22dof_partial/04_init_iter1_iter2_iter3.jpg)

iter3 has ω_z = **−0.39°** (twice iter1's −0.19°), Δt = (−106, +91, +96) mm. KB10 also
explodes: k4 = −83.6, k7 = +379, k10 = −11.7. **Roll, Δt, and KB conspire** to fit the
edges, and central pixels visibly drift in the wrong direction even though the
weighted-RMS number went down.

---

## 7. The real wall: model output saturation

The training perturbation was σ_rot = 1.5°. The duv distribution the model can ever
produce is bounded by:

- **center** (fx = 2325): tan(1.5°) × 2325 ≈ **60 px**
- **edge** (KB poly damps apparent sensitivity): **~25 px**

When the real drift at the edge is 50 px+, the model **saturates and outputs only
~20 px**. GN, being info-weighted and center-dominated, can't make up the remainder
through KB10 alone.

That is why iter2 plateaus around wrms 1.22 → 1.19 px and edge cells stay red even
after extending the model in dimensionality.

---

## 8. Counter-measures

### Plan A — re-train on TSS4 (kicked at the end of this session)

Bake the iter2 fit into the inst, then for each of the 1426 TSS4 frames, slice
**512×512 tiles** from u ∈ [576, 3264] (central 70%) × v ∈ [600, 1194] (the v-band
the GN was already running on). Frame split 80/20 → 1140 train / 286 val.

- σ_rot = **2.0°** (edge ±33 px ≈ half of img128's 64 px tile-half — safe)
- σ_trans = 20 cm
- img_size = 128, cs ∈ [256, 512] random crop
- oversample = 16, 30 epochs

Why drop the outer 10%: iter1-3 demonstrated that the edge non-linearity is too
strong for the model class to learn. Train on the central 70% first, get a clean
in-distribution model, **then** revisit the edges.

### Plan B — UV-non-linear augmentation (future)

A 128×128 tile sees duv that varies non-trivially across its width when the underlying
fish-eye geometry is steep. To make the model robust to this, inject ~20% augmented
samples per batch where a thin-plate spline / local-affine UV warp is applied to the
input image and the implied duv shift is added to the supervision. K_aug = A @ K does
not capture this — it has to be a duv-direct augmentation.

---

## 9. Numbers

| iter | dataset | wrms | ω (°) | Δt (mm) | notes |
|---|---|---|---|---|---|
| iter0 (INIT) | TSS4 | — | (0, 0, 0) | (0, 0, 0) | recalibration.json |
| iter1 (13-DoF) | TSS4 | 2.07 | — | — | KB4 + p1p2 |
| iter1 (22-DoF) | TSS4 | 1.22 | (0.07, 0.13, **−0.19**) | (−32, −2, +51) | KB10 + Δt |
| iter2 (22-DoF) | TSS4 | 1.19 | (0.16, 0.08, −0.20) | (−78, +78, +73) | iter1 baked into npz |
| iter3 (22-DoF, edge×10) | TSS4 | 1.18 | (0.03, 0.15, **−0.39**) | (−106, +91, +96) | edges forced; KB diverges |

Inner-80% / central-70% evaluation **already fully sub-pixel at iter2.** Pushing the
last 10% of the frame to sub-pixel requires re-training (Plan A).

---

## 10. Next session (kicking now)

- [ ] Build tss4_slow_tiled_v1 LMDB (1426 frames × central 70% × v-band × 512×512,
      iter2 fit baked in)
- [ ] tss4-only 30 ep, oversample=16, σ=2°/20cm
- [ ] If that converges, kick a 4-DS mix: km + wv + wm + tss4

---

*Author: hfunaya / base model: km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2*
