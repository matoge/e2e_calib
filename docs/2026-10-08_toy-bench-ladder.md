# Synthetic-data bench: where training breaks (2026-10-07 to 08)

## Summary

- With points on a LiDAR grid, CalibNetDepth with global cross-attention cannot correct the object points (8.51 px). Switching cross-attention to Deformable (reference point = each point's own uv) brings the same data down to 1.66 px. It does not go below 1 px.
- Keeping global attention and making the positional embedding sharper, or removing the absolute position from the local encoder's Q, stays at 7–9 px.
- The only run below 1 px is the old data (two thirds of the points packed onto the objects — too easy a problem) at 0.95 px.

## 1. Object-point error when changing one thing at a time from the old data

Model: CalibNetDepth (3 layers), 128 px, NLL loss (mean over all points), 80 epochs. Val: 800 images from seed 700000. No correction ≈ 11 px.

| step | change from the previous step | cross-attention | local encoder | ep80 object | ep80 background |
|---|---|---|---|---|---|
| 0 | old data (2 objects + background, 255 points at 1:1:1, greyscale, fixed depth) | global | no | 0.95 | 1.99 |
| 2 | greyscale → random-colour RGB | global | no | 1.04 | 1.83 |
| 3 | fixed depth → random | global | no | 1.14 | 1.90 |
| 4 | points on an 8 px LiDAR grid (by area; object points ≈ 170 → median 20) | global | no | **8.51** | 2.70 |
| 4a | uniform random instead of grid (same point count as step 4) | global | no | 10.72 | 1.19 |
| 4f | step 4 + local encoder | global | yes | 7.03 | 2.63 |
| 4f-1 | local encoder Q from (d, intensity) only | global | yes | 7.39 | 2.67 |
| 4f-PE | sharper positional embedding (same sinusoids for image and points, periods 2 images to 4 px) | global | yes | 8.83 | 2.05 |
| **4f-D** | Deformable cross-attention (reference point = own uv) | Deformable | yes | **1.66** | 1.02 |
| 4f-D g1 | layer 1 global, layers 2–3 Deformable | mixed | yes | 2.16 (stopped at ep47) | 1.31 |
| 4f-D grid4 | 4 px grid (≈ 1024 points per image) | Deformable | yes | 1.57 (stopped at ep31) | 0.92 |

Points of steps 3 and 4 (same seed; only the point placement differs):

![](_figs/2026-10-08/ladder3_vs_4_points.png)

Step 4 (global attention) and step 4f-D (Deformable), ep80, first val image:

![](_figs/2026-10-08/ladder4_ep80_val0.png)
![](_figs/2026-10-08/ladder4fD_ep80_val0.png)

## 2. What happens with global attention

- In some runs the prediction barely changes when the image is swapped for another sample's (in the run where only the object moves and the background is fixed, the prediction changes by 0.02 px). It ignores the image and keeps outputting the mean (zero shift).
- Even with the object shift a constant shared by all samples, the output stayed at 0 for about 375 updates and then dropped suddenly.
- The loss includes the object points and gradients do flow. On the same 16 images it memorises 9.99 → 1.03 px in 200 updates.
- Depth is visible: overwriting the object points' depth with the background value moves the step-4 prediction by 12.96 px. It knows a point is on the object but does not read that object's shift from the image.
- In a minimal 16×16 problem (points only on object pixels, all with the same ±3 px shift), global attention reaches 0.14–0.16 px.

## 3. Loss (1 object, 64 px bench)

| loss | object points |
|---|---|
| NLL (mean over all points) | 9.1–15.4 px even after 400 updates on one batch (σ widens, μ does not move) |
| μ by distance, σ by NLL with μ stopped (split) | 5.47 px (100 epochs) |
| split with object and background weighted half/half (split_grp) | 2.60 px (median 2.00 px even at depth difference 0.05–0.10) |

Object points are 6.8% of all points. With the all-point mean, background points make up over 90% of the loss.

## 4. Not done yet

- Variable density like a LiDAR (horizontal and vertical spacing separately 4–24 px), with points scattered outside the image too and only those inside kept after the shift. The two runs (with and without the local encoder) had not finished one epoch when the session died.
- Finishing the 4 px grid run and the layer-1-global run.
- Checking inference without GT (shuffled point order, unperturbed input, self-added shifts).
- Whether the gap between global attention and Deformable remains when shifts come from one pose (shared rotation, translation as 1/d) instead of per-object independent shifts.
- Going back to PandaSet (CalibNet2). ps_s1_refq with the fixed reference point: 6.50 px at 10 epochs. Reference point = own uv (`--ref-mode query`) stopped at ep3. (Later done: see [2026-10-08_pandaset-calib-fixes.md](2026-10-08_pandaset-calib-fixes.md).)

## Reproduce

- Training: `GRID_CFG=configs.<config> python -u train_grid_depth.py` (configs in `configs/grid_depth*.py`)
- Data: `make_image_and_points_depth` in `datasets/synthetic.py` (steps 0–4), `make_image_and_points_lidar` (64 px bench)
- ClearML: project e2e_calib/calib, tag `toy_bench`
