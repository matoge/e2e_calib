Re-ran yesterday's kamikado + WovenSequence calibration but warm-started from the nuScenes **6-cam** ckpt (`cam6_250x2_100ep`) instead of the CAM_FRONT-only one. Same data, same 30 → 50 ep recipe, held-out val:

- **F=1**: rot 0.0192° / t 5.33 mm  (was 0.0252° / 6.23 mm  → −24% / −14%)
- **F=32 gate3**: rot 0.0030° / t 0.54 mm  (was 0.0032° / 1.67 mm  → −6% / **−68%**)

Below the nuScenes-report benchmark of F=32 → 0.0068° / 1.43 mm on our own fisheye 4K data. **The model was pretrained on pinhole nuScenes and never saw a fisheye pixel during pretrain** — the transfer works because CalibNet2 doesn't bake the projection model (KB4/pinhole) into the network; the outer GN handles that, and the backbone only has to learn "what an edge looks like." cam6's rear/side cameras give it non-central-horizon priors that generalize to fisheye periphery.

Next: Waymo 5-cam pretrain (cache already built, 1.5 TB). Also worth pose-dumping our rear cameras with this ckpt — the mechanism suggests they should transfer for free.

Full write-up: `docs/assets/2026-08-31_kmwv_quick/slack_post_cam6.md`
