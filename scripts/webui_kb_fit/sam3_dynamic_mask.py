"""SAM3 video text-prompt segmentation → SplatAD-style 2D dynamic masks.

For one tss4_calib_raw_01 sequence, run SAM3 video with text prompts
('car', 'person', 'truck', 'bus', 'bicycle', 'motorcycle') across all
frames. Output: <out>/masks/<stem>.png (uint8, 255 dynamic, 0 static).

API used:
  Sam3VideoProcessor / Sam3VideoModel
  proc.init_video_session(video=...)        → session
  proc.add_text_prompt(session, text=...)   → register a text query
  model.propagate_in_video_iterator(session) → yields per-frame masks
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor


PROMPTS = ['car', 'truck', 'bus', 'van', 'trailer', 'person', 'pedestrian',
           'bicycle', 'motorcycle']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--prompts', nargs='+', default=PROMPTS)
    ap.add_argument('--model-id', default='facebook/sam3')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cam_dir = args.seq_dir / 'tss4_fcm'
    cam_files = sorted(cam_dir.glob('*.jpg'))
    if not cam_files:
        raise SystemExit(f'no jpgs in {cam_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.out_dir / 'masks'; masks_dir.mkdir(exist_ok=True)
    prev_dir = args.out_dir / 'preview'; prev_dir.mkdir(exist_ok=True)
    print(f'[seq] {args.seq_dir.name}  n_frames={len(cam_files)}')
    print(f'[prompts] {args.prompts}')

    print(f'[load] {args.model_id}  device={args.device}')
    proc = Sam3VideoProcessor.from_pretrained(args.model_id)
    model = Sam3VideoModel.from_pretrained(args.model_id).to(args.device).eval()

    images = [Image.open(p).convert('RGB') for p in cam_files]
    video = [np.asarray(im) for im in images]
    H, W = images[0].size[1], images[0].size[0]

    union_masks = [np.zeros((H, W), dtype=np.uint8) for _ in images]
    # also keep an instance-color map for preview (same H/W)
    inst_maps = [np.zeros((H, W), dtype=np.int32) for _ in images]
    next_iid = 1
    for prompt_i, prompt in enumerate(args.prompts):
        print(f'[run] prompt="{prompt}"')
        session = proc.init_video_session(video=video, inference_device=args.device)
        proc.add_text_prompt(session, text=prompt)
        for out in model.propagate_in_video_iterator(session):
            fi = int(out.frame_idx)
            obj_to_mask = getattr(out, 'obj_id_to_mask', {}) or {}
            if not obj_to_mask:
                continue
            pf = np.zeros((H, W), dtype=np.uint8)
            for obj_id, mask in obj_to_mask.items():
                # logit (1, h, w) float32 → bilinear upsample → threshold 0.0
                t = mask.detach().to(torch.float32) if hasattr(mask, 'detach') else torch.tensor(np.asarray(mask), dtype=torch.float32)
                if t.ndim == 2: t = t[None, None]
                elif t.ndim == 3: t = t[None]
                t = torch.nn.functional.interpolate(
                    t, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
                m_bin = (t > 0.0).cpu().numpy().astype(np.uint8)
                pf = np.maximum(pf, m_bin * 255)
                # global instance id keyed by (prompt_i, obj_id)
                # +1 offset so the first obj (obj_id=0, prompt_i=0) does not
                # collide with the "unset" sentinel value (0) in inst_maps.
                key = prompt_i * 10000 + int(obj_id) + 1
                inst_maps[fi] = np.where((m_bin > 0) & (inst_maps[fi] == 0),
                                          key, inst_maps[fi])
            union_masks[fi] = np.maximum(union_masks[fi], pf)

    # stable bright distinctive colors via HSV golden-ratio hue (any iid → vivid color)
    import colorsys
    color_cache: dict[int, np.ndarray] = {0: np.zeros(3)}

    def color_for(iid: int) -> np.ndarray:
        if iid not in color_cache:
            # golden-ratio hue cycling -> distinct hues, S=0.85, V=1.0
            h = (iid * 0.6180339887) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            color_cache[iid] = np.array([r * 255, g * 255, b * 255], dtype=np.float32)
        return color_cache[iid]

    # also save instance maps as int32 PNG (16-bit will lose info, use npy)
    inst_dir = args.out_dir / 'inst'; inst_dir.mkdir(exist_ok=True)
    for p, inst in zip(cam_files, inst_maps):
        np.save(inst_dir / (p.stem + '.npy'), inst.astype(np.int32))

    for p, m, inst in zip(cam_files, union_masks, inst_maps):
        Image.fromarray(m).save(masks_dir / (p.stem + '.png'))
        # preview overlay (per-instance color)
        img_np = np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
        ovr = img_np.copy()
        unique_ids = np.unique(inst)
        for iid in unique_ids:
            if iid == 0: continue
            sel = (inst == iid)
            c = color_for(int(iid))
            a = 0.45
            ovr[sel] = (1 - a) * img_np[sel] + a * c
        Image.fromarray(np.clip(ovr, 0, 255).astype(np.uint8)).save(
            prev_dir / (p.stem + '.jpg'), quality=85)
    print(f'[wrote] {masks_dir}  ({len(cam_files)} masks)')
    print(f'[wrote] {prev_dir}')


if __name__ == '__main__':
    main()
