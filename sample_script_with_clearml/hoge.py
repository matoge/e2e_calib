#!/usr/bin/env python3

import sys
from pathlib import Path
from clearml import Task, Model
import torch
import os
from util import create_data
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.model_depth import CalibNetDepth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 128
# expect image_*.png and points_V_*.txt are under the folder.
input_image_path = os.path.expanduser(
    "./data/image_0.png"
)
TASK_ID = "80f946cadb1747e49b8f51603e2107da"
CFG = dict(
    name="km_wv_wm_dgx1_n4_v4",
    n_layers=4,
    img_size=128,
    in_channels=3,
    use_convnext=False,
    use_frustum=True,
    epochs=50,
    batch_size=192,
    lr=0.001,
    lr_min=1e-06,
    val_fraction=0.1,
    split_seed=42,
    min_crop_px=128,
    max_crop_px=384,
    cache="/cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i",
    oversample=4,
    num_workers=8,
    deform_mode="sl",
    val_size=800,
)


def create_model():
    c = CFG
    model = CalibNetDepth(
        img_size=c["img_size"],
        in_channels=c["in_channels"],
        n_layers=c["n_layers"],
        self_first=c.get("self_first", False),
        use_convnext=c.get("use_convnext", False),
        use_frustum=c.get("use_frustum", True),
        frustum_dense=c.get("frustum_dense", False),
        use_lidar_kv=c.get("use_lidar_kv", False),
        use_pose_emb=c.get("use_pose_emb", False),
        deform_mode=c.get("deform_mode", "none"),
        use_frame_pose=c.get("use_frame_pose", False),
        frame_pose_dof=c.get("frame_pose_dof", 6),
        frame_pose_full_cov=c.get("frame_pose_full_cov", False),
        use_intensity=c.get("use_intensity", True),
    ).to(DEVICE)
    return model


def build_model_input(points, image, relative_center_from_img_center):
    # crop points and img
    if True:
        H, W = image.shape[:2]
        cx, cy = W // 2 + relative_center_from_img_center[0], H // 2 + relative_center_from_img_center[1]
        half = IMG_SIZE // 2
        x0 = int(cx - half)
        y0 = int(cy - half)
        x1 = x0 + IMG_SIZE
        y1 = y0 + IMG_SIZE
        # 画像境界処理
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)
        crop = image[y0:y1, x0:x1]
        # 足りない場合padding
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=crop.dtype)
        img[: crop.shape[0], : crop.shape[1]] = crop
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        imgs = img.unsqueeze(0)
        # --- lidar projection ---
        pts = []
        for p in points.copy():
            p[0] = p[0] - x0
            p[1] = p[1] - y0
            if p[0] < 0 or p[0] >= IMG_SIZE or p[1] < 0 or p[1] >= IMG_SIZE:
                continue
            pts.append(p)
        pts = np.array(pts, np.float32)
    # set points and img into torch style
    if True:
        point_in = torch.from_numpy(pts).unsqueeze(0)
        pad_mask = torch.zeros((1, pts.shape[0]), dtype=torch.bool)
        vfp = torch.tensor([IMG_SIZE], dtype=torch.float32)
        # --- frustum buckets ---
        G = 16
        K = 32
        bucket_uvd = torch.zeros((1, G * G, K, 4))
        bucket_valid = torch.zeros((1, G * G, K), dtype=torch.bool)
        counts = np.zeros(G * G, int)
        for u, v, d, intensity in pts:
            gx = int(u / IMG_SIZE * G)
            gy = int(v / IMG_SIZE * G)
            if gx < 0 or gx >= G or gy < 0 or gy >= G:
                continue
            idx = gy * G + gx
            k = counts[idx]
            if k >= K:
                continue
            bucket_uvd[0, idx, k] = torch.tensor([u, v, d, intensity])
            bucket_valid[0, idx, k] = True
            counts[idx] += 1
        imgs = imgs.to(DEVICE)
        point_in = point_in.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        vfp = vfp.to(DEVICE)
        bucket_uvd = bucket_uvd.to(DEVICE)
        bucket_valid = bucket_valid.to(DEVICE)
    return imgs, point_in, pad_mask, vfp, bucket_uvd, bucket_valid


def main():
    model = create_model()
    task = Task.get_task(task_id=TASK_ID)
    artifact = task.artifacts["best_model.pt"]
    ckpt_path = artifact.get_local_copy()
    print("checkpoint:", ckpt_path)
    # --- state_dict読み込み ---
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    lidar_points_in_frame, undistorted_image, cameraMatrix = create_data(input_image_path)
    relative_center_from_img_center = (-200, 400)
    imgs, point_in, pad_mask, vfp, bucket_uvd, bucket_valid = build_model_input(
        lidar_points_in_frame, undistorted_image, relative_center_from_img_center
    )
    with torch.no_grad():
        params = model(
            imgs,
            point_in,
            key_padding_mask=pad_mask,
            vfp=vfp,
            bucket_uvd=bucket_uvd,
            bucket_valid=bucket_valid,
            pose_emb_se3=None,
        )
    # plot original points
    if True:
        img = undistorted_image.copy()
        for p in lidar_points_in_frame:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
        H, W = img.shape[:2]
        center = (W // 2 + relative_center_from_img_center[0], H // 2 + relative_center_from_img_center[1])
        cv2.circle(img, center, 10, (0, 0, 255), -1)
        L = IMG_SIZE // 2
        cx, cy = center
        p1 = (int(cx - L), int(cy - L))
        p2 = (int(cx + L), int(cy + L))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.imwrite("hoge.png", img)
    # plot original points and deformed points
    if True:
        img = imgs[0].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img_ori = img.copy()
        x0 = int(center[0] - IMG_SIZE // 2) * 0
        y0 = int(center[1] - IMG_SIZE // 2) * 0
        pred = params[0].cpu().numpy()
        pts = point_in[0].cpu().numpy()
        for p, r in zip(pts, pred):
            u, v = p[0], p[1]
            du, dv = r[0], r[1]
            u2 = u + du
            v2 = v + dv
            p1 = (int(u), int(v))
            p2 = (int(u2), int(v2))
            cv2.circle(img_ori, p1, 2, (0, 0, 255), -1)
            cv2.circle(img, p2, 2, (0, 255, 0), -1)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        img_ori = cv2.resize(img_ori, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        img = cv2.vconcat([img_ori, img])
        cv2.imwrite("piyo.png", img)


if __name__ == "__main__":
    main()
