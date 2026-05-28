#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.model_depth import CalibNetDepth
from clearml import Task, Model
import torch
import os
from util import create_data
import cv2
import numpy as np

top_y = 1000
bottom_y = 1600
left_x = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
input_image_path = os.path.expanduser("./data/image_0.png")


def build_model_input(points, image, relative_center_from_img_center, cameraMatrix, cut_img_size):
    # crop points and img
    if True:
        H, W = image.shape[:2]
        cx, cy = W // 2 + relative_center_from_img_center[0], H // 2 + relative_center_from_img_center[1]
        half = cut_img_size // 2
        x0 = int(cx - half)
        y0 = int(cy - half)
        x1 = x0 + cut_img_size
        y1 = y0 + cut_img_size
        # 画像境界処理
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)
        crop = image[y0:y1, x0:x1]
        # 足りない場合padding
        img = np.zeros((cut_img_size, cut_img_size, 3), dtype=crop.dtype)
        img[: crop.shape[0], : crop.shape[1]] = crop
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        imgs = img.unsqueeze(0)
        # --- lidar projection ---
        mask = (
            (points[:, 0] >= x0)
            & (points[:, 0] < x0 + cut_img_size)
            & (points[:, 1] >= y0)
            & (points[:, 1] < y0 + cut_img_size)
        )

        pts = points[mask].astype(np.float32)
        pts[:, :2] -= np.array([x0, y0])
    # set points and img into torch style
    if True:
        point_in = torch.from_numpy(pts).unsqueeze(0)
        pad_mask = torch.zeros((1, pts.shape[0]), dtype=torch.bool)
        vfp = torch.tensor([cameraMatrix[0, 0]], dtype=torch.float32)
        # vfp = torch.tensor([cut_img_size], dtype=torch.float32)
        # --- frustum buckets ---
        G = 16
        K = 32
        bucket_uvd = torch.zeros((1, G * G, K, 4))
        bucket_valid = torch.zeros((1, G * G, K), dtype=torch.bool)
        counts = np.zeros(G * G, int)
        for p in pts:
            u, v, d, intensity = p[:4]
            gx = int(u / cut_img_size * G)
            gy = int(v / cut_img_size * G)
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


def plot(imgs, params, point_in):
    scale = 4
    right_x = W - left_x
    img = imgs[0].cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    img_input = img.copy()
    img_pred = img.copy()
    img_pred_with_ellipse = img.copy()
    pred = params[0].cpu().numpy()
    pts = point_in[0].cpu().numpy()
    depth = pts[:, 2]
    d_max = depth.max()
    d_min = depth.min()
    point_size = 3
    for p, r in zip(pts, pred):
        u, v, d = p[:3]
        du, dv = r[:2]
        log_sx, log_sy, rho = r[2:5]
        u2 = u + du
        v2 = v + dv
        p1 = (int(u * scale), int(v * scale))
        p2 = (int(u2 * scale), int(v2 * scale))
        c = int(255 * (d_max - d) / (d_max - d_min + 1e-6))
        cv2.circle(img_input, p1, point_size, (0, 0, c), -1)
        cv2.circle(img_pred, p2, point_size, (0, c, 0), -1)
        cv2.circle(img_pred_with_ellipse, p2, point_size, (0, c, 0), -1)
        # covariance ellipse
        sx = np.exp(log_sx)
        sy = np.exp(log_sy)
        Sigma = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
        eigval, eigvec = np.linalg.eig(Sigma)
        order = eigval.argsort()[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]
        angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
        axis1 = int(np.sqrt(eigval[0]) / 10 * 3 * scale)
        axis2 = int(np.sqrt(eigval[1]) / 10 * 3 * scale)
        cv2.ellipse(img_pred_with_ellipse, p2, (axis1, axis2), angle, 0, 360, (255, 255, 0), 1)
    vis = cv2.hconcat([img_input, img_pred, img_pred_with_ellipse])
    cv2.imwrite("bb.png", vis)


def perform_association(
    undistorted_image,
    model,
    lidar_points_in_frame,
    cameraMatrix,
    image_size
):
    H, W = undistorted_image.shape[:2]
    right_x = W - left_x
    index = 0
    point_ori = []
    point_moved = []
    ellip = []
    for y0 in range(0, H - image_size + 1, image_size):
        for x0 in range(0, W - image_size + 1, image_size):
            cx = x0 + image_size // 2
            cy = y0 + image_size // 2
            if cy < top_y or cy > bottom_y:
                continue
            if cx < left_x or cx > right_x:
                continue
            relative_center_from_img_center = (cx - W // 2, cy - H // 2)
            imgs, point_in, pad_mask, vfp, bucket_uvd, bucket_valid = build_model_input(
                lidar_points_in_frame,
                undistorted_image,
                relative_center_from_img_center,
                cameraMatrix,
                cut_img_size=image_size,
            )
            if len(point_in[0]) < 5:
                print("no point", len(point_in))
                continue
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
            # plot(imgs, params, point_in)
            index += 1
            if index%10==0:
                print(index)
            pred = params[0].cpu().numpy()
            pts = point_in[0].cpu().numpy()
            ellip += [r[2:5] for r in pred]
            point_ori += [(p[0] + x0, p[1] + y0, p[2]) for p in pts]
            point_moved += [(p[0] + r[0] + x0, p[1] + r[1] + y0, p[2]) for p, r in zip(pts, pred)]
    return point_ori, point_moved, ellip


def plot_result(undistorted_image, point_ori, point_moved, ellip):
    point_size = 3
    img = undistorted_image.copy()
    H, W = img.shape[:2]
    if True:
        img_before = img.copy()
        img_after = img.copy()
        img_ellip = img.copy()
        for (u, v, d), (u2, v2, _), e in zip(point_ori, point_moved, ellip):
            log_sx, log_sy, rho = e
            # depth色
            c = int(255 * (1.0 - d))
            p1 = (int(u), int(v))
            p2 = (int(u2), int(v2))
            cv2.circle(img_before, p1, point_size, (0, 0, c), -1)
            # 予測点
            cv2.circle(img_after, p2, point_size, (0, c, 0), -1)
            # # ---- ellipse ----
            sx = np.exp(log_sx)
            sy = np.exp(log_sy)
            Sigma = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
            eigval, eigvec = np.linalg.eig(Sigma)
            order = eigval.argsort()[::-1]
            eigval = eigval[order]
            eigvec = eigvec[:, order]
            angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
            axis1 = int(np.sqrt(eigval[0]) * 3 / 3)
            axis2 = int(np.sqrt(eigval[1]) * 3 / 3)
            cv2.ellipse(img_ellip, p2, (axis1, axis2), angle, 0, 360, (0, 0, 255), 1)
        cv2.imwrite("aa_before.png", img_before)
        cv2.imwrite("aa_after.png", img_after)
        cv2.imwrite("aa_ellip.png", img_ellip)
    img_before = cv2.imread("aa_before.png")
    img_after = cv2.imread("aa_after.png")
    img_ellip = cv2.imread("aa_ellip.png")
    window_size = 256
    x = 0
    y = 0
    right_x = W - left_x
    for x in range(0, W - window_size + 1, window_size):
        if x + window_size < left_x or x - window_size > right_x:
            continue
        crop_before = img_before[top_y:bottom_y:, x : x + window_size]
        crop_after = img_after[top_y:bottom_y:, x : x + window_size]
        crop_ellip = img_ellip[top_y:bottom_y:, x : x + window_size]
        vis = cv2.hconcat([crop_before, crop_after, crop_ellip])
        cv2.imwrite(f"slide_{x:04d}_{y:04d}.png", vis)


def create_model(c):
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


def get_model(task_id):
    task = Task.get_task(task_id=task_id)
    config = task.artifacts["config.py"]
    config_path = config.get_local_copy()
    print("config:", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
        scope = {}
        exec(content, scope)
        config = scope["CFG"]
        for key in config.keys():
            print(key, config[key])
    model = create_model(config)
    artifact = task.artifacts["best_model.pt"]
    ckpt_path = artifact.get_local_copy()
    print("checkpoint:", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config["img_size"]


def main():
    model, image_size = get_model(task_id="5e04d790db524038b3970c754bbf5fe2")
    lidar_points_in_frame, undistorted_image, cameraMatrix = create_data(input_image_path)
    point_ori, point_moved, ellip = perform_association(
        undistorted_image,
        model,
        lidar_points_in_frame,
        cameraMatrix,
        image_size
    )
    plot_result(undistorted_image, point_ori, point_moved, ellip)


if __name__ == "__main__":
    main()
