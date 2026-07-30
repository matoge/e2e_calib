"""Woven Sequence DataParser for SplatAD (PINHOLE input).

SplatAD's gsplat rasterizer expects pinhole / perspective cameras. We feed
it the *already-undistorted* fcm images that loom/tools/woven_sequence_gs/
stage 04 writes under <seq>/_gs/pinhole/camera/front_camera/.

Inputs (from one Woven sequence directory):
    metadata.json                        # poses + gicped_poses + per-frame timestamps + camera_delay_ms
    setting-<vehicle>.json               # POSLV / fcm extrinsics (used to compute T_rear_from_camera)
    saved_annotations/<stem>.json        # 3D actor cuboids (label_20000/20010/20027)
    vls128_rear_axle/<stem>.npz          # per-frame LiDAR (xs,ys,zs,intensity, lidar_point_gps_timestamp_ns)
    _gs/pinhole/                         # produced by stage 04
        camera/front_camera/*.jpg        # pinhole jpgs (3840x1350)
        camera/front_camera/intrinsics.json   # fx fy cx cy width height (no KB)
        recalib_pinhole.json             # cropped pinhole calib

Coordinate convention:
    world := rear_axle @ frame0 LiDAR-sweep time  (= identity)
    camera_to_world = T_world_from_rear @ T_rear_from_camera_static
        with T_world_from_rear interpolated to the camera shutter time
    lidar_to_world = T_world_from_rear at LiDAR sweep time
        (vls128 is mounted at rear_axle origin per setting-*.json::vls128.mp/rot)
    actors: lifted from saved_annotations.attributes.3dbb_rear_axle to world
        via T_world_from_rear at the per-frame LiDAR time.

Times are in seconds, anchored at the first frame's LiDAR-sweep time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _R

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.lidar_elevation_mappings import (
    VELODYNE_128_ELEVATION_MAPPING,
)


HORIZONTAL_BEAM_DIVERGENCE = 3e-3  # rad (Velodyne ~ 3 mrad)
VERTICAL_BEAM_DIVERGENCE = 1.5e-3

# rear_axle (x-forward, y-left, z-up) → camera optical (RDF, x-right, y-down, z-forward)
# matches scripts/splatad_kb/woven_parser_pinhole.py.
R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)

DYN_LABELS = ("label_20000", "label_20010", "label_20027")


@dataclass
class WovenDataParserConfig(ADDataParserConfig):
    """Config for one Woven sequence (pinhole-mode for SplatAD)."""

    _target: Type = field(default_factory=lambda: WovenDataParser)
    data: Path = Path(
        "/raid/home/hfunaya/woven_canary_local/canary_unilab/test01/"
        "sequence=ip708-lidar0-1451775030500023000-1451775035499970000"
    )
    """Absolute path to one Woven sequence directory."""
    vehicle: str = "ip708"
    """vehicle id; setting-<vehicle>.json must exist under `data`."""
    cameras: Tuple[Literal["fcm"], ...] = ("fcm",)
    lidars: Tuple[Literal["vls128"], ...] = ("vls128",)
    annotation_interval: float = 0.1
    correct_cuboid_time: bool = True
    min_lidar_dist: Tuple[float, float, float] = (1.0, 1.5, 1.5)
    rolling_shutter_time: float = 0.03
    time_to_center_pixel: float = 0.0
    add_missing_points: bool = False
    """vls128 channel layout != Velodyne128 physical, so leave off."""
    lidar_elevation_mapping: Dict[str, Dict] = field(
        default_factory=lambda: {"vls128": VELODYNE_128_ELEVATION_MAPPING}
    )
    """unused unless add_missing_points=True; provided so the base class
    config validates."""
    skip_elevation_channels: Dict[str, Tuple] = field(
        default_factory=lambda: {"vls128": tuple()}
    )
    lidar_azimuth_resolution: Dict[str, float] = field(
        default_factory=lambda: {"vls128": 0.2}
    )


# ───────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────

def _euler_zyx_to_R(euler_xyz_rad: List[float]) -> np.ndarray:
    """setting-*.json::fcm.rot is [roll, pitch, yaw] (rad), rot_order zyx."""
    roll, pitch, yaw = euler_xyz_rad
    return _R.from_euler("zyx", [yaw, pitch, roll]).as_matrix()


def _build_rear_to_camera(setting: dict) -> np.ndarray:
    """Static rigid: rear_axle frame → camera optical frame (= same as
    woven_parser_pinhole.py R_w2c_static / t_w2c_static, then inverted).
    """
    fcm = setting["fcm"]
    poslv = setting["poslv"]
    mp_fcm = np.asarray(fcm["mp"], dtype=np.float64)
    R_fcm = _euler_zyx_to_R(fcm["rot"])
    mp_poslv = np.asarray(poslv["mp"], dtype=np.float64)
    R_poslv = _euler_zyx_to_R(poslv["rot"])
    R_w2c_static = R_TO_RDF @ R_fcm.T @ R_poslv
    t_w2c_static = R_TO_RDF @ R_fcm.T @ mp_poslv - R_TO_RDF @ mp_fcm
    T_pscam_from_rear_static = np.eye(4)
    T_pscam_from_rear_static[:3, :3] = R_w2c_static
    T_pscam_from_rear_static[:3, 3] = t_w2c_static
    return np.linalg.inv(T_pscam_from_rear_static)


def _interp_T_world_from_rear(
    t_lidar_ns: np.ndarray,
    T_world_rear_at_lidar: np.ndarray,
    t_query_ns: np.ndarray,
) -> np.ndarray:
    from scipy.spatial.transform import Slerp
    R_knots = _R.from_matrix(T_world_rear_at_lidar[:, :3, :3])
    slerp = Slerp(t_lidar_ns.astype(np.float64), R_knots)
    out = np.tile(np.eye(4, dtype=np.float64), (len(t_query_ns), 1, 1))
    for i, t in enumerate(t_query_ns):
        t = float(np.clip(t, t_lidar_ns[0], t_lidar_ns[-1]))
        R_q = slerp([t]).as_matrix()[0]
        t_xyz = np.array([
            np.interp(t, t_lidar_ns, T_world_rear_at_lidar[:, k, 3])
            for k in range(3)
        ])
        out[i, :3, :3] = R_q
        out[i, :3, 3] = t_xyz
    return out


# ───────────────────────────────────────────────────────────────────────────
# parser
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class WovenDataParser(ADDataParser):
    """One-sequence Woven SplatAD dataparser (pinhole input)."""

    config: WovenDataParserConfig

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:  # noqa: ARG002
        return 1

    @staticmethod
    def _get_pose_dict(meta: dict) -> Dict[str, list]:
        if "gicped_poses" in meta:
            gp = meta["gicped_poses"]
            return gp.get("poses", gp)
        return meta["poses"]

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        seq = self.config.data
        meta = json.loads((seq / "metadata.json").read_text())
        camera_delay_ms = float(meta.get("camera_delay_ms", 0.0))

        keys = sorted(self._get_pose_dict(meta).keys())
        T_wr_knots = np.stack([np.array(self._get_pose_dict(meta)[k])
                                for k in keys])
        T_wr_knots = np.einsum("ij,kjl->kil",
                                np.linalg.inv(T_wr_knots[0]), T_wr_knots)
        t_lidar_ns = np.array([int(k.split("_")[1]) for k in keys],
                              dtype=np.float64)

        ts_block = meta.get("timestamps", {})
        if all(k in ts_block for k in keys):
            t_cam_ns = np.array(
                [ts_block[k]["camera_timestamps"]["fcm"] for k in keys],
                dtype=np.float64,
            )
        else:
            t_cam_ns = t_lidar_ns - camera_delay_ms * 1e6

        T_wr_at_cam = _interp_T_world_from_rear(
            t_lidar_ns, T_wr_knots, t_cam_ns)

        setting_path = seq / f"setting-{self.config.vehicle}.json"
        setting = json.loads(setting_path.read_text())
        if isinstance(setting, list):
            setting = setting[0]
        T_rear_from_cam_static = _build_rear_to_camera(setting)

        # Pinhole intrinsics from stage 04 output
        pin_dir = seq / "_gs" / "pinhole"
        cam_dir = pin_dir / "camera" / "front_camera"
        intr = json.loads((cam_dir / "intrinsics.json").read_text())
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
        W, H = int(intr["width"]), int(intr["height"])

        c2w_world = np.einsum("kij,jl->kil", T_wr_at_cam, T_rear_from_cam_static)
        c2w_world[:, :3, :3] = c2w_world[:, :3, :3] @ OPENCV_TO_NERFSTUDIO

        t0 = t_lidar_ns[0]
        cam_times = (t_cam_ns - t0) * 1e-9

        # pinhole jpg paths must match keys order
        jpgs: List[Path] = []
        for k in keys:
            p = cam_dir / f"{k}.jpg"
            if not p.is_file():
                raise FileNotFoundError(f"missing pinhole jpg {p} "
                                         "(run stage 04 first)")
            jpgs.append(p)

        n = len(jpgs)
        cameras = Cameras(
            fx=torch.full((n,), fx, dtype=torch.float32),
            fy=torch.full((n,), fy, dtype=torch.float32),
            cx=torch.full((n,), cx, dtype=torch.float32),
            cy=torch.full((n,), cy, dtype=torch.float32),
            height=torch.full((n,), H, dtype=torch.long),
            width=torch.full((n,), W, dtype=torch.long),
            camera_to_worlds=torch.tensor(c2w_world[:, :3, :4], dtype=torch.float32),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(cam_times, dtype=torch.float64),
            metadata={"sensor_idxs": torch.zeros((n, 1), dtype=torch.int32)},
        )
        return cameras, jpgs

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        seq = self.config.data
        meta = json.loads((seq / "metadata.json").read_text())
        keys = sorted(self._get_pose_dict(meta).keys())
        T_wr_knots = np.stack([np.array(self._get_pose_dict(meta)[k])
                                for k in keys])
        T_wr_knots = np.einsum("ij,kjl->kil",
                                np.linalg.inv(T_wr_knots[0]), T_wr_knots)
        t_lidar_ns = np.array([int(k.split("_")[1]) for k in keys],
                              dtype=np.float64)
        t0 = t_lidar_ns[0]
        lidar_times = (t_lidar_ns - t0) * 1e-9

        l2w = T_wr_knots.copy()  # vls128 ≡ rear_axle
        n = len(keys)
        files = [seq / "vls128_rear_axle" / f"{k}.npz" for k in keys]
        for p in files:
            if not p.is_file():
                raise FileNotFoundError(p)

        lidars = Lidars(
            lidar_to_worlds=torch.tensor(l2w[:, :3, :4], dtype=torch.float32),
            lidar_type=LidarType.VELODYNE128,
            times=torch.tensor(lidar_times, dtype=torch.float64),
            metadata={"sensor_idxs": torch.zeros((n, 1), dtype=torch.int32)},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        return lidars, files

    def _read_lidars(self, lidars: Lidars,
                     filepaths: List[Path]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        times = lidars.times.squeeze(-1).numpy()  # seconds since t0
        # frame0 LiDAR ns time = 0s in our anchor → ns = times[i]/1e-9 + t0_ns
        # (we stash t0 once and reuse to convert per-point ns to relative s)
        keys = [p.stem for p in filepaths]
        t0_ns = int(keys[0].split("_")[1])

        for i, p in enumerate(filepaths):
            d = np.load(p, allow_pickle=True)
            xs = d["xs"].astype(np.float64)
            ys = d["ys"].astype(np.float64)
            zs = d["zs"].astype(np.float64)
            inten = (d["intensity"].astype(np.float64) / 255.0).clip(0, 1)
            ts_pt_ns = d["lidar_point_gps_timestamp_ns"].astype(np.int64)
            # frame nominal ns from key stem
            t_frame_ns = int(p.stem.split("_")[1])
            # per-point time relative to this frame's nominal time (so the
            # window sits roughly on [-rolling_shutter_time, 0])
            ts_pt = (ts_pt_ns - t_frame_ns).astype(np.float64) * 1e-9

            mask = ((np.abs(xs) > self.config.min_lidar_dist[0])
                    | (np.abs(ys) > self.config.min_lidar_dist[1])
                    | (np.abs(zs) > self.config.min_lidar_dist[2]))
            mask &= np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            xs, ys, zs, inten, ts_pt = (xs[mask], ys[mask], zs[mask],
                                         inten[mask], ts_pt[mask])
            pc = np.stack([xs, ys, zs, inten, ts_pt], axis=-1).astype(np.float32)
            out.append(torch.from_numpy(pc))
        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return out

    def _get_actor_trajectories(self) -> List[Dict]:
        seq = self.config.data
        meta = json.loads((seq / "metadata.json").read_text())
        keys = sorted(self._get_pose_dict(meta).keys())
        T_wr = np.stack([np.array(self._get_pose_dict(meta)[k]) for k in keys])
        T_wr = np.einsum("ij,kjl->kil", np.linalg.inv(T_wr[0]), T_wr)
        t_lidar_ns = np.array([int(k.split("_")[1]) for k in keys],
                              dtype=np.float64)
        t0 = t_lidar_ns[0]

        per_actor: Dict[str, List[Dict]] = {}
        for fi, k in enumerate(keys):
            ann_p = seq / "saved_annotations" / f"{k}.json"
            if not ann_p.is_file():
                continue
            ann = json.loads(ann_p.read_text())
            for det in ann.get("details", []):
                if det.get("label") not in DYN_LABELS:
                    continue
                a = det.get("attributes", {})
                if "3dbb_rear_axle" not in a:
                    continue
                bb = a["3dbb_rear_axle"]
                c = np.asarray(bb["center_meter"], dtype=np.float64)
                size = np.asarray(bb["size_meter"], dtype=np.float64)
                cos_y, sin_y = bb["direction"]
                yaw = float(np.arctan2(sin_y, cos_y))
                R_rear = _R.from_euler("z", yaw).as_matrix()
                T_a_rear = np.eye(4)
                T_a_rear[:3, :3] = R_rear
                T_a_rear[:3, 3] = c
                T_a_world = T_wr[fi] @ T_a_rear
                uuid = (a.get("obj_id") or det.get("object_id")
                        or f"{det['label']}_{fi}")
                per_actor.setdefault(uuid, []).append({
                    "label": det["label"],
                    "pose": T_a_world,
                    "dims": np.array([size[0], size[1], size[2]],
                                      dtype=np.float32),
                    "time": (t_lidar_ns[fi] - t0) * 1e-9,
                })

        trajs: List[Dict] = []
        for uuid, recs in per_actor.items():
            recs = sorted(recs, key=lambda r: r["time"])
            poses = torch.from_numpy(
                np.stack([r["pose"] for r in recs])).float()
            times = torch.tensor([r["time"] for r in recs])
            dims = torch.from_numpy(
                np.stack([r["dims"] for r in recs]).max(axis=0)).float()
            trajs.append({
                "poses": poses,
                "timestamps": times,
                "dims": dims,
                "label": recs[0]["label"],
                "stationary": False,
                "symmetric": True,
                "deformable": False,
            })
        return trajs

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        if not self.config.data.is_dir():
            raise FileNotFoundError(self.config.data)
        return super()._generate_dataparser_outputs(split)
