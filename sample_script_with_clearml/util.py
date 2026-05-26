#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
from scipy.spatial.transform import Rotation
import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d

class SE3:
    def __init__(self, R=None, T=None):
        """
        Constructor for the SE3 class

        :param R: 3x3 rotation matrix (numpy.ndarray)
        :param T: 3D translation vector (numpy.ndarray)
        """
        if R is None:
            R = np.eye(3)  # Default rotation matrix is the identity matrix
        if T is None:
            T = np.zeros(3)  # Default translation is a zero vector

        self.R = np.asarray(R)
        self.T = np.asarray(T)

    def as_matrix(self):
        """
        Return the SE(3) matrix

        :return: 4x4 homogeneous transformation matrix
        """
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.T
        return M

    def transform_point(self, point):
        """
        Apply SE3 transformation to a 3D point

        :param point: 3D point (numpy.ndarray)
        :return: Transformed 3D point
        """
        point_homogeneous = np.append(point, 1)  # Homogeneous coordinates
        transformed_point = self.as_matrix().dot(point_homogeneous)
        return transformed_point[:3]

    def inverse(self):
        """
        Return the inverse of the SE3 transformation

        :return: SE3 object representing the inverse transformation
        """
        R_inv = self.R.T
        T_inv = -R_inv.dot(self.T)
        return SE3(R=R_inv, T=T_inv)

    def __mul__(self, other):
        """
        Compute the product of SE3 transformations

        :param other: Another SE3 object
        :return: Combined SE3 object
        """
        if not isinstance(other, SE3):
            raise ValueError("The multiplicand must be an instance of SE3.")

        # Compute the combined rotation matrix and translation vector
        R_combined = self.R.dot(other.R)
        T_combined = self.R.dot(other.T) + self.T
        return SE3(R=R_combined, T=T_combined)

    def to_quaternion_and_translation(self):
        """
        Represent using quaternion and translation vector

        :return: Quaternion (numpy.ndarray), Translation vector (numpy.ndarray)
        """
        quat = Rotation.from_matrix(self.R).as_quat()
        return quat, self.T

    @staticmethod
    def from_quaternion_and_translation(quat, T):
        """
        Generate SE3 object from quaternion and translation vector

        :param quat: Quaternion (numpy.ndarray)
        :param T: Translation vector (numpy.ndarray)
        :return: SE3 object
        """
        R_matrix = Rotation.from_quat(quat).as_matrix()
        return SE3(R=R_matrix, T=T)

    def to_euler_angles(self, seq="xyz"):
        """
        Convert to Roll, Pitch, Yaw (Euler angles)

        :param seq: Order of Euler angles (default is 'zyx', i.e., yaw, pitch, roll)
        :return: Euler angles (numpy.ndarray)
        """
        euler_angles = Rotation.from_matrix(self.R).as_euler(seq, degrees=True)
        return euler_angles

    def __repr__(self):
        return "SE3(R=\n{},\n T={})".format(self.R, self.T)


def get_camera_params(target_file_path):
    data = None
    with open(target_file_path, "r") as file:
        data = json.load(file)  # Load JSON file as a dictionary
    cameraMatrix = np.array(
        data["calibration"]["intrinsics"]["camera_model"]["pinhole_parameters"]["matrix_image_camera"]["matrix"]
    ).T
    distortion_coefs = np.array(
        data["calibration"]["intrinsics"]["distortion_model"]["generic_fisheye_parameters"]["coefficients"]
    )
    quat = data["calibration"]["extrinsics"]["transform_VS"]["so3"]
    quaternion = [quat["x"], quat["y"], quat["z"], quat["w"]]
    R_VS = Rotation.from_quat(quaternion)
    T_VS = np.array(data["calibration"]["extrinsics"]["transform_VS"]["translation"]["matrix"][0])
    transform_VS = SE3(R_VS.as_matrix(), T_VS)
    return cameraMatrix, distortion_coefs, transform_VS


class GenericFisheyeDistortion:
    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        rectification_resolution=None,
        publication_resolution=None,
        interpolation_flag=cv2.INTER_LINEAR,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rectification_resolution = rectification_resolution
        self.publication_resolution = publication_resolution
        self.interpolation_flag = interpolation_flag
        self.dmap1 = None
        self.dmap2 = None
        self.new_camera_matrix = None

    def rectify(self, input_image):
        if self.dmap1 is None or self.dmap2 is None:
            self.create_distortion_map(input_image.shape[0], input_image.shape[1])

        output = cv2.remap(input_image, self.dmap1, self.dmap2, self.interpolation_flag, borderMode=cv2.BORDER_CONSTANT)

        # Crop image (implement your own cropping logic as needed)
        output = self.crop(output)

        if self.publication_resolution and output.shape[:2] != self.publication_resolution:
            output = cv2.resize(output, self.publication_resolution)
        return output

    def create_distortion_map(self, num_rows, num_cols):
        if self.rectification_resolution is None:
            self.rectification_resolution = (num_cols, num_rows)
        new_camera_matrix = self.camera_matrix.copy()
        new_camera_matrix[0, 2] += (self.rectification_resolution[0] - num_cols) / 2.0
        new_camera_matrix[1, 2] += (self.rectification_resolution[1] - num_rows) / 2.0
        self.new_camera_matrix = new_camera_matrix
        self.dmap1, self.dmap2 = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3),
            new_camera_matrix,
            self.rectification_resolution,
            cv2.CV_32FC1,
        )

    def crop(self, image):
        # Implement cropping logic here if needed
        # For now, return the image as is
        return image


class Data:
    def __init__(self):
        self.point_V = None
        self.intensity = None

    def from_line(self, line):
        if len(line) == 0:
            return False
        if line[0] == "#":
            return False
        line = line.split(" ")
        if len(line) > 3:
            self.point_V = np.array([float(line[0]), float(line[1]), float(line[2])])
            if np.linalg.norm(self.point_V) > 60.0:
                return False
            self.intensity = float(line[3])
            return True
        return False


def prepare_data(input_image_path, read_points=True):
    point_file = input_image_path.replace("image", "points_V").replace("png", "txt")
    points = []
    intensity = []
    if read_points:
        with open(point_file, "r") as f:
            for line in f:
                data = Data()
                if data.from_line(line):
                    points.append(data.point_V)
                    intensity.append(data.intensity)
    points_V = np.asarray(points, dtype=np.float32)
    intensity = np.array(intensity)
    image = cv2.imread(input_image_path)
    calib_file = "/".join(input_image_path.split("/")[:-1]) + "/calib.calib"
    cameraMatrix, distortion_coefs, transform_VS = get_camera_params(calib_file)
    cameraMatrix = cameraMatrix.copy()
    distortion_model = GenericFisheyeDistortion(cameraMatrix, distortion_coefs)
    undistorted_image = distortion_model.rectify(image)
    return undistorted_image, cameraMatrix, points_V, transform_VS, intensity    



top_y = 1000
bottom_y = 1600
left_x = 0
R_S0V0 = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

def estimate_normals(points, k=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normals = np.asarray(pcd.normals)
    return normals


def make_lidar_img_original(points_S_gt, points_S_with_noize, K, H, W, intensity):
    # normals = estimate_normals(points_S_gt, k=10)
    log_intensity = np.log1p(intensity)
    log_intensity = (log_intensity - log_intensity.min()) / (log_intensity.max() - log_intensity.min() + 1e-6)
    points = []
    max_depth = 500.
    right_x = W - left_x
    for p_S, s in zip(points_S_with_noize, log_intensity):
        if p_S[2] <= 0 or p_S[2] > max_depth:
            continue
        proj = K @ p_S
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]
        if v < top_y or v > bottom_y or u < left_x or u > right_x:
            continue
        depth = p_S[2]
        points.append([u, v, depth, s])
    points = np.array(points, dtype=np.float32)
    return points


def create_data(input_image_path):
    undistorted_image, cameraMatrix, points_V, transform_VS, intensity = prepare_data(input_image_path)
    H, W = undistorted_image.shape[:2]
    points_S = (transform_VS.inverse().R @ points_V.T).T + transform_VS.inverse().T
    lidar_points_in_frame = make_lidar_img_original(points_S, points_S, cameraMatrix, H, W, intensity)
    return lidar_points_in_frame, undistorted_image, cameraMatrix

