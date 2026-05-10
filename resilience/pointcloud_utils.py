#!/usr/bin/env python3
"""
PointCloud Utils

Depth-to-world projection, voxelization, and PointCloud2 helpers.
"""

import numpy as np
from typing import Optional, Tuple
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


def depth_to_meters(depth, encoding: str):
    try:
        enc = (encoding or '').lower()
        if '16uc1' in enc or 'mono16' in enc:
            return depth.astype(np.float32) / 1000.0
        elif '32fc1' in enc or 'float32' in enc:
            return depth.astype(np.float32)
        else:
            return depth.astype(np.float32) / 1000.0
    except Exception:
        return None


def pose_position(pose: PoseStamped) -> np.ndarray:
    return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)


def pose_quat(pose: PoseStamped) -> np.ndarray:
    q = pose.pose.orientation
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)


def quat_to_rot(q: np.ndarray):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ], dtype=np.float32)

import os
import json
import time 
import open3d as o3d



import math
def _rpy_deg_to_rot(rpy_deg):
		try:
			roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
			cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
			Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
			Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
			Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
			return Rz @ Ry @ Rx
		except Exception:
			return np.eye(3, dtype=np.float32)

def _quat_to_rot( q: np.ndarray):
		x, y, z, w = q
		n = x*x + y*y + z*z + w*w
		if n < 1e-8:
			return np.eye(3, dtype=np.float32)
		s = 2.0 / n
		xx, yy, zz = x*x*s, y*y*s, z*z*s
		xy, xz, yz = x*y*s, x*z*s, y*z*s
		wx, wy, wz = w*x*s, w*y*s, w*z*s
		return np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)]
		], dtype=np.float32)

def depth_mask_to_world_points(depth_m: np.ndarray, mask: np.ndarray, intrinsics, pose: PoseStamped,
                              pose_is_base_link: bool = True,
                              apply_optical_frame_rotation: bool = True,
                              R_opt_to_base: Optional[np.ndarray] = None,
                              R_cam_to_base_extra: Optional[np.ndarray] = None,
                              t_cam_to_base_extra: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        fx, fy, cx, cy = intrinsics
        h, w = depth_m.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z_full = depth_m
        valid = np.isfinite(z_full) & (z_full > 0.0) & (mask > 0)
        if not np.any(valid):
            return None, None, None

        u, v, z = u[valid], v[valid], z_full[valid]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)

        if pose_is_base_link:
            if R_opt_to_base is None:
                R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
            if R_cam_to_base_extra is None:
                R_cam_to_base_extra = np.eye(3, dtype=np.float32)
            if t_cam_to_base_extra is None:
                t_cam_to_base_extra = np.zeros(3, dtype=np.float32)

            if apply_optical_frame_rotation:
                pts_cam = pts_cam @ R_opt_to_base.T
            pts_cam = pts_cam @ R_cam_to_base_extra.T + t_cam_to_base_extra

        R_world = quat_to_rot(pose_quat(pose))
        p_world = pose_position(pose)
        pts_world = pts_cam @ R_world.T + p_world
        return pts_world, u, v
    except Exception:
        return None, None, None


def voxelize_pointcloud(points: np.ndarray, voxel_size: float, max_points: int = 200) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.array([])
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxelized_points = []
    for i in range(len(unique_voxels)):
        voxel_mask = inverse_indices == i
        voxel_points = points[voxel_mask]
        centroid = np.mean(voxel_points, axis=0)
        voxelized_points.append(centroid)
    voxelized_points = np.array(voxelized_points)
    if len(voxelized_points) > max_points:
        indices = np.random.choice(len(voxelized_points), size=max_points, replace=False)
        voxelized_points = voxelized_points[indices]
    return voxelized_points


def create_cloud_xyz(points: np.ndarray, header: Header) -> PointCloud2:
    fields = [
        pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)
    ]
    if points is None or len(points) == 0:
        return pc2.create_cloud(header, fields, [])
    return pc2.create_cloud(header, fields, points.astype(np.float32)) 