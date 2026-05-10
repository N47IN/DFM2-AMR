#!/usr/bin/env python3
"""
Drift Calibration and Spline Editing Tool

- Reads pose data from a ROS2 rosbag directory
- Fits a spline (nominal trajectory)
- Allows interactive adjustment of spline control points
- Computes drift (distance from nominal spline)
- Visualizes drift with color-coded percentiles (50%, 90%, 95%)
- Provides an interactive slider to adjust drift threshold and update the plot
- Allows saving the adjusted spline as a discretized nominal trajectory

Usage:
    python calibrate_drift.py --rosbag-dir /path/to/rosbag_dir [--topic TOPIC_NAME]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PoseStamped
import rclpy
from scipy.interpolate import splprep, splev
import json

# Helper: List topics in rosbag

def list_rosbag_topics(rosbag_path: str) -> List[str]:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening rosbag {rosbag_path}: {e}")
        return []
    topic_types = reader.get_all_topics_and_types()
    return [topic.name for topic in topic_types]

# Helper: Extract poses from rosbag

def extract_poses_from_rosbag(rosbag_path: str, topic_name: str) -> List[Dict[str, Any]]:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {topic.name: topic.type for topic in topic_types}
    if topic_name not in topic_type_map:
        raise ValueError(f"Topic {topic_name} not found in rosbag.")
    msg_type = get_message(topic_type_map[topic_name])
    poses = []
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_type)
            poses.append({
                'timestamp': timestamp / 1e9,
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            })
    return poses

# Helper: Fit spline to trajectory

def fit_spline_with_n_control_points(poses: List[Dict[str, Any]], n_control: int = 10, n_points: int = 100, s=0.0, k=3) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.array([[p['x'], p['y'], p['z']] for p in poses])
    # Select n_control evenly spaced indices
    if len(xyz) < n_control:
        raise ValueError("Not enough points for the requested number of control points.")
    idxs = np.linspace(0, len(xyz) - 1, n_control, dtype=int)
    ctrl_xyz = xyz[idxs]
    # Fit spline through these control points
    tck, u = splprep([ctrl_xyz[:, 0], ctrl_xyz[:, 1], ctrl_xyz[:, 2]], s=s, k=min(k, n_control-1))
    u_fine = np.linspace(0, 1, n_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    return tck, np.vstack([x_fine, y_fine, z_fine]).T, u_fine, ctrl_xyz

# Helper: Compute drift (distance to spline)

def compute_drift_to_spline(poses: List[Dict[str, Any]], spline_points: np.ndarray) -> np.ndarray:
    pose_xyz = np.array([[p['x'], p['y'], p['z']] for p in poses])
    dists = np.linalg.norm(pose_xyz[:, None, :] - spline_points[None, :, :], axis=2)
    min_dists = np.min(dists, axis=1)
    nearest_idxs = np.argmin(dists, axis=1)
    return min_dists, nearest_idxs

# Helper: Color by percentile

def get_drift_colors(drifts: np.ndarray, soft: float, hard: float) -> list:
    colors = []
    for d in drifts:
        if d <= soft:
            colors.append('green')
        elif d <= hard:
            colors.append('yellow')
        else:
            colors.append('red')
    return colors

# Interactive spline editor and drift plot

class SplineDriftEditor:
    def __init__(self, poses, tck, spline_points, drifts, control_points, n_points=100, soft_thresh=0.1, hard_thresh=0.2, save_callback=None):
        self.poses = poses
        self.tck = tck
        self.n_points = n_points
        self.drifts = drifts
        self.spline_points = spline_points
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.32)
        self.pose_xyz = np.array([[p['x'], p['y'], p['z']] for p in poses])
        self.control_points = control_points.copy()  # shape (n_ctrl, 3)
        self.dragging_idx = None
        self.soft_thresh = soft_thresh
        self.hard_thresh = hard_thresh
        self.save_callback = save_callback
        self._init_plot()

    def _init_plot(self):
        self.colors = get_drift_colors(self.drifts, self.soft_thresh, self.hard_thresh)
        self.sc = self.ax.scatter(self.pose_xyz[:, 0], self.pose_xyz[:, 1], c=self.colors, s=30, label='Actual Trajectory')
        self.spline_line, = self.ax.plot(self.spline_points[:, 0], self.spline_points[:, 1], 'b-', lw=2, label='Nominal Spline')
        self.ctrl_sc = self.ax.scatter(self.control_points[:, 0], self.control_points[:, 1], c='magenta', s=80, marker='o', label='Spline Control Points', zorder=5, picker=True)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Drift Calibration: Spline Nominal Path (Drag Magenta Points)')
        self.ax.legend()
        self.ax.axis('equal')
        # Sliders
        axcolor = 'lightgoldenrodyellow'
        self.ax_soft = plt.axes((0.15, 0.18, 0.65, 0.04), facecolor=axcolor)
        self.ax_hard = plt.axes((0.15, 0.12, 0.65, 0.04), facecolor=axcolor)
        self.s_soft = Slider(self.ax_soft, 'Soft Threshold', 0, max(0.5, np.max(self.drifts)), valinit=self.soft_thresh, valstep=0.001)
        self.s_hard = Slider(self.ax_hard, 'Hard Threshold', 0, max(0.5, np.max(self.drifts)), valinit=self.hard_thresh, valstep=0.001)
        self.s_soft.on_changed(self.update_thresholds)
        self.s_hard.on_changed(self.update_thresholds)
        # Save button
        self.save_ax = plt.axes((0.8, 0.04, 0.15, 0.05))
        self.save_button = Button(self.save_ax, 'Save Spline', color='lightblue', hovercolor='skyblue')
        self.save_button.on_clicked(self.save_spline)
        # Connect events
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def update_thresholds(self, val):
        self.soft_thresh = self.s_soft.val
        self.hard_thresh = self.s_hard.val
        self.colors = get_drift_colors(self.drifts, self.soft_thresh, self.hard_thresh)
        self.sc.set_color(self.colors)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        xy = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.control_points[:, :2] - xy, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < 0.1:
            self.dragging_idx = idx

    def on_release(self, event):
        self.dragging_idx = None

    def on_motion(self, event):
        if self.dragging_idx is None or event.inaxes != self.ax:
            return
        self.control_points[self.dragging_idx, 0] = event.xdata
        self.control_points[self.dragging_idx, 1] = event.ydata
        tck = (self.tck[0], [self.control_points[:, 0], self.control_points[:, 1], self.control_points[:, 2]], self.tck[2])
        x_fine, y_fine, z_fine = splev(np.linspace(0, 1, self.n_points), tck)
        self.spline_points = np.vstack([x_fine, y_fine, z_fine]).T
        self.spline_line.set_data(self.spline_points[:, 0], self.spline_points[:, 1])
        self.ctrl_sc.set_offsets(self.control_points[:, :2])
        self.drifts, _ = compute_drift_to_spline(self.poses, self.spline_points)
        self.colors = get_drift_colors(self.drifts, self.soft_thresh, self.hard_thresh)
        self.sc.set_color(self.colors)
        self.fig.canvas.draw_idle()

    def save_spline(self, event):
        if self.save_callback:
            self.save_callback(self)
        else:
            print("No save callback provided!")

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Calibrate and visualize drift from rosbag trajectory with spline editing')
    parser.add_argument('--rosbag-dir', required=True, help='Path to rosbag2 directory')
    parser.add_argument('--topic', default='/robot_1/sensors/front_stereo/pose', help='Pose topic name (default: /robot_1/sensors/front_stereo/pose)')
    parser.add_argument('--discretization', type=float, default=0.1, help='Discretization for nominal line (meters)')
    parser.add_argument('--spline-points', type=int, default=100, help='Number of points to discretize spline')
    parser.add_argument('--n-control', type=int, default=10, help='Number of spline control points')
    parser.add_argument('--soft-threshold', type=float, default=0.1, help='Initial soft drift threshold (green)')
    parser.add_argument('--hard-threshold', type=float, default=0.2, help='Initial hard drift threshold (yellow/red)')
    args = parser.parse_args()

    rosbag_dir = Path(args.rosbag_dir)
    db3_files = list(rosbag_dir.glob('*.db3'))
    if not db3_files:
        print(f"No .db3 files found in {rosbag_dir}")
        sys.exit(1)
    db3_path = str(db3_files[0])

    topic = args.topic
    print(f"Extracting poses from topic: {topic}")
    poses = extract_poses_from_rosbag(str(rosbag_dir), topic)
    if len(poses) < 2:
        print("Not enough poses found!")
        sys.exit(1)

    tck, spline_points, u_fine, ctrl_xyz = fit_spline_with_n_control_points(poses, n_control=args.n_control, n_points=args.spline_points)
    drifts, nearest_idxs = compute_drift_to_spline(poses, spline_points)
    avg_drift = np.mean(drifts)

    def save_spline_with_params(editor):
        save_path = 'adjusted_nominal_spline.json'
        points = [
            {
                'index': i,
                'position': {
                    'x': float(editor.spline_points[i, 0]),
                    'y': float(editor.spline_points[i, 1]),
                    'z': float(editor.spline_points[i, 2])
                }
            }
            for i in range(editor.spline_points.shape[0])
        ]
        calib_params = {
            'pose_topic': topic,
            'n_control_points': args.n_control,
            'discretization': args.discretization,
            'spline_points': args.spline_points,
            'soft_threshold': editor.soft_thresh,
            'hard_threshold': editor.hard_thresh,
            'avg_drift': float(avg_drift)
        }
        with open(save_path, 'w') as f:
            json.dump({'points': points, 'calibration': calib_params}, f, indent=2)
        print(f"Saved adjusted nominal spline and calibration params to {save_path}")

    editor = SplineDriftEditor(poses, tck, spline_points, drifts, ctrl_xyz, n_points=args.spline_points, soft_thresh=args.soft_threshold, hard_thresh=args.hard_threshold, save_callback=save_spline_with_params)
    editor.show()

if __name__ == "__main__":
    main() 