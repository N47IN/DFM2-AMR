#!/usr/bin/env python3
"""
Analyze Latest GP Fits: Sequential 3D Visualization per Buffer

- Finds the latest run_* directory under /home/navin/ros2_ws/src/buffers
- For each buffer (buffer1, buffer2, ...), loads:
  * poses.npy (actual trajectory)
  * points.pcd (cause points saved by depth_octomap_node for narration)
  * voxel_gp_fit.json (fit parameters from background GP)
- Predicts disturbance field on a 3D grid using voxel_gp_helper (same workflow as voxel_gp.py)
- Visualizes sequentially (PyVista if available; otherwise Matplotlib isosurfaces/pointcloud)

Usage:
  rosrun/ros2 run not required; run standalone:
    python3 resilience/scripts/analyze_latest_gps.py

Optional args:
  --buffers-root <path>  (default: /home/navin/ros2_ws/src/buffers)
  --res-xy <float>       (default: 0.06)
  --res-z <float>        (default: 0.06)
  --matplotlib           (force Matplotlib even if PyVista is available)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Try PyVista
try:
	import pyvista as _pv
	from pyvista import themes as _pv_themes
	_HAS_PYVISTA = True
except Exception:
	_HAS_PYVISTA = False

# Helper API (mirrors voxel_gp.py)
from resilience.voxel_gp_helper import (
	load_buffer_xyz_drift,
	create_3d_prediction_grid,
	predict_direct_field_3d,
	plot_3d_volume_with_cause_points,
	plot_3d_pyvista_volume_with_points,
)


def _load_pcd_points(pcd_path: str) -> np.ndarray:
	"""Minimal PCD loader: try Open3D, else simple ASCII parse; returns (N,3) or empty array."""
	try:
		import open3d as _o3d  # type: ignore
		try:
			pc = _o3d.io.read_point_cloud(str(pcd_path))
			pts = np.asarray(pc.points, dtype=float)
			if pts.ndim == 2 and pts.shape[1] >= 3:
				return pts[:, :3]
		except Exception:
			pass
	except Exception:
		pass
	# Fallback ASCII parser (common PCD ASCII format)
	pts = []
	try:
		with open(pcd_path, 'r') as f:
			header = True
			data_started = False
			for line in f:
				line = line.strip()
				if header:
					if line.startswith('DATA'):
						data_started = True
						header = False
					continue
				if data_started and line and not line.startswith('#'):
					parts = line.split()
					if len(parts) >= 3:
						try:
							x = float(parts[0]); y = float(parts[1]); z = float(parts[2])
							pts.append((x, y, z))
						except Exception:
							pass
		if len(pts) == 0:
			return np.empty((0, 3), dtype=float)
		return np.array(pts, dtype=float)
	except Exception:
		return np.empty((0, 3), dtype=float)


def _find_latest_run_dir(buffers_root: Path) -> Path:
	runs = [p for p in buffers_root.iterdir() if p.is_dir() and p.name.startswith('run_')]
	if not runs:
		raise FileNotFoundError(f"No run_* directories under {buffers_root}")
	return max(runs, key=lambda p: p.stat().st_mtime)


def _iter_buffer_dirs(run_dir: Path):
	for entry in sorted(run_dir.iterdir()):
		if entry.is_dir() and entry.name.startswith('buffer'):
			yield entry


def visualize_buffer(buffer_dir: Path, res_xy: float, res_z: float, force_matplotlib: bool = False):
	poses_path = buffer_dir / 'poses.npy'
	fit_path = buffer_dir / 'voxel_gp_fit.json'
	pcd_path = buffer_dir / 'points.pcd'
	if not (poses_path.exists() and fit_path.exists() and pcd_path.exists()):
		print(f"Skipping {buffer_dir.name}: missing required files")
		return

	# Load actual xyz and GP fit
	xyz = np.load(poses_path)[:, 1:4]
	with open(fit_path, 'r') as f:
		fit_obj = json.load(f)
	fit = fit_obj.get('fit_params', {})
	# Load cause points
	cause_points = _load_pcd_points(str(pcd_path))
	if cause_points.size == 0:
		print(f"Skipping {buffer_dir.name}: empty PCD")
		return
	# Cause location for display (centroid)
	cause_xyz = np.mean(cause_points, axis=0) if cause_points.size > 0 else None

	print(f"\nBuffer: {buffer_dir.name}")
	print(f"  Trajectory points: {len(xyz)}  Cause points: {cause_points.shape[0]}")
	print(f"  Fit: lxy={fit.get('lxy')} lz={fit.get('lz')} A={fit.get('A')} b={fit.get('b')} R2={fit.get('r2_score')}")

	# Grid + prediction
	Xg, Yg, Zg, grid_points, xs, ys, zs = create_3d_prediction_grid(xyz, cause_xyz, resolution_xy=res_xy, resolution_z=res_z)
	mean_pred, std_pred = predict_direct_field_3d(fit, grid_points, cause_points)

	used_pyvista = False
	if _HAS_PYVISTA and not force_matplotlib:
		try:
			print("  Rendering 3D (PyVista)... close the window to continue")
			plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_pred, xyz, cause_points)
			used_pyvista = True
		except Exception as e:
			print(f"  PyVista rendering failed: {e}; falling back to Matplotlib")

	if not used_pyvista:
		print("  Rendering 3D (Matplotlib)...")
		_ = plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_pred, xs, ys, zs, xyz, cause_points, use_isosurfaces=True)
		plt.show()


def main():
	parser = argparse.ArgumentParser(description='Analyze and visualize latest GP fits per buffer')
	parser.add_argument('--buffers-root', type=str, default='/home/navin/ros2_ws/src/buffers')
	parser.add_argument('--res-xy', type=float, default=0.06)
	parser.add_argument('--res-z', type=float, default=0.06)
	parser.add_argument('--matplotlib', action='store_true', help='Force Matplotlib renderer')
	args = parser.parse_args()

	buffers_root = Path(args.buffers_root)
	run_dir = _find_latest_run_dir(buffers_root)
	print(f"Latest run: {run_dir}")

	for buffer_dir in _iter_buffer_dirs(run_dir):
		visualize_buffer(buffer_dir, args.res_xy, args.res_z, force_matplotlib=args.matplotlib)

	print("\nDone.")


if __name__ == '__main__':
	main() 