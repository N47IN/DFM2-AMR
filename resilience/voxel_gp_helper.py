#!/usr/bin/env python3
"""
Disturbance Field Helper - Class-based utility for 3D disturbance modeling with GP uncertainty

This helper provides a reusable class for Gaussian Process (GP) modeling of 3D disturbance fields.
It is designed for use in ROS2 nodes (e.g., frontier_mapping_node) to model spatial disturbances
caused by environmental factors (e.g., obstacles, wind, terrain effects).

Key Features:
- Superposed Anisotropic RBF Model: Uses sum of anisotropic radial basis functions with
  different length scales for horizontal (xy) vs vertical (z) dimensions
- Bayesian Uncertainty Estimation: Supports both MSE and NLL objectives for fitting,
  with optional epistemic uncertainty quantification via inverse Hessian
- Flexible Input Sources: Can load data from buffer directories or accept direct inputs
- Visualization Support: Provides 2D/3D plotting utilities for field visualization

Primary capabilities:
- Load actual trajectory and cause metadata from a buffer directory
- Accept a nominal trajectory path or pass nominal directly
- Accept a point cloud (Nx3 numpy array) representing cause points (PCD content)
- Compute disturbances against nominal, fit a superposed anisotropic kernel model
- Estimate epistemic uncertainty in GP parameters (for risk-aware planning)
- Predict a 3D field on a grid around the trajectory and cause
- Provide Matplotlib and optional PyVista visualizations

GP Model:
	disturbance(x) = A * phi(x) + b
	where phi(x) = Σ_j exp(-0.5 * d²_j(x, c_j))
	and d² uses anisotropic distance: (dx² + dy²)/lxy² + dz²/lz²

Uncertainty Quantification:
	- Aleatoric uncertainty: Observation noise (sigma² = MSE)
	- Epistemic uncertainty: Parameter uncertainty (from inverse Hessian of lxy, lz)
	- Predictive variance: Var(y*) = sigma² * (1 + v^T * (X^T X)^-1 * v)

Inputs (main methods):
- pointcloud: numpy.ndarray of shape (N, 3) - cause points
- buffer_dir: directory containing `poses.npy` and `cause_location.json`/`metadata.json`
- nominal_path or nominal_xyz: nominal trajectory for comparison

Outputs:
- Fitted parameters (lxy, lz, A, b) and quality metrics (mse, rmse, r2_score)
- Uncertainty estimates (sigma2, nll, param_std, hess_inv)
- Optional visualizations (2D scatter, orthogonal slices, 3D volume)

Dependency policy:
- Functions are directly adapted from `voxel_gp.py` and `gp_sampler.py` for clean reuse
- All public APIs accept numpy arrays and paths; no ROS messages are required
- Maintains backward compatibility with existing frontier_mapping_node usage
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, lsq_linear

# Optional dependencies
try:
	from skimage import measure as _sk_measure
	_HAS_SKIMAGE = True
except Exception:
	_HAS_SKIMAGE = False

try:
	import pyvista as _pv
	from pyvista import themes as _pv_themes
	_HAS_PYVISTA = True
except Exception:
	_HAS_PYVISTA = False

# ----------------------------
# Configuration defaults
# ----------------------------

_DEFAULTS = {
	'resolution_xy': 0.06,
	'resolution_z': 0.06,
	'pad_bounds': 0.5,
	'percentile_range_2d': (5, 95),
	'percentile_range_3d': (5, 97),
	'max_points_pointcloud': 50000,
	'point_cloud_alpha': 0.28,
	# Disturbance sampling / validation
	'disturbance_match_threshold_m': 0.30,
	'min_nominal_actual_overlap_fraction': 0.10,
}


class DisturbanceFieldHelper:
	"""
	Helper for computing and visualizing 3D disturbance fields using a superposed
	anisotropic RBF (Radial Basis Function) GP model.
	
	This class provides a unified interface for GP-based disturbance field modeling,
	supporting both traditional MSE-based fitting and Bayesian NLL-based fitting with
	epistemic uncertainty quantification. It maintains backward compatibility with
	existing code (e.g., frontier_mapping_node) while adding new capabilities.
	
	Key Features:
		- Superposed Anisotropic RBF Model: Sum of RBF kernels with different
		  length scales for horizontal (lxy) vs vertical (lz) dimensions
		- Dual Objective Support: "mse" (default, backward compatible) or "nll" (Bayesian)
		- Uncertainty Quantification: Optional epistemic uncertainty via inverse Hessian
		- Efficient Implementation: Chunked processing for large grids, vectorized operations
	
	Usage:
		helper = DisturbanceFieldHelper()
		result = helper.fit_from_pointcloud_and_buffer(
			pointcloud_xyz=cause_points,
			buffer_dir="/path/to/buffer",
			nominal_path="/path/to/nominal.json",
			objective="nll"  # or "mse" for backward compatibility
		)
		fit_params = result['fit']
		# fit_params contains: lxy, lz, A, b, mse, rmse, r2_score, sigma2, nll, param_std, etc.
	"""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		cfg = dict(_DEFAULTS)
		if config:
			cfg.update(config)
		self.cfg = cfg

	# ----------------------------
	# Data loading utilities
	# ----------------------------

	@staticmethod
	def load_buffer_xyz_drift(buffer_dir: str) -> Tuple[np.ndarray, Optional[str], Optional[np.ndarray]]:
		"""Load actual trajectory xyz and cause metadata from a buffer directory.
		Returns (xyz, cause, cause_xyz).
		"""
		buffer_path = Path(buffer_dir)
		poses_path = buffer_path / "poses.npy"
		meta_path = buffer_path / "metadata.json"
		cause_loc_path = buffer_path / "cause_location.json"

		if not poses_path.exists():
			raise FileNotFoundError(f"poses.npy missing in {buffer_dir}")

		poses = np.load(poses_path)
		xyz = poses[:, 1:4]

		cause = None
		cause_loc = None
		if meta_path.exists():
			with open(meta_path, "r") as f:
				meta = json.load(f)
			cause = meta.get("cause")
			cause_loc = meta.get("cause_location")
		if cause_loc is None and cause_loc_path.exists():
			with open(cause_loc_path, "r") as f:
				d = json.load(f)
			cause = d.get("cause", cause)
			cause_loc = d.get("location_3d")

		cause_xyz = None
		if cause_loc is not None:
			cause_xyz = np.array(cause_loc[:3], dtype=float)

		return xyz, cause, cause_xyz

	@staticmethod
	def load_nominal_xyz(nominal_path: str) -> Optional[np.ndarray]:
		"""Load nominal trajectory from JSON format used in assets.
		Accepts dict with `points` -> `position` (x,y,z) or list of lists.
		"""
		p = Path(nominal_path)
		if not p.exists():
			return None
		with open(p, "r") as f:
			data = json.load(f)
		pts = data.get("points") if isinstance(data, dict) else data
		if isinstance(pts, list) and len(pts) > 0:
			if isinstance(pts[0], dict):
				xyz_list = []
				for item in pts:
					pos = item.get("position") if isinstance(item, dict) else None
					if pos and all(k in pos for k in ("x", "y", "z")):
						xyz_list.append([float(pos["x"]), float(pos["y"]), float(pos["z"])] )
				if xyz_list:
					return np.array(xyz_list, dtype=float)
			else:
				arr = np.array(pts, dtype=float)
				if arr.ndim == 2 and arr.shape[1] >= 3:
					return arr[:, :3]
		return None

	# ----------------------------
	# Core computations (adapted)
	# ----------------------------

	@staticmethod
	def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy') -> np.ndarray:
		"""
		Clip nominal trajectory to the segment corresponding to the actual trajectory.
		
		Finds the start and end indices of the nominal trajectory that best match
		the actual trajectory endpoints, then returns the clipped segment. This ensures
		we only fit the GP model to the relevant portion of the nominal trajectory.
		
		Args:
			nominal_xyz: (N, 3) full nominal trajectory
			actual_xyz: (M, 3) actual trajectory (subset of nominal)
			plane: 'xy' or 'xz' projection plane for matching endpoints
		
		Returns:
			Clipped nominal trajectory segment (K, 3)
		"""
		if nominal_xyz is None or len(nominal_xyz) == 0 or actual_xyz is None or len(actual_xyz) == 0:
			return nominal_xyz
		plane = plane.lower()
		if plane not in ('xy', 'xz'):
			plane = 'xy'
		# Project to 2D plane for endpoint matching
		if plane == 'xy':
			nom_proj = nominal_xyz[:, [0, 1]]
			act_start = actual_xyz[0, [0, 1]]
			act_end = actual_xyz[-1, [0, 1]]
		else:
			nom_proj = nominal_xyz[:, [0, 2]]
			act_start = actual_xyz[0, [0, 2]]
			act_end = actual_xyz[-1, [0, 2]]

		# Find closest nominal points to actual start/end
		d_start = np.linalg.norm(nom_proj - act_start[None, :], axis=1)
		d_end = np.linalg.norm(nom_proj - act_end[None, :], axis=1)
		i_start = int(np.argmin(d_start))
		i_end = int(np.argmin(d_end))

		# Extract segment (handle reverse direction)
		lo, hi = (i_start, i_end) if i_start <= i_end else (i_end, i_start)
		lo = max(0, lo)
		hi = min(len(nominal_xyz) - 1, hi)
		if hi <= lo:
			return nominal_xyz
		return nominal_xyz[lo:hi + 1]

	@staticmethod
	def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
		"""
		Compute drift vectors from nominal to actual trajectory.
		
		For each actual trajectory point, finds the closest nominal point and
		computes the drift vector (difference). Used as a fallback when
		compute_disturbance_at_nominal_points returns no valid points.
		
		Args:
			actual_xyz: (N, 3) actual trajectory points
			nominal_xyz: (M, 3) nominal trajectory points
		
		Returns:
			tuple (drift_vectors, drift_magnitudes):
				- drift_vectors: (N, 3) vectors from nominal to actual
				- drift_magnitudes: (N,) Euclidean distances (disturbance magnitudes)
		"""
		if nominal_xyz is None or len(nominal_xyz) == 0:
			return None, None
		drift_vectors = []
		drift_magnitudes = []
		for actual_point in actual_xyz:
			# Find closest nominal point
			diffs = nominal_xyz - actual_point
			dists = np.linalg.norm(diffs, axis=1)
			closest_idx = int(np.argmin(dists))
			# Compute drift vector and magnitude
			drift_vec = actual_point - nominal_xyz[closest_idx]
			drift_mag = float(np.linalg.norm(drift_vec))
			drift_vectors.append(drift_vec)
			drift_magnitudes.append(drift_mag)
		return np.array(drift_vectors), np.array(drift_magnitudes)

	@staticmethod
	def compute_disturbance_at_nominal_points(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Compute disturbance magnitudes at nominal trajectory points.
		
		For each nominal point, finds the closest actual trajectory point and computes
		the Euclidean distance (disturbance magnitude). Only includes nominal points
		that are within a reasonable distance (0.3m) of the actual trajectory.
		
		This provides the training data for GP fitting: we want to learn a field that
		predicts disturbance at any nominal point based on the spatial distribution
		of cause points.
		
		Args:
			nominal_xyz: (N, 3) nominal trajectory points
			actual_xyz: (M, 3) actual trajectory points (observed)
			cause_xyz: Unused (kept for API compatibility with voxel_gp.py)
		
		Returns:
			tuple (nominal_points_used, disturbance_magnitudes):
				- nominal_points_used: (K, 3) nominal points with valid disturbances
				- disturbance_magnitudes: (K,) disturbance magnitudes (Euclidean distances)
		"""
		disturbances = []
		nominal_points_used = []
		actual_bounds = {
			'x': (actual_xyz[:, 0].min(), actual_xyz[:, 0].max()),
			'y': (actual_xyz[:, 1].min(), actual_xyz[:, 1].max()),
			'z': (actual_xyz[:, 2].min(), actual_xyz[:, 2].max())
		}
		pad = 0.3
		mask = (
			(nominal_xyz[:, 0] >= actual_bounds['x'][0] - pad) &
			(nominal_xyz[:, 0] <= actual_bounds['x'][1] + pad) &
			(nominal_xyz[:, 1] >= actual_bounds['y'][0] - pad) &
			(nominal_xyz[:, 1] <= actual_bounds['y'][1] + pad) &
			(nominal_xyz[:, 2] >= actual_bounds['z'][0] - pad) &
			(nominal_xyz[:, 2] <= actual_bounds['z'][1] + pad)
		)
		relevant_nominal = nominal_xyz[mask]
		if len(relevant_nominal) == 0:
			return np.array([]), np.array([])
		for nominal_point in relevant_nominal:
			distances = np.linalg.norm(actual_xyz - nominal_point, axis=1)
			closest_idx = np.argmin(distances)
			closest_actual = actual_xyz[closest_idx]
			disturbance = np.linalg.norm(closest_actual - nominal_point)
			if distances[closest_idx] < 0.3:
				disturbances.append(disturbance)
				nominal_points_used.append(nominal_point)
		return np.array(nominal_points_used), np.array(disturbances)

	@staticmethod
	def _min_distances_to_set(src_xyz: np.ndarray, dst_xyz: np.ndarray, chunk: int = 512) -> np.ndarray:
		"""Compute per-point min Euclidean distance from src points to dst points."""
		if src_xyz is None or dst_xyz is None or src_xyz.size == 0 or dst_xyz.size == 0:
			return np.array([], dtype=float)
		dst = np.asarray(dst_xyz, dtype=float)
		src = np.asarray(src_xyz, dtype=float)
		out = np.empty((src.shape[0],), dtype=float)
		for i in range(0, src.shape[0], chunk):
			j = min(src.shape[0], i + chunk)
			d = src[i:j, None, :] - dst[None, :, :]
			# (chunk, M)
			dist2 = np.sum(d * d, axis=2)
			out[i:j] = np.sqrt(np.min(dist2, axis=1))
		return out

	@staticmethod
	def _sum_of_anisotropic_rbf(grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
		"""
		Compute superposed anisotropic RBF basis function phi(x) = Σ_j exp(-0.5 * d²_j).
		
		Uses anisotropic distance metric:
			d²_j = (dx² + dy²) / lxy² + dz² / lz²
		where dx, dy, dz are the differences in x, y, z coordinates between grid points
		and cause centers, and lxy, lz are the length scales (different for horizontal vs vertical).
		
		This allows the GP to model different correlation scales in horizontal (xy) vs vertical (z)
		directions, which is important for spatial disturbances that may vary more slowly
		horizontally than vertically (e.g., wind, terrain effects).
		
		Args:
			grid_points: (N, 3) query points where we evaluate the basis function
			centers: (M, 3) cause points (centers of RBF kernels)
			lxy: Length scale for horizontal (xy) dimensions (meters)
			lz: Length scale for vertical (z) dimension (meters)
		
		Returns:
			phi: (N,) array of basis function values (sum of RBF contributions)
		"""
		if centers.size == 0:
			return np.zeros(grid_points.shape[0], dtype=float)
		num_points = grid_points.shape[0]
		phi = np.zeros(num_points, dtype=float)
		chunk = 200000  # Process in chunks to manage memory for large grids
		# Precompute inverse squared length scales for efficiency
		inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
		inv_lz2 = 1.0 / (lz * lz + 1e-12)
		for start in range(0, num_points, chunk):
			end = min(num_points, start + chunk)
			gp_chunk = grid_points[start:end]
			# Vectorized distance computation: (chunk_size, M, 3) difference array
			dx = gp_chunk[:, None, 0] - centers[None, :, 0]
			dy = gp_chunk[:, None, 1] - centers[None, :, 1]
			dz = gp_chunk[:, None, 2] - centers[None, :, 2]
			# Anisotropic squared distance: horizontal and vertical components scaled separately
			d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
			# Compute RBF contributions: exp(-0.5 * d²) and sum over all centers
			np.exp(-0.5 * d2, out=d2)
			phi[start:end] = d2.sum(axis=1)  # Sum over M centers for each query point
		return phi

	@staticmethod
	def _rbf_phi_and_grads(grid_points: np.ndarray, centers: np.ndarray,
						   lxy: float, lz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Compute phi AND analytic gradients ∂phi/∂lxy, ∂phi/∂lz in a single pass.

		These gradients are needed for propagating length-scale uncertainty into
		predictive variance via the delta method:
			Var_ls = A² · [∂φ/∂lxy, ∂φ/∂lz]ᵀ · H⁻¹ · [∂φ/∂lxy, ∂φ/∂lz]

		Derivation (from chain rule on φ = Σ_j exp(-0.5·r_j)):
			∂φ/∂lxy = Σ_j (dx²+dy²)/lxy³ · exp(-0.5·r_j)   (always ≥ 0)
			∂φ/∂lz  = Σ_j dz²/lz³ · exp(-0.5·r_j)           (always ≥ 0)

		Args:
			grid_points: (N, 3) query points
			centers: (M, 3) RBF centers (cause points)
			lxy: horizontal length scale
			lz: vertical length scale

		Returns:
			(phi, dphi_dlxy, dphi_dlz) — each shape (N,)
		"""
		n = grid_points.shape[0]
		if centers.size == 0:
			return np.zeros(n), np.zeros(n), np.zeros(n)

		inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
		inv_lz2 = 1.0 / (lz * lz + 1e-12)
		inv_lxy3 = 1.0 / (lxy ** 3 + 1e-12)
		inv_lz3 = 1.0 / (lz ** 3 + 1e-12)

		phi = np.zeros(n, dtype=float)
		dphi_dlxy = np.zeros(n, dtype=float)
		dphi_dlz = np.zeros(n, dtype=float)
		chunk = 200000
		for start in range(0, n, chunk):
			end = min(n, start + chunk)
			gp = grid_points[start:end]
			dx = gp[:, None, 0] - centers[None, :, 0]
			dy = gp[:, None, 1] - centers[None, :, 1]
			dz = gp[:, None, 2] - centers[None, :, 2]
			rxy2 = dx * dx + dy * dy        # (chunk, M)
			rz2 = dz * dz                    # (chunk, M)
			d2 = rxy2 * inv_lxy2 + rz2 * inv_lz2
			rbf = np.exp(-0.5 * d2)          # (chunk, M)
			phi[start:end] = rbf.sum(axis=1)
			dphi_dlxy[start:end] = (rbf * rxy2).sum(axis=1) * inv_lxy3
			dphi_dlz[start:end] = (rbf * rz2).sum(axis=1) * inv_lz3
		return phi, dphi_dlxy, dphi_dlz

	def fit_direct_superposition_to_disturbances(self, nominal_points: np.ndarray, disturbance_magnitudes: np.ndarray, cause_points: np.ndarray, objective: str = "nll", fixed_b: Optional[float] = None) -> Dict[str, Any]:
		"""
		Fit GP parameters (lxy, lz, A, b) using superposed anisotropic RBF model.
		
		Model: disturbance(x) = A * phi(x) + b
		where phi(x) = Σ_j exp(-0.5 * d²_j) is the sum of anisotropic RBF kernels
		centered at cause points, and d² uses different length scales for xy and z.
		
		Two modes:
		  1. fixed_b=None (default): Fit both A and b via bounded least squares
		     (A >= 0, b >= 0). Standard approach.
		  2. fixed_b=value: Fix b to a known ambient disturbance level and only
		     fit A >= 0. This is the preferred mode when we know the ambient
		     disturbance (e.g., from pre-breach safe evidence). It prevents b
		     from absorbing spatial structure that should be in A*phi, yielding
		     sharper and more physically meaningful fits.
		
		Fits lxy and lz (hyperparameters) via L-BFGS-B with multi-start.
		
		Objective functions:
			- "mse": Minimize mean squared error (standard approach)
			- "nll": Minimize negative log-likelihood (Bayesian approach, better uncertainty)
		
		Args:
			nominal_points: (N, 3) training locations where disturbances were measured
			disturbance_magnitudes: (N,) observed disturbance magnitudes
			cause_points: (M, 3) cause points (e.g., from PCD file)
			objective: "mse" or "nll"
			fixed_b: If provided, fix b to this value (meters) and only fit A.
				Typically set to the known ambient disturbance (e.g., 0.0005m).
		
		Returns:
			Dictionary with fitted parameters:
				- lxy, lz: Optimized length scales (meters)
				- A, b: Linear coefficients (A * phi + b)
				- mse, rmse, mae, r2_score: Quality metrics
				- sigma2: Noise variance estimate (for uncertainty quantification)
				- nll: Negative log-likelihood (if objective="nll")
				- hess_inv: Inverse Hessian matrix (for parameter uncertainty)
				- param_std: Standard deviations of lxy, lz (for epistemic uncertainty)
				- optimization_result: scipy.optimize.OptimizeResult object
		"""
		if cause_points.size == 0:
			return {
				'lxy': None,
				'lz': None,
				'A': 0.0,
				'b': 0.0,
				'recon': np.zeros_like(disturbance_magnitudes),
				'mse': float('inf'),
				'r2_score': 0.0,
				'mae': float('inf'),
				'rmse': float('inf'),
				'sigma2': float('inf'),
				'nll': float('inf'),
			}
		obj = objective.lower()
		target = disturbance_magnitudes.astype(float)

		if fixed_b is None:
			# Original mode: normalize target for stable 2-param optimization
			target_mean = np.mean(target)
			target_std = np.std(target)
			if target_std < 1e-8:
				target_std = 1.0
			target_norm = (target - target_mean) / target_std

		def objective_fn(params):
			"""Optimization objective: find best lxy, lz hyperparameters."""
			lxy, lz = params
			lxy = max(lxy, 0.01)  # Enforce minimum bounds
			lz = max(lz, 0.01)
			# Compute basis function phi at training points
			phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
			n = phi.shape[0]

			if fixed_b is not None:
				# Fixed-ambient mode: y - b_fixed = A * phi, solve for A >= 0 only
				# This prevents b from absorbing spatial structure meant for A*phi
				y_adj = target - fixed_b
				# Check if phi has sufficient variation (not constant)
				phi_std = np.std(phi)
				if phi_std < 1e-8:
					return float('inf')  # Invalid: constant phi
				phi_dot = np.dot(phi, phi)
				if phi_dot < 1e-12:
					return float('inf')
				# Use least squares with non-negativity constraint for robustness
				# This handles cases where phi values are small but not constant
				phi_dot_y = np.dot(phi, y_adj)
				# If correlation is negative or very weak, A should be 0
				# But we need to be careful: if phi values are small, we need to scale appropriately
				if phi_dot_y <= 0:
					A = 0.0
				else:
					# Regularize to avoid numerical issues when phi_dot is very small
					# Add small regularization term proportional to phi variance
					reg = 1e-10 * phi_std * phi_std * n
					A = phi_dot_y / (phi_dot + reg)
					A = max(0.0, A)
				recon = A * phi + fixed_b
				sse = np.sum((recon - target) ** 2)
				mse = sse / n
			else:
				# Original mode: fit both A and b via normalized 2-param LS
				phi_mean = np.mean(phi)
				phi_std = np.std(phi)
				if phi_std < 1e-8:
					return float('inf')  # Invalid: constant phi
				phi_norm = (phi - phi_mean) / phi_std
				X = np.column_stack([phi_norm, np.ones(n, dtype=float)])
				try:
					res = lsq_linear(X, target_norm, bounds=([0, -np.inf], [np.inf, np.inf]))
					params_ab = res.x
				except Exception:
					params_ab = np.linalg.lstsq(X, target_norm, rcond=None)[0]
					params_ab[0] = max(params_ab[0], 0.0)
				A_norm, b_norm = params_ab[0], params_ab[1]
				recon_norm = A_norm * phi_norm + b_norm
				sse = np.sum((recon_norm - target_norm) ** 2)
				mse = sse / n

			# Choose objective: MSE or NLL
			if obj == "nll":
				nll = 0.5 * n * (np.log(mse + 1e-12) + 1.0)
				loss = nll
			else:
				loss = mse
			# Regularization: prefer moderate length scales (avoid overfitting)
			reg_term = 0.05 * (1.0 / (lxy + 0.05) + 1.0 / (lz + 0.05))
			return loss + reg_term

		# Multi-start optimization: try multiple initial guesses to avoid local minima
		initial_guesses = [
			[0.02, 0.02], [0.05, 0.05], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3],
			[0.1, 0.05], [0.05, 0.1], [0.15, 0.08], [0.08, 0.15],
		]
		bounds = [(0.005, 1.0), (0.005, 1.0)]  # Reasonable bounds for length scales (5mm to 1m)

		best_result = None
		best_loss = float('inf')
		for x0 in initial_guesses:
			try:
				result = minimize(objective_fn, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
				if result.success and result.fun < best_loss:
					best_result = result
					best_loss = result.fun
			except Exception:
				pass

		if best_result is None:
			# Grid search fallback if optimization fails
			lxy_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5], dtype=float)
			lz_grid = np.array([0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.24, 0.35, 0.5], dtype=float)
			best = {'mse': float('inf'), 'nll': float('inf')}
			for lxy in lxy_grid:
				for lz in lz_grid:
					phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
					n = phi.shape[0]
					if fixed_b is not None:
						# Fixed-ambient mode: solve for A only
						y_adj = target - fixed_b
						# Check if phi has sufficient variation (not constant)
						phi_std = np.std(phi)
						if phi_std < 1e-8:
							A = 0.0
						else:
							phi_dot = np.dot(phi, phi)
							phi_dot_y = np.dot(phi, y_adj)
							if phi_dot < 1e-12 or phi_dot_y <= 0:
								A = 0.0
							else:
								# Regularize to avoid numerical issues when phi_dot is very small
								reg = 1e-10 * phi_std * phi_std * n
								A = phi_dot_y / (phi_dot + reg)
								A = max(0.0, A)
						b = float(fixed_b)
					else:
						X = np.column_stack([phi, np.ones(n, dtype=float)])
						try:
							res = lsq_linear(X, target, bounds=([0, 0], [np.inf, np.inf]))
							params = res.x
						except Exception:
							params = np.linalg.lstsq(X, target, rcond=None)[0]
							params[0] = max(params[0], 0.0)
							params[1] = max(params[1], 0.0)
						A, b = float(params[0]), float(params[1])
					recon = A * phi + b
					mse_ = float(np.mean((recon - target) ** 2))
					nll_ = 0.5 * n * (np.log(mse_ + 1e-12) + 1.0)
					# Choose best based on objective
					if (obj == "nll" and nll_ < best.get('nll', float('inf'))) or (obj != "nll" and mse_ < best['mse']):
						best = {'lxy': lxy, 'lz': lz, 'A': A, 'b': b, 'recon': recon, 'mse': mse_, 'nll': nll_, 'sigma2': mse_}
			# Add missing fields for consistency
			best.setdefault('rmse', float(np.sqrt(best['mse'])))
			best.setdefault('mae', float(np.mean(np.abs(best['recon'] - target))))
			ss_res = np.sum((target - best['recon']) ** 2)
			ss_tot = np.sum((target - np.mean(target)) ** 2)
			best.setdefault('r2_score', float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0)
			best.setdefault('optimization_result', None)
			return best

		# Extract optimal hyperparameters
		lxy_opt, lz_opt = best_result.x
		# Compute final basis function with optimal hyperparameters
		phi_opt = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy_opt, lz=lz_opt)
		n = phi_opt.shape[0]
		if fixed_b is not None:
			# Fixed-ambient mode: solve for A only, b = fixed_b
			y_adj = target - fixed_b
			# Check if phi has sufficient variation (not constant)
			phi_std = np.std(phi_opt)
			if phi_std < 1e-8:
				# Constant phi: cannot fit A, set to 0
				A_opt = 0.0
			else:
				phi_dot = np.dot(phi_opt, phi_opt)
				phi_dot_y = np.dot(phi_opt, y_adj)
				if phi_dot < 1e-12 or phi_dot_y <= 0:
					A_opt = 0.0
				else:
					# Regularize to avoid numerical issues when phi_dot is very small
					# Add small regularization term proportional to phi variance
					reg = 1e-10 * phi_std * phi_std * n
					A_opt = phi_dot_y / (phi_dot + reg)
					A_opt = max(0.0, A_opt)
			b_opt = float(fixed_b)
		else:
			# Solve for linear coefficients A, b (in original scale, not normalized)
			# Constrain A >= 0, b >= 0 (disturbance magnitudes are L2 norms)
			X = np.column_stack([phi_opt, np.ones(n, dtype=float)])
			try:
				res = lsq_linear(X, target, bounds=([0, 0], [np.inf, np.inf]))
				params_ab = res.x
			except Exception:
				params_ab = np.linalg.lstsq(X, target, rcond=None)[0]
				params_ab[0] = max(params_ab[0], 0.0)
				params_ab[1] = max(params_ab[1], 0.0)
			A_opt, b_opt = float(params_ab[0]), float(params_ab[1])
		# Compute predictions and metrics
		recon_opt = A_opt * phi_opt + b_opt
		mse_opt = float(np.mean((recon_opt - target) ** 2))
		rmse_opt = float(np.sqrt(mse_opt))
		mae_opt = float(np.mean(np.abs(recon_opt - target)))
		ss_res = np.sum((target - recon_opt) ** 2)
		ss_tot = np.sum((target - np.mean(target)) ** 2)
		r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
		
		# Noise variance estimate (used for uncertainty quantification)
		sigma2_opt = mse_opt
		
		# Negative log-likelihood (for Bayesian methods)
		nll_opt = 0.5 * n * (np.log(mse_opt + 1e-12) + 1.0)
		
		# Precompute (XᵀX)⁻¹ for downstream BLR uncertainty quantification.
		# This avoids recomputing phi on training data during every prediction call.
		# Shape: (1,1) when b is fixed, (2,2) when both A,b are fitted.
		XtX_inv_mat = None
		try:
			if fixed_b is not None:
				# Fixed b: only A is variable, design matrix is phi (N,1)
				phi_ss = float(np.dot(phi_opt, phi_opt))
				if phi_ss > 1e-12:
					XtX_inv_mat = np.array([[1.0 / phi_ss]])
			else:
				# Both A,b fitted: design matrix is [phi, 1] (N,2)
				X_train = np.column_stack([phi_opt, np.ones(n, dtype=float)])
				XtX = X_train.T @ X_train
				XtX[0, 0] += 1e-6  # Regularize for numerical stability
				XtX[1, 1] += 1e-6
				XtX_inv_mat = np.linalg.inv(XtX)
		except Exception:
			XtX_inv_mat = None
		
		# Extract optimization diagnostics (JSON-friendly)
		opt_info = {
			"nit": int(getattr(best_result, "nit", -1)),
			"nfev": int(getattr(best_result, "nfev", -1)),
			"success": bool(getattr(best_result, "success", False)),
			"message": str(getattr(best_result, "message", "")),
		}
		
		# Approximate inverse Hessian (from L-BFGS-B) for parameter uncertainty
		# This gives us epistemic uncertainty in hyperparameters lxy, lz
		hess_inv_mat = None
		param_std = None
		try:
			hinv = getattr(best_result, "hess_inv", None)
			if hinv is not None:
				# For L-BFGS-B this is an LbfgsInvHessProduct; convert to dense
				if hasattr(hinv, "todense"):
					hess_inv_mat = np.asarray(hinv.todense(), dtype=float)
				else:
					hess_inv_mat = np.asarray(hinv, dtype=float)
				if hess_inv_mat.shape == (2, 2):
					# Standard deviations ≈ sqrt(diag(H^{-1}))
					stds = np.sqrt(np.maximum(np.diag(hess_inv_mat), 0.0))
					param_std = {"lxy": float(stds[0]), "lz": float(stds[1])}
		except Exception:
			hess_inv_mat = None
			param_std = None
		
		return {
			'lxy': float(lxy_opt),
			'lz': float(lz_opt),
			'A': A_opt,
			'b': b_opt,
			'recon': recon_opt,
			'mse': mse_opt,
			'rmse': rmse_opt,
			'mae': mae_opt,
			'r2_score': r2_score,
			'sigma2': sigma2_opt,  # Noise variance σ² (aleatoric)
			'nll': nll_opt,  # Negative log-likelihood
			'XtX_inv': (XtX_inv_mat.tolist() if XtX_inv_mat is not None else None),  # (XᵀX)⁻¹ for BLR uncertainty
			'fixed_b': fixed_b,  # Whether b was fixed (affects uncertainty formula)
			'optimization_result': best_result,
			'optimization': opt_info,  # JSON-friendly optimization info
			'hess_inv': (hess_inv_mat.tolist() if hess_inv_mat is not None else None),  # Cov(lxy, lz) for delta method
			'param_std': param_std,  # Standard deviations of lxy, lz
		}

	# ----------------------------
	# Grid and prediction
	# ----------------------------

	def create_3d_prediction_grid(self, xyz: np.ndarray, cause_xyz: Optional[np.ndarray], resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
		"""
		Create a 3D prediction grid around the trajectory and cause points.
		
		The grid is padded around the bounding box of the trajectory and cause points,
		allowing us to predict the GP field at regular intervals in 3D space for
		visualization and planning purposes.
		
		Args:
			xyz: (N, 3) trajectory points to bound the grid
			cause_xyz: Optional (3,) cause location to include in bounding box
			resolution_xy: Grid resolution in xy plane (meters, default from config)
			resolution_z: Grid resolution in z direction (meters, default from config)
		
		Returns:
			tuple (Xg, Yg, Zg, grid_points, xs, ys, zs):
				- Xg, Yg, Zg: 3D meshgrid arrays (for visualization)
				- grid_points: (N_grid, 3) flattened grid points
				- xs, ys, zs: 1D coordinate arrays
		"""
		pad = float(self.cfg['pad_bounds'])
		res_xy = float(resolution_xy or self.cfg['resolution_xy'])
		res_z = float(resolution_z or self.cfg['resolution_z'])
		xmin, xmax = xyz[:, 0].min() - pad, xyz[:, 0].max() + pad
		ymin, ymax = xyz[:, 1].min() - pad, xyz[:, 1].max() + pad
		zmin, zmax = xyz[:, 2].min() - pad, xyz[:, 2].max() + pad
		if cause_xyz is not None:
			xmin = min(xmin, cause_xyz[0] - pad)
			xmax = max(xmax, cause_xyz[0] + pad)
			ymin = min(ymin, cause_xyz[1] - pad)
			ymax = max(ymax, cause_xyz[1] + pad)
			zmin = min(zmin, cause_xyz[2] - pad)
			zmax = max(zmax, cause_xyz[2] + pad)
		xs = np.arange(xmin, xmax + res_xy, res_xy)
		ys = np.arange(ymin, ymax + res_xy, res_xy)
		zs = np.arange(zmin, zmax + res_z, res_z)
		Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='xy')
		grid_points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
		return Xg, Yg, Zg, grid_points, xs, ys, zs

	def predict_direct_field_3d(self, fit_params: Dict[str, Any], grid_points: np.ndarray, cause_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Predict GP field mean and full predictive standard deviation.

		Combines three uncertainty components into a single predictive variance:

		  Var[y*] = σ²                              (1. aleatoric noise)
		          + σ² · vᵀ (XᵀX)⁻¹ v               (2. linear param BLR)
		          + A² · [∂φ/∂ℓ]ᵀ H⁻¹ [∂φ/∂ℓ]       (3. length-scale delta)

		Component 1 — Aleatoric noise (σ² = MSE):
		  Irreducible observation noise.

		Component 2 — Linear parameter uncertainty (Bayesian Linear Regression):
		  Posterior variance of (A, b) given fixed (lxy, lz).
		  Feature vector v = [φ(x*), 1] (free b) or v = φ(x*) (fixed b).
		  Requires XtX_inv from fitting.

		Component 3 — Length-scale uncertainty (Delta method):
		  Propagates (lxy, lz) covariance from L-BFGS-B inverse Hessian.
		  Jacobian: J = A · [∂φ/∂lxy, ∂φ/∂lz].
		  Requires hess_inv from fitting and ∂φ/∂ℓ gradients.

		Graceful degradation:
		  - If XtX_inv missing → skip component 2
		  - If hess_inv missing → skip component 3
		  - If both missing → returns σ² only (pure noise)

		Args:
			fit_params: Dictionary from fit_direct_superposition_to_disturbances.
				Required: lxy, lz, A, b.
				For full uncertainty: sigma2, XtX_inv, hess_inv, fixed_b.
			grid_points: (N, 3) query points
			cause_points: (M, 3) cause points (RBF centers)

		Returns:
			(mean_pred, std_pred): each (N,) arrays.
		"""
		if fit_params is None or 'lxy' not in fit_params or fit_params['lxy'] is None:
			return np.zeros(grid_points.shape[0]), np.zeros(grid_points.shape[0])

		lxy = float(fit_params['lxy'])
		lz = float(fit_params['lz'])
		A = float(fit_params['A'])
		b = float(fit_params['b'])
		sigma2 = fit_params.get('sigma2', 0.0)
		XtX_inv = fit_params.get('XtX_inv')
		hess_inv = fit_params.get('hess_inv')
		is_fixed_b = fit_params.get('fixed_b') is not None

		# Decide whether to compute gradients (only needed for component 3)
		need_grads = (hess_inv is not None and A != 0.0)

		if need_grads:
			phi, dphi_dlxy, dphi_dlz = self._rbf_phi_and_grads(
				grid_points, cause_points, lxy, lz
			)
		else:
			phi = self._sum_of_anisotropic_rbf(
				grid_points, cause_points, lxy=lxy, lz=lz
			)

		# ── Mean prediction ──
		mean_pred = A * phi + b

		# ── Variance: start with aleatoric noise (component 1) ──
		n_pts = grid_points.shape[0]
		if sigma2 is None or sigma2 <= 0 or sigma2 == float('inf'):
			# No noise info → heuristic fallback
			std_pred = np.full(n_pts, 0.1 * np.std(mean_pred))
			return mean_pred, std_pred

		var_pred = np.full(n_pts, float(sigma2))

		# ── Component 2: BLR (linear parameter uncertainty) ──
		if XtX_inv is not None:
			M = np.asarray(XtX_inv, dtype=float)
			if is_fixed_b and M.shape == (1, 1):
				# Only A is uncertain: Var_blr = σ² · M₀₀ · φ²
				var_pred += sigma2 * M[0, 0] * (phi ** 2)
			elif not is_fixed_b and M.shape == (2, 2):
				# Both A, b uncertain: Var_blr = σ² · vᵀ M v, v=[φ, 1]
				var_pred += sigma2 * (
					M[0, 0] * (phi ** 2)
					+ 2.0 * M[0, 1] * phi
					+ M[1, 1]
				)

		# ── Component 3: length-scale uncertainty (delta method) ──
		if need_grads:
			H = np.asarray(hess_inv, dtype=float)
			if H.shape == (2, 2):
				j_lxy = A * dphi_dlxy
				j_lz = A * dphi_dlz
				var_ls = (
					H[0, 0] * (j_lxy ** 2)
					+ 2.0 * H[0, 1] * j_lxy * j_lz
					+ H[1, 1] * (j_lz ** 2)
				)
				var_pred += np.maximum(var_ls, 0.0)

		std_pred = np.sqrt(np.maximum(var_pred, 0.0))
		return mean_pred, std_pred

	# ----------------------------
	# Visualizations
	# ----------------------------

	def _normalize_percentile(self, values: np.ndarray, lower_pct: float, upper_pct: float):
		lo = np.percentile(values, lower_pct)
		hi = np.percentile(values, upper_pct)
		if hi <= lo:
			hi = lo + 1e-9
		v = np.clip(values, lo, hi)
		v = (v - lo) / (hi - lo)
		return v, lo, hi

	def plot_2d_points(self, xyz: np.ndarray, nominal_points_used: np.ndarray, disturbance_magnitudes: np.ndarray, cause_xyz: Optional[np.ndarray], cause: Optional[str] = None):
		pr = self.cfg['percentile_range_2d']
		norm_vals, _, _ = self._normalize_percentile(disturbance_magnitudes, *pr)
		fig, ax = plt.subplots(1, 1, figsize=(8, 6))
		ax.plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=2, alpha=0.55, label='Actual trajectory')
		if nominal_points_used is not None and nominal_points_used.size > 0:
			sc = ax.scatter(nominal_points_used[:, 0], nominal_points_used[:, 1], c=norm_vals, cmap='plasma', s=42, alpha=0.95, edgecolors='white', linewidths=0.4)
			ax.plot(nominal_points_used[:, 0], nominal_points_used[:, 1], '--', color='orange', linewidth=1.3, alpha=0.8, label='Nominal (sampled)')
		else:
			sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=norm_vals, cmap='plasma', s=42, alpha=0.95, edgecolors='white', linewidths=0.4)
		if cause_xyz is not None:
			ax.scatter(cause_xyz[0], cause_xyz[1], marker='*', s=320, color='red', edgecolors='white', linewidths=1.2, label=f'Cause: {cause or "Unknown"}')
		ax.set_xlabel('X (m)')
		ax.set_ylabel('Y (m)')
		ax.set_title('2D Points: Normalized Disturbance (XY)')
		ax.set_aspect('equal')
		ax.legend(fontsize=9)
		cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
		cbar.set_label('Normalized disturbance')
		fig.tight_layout()
		return fig

	def plot_gp_orthogonal_views(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, mean_field: np.ndarray, xyz: np.ndarray, cause_xyz: Optional[np.ndarray]):
		Ny, Nx, Nz = len(ys), len(xs), len(zs)
		volume = mean_field.reshape(Ny, Nx, Nz)
		kz = Nz // 2
		kx = Nx // 2
		ky = Ny // 2
		fig, axes = plt.subplots(1, 3, figsize=(15, 5))
		xy_slice = volume[:, :, kz]
		xy_norm, _, _ = self._normalize_percentile(xy_slice, *self.cfg['percentile_range_3d'])
		im0 = axes[0].imshow(xy_norm, extent=(xs.min(), xs.max(), ys.min(), ys.max()), origin='lower', aspect='equal', cmap='viridis')
		axes[0].plot(xyz[:, 0], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
		if cause_xyz is not None:
			axes[0].scatter(cause_xyz[0], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
		axes[0].set_xlim(xs.min(), xs.max())
		axes[0].set_ylim(ys.min(), ys.max())
		axes[0].set_title('GP slice XY (z = mid)')
		axes[0].set_xlabel('X (m)')
		axes[0].set_ylabel('Y (m)')
		fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
		yz_slice = volume[:, kx, :]
		yz_norm, _, _ = self._normalize_percentile(yz_slice, *self.cfg['percentile_range_3d'])
		im1 = axes[1].imshow(yz_norm, extent=(zs.min(), zs.max(), ys.min(), ys.max()), origin='lower', aspect='equal', cmap='viridis')
		axes[1].plot(xyz[:, 2], xyz[:, 1], 'b-', linewidth=1.5, alpha=0.8)
		if cause_xyz is not None:
			axes[1].scatter(cause_xyz[2], cause_xyz[1], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
		axes[1].set_xlim(zs.min(), zs.max())
		axes[1].set_ylim(ys.min(), ys.max())
		axes[1].set_title('GP slice YZ (x = mid)')
		axes[1].set_xlabel('Z (m)')
		axes[1].set_ylabel('Y (m)')
		fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
		zx_slice = volume[ky, :, :].T
		zx_norm, _, _ = self._normalize_percentile(zx_slice, *self.cfg['percentile_range_3d'])
		im2 = axes[2].imshow(zx_norm, extent=(xs.min(), xs.max(), zs.min(), zs.max()), origin='lower', aspect='equal', cmap='viridis')
		axes[2].plot(xyz[:, 0], xyz[:, 2], 'b-', linewidth=1.5, alpha=0.8)
		if cause_xyz is not None:
			axes[2].scatter(cause_xyz[0], cause_xyz[2], marker='*', s=160, color='red', edgecolors='white', linewidths=0.6)
		axes[2].set_xlim(xs.min(), xs.max())
		axes[2].set_ylim(zs.min(), zs.max())
		axes[2].set_title('GP slice ZX (y = mid)')
		axes[2].set_xlabel('X (m)')
		axes[2].set_ylabel('Z (m)')
		fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
		fig.tight_layout()
		return fig

	def plot_3d_volume_with_cause_points(self, Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces: bool = True, max_cause_points: int = 5000):
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		if use_isosurfaces and _HAS_SKIMAGE:
			self._plot_3d_isosurfaces(mean_field, xs, ys, zs, ax=ax, num_levels=5)
		else:
			self._plot_3d_pointcloud(mean_field.reshape(Xg.shape), Xg, Yg, Zg, ax=ax, max_points=self.cfg['max_points_pointcloud'])
		ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue', linewidth=2.0, alpha=0.9, label='Actual trajectory')
		if cause_points is not None and cause_points.size > 0:
			pts = cause_points
			if pts.shape[0] > max_cause_points:
				idx = np.random.RandomState(42).choice(pts.shape[0], size=max_cause_points, replace=False)
				pts = pts[idx]
			ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, c='red', alpha=0.6, edgecolors='none', label='Cause points')
		ax.set_xlabel('X (m)')
		ax.set_ylabel('Y (m)')
		ax.set_zlabel('Z (m)')
		ax.set_title('Reconstructed 3D Field (Superposed) + Cause Points')
		ax.legend(loc='upper right')
		ax.set_xlim(xs.min(), xs.max())
		ax.set_ylim(ys.min(), ys.max())
		ax.set_zlim(zs.min(), zs.max())
		fig.tight_layout()
		return fig

	def _plot_3d_pointcloud(self, mean_field: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, Zg: np.ndarray, ax, max_points: int = 50000):
		X = Xg.ravel(); Y = Yg.ravel(); Z = Zg.ravel(); V = mean_field.ravel()
		N = X.shape[0]
		if N > max_points:
			idx = np.random.RandomState(42).choice(N, size=max_points, replace=False)
			X, Y, Z, V = X[idx], Y[idx], Z[idx], V[idx]
		Vn, _, _ = self._normalize_percentile(V, *self.cfg['percentile_range_3d'])
		ax.scatter(X, Y, Z, c=Vn, cmap='viridis', s=2, alpha=self.cfg['point_cloud_alpha'])

	def _plot_3d_isosurfaces(self, mean_field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, ax, num_levels: int = 8):
		if not _HAS_SKIMAGE:
			raise RuntimeError("skimage not available for isosurface rendering")
		Ny, Nx, Nz = len(ys), len(xs), len(zs)
		field_3d = mean_field.reshape(Ny, Nx, Nz)
		volume = np.transpose(field_3d, (2, 0, 1))
		field_min, field_max = mean_field.min(), mean_field.max()
		field_range = field_max - field_min
		if field_range < 1e-8:
			return
		percentiles = np.linspace(50, 98, num_levels)
		levels = np.percentile(mean_field, percentiles)
		levels = np.clip(levels, field_min + 0.005 * field_range, field_max - 0.005 * field_range)
		dx = xs[1] - xs[0] if len(xs) > 1 else 1.0
		dy = ys[1] - ys[0] if len(ys) > 1 else 1.0
		dz = zs[1] - zs[0] if len(zs) > 1 else 1.0
		xmin, ymin, zmin = xs.min(), ys.min(), zs.min()
		colors = ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#795548']
		alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]
		for i, (level, color, alpha) in enumerate(zip(levels, colors, alphas)):
			try:
				verts, faces, normals, values = _sk_measure.marching_cubes(volume=volume, level=level, spacing=(dz, dy, dx))
				if len(verts) > 0:
					verts_world = np.column_stack([
						verts[:, 2] + xmin,
						verts[:, 1] + ymin,
						verts[:, 0] + zmin,
					])
					ax.plot_trisurf(verts_world[:, 0], verts_world[:, 1], faces, verts_world[:, 2], color=color, lw=0.0, edgecolor='none', alpha=alpha)
			except Exception:
				pass

	# ----------------------------
	# High-level pipeline
	# ----------------------------

		  
	def fit_from_pointcloud_and_buffer(self, pointcloud_xyz: np.ndarray, buffer_dir: str, nominal_path: Optional[str] = None, nominal_xyz: Optional[np.ndarray] = None, clip_plane: str = 'xy', objective: str = "nll", safe_points_xyz: Optional[np.ndarray] = None, ambient_disturbance: Optional[float] = None) -> Dict[str, Any]:
		"""
		End-to-end fitting pipeline using a provided cause pointcloud and buffer directory.
		
		This is the main entry point for GP fitting. It:
		1. Loads actual trajectory and cause metadata from buffer directory
		2. Loads/clips nominal trajectory to match actual trajectory segment
		3. Computes disturbance magnitudes at nominal points
		4. (Optional) Augments training data with safe evidence — pre-breach
		   trajectory points with known ambient disturbance
		5. Fits GP parameters (lxy, lz, A, b) using the superposed RBF model
		
		When ambient_disturbance is provided (step 4), b is FIXED to the known
		ambient level instead of being fitted. This prevents b from absorbing
		spatial structure that should be in A*phi, yielding sharper fits.
		Without it, b is unconstrained and may absorb the mean disturbance.
		
		Args:
			pointcloud_xyz: numpy array (N,3) of cause points (e.g., from PCD file)
			buffer_dir: path to buffer directory containing poses.npy and metadata
			nominal_path: optional path to nominal trajectory JSON file
			nominal_xyz: alternatively, pass nominal points directly as numpy array
			clip_plane: 'xy' or 'xz' for clipping nominal to actual segment
			objective: "mse" or "nll" (default: "nll" for Bayesian, better uncertainty)
			safe_points_xyz: optional (K,3) array of pre-breach trajectory positions.
				Used for diagnostics/logging to validate the ambient assumption.
			ambient_disturbance: if provided, fixes b to this value (meters) and only
				fits A. Typically 0.0005m (0.05cm) — the system's nominal tracking error.
				If None, fits both A and b freely (backward compatible).
		
		Returns:
			Dictionary with:
				- 'fit': Fitted GP parameters (lxy, lz, A, b) and metrics
				- 'actual_xyz': Actual trajectory points
				- 'nominal_used': Training locations (breach + safe evidence)
				- 'disturbances': Disturbance magnitudes at training locations
				- 'cause': Cause name/description (if available)
				- 'cause_xyz': Cause location (if available)
		"""
		def _no_fit(reason: str, diagnostics: Optional[Dict[str, Any]] = None):
			return {
				'lxy': None,
				'lz': None,
				'A': 0.0,
				'b': 0.0,
				'recon': np.array([], dtype=float),
				'mse': float('inf'),
				'r2_score': 0.0,
				'mae': float('inf'),
				'rmse': float('inf'),
				'sigma2': float('inf'),
				'nll': float('inf'),
				'optimization_result': None,
				'optimization': {"success": False, "message": f"no_fit:{reason}"},
				'status': 'no_fit',
				'reason': reason,
				'diagnostics': (diagnostics or {}),
			}

		# Load actual trajectory and cause metadata
		actual_xyz, cause, cause_xyz = self.load_buffer_xyz_drift(buffer_dir)

		# Resolve nominal trajectory (nominal_xyz overrides nominal_path)
		nominal_resolved = nominal_xyz
		nominal_source = "provided_array" if nominal_xyz is not None else None
		if nominal_resolved is None and nominal_path:
			nominal_resolved = self.load_nominal_xyz(nominal_path)
			nominal_source = "file" if nominal_resolved is not None else None

		if nominal_resolved is None or len(nominal_resolved) == 0:
			fit = _no_fit("missing_nominal", {"nominal_source": nominal_source, "nominal_path": nominal_path})
			return {
				'fit': fit,
				'status': 'no_fit',
				'actual_xyz': actual_xyz,
				'nominal_used': np.array([], dtype=float).reshape(0, 3),
				'disturbances': np.array([], dtype=float),
				'cause': cause,
				'cause_xyz': cause_xyz,
			}

		# Clip nominal to match actual segment
		clipped_nominal = self.clip_nominal_to_actual_segment(nominal_resolved, actual_xyz, plane=clip_plane)
		if clipped_nominal is None or len(clipped_nominal) == 0:
			clipped_nominal = nominal_resolved

		# Validate nominal/actual overlap before fitting (no fallback training semantics)
		match_thresh = float(self.cfg.get('disturbance_match_threshold_m', 0.30))
		min_overlap = float(self.cfg.get('min_nominal_actual_overlap_fraction', 0.10))
		min_dists = self._min_distances_to_set(clipped_nominal, actual_xyz, chunk=512)
		if min_dists.size == 0:
			fit = _no_fit("empty_distance_check", {"nominal_source": nominal_source})
			return {
				'fit': fit,
				'status': 'no_fit',
				'actual_xyz': actual_xyz,
				'nominal_used': np.array([], dtype=float).reshape(0, 3),
				'disturbances': np.array([], dtype=float),
				'cause': cause,
				'cause_xyz': cause_xyz,
			}
		overlap_frac = float(np.mean(min_dists <= match_thresh))
		diag = {
			'nominal_source': nominal_source,
			'nominal_path': nominal_path,
			'clip_plane': clip_plane,
			'match_threshold_m': match_thresh,
			'min_overlap_fraction': min_overlap,
			'overlap_fraction': overlap_frac,
			'min_nominal_to_actual_dist_m': float(np.min(min_dists)) if min_dists.size else None,
			'median_nominal_to_actual_dist_m': float(np.median(min_dists)) if min_dists.size else None,
		}
		if overlap_frac < min_overlap:
			fit = _no_fit("insufficient_nominal_actual_overlap", diag)
			return {
				'fit': fit,
				'status': 'no_fit',
				'actual_xyz': actual_xyz,
				'nominal_used': np.array([], dtype=float).reshape(0, 3),
				'disturbances': np.array([], dtype=float),
				'cause': cause,
				'cause_xyz': cause_xyz,
			}

		# Compute disturbances strictly at nominal points near the observed (no fallback)
		nominal_points_used, disturbance_magnitudes = self.compute_disturbance_at_nominal_points(clipped_nominal, actual_xyz, cause_xyz)
		
		# Determine whether to fix b = ambient (instead of fitting it)
		# When ambient_disturbance is provided, fix b to that value.
		# This prevents b from absorbing spatial structure that A*phi should capture.
		# Safe evidence (pre-breach poses) validates the ambient level; logged for diagnostics.
		n_safe = len(safe_points_xyz) if safe_points_xyz is not None else 0
		fixed_b = float(ambient_disturbance) if ambient_disturbance is not None else None
		diag['safe_evidence_points'] = n_safe
		diag['ambient_disturbance_m'] = ambient_disturbance
		diag['fixed_b'] = fixed_b
		
		if nominal_points_used.size == 0 or disturbance_magnitudes.size == 0:
			fit = _no_fit("no_valid_disturbance_samples", diag)
			return {
				'fit': fit,
				'status': 'no_fit',
				'actual_xyz': actual_xyz,
				'nominal_used': np.array([], dtype=float).reshape(0, 3),
				'disturbances': np.array([], dtype=float),
				'cause': cause,
				'cause_xyz': cause_xyz,
			}

		# Fit GP parameters using the computed disturbances
		fit = self.fit_direct_superposition_to_disturbances(
			nominal_points_used,
			disturbance_magnitudes,
			pointcloud_xyz if pointcloud_xyz is not None else np.empty((0, 3), dtype=float),
			objective=objective,
			fixed_b=fixed_b
		)
		# Attach status + diagnostics for downstream logging/robustness
		if isinstance(fit, dict):
			fit.setdefault('status', 'ok')
			fit.setdefault('reason', None)
			fit.setdefault('diagnostics', diag)

		return {
			'fit': fit,
			'status': (fit.get('status') if isinstance(fit, dict) else 'ok'),
			'actual_xyz': actual_xyz,
			'nominal_used': nominal_points_used,
			'disturbances': disturbance_magnitudes,
			'cause': cause,
			'cause_xyz': cause_xyz,
		}

	def predict_grid_from_fit(self, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray], fit: Dict[str, Any], pointcloud_xyz: np.ndarray, resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
		Xg, Yg, Zg, grid_points, xs, ys, zs = self.create_3d_prediction_grid(actual_xyz, cause_xyz, resolution_xy, resolution_z)
		mean_pred, std_pred = self.predict_direct_field_3d(fit, grid_points, pointcloud_xyz)
		return Xg, Yg, Zg, grid_points, xs, ys, zs, mean_pred, std_pred

	# ----------------------------
	# PyVista visualization (optional, off by default)
	# ----------------------------

	def plot_3d_pyvista_volume_with_points(self, xs, ys, zs, mean_field, xyz, cause_points):
		if not _HAS_PYVISTA:
			raise RuntimeError("PyVista not available")
		Ny, Nx, Nz = len(ys), len(xs), len(zs)
		volume = mean_field.reshape(Ny, Nx, Nz)
		theme = _pv_themes.DefaultTheme()
		theme.cmap = 'viridis'
		theme.colorbar_orientation = 'vertical'
		plotter = _pv.Plotter(window_size=(1024, 768), theme=theme)
		grid = _pv.UniformGrid()
		grid.dimensions = np.array(volume.shape) + 1
		spacing = (ys[1] - ys[0] if Ny > 1 else 1.0, xs[1] - xs[0] if Nx > 1 else 1.0, zs[1] - zs[0] if Nz > 1 else 1.0)
		origin = (ys.min(), xs.min(), zs.min())
		grid.spacing = spacing
		grid.origin = origin
		grid.cell_data['disturbance'] = volume.ravel(order='F')
		Vn, vmin, vmax = self._normalize_percentile(mean_field, *self.cfg['percentile_range_3d'])
		levels = np.percentile(Vn, [55, 65, 75, 82, 88, 93, 96, 98])
		raw_levels = vmin + levels * (vmax - vmin)
		try:
			contour = grid.contour(isosurfaces=list(raw_levels), scalars='disturbance')
			plotter.add_mesh(contour, opacity=0.20, cmap='viridis', show_edges=False)
		except Exception:
			plotter.add_volume(grid, scalars='disturbance', opacity='sigmoid', cmap='viridis', shade=True)
		traj = _pv.Spline(xyz, len(xyz)) if xyz.shape[0] >= 2 else _pv.PolyData(xyz)
		plotter.add_mesh(traj, color='blue', line_width=3, label='Actual trajectory')
		if cause_points is not None and cause_points.size > 0:
			cloud = _pv.PolyData(cause_points)
			plotter.add_mesh(cloud, color='red', point_size=6, render_points_as_spheres=True, opacity=0.6)
		plotter.add_axes(interactive=False)
		plotter.show_bounds(grid='front', location='outer', ticks='outside')
		plotter.add_scalar_bar(title='Disturbance', n_labels=4)
		plotter.set_background('white')
		plotter.show() 

# ============================================================
# Module-level wrappers to mirror voxel_gp.py function API
# ============================================================

def load_buffer_xyz_drift(buffer_dir: str):
	return DisturbanceFieldHelper.load_buffer_xyz_drift(buffer_dir)

def _sum_of_anisotropic_rbf_fast(grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
		"""OPTIMIZED anisotropic RBF computation for speed."""
		try:
			if centers.size == 0:
				return np.zeros(grid_points.shape[0], dtype=float)
			
			# Precompute inverse squared length scales
			inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
			inv_lz2 = 1.0 / (lz * lz + 1e-12)
			
			# Vectorized computation - much faster than chunked approach
			dx = grid_points[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
			dy = grid_points[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
			dz = grid_points[:, np.newaxis, 2] - centers[np.newaxis, :, 2]
			
			# Compute anisotropic distance squared
			d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
			
			# Compute RBF contributions and sum over all centers
			phi = np.sum(np.exp(-0.5 * d2), axis=1)
			
			return phi
			
		except Exception as e:
			return np.zeros(grid_points.shape[0], dtype=float)
		
def load_nominal_xyz(nominal_path: str):
	return DisturbanceFieldHelper.load_nominal_xyz(nominal_path)

def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = 'xy') -> np.ndarray:
	return DisturbanceFieldHelper.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane)

def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
	return DisturbanceFieldHelper.compute_trajectory_drift_vectors(actual_xyz, nominal_xyz)

def compute_disturbance_at_nominal_points(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray] = None):
	return DisturbanceFieldHelper.compute_disturbance_at_nominal_points(nominal_xyz, actual_xyz, cause_xyz)

def create_3d_prediction_grid(xyz: np.ndarray, cause_xyz: Optional[np.ndarray], resolution_xy: Optional[float] = None, resolution_z: Optional[float] = None):
	helper = DisturbanceFieldHelper()
	return helper.create_3d_prediction_grid(xyz, cause_xyz, resolution_xy, resolution_z)

def fit_direct_superposition_to_disturbances(nominal_points: np.ndarray, disturbance_magnitudes: np.ndarray, cause_points: np.ndarray):
	helper = DisturbanceFieldHelper()
	return helper.fit_direct_superposition_to_disturbances(nominal_points, disturbance_magnitudes, cause_points)

def predict_direct_field_3d(fit_params: Dict[str, Any], grid_points: np.ndarray, cause_points: np.ndarray):
	helper = DisturbanceFieldHelper()
	return helper.predict_direct_field_3d(fit_params, grid_points, cause_points)

def plot_2d_points(xyz: np.ndarray, nominal_points_used: np.ndarray, disturbance_magnitudes: np.ndarray, cause_xyz: Optional[np.ndarray], cause: Optional[str] = None):
	helper = DisturbanceFieldHelper()
	return helper.plot_2d_points(xyz, nominal_points_used, disturbance_magnitudes, cause_xyz, cause)

def plot_gp_orthogonal_views(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, mean_field: np.ndarray, xyz: np.ndarray, cause_xyz: Optional[np.ndarray]):
	helper = DisturbanceFieldHelper()
	return helper.plot_gp_orthogonal_views(xs, ys, zs, mean_field, xyz, cause_xyz)

def plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces: bool = True, max_cause_points: int = 5000):
	helper = DisturbanceFieldHelper()
	return helper.plot_3d_volume_with_cause_points(Xg, Yg, Zg, mean_field, xs, ys, zs, xyz, cause_points, use_isosurfaces, max_cause_points)

def plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_field, xyz, cause_points):
	helper = DisturbanceFieldHelper()
	return helper.plot_3d_pyvista_volume_with_points(xs, ys, zs, mean_field, xyz, cause_points) 