#!/usr/bin/env python3
"""
Visualize GP uncertainty around cause points (points.pcd)

Shows:
1. Mean GP field (risk)
2. Bayesian uncertainty in A, b parameters
3. Conservative risk (mean + 2σ)

This script:
- Runs NLL optimization to fit GP parameters
- Computes Bayesian linear regression uncertainty
- Visualizes risk, uncertainty, and conservative risk
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import sys
import os
import argparse
from pathlib import Path

# Import fitting helper from offline_gp_fit_and_viz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from offline_gp_fit_and_viz import DisturbanceFieldHelper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

# Optional deps (mirroring voxel_gp_helper)
try:
    from skimage import measure as _sk_measure  # type: ignore
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import pyvista as _pv  # type: ignore
    from pyvista import themes as _pv_themes  # type: ignore
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False


_DEFAULTS = {
    "resolution_xy": 0.06,
    "resolution_z": 0.06,
    "pad_bounds": 0.5,
    "percentile_range_2d": (5, 95),
    "percentile_range_3d": (5, 97),
    "max_points_pointcloud": 50000,
    "point_cloud_alpha": 0.28,
}
# ============================================================================
# GP UNCERTAINTY FIELD CLASS (Modular Interface)
# ============================================================================
class DisturbanceFieldHelper:
    """Helper for computing and visualizing 3D disturbance fields."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update(config)
        self.cfg = cfg

    # ---------- Data loading ----------

    @staticmethod
    def load_buffer_xyz_drift(buffer_dir: str) -> Tuple[np.ndarray, Optional[str], Optional[np.ndarray]]:
        """Load actual trajectory xyz and cause metadata from buffer."""
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
        """Load nominal trajectory from JSON format used in assets."""
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
                        xyz_list.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
                if xyz_list:
                    return np.array(xyz_list, dtype=float)
            else:
                arr = np.array(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
        return None

    # ---------- Drift / disturbance computation ----------

    @staticmethod
    def clip_nominal_to_actual_segment(nominal_xyz: np.ndarray, actual_xyz: np.ndarray, plane: str = "xy") -> np.ndarray:
        if nominal_xyz is None or len(nominal_xyz) == 0 or actual_xyz is None or len(actual_xyz) == 0:
            return nominal_xyz
        plane = plane.lower()
        if plane not in ("xy", "xz"):
            plane = "xy"
        if plane == "xy":
            nom_proj = nominal_xyz[:, [0, 1]]
            act_start = actual_xyz[0, [0, 1]]
            act_end = actual_xyz[-1, [0, 1]]
        else:
            nom_proj = nominal_xyz[:, [0, 2]]
            act_start = actual_xyz[0, [0, 2]]
            act_end = actual_xyz[-1, [0, 2]]

        d_start = np.linalg.norm(nom_proj - act_start[None, :], axis=1)
        d_end = np.linalg.norm(nom_proj - act_end[None, :], axis=1)
        i_start = int(np.argmin(d_start))
        i_end = int(np.argmin(d_end))

        lo, hi = (i_start, i_end) if i_start <= i_end else (i_end, i_start)
        lo = max(0, lo)
        hi = min(len(nominal_xyz) - 1, hi)
        if hi <= lo:
            return nominal_xyz
        return nominal_xyz[lo : hi + 1]

    @staticmethod
    def compute_trajectory_drift_vectors(actual_xyz: np.ndarray, nominal_xyz: np.ndarray):
        if nominal_xyz is None or len(nominal_xyz) == 0:
            return None, None
        drift_vectors = []
        drift_magnitudes = []
        for actual_point in actual_xyz:
            diffs = nominal_xyz - actual_point
            dists = np.linalg.norm(diffs, axis=1)
            closest_idx = int(np.argmin(dists))
            drift_vec = actual_point - nominal_xyz[closest_idx]
            drift_mag = float(np.linalg.norm(drift_vec))
            drift_vectors.append(drift_vec)
            drift_magnitudes.append(drift_mag)
        return np.array(drift_vectors), np.array(drift_magnitudes)

    @staticmethod
    def compute_disturbance_at_nominal_points(
        nominal_xyz: np.ndarray, actual_xyz: np.ndarray, cause_xyz: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute disturbance at nominal trajectory points near the actual trajectory bounds."""
        disturbances = []
        nominal_points_used = []
        actual_bounds = {
            "x": (actual_xyz[:, 0].min(), actual_xyz[:, 0].max()),
            "y": (actual_xyz[:, 1].min(), actual_xyz[:, 1].max()),
            "z": (actual_xyz[:, 2].min(), actual_xyz[:, 2].max()),
        }
        pad = 0.3
        mask = (
            (nominal_xyz[:, 0] >= actual_bounds["x"][0] - pad)
            & (nominal_xyz[:, 0] <= actual_bounds["x"][1] + pad)
            & (nominal_xyz[:, 1] >= actual_bounds["y"][0] - pad)
            & (nominal_xyz[:, 1] <= actual_bounds["y"][1] + pad)
            & (nominal_xyz[:, 2] >= actual_bounds["z"][0] - pad)
            & (nominal_xyz[:, 2] <= actual_bounds["z"][1] + pad)
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

    # ---------- Core kernel + fitting ----------

    @staticmethod
    def _sum_of_anisotropic_rbf(grid_points: np.ndarray, centers: np.ndarray, lxy: float, lz: float) -> np.ndarray:
        if centers.size == 0:
            return np.zeros(grid_points.shape[0], dtype=float)
        num_points = grid_points.shape[0]
        phi = np.zeros(num_points, dtype=float)
        chunk = 200000
        inv_lxy2 = 1.0 / (lxy * lxy + 1e-12)
        inv_lz2 = 1.0 / (lz * lz + 1e-12)
        for start in range(0, num_points, chunk):
            end = min(num_points, start + chunk)
            gp_chunk = grid_points[start:end]
            dx = gp_chunk[:, None, 0] - centers[None, :, 0]
            dy = gp_chunk[:, None, 1] - centers[None, :, 1]
            dz = gp_chunk[:, None, 2] - centers[None, :, 2]
            d2 = (dx * dx + dy * dy) * inv_lxy2 + (dz * dz) * inv_lz2
            np.exp(-0.5 * d2, out=d2)
            phi[start:end] = d2.sum(axis=1)
        return phi

    def fit_direct_superposition_to_disturbances(
        self,
        nominal_points: np.ndarray,
        disturbance_magnitudes: np.ndarray,
        cause_points: np.ndarray,
        objective: str = "mse",
    ) -> Dict[str, Any]:
        """Fit lxy, lz, A, b for the superposed anisotropic RBF model.

        objective: "mse" or "nll"
            - mse: minimize sum of squared errors (what we had)
            - nll: Gaussian NLL with sigma^2 = SSE / n (adds a log term)
        """
        if cause_points.size == 0:
            return {
                "lxy": None,
                "lz": None,
                "A": 0.0,
                "b": 0.0,
                "recon": np.zeros_like(disturbance_magnitudes),
                "mse": float("inf"),
                "r2_score": 0.0,
                "mae": float("inf"),
                "rmse": float("inf"),
            }
        obj = objective.lower()
        target = disturbance_magnitudes.astype(float)
        target_mean = np.mean(target)
        target_std = np.std(target)
        if target_std < 1e-8:
            target_std = 1.0
        target_norm = (target - target_mean) / target_std

        from scipy.optimize import minimize

        def objective_fn(params):
            lxy, lz = params
            lxy = max(lxy, 0.01)
            lz = max(lz, 0.01)
            phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
            phi_mean = np.mean(phi)
            phi_std = np.std(phi)
            if phi_std < 1e-8:
                return float("inf")
            phi_norm = (phi - phi_mean) / phi_std
            n = phi_norm.shape[0]
            X = np.column_stack([phi_norm, np.ones(n, dtype=float)])
            try:
                XtX = X.T @ X
                XtY = X.T @ target_norm
                params_ab = np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                params_ab = np.linalg.lstsq(X, target_norm, rcond=None)[0]
            A_norm, b_norm = params_ab[0], params_ab[1]
            recon_norm = A_norm * phi_norm + b_norm
            sse = np.sum((recon_norm - target_norm) ** 2)
            mse = sse / n
            if obj == "nll":
                # Gaussian NLL with sigma^2 = sse/n (constant dropped)
                nll = 0.5 * n * (np.log(mse + 1e-12) + 1.0)
                loss = nll
            else:
                loss = mse
            reg_term = 0.05 * (1.0 / (lxy + 0.05) + 1.0 / (lz + 0.05))
            return loss + reg_term

        initial_guesses = [
            [0.02, 0.02],
            [0.05, 0.05],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.1, 0.05],
            [0.05, 0.1],
            [0.15, 0.08],
            [0.08, 0.15],
        ]
        bounds = [(0.005, 1.0), (0.005, 1.0)]

        best_result = None
        best_mse = float("inf")
        for x0 in initial_guesses:
            try:
                result = minimize(
                    objective_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-8},
                )
                if result.success and result.fun < best_mse:
                    best_result = result
                    best_mse = result.fun
            except Exception:
                pass

        if best_result is None:
            # Simple grid-search fallback
            lxy_grid = np.array(
                [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5], dtype=float
            )
            lz_grid = np.array(
                [0.01, 0.02, 0.04, 0.06, 0.10, 0.16, 0.24, 0.35, 0.5], dtype=float
            )
            best = {"mse": float("inf")}
            for lxy in lxy_grid:
                for lz in lz_grid:
                    phi = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy, lz=lz)
                    n = phi.shape[0]
                    X = np.column_stack([phi, np.ones(n, dtype=float)])
                    try:
                        XtX = X.T @ X
                        XtY = X.T @ target
                        params_ = np.linalg.solve(XtX, XtY)
                    except np.linalg.LinAlgError:
                        params_ = np.linalg.lstsq(X, target, rcond=None)[0]
                    A_, b_ = float(params_[0]), float(params_[1])
                    recon_ = A_ * phi + b_
                    mse_ = float(np.mean((recon_ - target) ** 2))
                    sse_ = float(np.sum((recon_ - target) ** 2))
                    nll_ = 0.5 * n * (np.log(mse_ + 1e-12) + 1.0)
                    if (obj == "nll" and nll_ < best.get("nll", float("inf"))) or (
                        obj != "nll" and mse_ < best["mse"]
                    ):
                        best = {
                            "lxy": lxy,
                            "lz": lz,
                            "A": A_,
                            "b": b_,
                            "recon": recon_,
                            "mse": mse_,
                            "nll": nll_,
                            "sigma2": mse_,  # with normalized target, sigma2 ≈ mse
                            "optimization": None,
                            "hess_inv": None,
                            "param_std": None,
                        }
            return best

        # Extract optimal parameters
        lxy_opt, lz_opt = best_result.x
        phi_opt = self._sum_of_anisotropic_rbf(nominal_points, cause_points, lxy=lxy_opt, lz=lz_opt)
        n = phi_opt.shape[0]
        X = np.column_stack([phi_opt, np.ones(n, dtype=float)])
        try:
            XtX = X.T @ X
            XtY = X.T @ target
            params_ab = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            params_ab = np.linalg.lstsq(X, target, rcond=None)[0]
        A_opt, b_opt = float(params_ab[0]), float(params_ab[1])
        recon_opt = A_opt * phi_opt + b_opt
        mse_opt = float(np.mean((recon_opt - target) ** 2))
        rmse_opt = float(np.sqrt(mse_opt))
        mae_opt = float(np.mean(np.abs(recon_opt - target)))
        nll_opt = 0.5 * n * (np.log(mse_opt + 1e-12) + 1.0)  # Gaussian NLL with sigma2 = mse_opt
        sigma2_opt = mse_opt
        ss_res = np.sum((target - recon_opt) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2_score = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Optimization diagnostics (JSON‑friendly)
        opt = best_result
        opt_info = {
            "nit": int(getattr(opt, "nit", -1)),
            "nfev": int(getattr(opt, "nfev", -1)),
            "success": bool(getattr(opt, "success", False)),
            "message": str(getattr(opt, "message", "")),
        }

        # Approximate inverse Hessian (from L‑BFGS‑B) and parameter std devs
        hess_inv_mat = None
        param_std = None
        try:
            hinv = getattr(opt, "hess_inv", None)
            if hinv is not None:
                # For L‑BFGS‑B this is an LbfgsInvHessProduct; convert to dense
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
            "lxy": float(lxy_opt),
            "lz": float(lz_opt),
            "A": A_opt,
            "b": b_opt,
            "recon": recon_opt,
            "mse": mse_opt,
            "rmse": rmse_opt,
            "mae": mae_opt,
            "r2_score": r2_score,
            "nll": nll_opt,
            "sigma2": sigma2_opt,
            "optimization": opt_info,
            "hess_inv": (hess_inv_mat.tolist() if hess_inv_mat is not None else None),
            "param_std": param_std,
        }

    # ---------- Grid + prediction ----------

    def create_3d_prediction_grid(
        self,
        xyz: np.ndarray,
        cause_xyz: Optional[np.ndarray],
        resolution_xy: Optional[float] = None,
        resolution_z: Optional[float] = None,
    ):
        pad = float(self.cfg["pad_bounds"])
        res_xy = float(resolution_xy or self.cfg["resolution_xy"])
        res_z = float(resolution_z or self.cfg["resolution_z"])
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
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="xy")
        grid_points = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])
        return Xg, Yg, Zg, grid_points, xs, ys, zs

    def predict_direct_field_3d(
        self, fit_params: Dict[str, Any], grid_points: np.ndarray, cause_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if fit_params is None or "lxy" not in fit_params or fit_params["lxy"] is None:
            return np.zeros(grid_points.shape[0]), np.zeros(grid_points.shape[0])
        lxy = float(fit_params["lxy"])
        lz = float(fit_params["lz"])
        A = float(fit_params["A"])
        b = float(fit_params["b"])
        phi = self._sum_of_anisotropic_rbf(grid_points, cause_points, lxy=lxy, lz=lz)
        mean_pred = A * phi + b
        std_pred = np.full(grid_points.shape[0], 0.1 * np.std(mean_pred))
        return mean_pred, std_pred

    # ---------- Normalization + visualization ----------

    def _normalize_percentile(self, values: np.ndarray, lower_pct: float, upper_pct: float):
        lo = np.percentile(values, lower_pct)
        hi = np.percentile(values, upper_pct)
        if hi <= lo:
            hi = lo + 1e-9
        v = np.clip(values, lo, hi)
        v = (v - lo) / (hi - lo)
        return v, lo, hi

    def plot_3d_volume_with_cause_points(
        self,
        Xg,
        Yg,
        Zg,
        mean_field,
        xs,
        ys,
        zs,
        xyz,
        cause_points,
        use_isosurfaces: bool = True,
        max_cause_points: int = 5000,
    ):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        if use_isosurfaces and _HAS_SKIMAGE:
            self._plot_3d_isosurfaces(mean_field, xs, ys, zs, ax=ax, num_levels=5)
        else:
            self._plot_3d_pointcloud(mean_field.reshape(Xg.shape), Xg, Yg, Zg, ax=ax, max_points=self.cfg["max_points_pointcloud"])
        ax.plot3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="blue", linewidth=2.0, alpha=0.9, label="Actual trajectory")
        if cause_points is not None and cause_points.size > 0:
            pts = cause_points
            if pts.shape[0] > max_cause_points:
                idx = np.random.RandomState(42).choice(pts.shape[0], size=max_cause_points, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, c="red", alpha=0.6, edgecolors="none", label="Cause points")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Reconstructed 3D Field (Superposed) + Cause Points")
        ax.legend(loc="upper right")
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_zlim(zs.min(), zs.max())
        fig.tight_layout()
        return fig

    def _plot_3d_pointcloud(self, mean_field: np.ndarray, Xg: np.ndarray, Yg: np.ndarray, Zg: np.ndarray, ax, max_points: int = 50000):
        X = Xg.ravel()
        Y = Yg.ravel()
        Z = Zg.ravel()
        V = mean_field.ravel()
        N = X.shape[0]
        if N > max_points:
            idx = np.random.RandomState(42).choice(N, size=max_points, replace=False)
            X, Y, Z, V = X[idx], Y[idx], Z[idx], V[idx]
        Vn, _, _ = self._normalize_percentile(V, *self.cfg["percentile_range_3d"])
        ax.scatter(X, Y, Z, c=Vn, cmap="viridis", s=2, alpha=self.cfg["point_cloud_alpha"])

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
        colors = ["#2196F3", "#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]
        alphas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]
        for level, color, alpha in zip(levels, colors, alphas):
            try:
                verts, faces, normals, values = _sk_measure.marching_cubes(
                    volume=volume, level=level, spacing=(dz, dy, dx)
                )
                if len(verts) > 0:
                    verts_world = np.column_stack(
                        [verts[:, 2] + xmin, verts[:, 1] + ymin, verts[:, 0] + zmin]
                    )
                    ax.plot_trisurf(
                        verts_world[:, 0],
                        verts_world[:, 1],
                        faces,
                        verts_world[:, 2],
                        color=color,
                        lw=0.0,
                        edgecolor="none",
                        alpha=alpha,
                    )
            except Exception:
                pass

    # ---------- High-level pipeline ----------

    def fit_from_pointcloud_and_buffer(
        self,
        pointcloud_xyz: np.ndarray,
        buffer_dir: str,
        nominal_path: Optional[str] = None,
        nominal_xyz: Optional[np.ndarray] = None,
        clip_plane: str = "xy",
        objective: str = "mse",
    ) -> Dict[str, Any]:
        """End-to-end fitting using a provided cause pointcloud and buffer directory."""
        actual_xyz, cause, cause_xyz = self.load_buffer_xyz_drift(buffer_dir)
        if nominal_xyz is None and nominal_path:
            nominal_xyz = self.load_nominal_xyz(nominal_path)
        clipped_nominal = None
        if nominal_xyz is not None:
            clipped_nominal = self.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane=clip_plane)
            if clipped_nominal is None or len(clipped_nominal) == 0:
                clipped_nominal = nominal_xyz
        if clipped_nominal is not None:
            nominal_points_used, disturbance_magnitudes = self.compute_disturbance_at_nominal_points(
                clipped_nominal, actual_xyz, cause_xyz
            )
            if nominal_points_used.size == 0:
                drift_vectors, drift_magnitudes = self.compute_trajectory_drift_vectors(actual_xyz, clipped_nominal)
                if drift_vectors is None:
                    disturbance_magnitudes = np.zeros(len(clipped_nominal))
                    nominal_points_used = clipped_nominal
                else:
                    disturbance_magnitudes = drift_magnitudes
                    nominal_points_used = clipped_nominal
        else:
            disturbance_magnitudes = np.zeros(len(actual_xyz))
            nominal_points_used = actual_xyz
        fit = self.fit_direct_superposition_to_disturbances(
            nominal_points_used,
            disturbance_magnitudes,
            pointcloud_xyz if pointcloud_xyz is not None else np.empty((0, 3), dtype=float),
            objective=objective,
        )
        return {
            "fit": fit,
            "actual_xyz": actual_xyz,
            "nominal_used": nominal_points_used,
            "disturbances": disturbance_magnitudes,
            "cause": cause,
            "cause_xyz": cause_xyz,
        }

    def predict_grid_from_fit(
        self,
        actual_xyz: np.ndarray,
        cause_xyz: Optional[np.ndarray],
        fit: Dict[str, Any],
        pointcloud_xyz: np.ndarray,
        resolution_xy: Optional[float] = None,
        resolution_z: Optional[float] = None,
    ):
        Xg, Yg, Zg, grid_points, xs, ys, zs = self.create_3d_prediction_grid(
            actual_xyz, cause_xyz, resolution_xy, resolution_z
        )
        mean_pred, std_pred = self.predict_direct_field_3d(fit, grid_points, pointcloud_xyz)
        return Xg, Yg, Zg, grid_points, xs, ys, zs, mean_pred, std_pred

# Load PCD
def load_pcd(pcd_path):
    """Load points from PCD file"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        return np.asarray(pcd.points)
    except:
        # ASCII fallback
        with open(pcd_path, 'r') as f:
            lines = f.readlines()
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start = i + 1
                break
        if data_start:
            points = []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
            return np.array(points)
    return np.array([])


def gp_field(positions, cause_points, lxy, lz, A, b):
    """Compute GP field at positions
    
    Args:
        positions: (N, 3) query points
        cause_points: (M, 3) cause points
        lxy, lz: length scales
        A, b: amplitude and bias
    Returns:
        (N,) field values
    """
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    
    # Distance computation
    inv_lxy2 = 1.0 / (lxy**2 + 1e-12)
    inv_lz2 = 1.0 / (lz**2 + 1e-12)
    
    # Vectorized: (N, M, 3)
    diff = positions[:, np.newaxis, :] - cause_points[np.newaxis, :, :]
    d2 = (diff[:, :, 0]**2 + diff[:, :, 1]**2) * inv_lxy2 + diff[:, :, 2]**2 * inv_lz2
    
    phi = np.exp(-0.5 * d2).sum(axis=1)  # (N,)
    return A * phi + b


def compute_uncertainty_bayesian_linear(positions, cause_points, lxy, lz, sigma2_noise, nominal_points, disturbances):
    """Compute predictive uncertainty via Bayesian linear regression (A, b uncertainty)
    
    Mathematically: Var(y*) = sigma^2 * (1 + v^T * (X^T X)^-1 * v)
    where v = [phi(x), 1] is the feature vector at query point x.
    
    This captures uncertainty in A and b given fixed lxy, lz.
    
    Args:
        positions: (N, 3) query points
        cause_points: (M, 3) cause points
        lxy, lz: fixed length scales
        sigma2_noise: observation noise variance (MSE from fit)
        nominal_points: (K, 3) training points where disturbances were measured
        disturbances: (K,) observed disturbance magnitudes
    Returns:
        (N,) predictive standard deviation
    """
    # 1. Compute training feature matrix X
    phi_train = np.zeros(len(nominal_points))
    inv_lxy2 = 1.0 / (lxy**2 + 1e-12)
    inv_lz2 = 1.0 / (lz**2 + 1e-12)
    
    for i, nom_pt in enumerate(nominal_points):
        diff = nom_pt - cause_points
        d2 = (diff[:, 0]**2 + diff[:, 1]**2) * inv_lxy2 + diff[:, 2]**2 * inv_lz2
        phi_train[i] = np.exp(-0.5 * d2).sum()
    
    X_train = np.column_stack([phi_train, np.ones(len(phi_train))])  # (K, 2)
    
    # 2. Compute parameter covariance: Cov(A, b) = sigma^2 * (X^T X)^-1
    XtX = X_train.T @ X_train
    XtX[0, 0] += 1e-6  # Regularization for stability
    XtX[1, 1] += 1e-6
    
    try:
        XtX_inv = np.linalg.inv(XtX)
        Cov_params = sigma2_noise * XtX_inv  # (2, 2)
    except np.linalg.LinAlgError:
        # Fallback: just return noise level
        return np.full(len(positions), np.sqrt(sigma2_noise))
    
    # 3. Compute phi at query points
    phi_query = np.zeros(len(positions))
    for i, pos in enumerate(positions):
        diff = pos - cause_points
        d2 = (diff[:, 0]**2 + diff[:, 1]**2) * inv_lxy2 + diff[:, 2]**2 * inv_lz2
        phi_query[i] = np.exp(-0.5 * d2).sum()
    
    # 4. Epistemic variance: v^T * Cov * v where v = [phi, 1]
    epistemic_var = (Cov_params[0, 0] * phi_query**2 + 
                     2 * Cov_params[0, 1] * phi_query + 
                     Cov_params[1, 1])
    
    # 5. Total variance = epistemic + aleatoric
    total_variance = epistemic_var + sigma2_noise
    
    return np.sqrt(np.maximum(total_variance, 0.0))


def compute_uncertainty_bayesian(positions, cause_points, lxy, lz, A, b, sigma2_noise, 
                                  nominal_points, disturbances):
    """Compute Bayesian uncertainty in A, b parameters
    
    Returns:
        (N,) uncertainty (standard deviation)
    """
    return compute_uncertainty_bayesian_linear(positions, cause_points, lxy, lz, sigma2_noise,
                                               nominal_points, disturbances)


def visualize_uncertainty_2d_slice(cause_points, lxy, lz, A, b, sigma2_noise, 
                                    nominal_points, disturbances, z_slice=None):
    """2D slice showing mean risk, Bayesian uncertainty, and conservative risk"""
    
    # Determine z_slice (middle of cause points)
    if z_slice is None:
        z_slice = cause_points[:, 2].mean()
    
    # Create 2D grid at z=z_slice
    x_min, x_max = cause_points[:, 0].min() - 1, cause_points[:, 0].max() + 1
    y_min, y_max = cause_points[:, 1].min() - 1, cause_points[:, 1].max() + 1
    
    res = 0.05
    x_grid = np.arange(x_min, x_max, res)
    y_grid = np.arange(y_min, y_max, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    positions = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_slice)])
    
    # Compute mean field
    mean = gp_field(positions, cause_points, lxy, lz, A, b)
    
    # Compute Bayesian uncertainty
    uncertainty = compute_uncertainty_bayesian(positions, cause_points, lxy, lz, A, b, sigma2_noise,
                                               nominal_points, disturbances)
    
    # Reshape for plotting
    mean_2d = mean.reshape(X.shape)
    uncertainty_2d = uncertainty.reshape(X.shape)
    
    # Normalize
    mean_norm = (mean_2d - mean_2d.min()) / (mean_2d.max() - mean_2d.min() + 1e-9)
    unc_norm = (uncertainty_2d - uncertainty_2d.min()) / (uncertainty_2d.max() - uncertainty_2d.min() + 1e-9)
    
    # Conservative risk
    conservative = mean + 2 * uncertainty
    conservative_2d = conservative.reshape(X.shape)
    conservative_norm = (conservative_2d - conservative_2d.min()) / (conservative_2d.max() - conservative_2d.min() + 1e-9)
    
    # Plot (1 row x 3 cols)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Mean risk
    im1 = axes[0].contourf(X, Y, mean_norm, levels=30, cmap='YlOrRd')
    axes[0].scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='blue', alpha=0.5, label='Cause (PCD)')
    axes[0].set_title(f'Mean GP Field (Risk) | z={z_slice:.2f}m')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].axis('equal')
    plt.colorbar(im1, ax=axes[0], label='Risk', shrink=0.8)
    axes[0].legend(fontsize=8)
    
    # 2. Bayesian uncertainty (A, b)
    im2 = axes[1].contourf(X, Y, unc_norm, levels=30, cmap='plasma')
    axes[1].scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='cyan', alpha=0.5)
    axes[1].set_title('Bayesian Uncertainty (A, b)')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].axis('equal')
    plt.colorbar(im2, ax=axes[1], label='σ (Bayes)', shrink=0.8)
    
    # 3. Conservative risk (mean + 2σ)
    im3 = axes[2].contourf(X, Y, conservative_norm, levels=30, cmap='coolwarm')
    axes[2].scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='yellow', alpha=0.7)
    axes[2].set_title('Conservative Risk (mean + 2σ)')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].axis('equal')
    plt.colorbar(im3, ax=axes[2], label='Risk (95% CI)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('/tmp/gp_uncertainty_2d.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 2D uncertainty comparison saved to /tmp/gp_uncertainty_2d.png")
    plt.show()
    
    return mean, uncertainty, conservative


def visualize_uncertainty_3d(cause_points, lxy, lz, A, b, sigma2_noise,
                              nominal_points, disturbances, trajectory_points=None):
    """3D scatter showing Bayesian uncertainty regions"""
    
    # Create 3D grid around cause points
    pad = 0.5
    x_min, x_max = cause_points[:, 0].min() - pad, cause_points[:, 0].max() + pad
    y_min, y_max = cause_points[:, 1].min() - pad, cause_points[:, 1].max() + pad
    z_min, z_max = cause_points[:, 2].min() - pad, cause_points[:, 2].max() + pad
    
    res = 0.15  # Coarser for 3D
    x_grid = np.arange(x_min, x_max, res)
    y_grid = np.arange(y_min, y_max, res)
    z_grid = np.arange(z_min, z_max, res)
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)
    
    positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Compute Bayesian uncertainty
    uncertainty = compute_uncertainty_bayesian(positions, cause_points, lxy, lz, A, b, sigma2_noise,
                                               nominal_points, disturbances)
    
    # Filter: only show high-uncertainty regions (top 20%)
    threshold = np.percentile(uncertainty, 80)
    mask = uncertainty >= threshold
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot high-uncertainty points
    scatter = ax.scatter(
        positions[mask, 0], positions[mask, 1], positions[mask, 2],
        c=uncertainty[mask], cmap='plasma', s=30, alpha=0.6,
        label='High uncertainty (Bayesian)'
    )
    
    # Plot cause points
    ax.scatter(cause_points[:, 0], cause_points[:, 1], cause_points[:, 2],
               c='red', s=8, alpha=0.7, label='Cause points (PCD)')
    
    # Plot trajectory if provided
    if trajectory_points is not None:
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
                'b-', linewidth=2, alpha=0.8, label='Actual trajectory')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('High-Uncertainty Regions (Bayesian, Top 20%)')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Uncertainty (σ)')
    
    plt.tight_layout()
    plt.savefig('/tmp/gp_uncertainty_3d.png', dpi=150, bbox_inches='tight')
    print(f"✓ 3D uncertainty visualization saved to /tmp/gp_uncertainty_3d.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fit GP with NLL and visualize uncertainty")
    parser.add_argument("--buffer-dir", type=str, 
                        default="/home/navin/ros2_ws/src/buffers/run_20251221_144638_231_738a9b22/buffer1",
                        help="Buffer directory with poses.npy and points.pcd")
    parser.add_argument("--nominal-path", type=str,
                        default="/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json",
                        help="Path to nominal trajectory JSON")
    args = parser.parse_args()
    
    buffer_dir = Path(args.buffer_dir)
    pcd_path = buffer_dir / "points.pcd"
    poses_path = buffer_dir / "poses.npy"
    
    print("="*70)
    print("GP FITTING AND UNCERTAINTY VISUALIZATION (NLL)")
    print("="*70)
    print(f"\nBuffer directory: {buffer_dir}")
    print(f"PCD path:         {pcd_path}")
    print(f"Nominal path:     {args.nominal_path}")
    
    # Load cause points
    cause_points = load_pcd(str(pcd_path))
    if len(cause_points) == 0:
        print("ERROR: No points loaded from PCD!")
        return
    print(f"\n✓ Loaded {len(cause_points)} cause points from PCD")
    
    # Load actual trajectory
    if not poses_path.exists():
        print(f"ERROR: poses.npy not found at {poses_path}")
        return
    poses = np.load(poses_path)
    actual_xyz = poses[:, 1:4]
    print(f"✓ Loaded {len(actual_xyz)} actual trajectory points")
    
    # Load nominal trajectory
    nominal_points = None
    nominal_path = Path(args.nominal_path)
    if nominal_path.exists():
        helper = DisturbanceFieldHelper()
        nominal_xyz = helper.load_nominal_xyz(str(nominal_path))
        if nominal_xyz is not None:
            # Clip to actual segment
            nominal_points = helper.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane='xy')
            print(f"✓ Loaded and clipped {len(nominal_points)} nominal trajectory points")
        else:
            print("Warning: Failed to parse nominal trajectory")
    else:
        print(f"Warning: Nominal path not found: {nominal_path}")
    
    # Compute disturbances
    print(f"\n{'='*70}")
    print("COMPUTING DISTURBANCES")
    print("="*70)
    helper = DisturbanceFieldHelper()
    
    if nominal_points is not None and len(nominal_points) > 0:
        nominal_used, disturbances = helper.compute_disturbance_at_nominal_points(
            nominal_points, actual_xyz, None
        )
        if len(nominal_used) == 0:
            print("Warning: No nominal points near actual trajectory, using drift vectors")
            drift_vecs, drift_mags = helper.compute_trajectory_drift_vectors(actual_xyz, nominal_points)
            if drift_mags is not None:
                nominal_used = nominal_points
                disturbances = drift_mags
            else:
                print("ERROR: Could not compute disturbances")
                return
    else:
        print("ERROR: No nominal trajectory available")
        return
    
    print(f"✓ Computed disturbances at {len(nominal_used)} points")
    print(f"  Mean disturbance: {disturbances.mean():.4f} m")
    print(f"  Std disturbance:  {disturbances.std():.4f} m")
    print(f"  Max disturbance:  {disturbances.max():.4f} m")
    
    # Fit GP with NLL
    print(f"\n{'='*70}")
    print("FITTING GP WITH NLL OBJECTIVE")
    print("="*70)
    
    fit_result = helper.fit_direct_superposition_to_disturbances(
        nominal_used, disturbances, cause_points, objective="nll"
    )
    
    lxy = fit_result['lxy']
    lz = fit_result['lz']
    A = fit_result['A']
    b = fit_result['b']
    mse = fit_result['mse']
    r2 = fit_result['r2_score']
    nll = fit_result.get('nll', 0.0)
    sigma2 = fit_result.get('sigma2', mse)
    
    print(f"\n✓ NLL Fit Results:")
    print(f"  lxy:  {lxy:.6f} m")
    print(f"  lz:   {lz:.6f} m")
    print(f"  A:    {A:.6f}")
    print(f"  b:    {b:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f} m")
    print(f"  R²:   {r2:.6f}")
    print(f"  NLL:  {nll:.6f}")
    print(f"  σ²:   {sigma2:.6f}")
    
    # Optimization statistics
    opt_info = fit_result.get('optimization')
    if opt_info is not None:
        print(f"\n  Optimization Statistics:")
        print(f"    Iterations:   {opt_info.get('nit', 'N/A')}")
        print(f"    Func evals:   {opt_info.get('nfev', 'N/A')}")
        print(f"    Success:      {opt_info.get('success', 'N/A')}")
        print(f"    Message:      {opt_info.get('message', 'N/A')}")
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 2D slice (Bayesian uncertainty)
    print("\n[1/2] 2D slice with Bayesian uncertainty...")
    z_mean = cause_points[:, 2].mean()
    mean, uncertainty, conservative = visualize_uncertainty_2d_slice(
        cause_points, lxy, lz, A, b, sigma2, 
        nominal_used, disturbances, z_slice=z_mean
    )
    
    # 3D scatter
    print("\n[2/2] 3D high-uncertainty regions...")
    visualize_uncertainty_3d(cause_points, lxy, lz, A, b, sigma2,
                             nominal_used, disturbances, actual_xyz)
    
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print("="*70)
    print(f"""
🎯 PLOT 1: MEAN GP FIELD (Risk)
   - Shows expected disturbance at each location
   - High near cause points (< {lxy:.2f}m horiz, < {lz:.2f}m vert)
   - This is f(x) = A*Σ exp(-d²/2) + b

🧮 PLOT 2: BAYESIAN UNCERTAINTY (A, b)
   - Uncertainty from amplitude/bias parameter variance
   - Uses (X^T X)^-1 from Bayesian linear regression
   - σ = sqrt(σ²_noise * (1 + v^T (X^T X)^-1 v))
   - Captures: "How uncertain are A and b given lxy, lz?"
   - High where far from training data (low feature coverage)
   - Principled & computationally efficient ✅

⚠️  PLOT 3: CONSERVATIVE RISK
   - mean + 2σ
   - 95% confidence upper bound
   - **USE THIS FOR MPPI PLANNING** ⭐
   - Inflates cost in both risky AND uncertain regions

🧭 TWO TYPES OF UNCERTAINTY:
   1. Aleatoric (irreducible noise): σ² = {sigma2:.6f}
   2. Epistemic (parameter uncertainty): from A/b uncertainty (Bayesian LR)
   
   Total: sqrt(aleatoric + epistemic)

📊 FIT QUALITY:
   - R² = {r2:.3f} (variance explained)
   - RMSE = {np.sqrt(mse):.4f} m (prediction error)
   - NLL = {nll:.2f} (negative log-likelihood)
   - σ²_noise = {sigma2:.6f} (observation noise variance)

🚀 FOR MPPI PLANNING:
   Use conservative risk (Plot 3):
   - Exact Bayesian treatment of A, b uncertainty
   - Fast computation (no gradient required)
   - Captures dominant uncertainty source
   - Conservative: inflates risk far from training data
   - Epistemic uncertainty decreases as more data is collected
    """)


if __name__ == "__main__":
    main()

