#!/usr/bin/env python3
"""
Analyze 2D slice of GP and visualize Complete MPPI Cost Function

Shows:
1. Mean GP field (Eman GP - expected risk)
2. Epistemic uncertainty (parameter uncertainty)
3. Aleatoric uncertainty (observation noise)
4. Probabilistic Safety Barrier Cost
5. Combined Full MPPI Cost (all components together)
6. Cost breakdown statistics

This script implements and visualizes the complete cost function for MPPI:
    
    TOTAL COST = w_gp × mean_gp + w_unc × σ_total + w_safe × exp(-γ × (d_obs - β × σ_total))
    
    where:
    - mean_gp: GP predicted disturbance
    - σ_total: Total uncertainty (epistemic + aleatoric)
    - d_obs: Distance to nearest obstacle
    - β: Risk factor (safety margin in std devs)
    - γ: Barrier decay rate (wall steepness)

Based on Risk-Sensitive Control & Chance-Constrained Optimization.
Treats uncertainty as virtual reduction in free space rather than additive cost.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import sys
import os
import argparse
from pathlib import Path
import json

# Import from sample_gp.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.utils.sample_gp import (
    DisturbanceFieldHelper,
    load_pcd,
    gp_field,
    compute_uncertainty_bayesian_linear
)


def compute_total_variance(epistemic_var, aleatoric_var):
    """Compute total variance from epistemic and aleatoric components"""
    return epistemic_var + aleatoric_var


def compute_epistemic_variance(positions, cause_points, lxy, lz, sigma2_noise, 
                                nominal_points, disturbances):
    """Compute epistemic variance (from parameter uncertainty) only"""
    # Get uncertainty (std) and convert to variance
    uncertainty_std = compute_uncertainty_bayesian_linear(
        positions, cause_points, lxy, lz, sigma2_noise,
        nominal_points, disturbances
    )
    
    # Total variance = (uncertainty_std)^2
    total_var = uncertainty_std ** 2
    
    # Epistemic variance = total - aleatoric
    epistemic_var = total_var - sigma2_noise
    
    # Ensure non-negative
    epistemic_var = np.maximum(epistemic_var, 0.0)
    
    return epistemic_var


def point_in_obstacle(point, obstacles):
    """Check if a point is inside any obstacle"""
    x, y = point
    for obs_x, obs_y, obs_w, obs_h in obstacles:
        if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
            return True
    return False


def path_intersects_obstacle(path_points, obstacles, margin=0.1):
    """Check if path intersects any obstacle (with margin)"""
    for point in path_points:
        for obs_x, obs_y, obs_w, obs_h in obstacles:
            # Expand obstacle by margin
            if (obs_x - margin <= point[0] <= obs_x + obs_w + margin and 
                obs_y - margin <= point[1] <= obs_y + obs_h + margin):
                return True
    return False


def create_obstacles_and_path(cause_points, z_slice):
    """Create obstacles and a path in a 10x10 arena
    
    Returns:
        arena_bounds: (x_min, x_max, y_min, y_max) tuple
        obstacles: list of (x, y, width, height) tuples
        start_pos: (x, y) start position
        goal_pos: (x, y) goal position
        path_points: (N, 2) array of path points
    """
    # Fixed 10x10 arena centered at origin
    arena_size = 10.0
    x_min, x_max = -arena_size/2, arena_size/2
    y_min, y_max = -arena_size/2, arena_size/2
    arena_bounds = (x_min, x_max, y_min, y_max)
    
    # Start position: bottom-left area
    start_pos = np.array([x_min + 1.5, y_min + 1.5])
    
    # Goal position: top-right area
    goal_pos = np.array([x_max - 1.5, y_max - 1.5])
    
    # Create straight line path between start and goal
    num_path_points = 200
    t = np.linspace(0, 1, num_path_points)
    path_points = start_pos[None, :] + t[:, None] * (goal_pos - start_pos)[None, :]
    
    # Place obstacles that don't intersect the path
    # Position them near the path but with clearance
    obstacles = []
    
    # Get path center and direction
    path_center = (start_pos + goal_pos) / 2
    path_dir = goal_pos - start_pos
    path_dir_norm = path_dir / np.linalg.norm(path_dir)
    perp_dir = np.array([-path_dir_norm[1], path_dir_norm[0]])  # Perpendicular
    
    # Obstacle 1: Left side of path (above the path)
    obs1_offset = perp_dir * 1.5  # 1.5m offset perpendicular
    obs1_center = path_center + obs1_offset
    obs1_w, obs1_h = 1.2, 1.5
    obs1_x = obs1_center[0] - obs1_w / 2
    obs1_y = obs1_center[1] - obs1_h / 2
    obstacles.append((obs1_x, obs1_y, obs1_w, obs1_h))
    
    # Obstacle 2: Right side of path (below the path)
    obs2_offset = -perp_dir * 1.5  # Opposite side
    obs2_center = path_center + obs2_offset
    obs2_w, obs2_h = 1.2, 1.5
    obs2_x = obs2_center[0] - obs2_w / 2
    obs2_y = obs2_center[1] - obs2_h / 2
    obstacles.append((obs2_x, obs2_y, obs2_w, obs2_h))
    
    # Obstacle 3: Near start, offset from path
    obs3_offset = perp_dir * 1.8
    obs3_center = start_pos + (goal_pos - start_pos) * 0.3 + obs3_offset
    obs3_w, obs3_h = 1.0, 1.2
    obs3_x = obs3_center[0] - obs3_w / 2
    obs3_y = obs3_center[1] - obs3_h / 2
    obstacles.append((obs3_x, obs3_y, obs3_w, obs3_h))
    
    # Verify obstacles don't intersect path
    if path_intersects_obstacle(path_points, obstacles, margin=0.2):
        print("WARNING: Path may intersect obstacles, adjusting...")
        # If intersection detected, move obstacles further away
        for i in range(len(obstacles)):
            obs_x, obs_y, obs_w, obs_h = obstacles[i]
            # Move obstacle further from path center
            if i == 0:  # First obstacle
                obs_y += 0.5
            elif i == 1:  # Second obstacle
                obs_y -= 0.5
            else:  # Third obstacle
                obs_x += 0.5
            obstacles[i] = (obs_x, obs_y, obs_w, obs_h)
    
    return arena_bounds, obstacles, start_pos, goal_pos, path_points


def compute_distance_to_obstacles(positions, obstacles):
    """Compute distance from each position to nearest obstacle
    
    Args:
        positions: (N, 3) array of positions
        obstacles: list of (x, y, width, height) tuples
    
    Returns:
        d_obs: (N,) array of distances to nearest obstacle
    """
    N = len(positions)
    d_obs = np.full(N, np.inf)
    
    for obs_x, obs_y, obs_w, obs_h in obstacles:
        # Get obstacle corners
        obs_corners = np.array([
            [obs_x, obs_y],
            [obs_x + obs_w, obs_y],
            [obs_x + obs_w, obs_y + obs_h],
            [obs_x, obs_y + obs_h]
        ])
        
        # For each position, compute distance to obstacle
        for i, pos in enumerate(positions):
            px, py = pos[0], pos[1]
            
            # Check if inside obstacle
            if obs_x <= px <= obs_x + obs_w and obs_y <= py <= obs_y + obs_h:
                d_obs[i] = 0.0
                continue
            
            # Compute distance to nearest edge/corner
            # Clamp point to obstacle bounds to get nearest point on obstacle
            nearest_x = np.clip(px, obs_x, obs_x + obs_w)
            nearest_y = np.clip(py, obs_y, obs_y + obs_h)
            
            dist = np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
            d_obs[i] = min(d_obs[i], dist)
    
    return d_obs


def compute_probabilistic_safety_barrier_cost(positions, obstacles, sigma_total,
                                              w_safe=100.0, gamma=10.0, beta=2.0):
    """Compute the Probabilistic Safety Barrier cost
    
    This implements the theoretically sound cost function:
        cost_barrier = w_safe * exp(-gamma * (d_obs - beta * sigma_total))
    
    where:
        - d_obs: distance to nearest obstacle
        - sigma_total: total uncertainty (epistemic + aleatoric)
        - beta: safety factor (number of std devs for confidence)
        - gamma: barrier decay rate (steepness of the wall)
        - w_safe: safety weight
    
    Args:
        positions: (N, 3) array of positions
        obstacles: list of (x, y, width, height) tuples
        sigma_total: (N,) array of total uncertainty
        w_safe: safety weight (default: 100.0)
        gamma: barrier decay rate (default: 10.0)
        beta: risk factor (default: 2.0 for ~95% confidence)
    
    Returns:
        cost_barrier: (N,) array of barrier costs
        margin: (N,) array of effective safety margins
    """
    # Compute distance to nearest obstacle
    d_obs = compute_distance_to_obstacles(positions, obstacles)
    
    # Compute effective safety margin
    # This is the "physical distance" minus "uncertainty buffer"
    margin = d_obs - beta * sigma_total
    
    # Compute exponential barrier cost
    # Provides smooth gradients for MPPI while acting as a "soft wall"
    cost_barrier = w_safe * np.exp(-gamma * margin)
    
    # Numerical stability: clip massive costs
    cost_barrier = np.clip(cost_barrier, 0.0, 1e4)
    
    return cost_barrier, margin


def compute_combined_mppi_cost(positions, cause_points, lxy, lz, A, b, sigma2_noise,
                               nominal_points, disturbances, obstacles,
                               w_gp=1.0, w_uncertainty=2.0, w_safe=100.0, 
                               gamma=10.0, beta=2.0):
    """Compute the full MPPI cost combining GP effects and safety barrier
    
    Full Cost = w_gp * mean_gp + w_uncertainty * sigma_total + barrier_cost
    
    where:
        - mean_gp: Expected disturbance from GP
        - sigma_total: Total uncertainty (epistemic + aleatoric)
        - barrier_cost: Probabilistic safety barrier
    
    Args:
        positions: (N, 3) array of positions
        w_gp: weight for GP mean disturbance
        w_uncertainty: weight for uncertainty
        w_safe: weight for barrier cost
        gamma: barrier decay rate
        beta: risk factor for safety margin
    
    Returns:
        total_cost: (N,) array of combined costs
        components: dict with individual cost components
    """
    # 1. GP mean (expected disturbance)
    mean_gp = gp_field(positions, cause_points, lxy, lz, A, b)
    
    # 2. Total uncertainty
    sigma_total = compute_uncertainty_bayesian_linear(
        positions, cause_points, lxy, lz, sigma2_noise,
        nominal_points, disturbances
    )
    
    # 3. Barrier cost
    barrier_cost, safety_margin = compute_probabilistic_safety_barrier_cost(
        positions, obstacles, sigma_total, w_safe, gamma, beta
    )
    
    # 4. Combined cost
    total_cost = w_gp * mean_gp + w_uncertainty * sigma_total + barrier_cost
    
    return total_cost, {
        'mean_gp': mean_gp,
        'sigma_total': sigma_total,
        'barrier_cost': barrier_cost,
        'safety_margin': safety_margin,
        'gp_component': w_gp * mean_gp,
        'uncertainty_component': w_uncertainty * sigma_total,
    }


def compute_cost_components(positions, cause_points, lxy, lz, A, b, sigma2_noise,
                            nominal_points, disturbances, lambda_uncertainty=2.0,
                            obstacles=None, w_safe=100.0, gamma=10.0, beta=2.0):
    """Compute all cost components at given positions
    
    Mathematical relationship:
        total_variance = epistemic_variance + aleatoric_variance
        where:
            - epistemic_variance = σ²_noise * v^T * (X^T X)^-1 * v  (parameter uncertainty)
            - aleatoric_variance = σ²_noise (observation noise, constant)
        
        Therefore:
            total_variance = σ²_noise * (1 + v^T * (X^T X)^-1 * v)
    
    Returns:
        mean_risk: (N,) expected risk
        uncertainty_std: (N,) standard deviation (total)
        epistemic_var: (N,) epistemic variance
        aleatoric_var: (N,) aleatoric variance (constant = sigma2_noise)
        total_variance: (N,) total variance
        cost: (N,) cost = mean + lambda * uncertainty
        conservative_cost: (N,) mean + 2*uncertainty
    """
    # 1. Mean risk
    mean_risk = gp_field(positions, cause_points, lxy, lz, A, b)
    
    # 2. Total uncertainty (std) from Bayesian linear regression
    # Returns: sqrt(epistemic_var + aleatoric_var)
    uncertainty_std = compute_uncertainty_bayesian_linear(
        positions, cause_points, lxy, lz, sigma2_noise,
        nominal_points, disturbances
    )
    
    # 3. Variance components
    total_variance = uncertainty_std ** 2  # Squaring gives total variance
    aleatoric_var = np.full(len(positions), sigma2_noise)
    epistemic_var = np.maximum(total_variance - aleatoric_var, 0.0)
    
    # Verify: epistemic + aleatoric should equal total (within numerical precision)
    reconstructed_total = epistemic_var + aleatoric_var
    variance_error = np.abs(reconstructed_total - total_variance).max()
    if variance_error > 1e-8:
        print(f"WARNING: Variance decomposition error: {variance_error:.2e}")
        print(f"  Max epistemic: {epistemic_var.max():.6f}")
        print(f"  Aleatoric (constant): {sigma2_noise:.6f}")
        print(f"  Max total: {total_variance.max():.6f}")
        print(f"  Max reconstructed: {reconstructed_total.max():.6f}")
    
    # 4. Cost functions
    cost = mean_risk + lambda_uncertainty * uncertainty_std
    conservative_cost = mean_risk + 2.0 * uncertainty_std
    
    # 5. Probabilistic Safety Barrier Cost (if obstacles provided)
    if obstacles is not None:
        barrier_cost, safety_margin = compute_probabilistic_safety_barrier_cost(
            positions, obstacles, uncertainty_std, w_safe, gamma, beta
        )
    else:
        barrier_cost = np.zeros(len(positions))
        safety_margin = np.full(len(positions), np.inf)
    
    return {
        'mean_risk': mean_risk,
        'uncertainty_std': uncertainty_std,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_variance': total_variance,
        'cost': cost,
        'conservative_cost': conservative_cost,
        'barrier_cost': barrier_cost,
        'safety_margin': safety_margin,
    }


def visualize_cost_2d_slice(cause_points, lxy, lz, A, b, sigma2_noise,
                            nominal_points, disturbances, z_slice=None,
                            lambda_uncertainty=2.0, w_gp=1.0, w_safe=100.0, 
                            gamma=10.0, beta=2.0, save_path='/tmp/gp_cost_2d.png'):
    """Visualize cost function components on 2D slice"""
    
    # Determine z_slice (middle of cause points)
    if z_slice is None:
        z_slice = cause_points[:, 2].mean()
    
    # Create obstacles and path first to get arena bounds
    arena_bounds, obstacles, start_pos, goal_pos, path_points = create_obstacles_and_path(cause_points, z_slice)
    x_min, x_max, y_min, y_max = arena_bounds
    
    # Create 2D grid covering the arena
    res = 0.05
    x_grid = np.arange(x_min, x_max, res)
    y_grid = np.arange(y_min, y_max, res)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    positions = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_slice)])
    
    print(f"Computing cost components for {len(positions)} grid points...")
    print(f"Arena bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    # Compute traditional cost components
    components = compute_cost_components(
        positions, cause_points, lxy, lz, A, b, sigma2_noise,
        nominal_points, disturbances, lambda_uncertainty,
        obstacles=obstacles, w_safe=w_safe, gamma=gamma, beta=beta
    )
    
    # Compute combined MPPI cost
    combined_cost, combined_components = compute_combined_mppi_cost(
        positions, cause_points, lxy, lz, A, b, sigma2_noise,
        nominal_points, disturbances, obstacles,
        w_gp=w_gp, w_uncertainty=lambda_uncertainty, w_safe=w_safe,
        gamma=gamma, beta=beta
    )
    
    # Add combined cost to components
    components['combined_cost'] = combined_cost
    components['gp_component'] = combined_components['gp_component']
    components['uncertainty_component'] = combined_components['uncertainty_component']
    
    print(f"\n📦 Obstacle Configuration:")
    print(f"  Number of obstacles: {len(obstacles)}")
    for i, (obs_x, obs_y, obs_w, obs_h) in enumerate(obstacles, 1):
        print(f"  Obstacle {i}: center=({obs_x+obs_w/2:.2f}, {obs_y+obs_h/2:.2f}), "
              f"size=({obs_w:.2f} x {obs_h:.2f})")
    print(f"\n🎯 Path Configuration:")
    print(f"  Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    print(f"  Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    print(f"  Path length: {np.linalg.norm(goal_pos - start_pos):.2f} m")
    print(f"  Path points: {len(path_points)}")
    
    # Sample GP along the path
    path_positions_3d = np.column_stack([path_points, np.full(len(path_points), z_slice)])
    path_components = compute_cost_components(
        path_positions_3d, cause_points, lxy, lz, A, b, sigma2_noise,
        nominal_points, disturbances, lambda_uncertainty,
        obstacles=obstacles, w_safe=w_safe, gamma=gamma, beta=beta
    )
    path_mean = path_components['mean_risk']
    path_epistemic_std = np.sqrt(path_components['epistemic_var'])
    path_aleatoric_std = np.sqrt(path_components['aleatoric_var'])
    path_barrier_cost = path_components['barrier_cost']
    path_safety_margin = path_components['safety_margin']
    
    # Compute combined cost along path
    path_combined_cost, path_combined_components = compute_combined_mppi_cost(
        path_positions_3d, cause_points, lxy, lz, A, b, sigma2_noise,
        nominal_points, disturbances, obstacles,
        w_gp=w_gp, w_uncertainty=lambda_uncertainty, w_safe=w_safe,
        gamma=gamma, beta=beta
    )
    
    print(f"\n📊 GP Values Along Path:")
    print(f"  Mean GP: min={path_mean.min():.4f}, max={path_mean.max():.4f}, "
          f"mean={path_mean.mean():.4f}")
    print(f"  Epistemic σ: min={path_epistemic_std.min():.6f}, "
          f"max={path_epistemic_std.max():.6f}, mean={path_epistemic_std.mean():.6f}")
    print(f"  Aleatoric σ: constant={path_aleatoric_std[0]:.6f}")
    print(f"\n🛡️ Safety Barrier Cost Along Path:")
    print(f"  Barrier cost: min={path_barrier_cost.min():.4f}, max={path_barrier_cost.max():.4f}, "
          f"mean={path_barrier_cost.mean():.4f}")
    print(f"  Safety margin: min={path_safety_margin.min():.4f}, max={path_safety_margin.max():.4f}, "
          f"mean={path_safety_margin.mean():.4f}")
    print(f"  Parameters: w_safe={w_safe:.1f}, gamma={gamma:.1f}, beta={beta:.1f}")
    print(f"\n💰 COMBINED FULL COST Along Path:")
    print(f"  Total cost: min={path_combined_cost.min():.4f}, max={path_combined_cost.max():.4f}, "
          f"mean={path_combined_cost.mean():.4f}")
    print(f"  Components:")
    print(f"    - GP component (w_gp={w_gp:.1f}): "
          f"mean={path_combined_components['gp_component'].mean():.4f}")
    print(f"    - Uncertainty component (w_unc={lambda_uncertainty:.1f}): "
          f"mean={path_combined_components['uncertainty_component'].mean():.4f}")
    print(f"    - Barrier component: mean={path_combined_components['barrier_cost'].mean():.4f}")
    
    # Reshape for plotting
    mean_2d = components['mean_risk'].reshape(X.shape)
    epistemic_var_2d = components['epistemic_var'].reshape(X.shape)
    aleatoric_var_2d = components['aleatoric_var'].reshape(X.shape)
    barrier_cost_2d = components['barrier_cost'].reshape(X.shape)
    safety_margin_2d = components['safety_margin'].reshape(X.shape)
    combined_cost_2d = components['combined_cost'].reshape(X.shape)
    gp_component_2d = components['gp_component'].reshape(X.shape)
    uncertainty_component_2d = components['uncertainty_component'].reshape(X.shape)
    
    # Convert variance to standard deviation for epistemic uncertainty
    epistemic_std_2d = np.sqrt(epistemic_var_2d)
    aleatoric_std_2d = np.sqrt(aleatoric_var_2d)
    
    # Create figure with 6 subplots in 3x2 grid
    fig = plt.figure(figsize=(18, 24))
    
    # Helper function to draw obstacles and path
    def draw_obstacles_and_path(ax, path_values=None, cmap='coolwarm', label='GP Value'):
        # Draw arena boundary
        arena_rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor='gray', linewidth=2, 
                              linestyle='--', alpha=0.5, zorder=0, label='Arena (10x10m)')
        ax.add_patch(arena_rect)
        
        # Draw obstacles (add one to legend)
        obstacle_added_to_legend = False
        for i, (obs_x, obs_y, obs_w, obs_h) in enumerate(obstacles):
            rect = Rectangle((obs_x, obs_y), obs_w, obs_h, 
                           facecolor='darkred', edgecolor='black', 
                           linewidth=2, alpha=0.7, zorder=10,
                           label='Obstacle' if not obstacle_added_to_legend else '')
            ax.add_patch(rect)
            obstacle_added_to_legend = True
        
        # Draw start and goal
        ax.scatter(*start_pos, s=200, c='green', marker='o', 
                  edgecolors='black', linewidths=2, zorder=11, label='Start')
        ax.scatter(*goal_pos, s=200, c='red', marker='*', 
                  edgecolors='black', linewidths=2, zorder=11, label='Goal')
        
        # Draw path
        if path_values is not None:
            # Color path by GP values
            scatter = ax.scatter(path_points[:, 0], path_points[:, 1], 
                               c=path_values, s=20, cmap=cmap, 
                               edgecolors='white', linewidths=0.5, 
                               zorder=9, alpha=0.8)
            # Also draw a line connecting the points
            ax.plot(path_points[:, 0], path_points[:, 1], 'k-', 
                   linewidth=1, alpha=0.3, zorder=8)
        else:
            # Just draw a line
            ax.plot(path_points[:, 0], path_points[:, 1], 'cyan', 
                   linewidth=2, alpha=0.7, zorder=9, label='Path')
    
    # 1. Mean GP Field (Eman GP)
    ax1 = plt.subplot(3, 2, 1)
    im1 = ax1.contourf(X, Y, mean_2d, levels=30, cmap='YlOrRd')
    ax1.scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='blue', 
                alpha=0.5, label='Cause points', edgecolors='k', linewidths=0.3)
    draw_obstacles_and_path(ax1, path_mean, cmap='YlOrRd', label='GP Risk')
    ax1.set_title(f'Mean GP Field (Eman GP)\nz={z_slice:.2f}m', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.axis('equal')
    plt.colorbar(im1, ax=ax1, label='Risk', shrink=0.8)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Epistemic Uncertainty (std)
    ax2 = plt.subplot(3, 2, 2)
    im2 = ax2.contourf(X, Y, epistemic_std_2d, levels=30, cmap='viridis')
    ax2.scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='yellow', 
                alpha=0.5, label='Cause points', edgecolors='k', linewidths=0.3)
    draw_obstacles_and_path(ax2, path_epistemic_std, cmap='viridis', label='Epistemic σ')
    ax2.set_title('Epistemic Uncertainty (σ)\n(Parameter Uncertainty)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.axis('equal')
    plt.colorbar(im2, ax=ax2, label='σ (epistemic)', shrink=0.8)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Aleatoric Uncertainty (std)
    ax3 = plt.subplot(3, 2, 3)
    im3 = ax3.contourf(X, Y, aleatoric_std_2d, levels=30, cmap='cividis')
    ax3.scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='orange', 
                alpha=0.5, label='Cause points', edgecolors='k', linewidths=0.3)
    draw_obstacles_and_path(ax3, path_aleatoric_std, cmap='cividis', label='Aleatoric σ')
    ax3.set_title(f'Aleatoric Uncertainty (σ)\n(Observation Noise) σ = {np.sqrt(sigma2_noise):.6f}', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Y (m)', fontsize=12)
    ax3.axis('equal')
    plt.colorbar(im3, ax=ax3, label='σ (aleatoric)', shrink=0.8)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Probabilistic Safety Barrier Cost
    ax4 = plt.subplot(3, 2, 4)
    # Use log scale for better visualization since cost can vary widely
    barrier_cost_plot = np.log10(barrier_cost_2d + 1e-6)  # Add small epsilon to avoid log(0)
    im4 = ax4.contourf(X, Y, barrier_cost_plot, levels=30, cmap='RdPu')
    ax4.scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='cyan', 
                alpha=0.5, label='Cause points', edgecolors='k', linewidths=0.3)
    # Color path by barrier cost (log scale)
    path_barrier_log = np.log10(path_barrier_cost + 1e-6)
    draw_obstacles_and_path(ax4, path_barrier_log, cmap='RdPu', label='Barrier Cost')
    ax4.set_title(f'Probabilistic Safety Barrier Cost (log₁₀)\n' + 
                  f'w_safe={w_safe:.0f}, γ={gamma:.1f}, β={beta:.1f}', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('X (m)', fontsize=12)
    ax4.set_ylabel('Y (m)', fontsize=12)
    ax4.axis('equal')
    cbar4 = plt.colorbar(im4, ax=ax4, label='log₁₀(cost)', shrink=0.8)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Combined Full MPPI Cost
    ax5 = plt.subplot(3, 2, 5)
    # Use log scale for visualization
    combined_cost_plot = np.log10(combined_cost_2d + 1e-6)
    im5 = ax5.contourf(X, Y, combined_cost_plot, levels=30, cmap='plasma')
    ax5.scatter(cause_points[:, 0], cause_points[:, 1], s=8, c='white', 
                alpha=0.7, label='Cause points', edgecolors='k', linewidths=0.3)
    # Color path by combined cost (log scale)
    path_combined_log = np.log10(path_combined_cost + 1e-6)
    draw_obstacles_and_path(ax5, path_combined_log, cmap='plasma', label='Combined Cost')
    ax5.set_title(f'COMBINED FULL COST (log₁₀)\n' + 
                  f'w_gp×mean + w_unc×σ + barrier', 
                  fontsize=14, fontweight='bold')
    ax5.set_xlabel('X (m)', fontsize=12)
    ax5.set_ylabel('Y (m)', fontsize=12)
    ax5.axis('equal')
    cbar5 = plt.colorbar(im5, ax=ax5, label='log₁₀(total cost)', shrink=0.8)
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cost Components Breakdown
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Statistics text
    stats_text = f"""
    ╔═══════════════════════════════════════════════════╗
    ║     COMBINED COST BREAKDOWN (Arena Stats)         ║
    ╚═══════════════════════════════════════════════════╝
    
    📊 INDIVIDUAL COMPONENTS:
    ────────────────────────────────────────────────────
    1. GP Mean Disturbance:
       Weight (w_gp):     {w_gp:.2f}
       Raw range:         [{mean_2d.min():.4f}, {mean_2d.max():.4f}]
       Weighted range:    [{gp_component_2d.min():.4f}, {gp_component_2d.max():.4f}]
       Mean contribution: {gp_component_2d.mean():.4f}
    
    2. Total Uncertainty (σ):
       Weight (w_unc):    {lambda_uncertainty:.2f}
       Raw range:         [{components['uncertainty_std'].min():.6f}, {components['uncertainty_std'].max():.6f}]
       Weighted range:    [{uncertainty_component_2d.min():.4f}, {uncertainty_component_2d.max():.4f}]
       Mean contribution: {uncertainty_component_2d.mean():.4f}
    
    3. Safety Barrier:
       Weight (w_safe):   {w_safe:.0f}
       Decay (γ):         {gamma:.1f}
       Risk factor (β):   {beta:.1f}
       Raw range:         [{barrier_cost_2d.min():.4f}, {barrier_cost_2d.max():.4f}]
       Mean contribution: {barrier_cost_2d.mean():.4f}
    
    💰 COMBINED COST:
    ────────────────────────────────────────────────────
       Total range:       [{combined_cost_2d.min():.4f}, {combined_cost_2d.max():.4f}]
       Mean:              {combined_cost_2d.mean():.4f}
       Std dev:           {combined_cost_2d.std():.4f}
    
    🛤️  ALONG PATH:
    ────────────────────────────────────────────────────
       GP contribution:   {path_combined_components['gp_component'].mean():.4f}
       Unc contribution:  {path_combined_components['uncertainty_component'].mean():.4f}
       Barrier contrib:   {path_combined_components['barrier_cost'].mean():.4f}
       Total path cost:   {path_combined_cost.mean():.4f}
    
    📐 INTERPRETATION:
    ────────────────────────────────────────────────────
    The COMBINED COST includes ALL effects:
    • GP mean: Expected disturbance magnitude
    • Uncertainty: Lack of knowledge penalty
    • Barrier: Distance-based safety cost
    
    High cost regions indicate:
    ✗ Near obstacles (barrier dominates)
    ✗ High GP disturbance zones
    ✗ High uncertainty regions
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Cost visualization saved to {save_path}")
    plt.show()
    
    return components


def main():
    parser = argparse.ArgumentParser(description="Analyze 2D GP slice and visualize cost function")
    parser.add_argument("--buffer-dir", type=str, 
                        default="/home/navin/ros2_ws/src/buffers/run_20251221_144638_231_738a9b22/buffer1",
                        help="Buffer directory with poses.npy and points.pcd")
    parser.add_argument("--nominal-path", type=str,
                        default="/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json",
                        help="Path to nominal trajectory JSON")
    parser.add_argument("--z-slice", type=float, default=None,
                        help="Z coordinate for 2D slice (default: mean of cause points)")
    parser.add_argument("--w-gp", type=float, default=1.0,
                        help="Weight for GP mean disturbance (default: 1.0)")
    parser.add_argument("--lambda", type=float, default=2.0, dest='lambda_unc',
                        help="Uncertainty weight in cost function (default: 2.0)")
    parser.add_argument("--w-safe", type=float, default=100.0,
                        help="Safety weight for barrier cost (default: 100.0)")
    parser.add_argument("--gamma", type=float, default=10.0,
                        help="Barrier decay rate - steepness of safety wall (default: 10.0)")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="Risk factor - number of std devs for safety buffer (default: 2.0)")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Grid resolution in meters (default: 0.05)")
    parser.add_argument("--output", type=str, default="/tmp/gp_cost_2d.png",
                        help="Output path for visualization (default: /tmp/gp_cost_2d.png)")
    args = parser.parse_args()
    
    buffer_dir = Path(args.buffer_dir)
    pcd_path = buffer_dir / "points.pcd"
    poses_path = buffer_dir / "poses.npy"
    
    print("="*70)
    print("GP COST FUNCTION ANALYSIS (2D SLICE)")
    print("="*70)
    print(f"\nBuffer directory: {buffer_dir}")
    print(f"PCD path:         {pcd_path}")
    print(f"Nominal path:     {args.nominal_path}")
    print(f"Lambda (λ):       {args.lambda_unc}")
    print(f"Grid resolution:  {args.resolution} m")
    
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
    helper = DisturbanceFieldHelper()
    nominal_xyz = helper.load_nominal_xyz(args.nominal_path)
    if nominal_xyz is not None:
        nominal_points = helper.clip_nominal_to_actual_segment(nominal_xyz, actual_xyz, plane='xy')
        print(f"✓ Loaded and clipped {len(nominal_points)} nominal trajectory points")
    else:
        print("ERROR: Failed to load nominal trajectory")
        return
    
    # Compute disturbances
    print(f"\n{'='*70}")
    print("COMPUTING DISTURBANCES")
    print("="*70)
    
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
    
    print(f"✓ Computed disturbances at {len(nominal_used)} points")
    print(f"  Mean: {disturbances.mean():.4f} m, Std: {disturbances.std():.4f} m")
    
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
    sigma2 = fit_result.get('sigma2', mse)
    
    print(f"\n✓ Fit Results:")
    print(f"  lxy:  {lxy:.6f} m")
    print(f"  lz:   {lz:.6f} m")
    print(f"  A:    {A:.6f}")
    print(f"  b:    {b:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  σ²:   {sigma2:.6f}")
    
    # Visualize cost components
    print(f"\n{'='*70}")
    print("GENERATING COST VISUALIZATION")
    print("="*70)
    
    components = visualize_cost_2d_slice(
        cause_points, lxy, lz, A, b, sigma2,
        nominal_used, disturbances,
        z_slice=args.z_slice,
        lambda_uncertainty=args.lambda_unc,
        w_gp=args.w_gp,
        w_safe=args.w_safe,
        gamma=args.gamma,
        beta=args.beta,
        save_path=args.output
    )
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)
    print(f"""
📊 GP COMPONENTS & COST FUNCTION VISUALIZATION:

1. MEAN GP FIELD (Eman GP - Expected Risk)
   - GP prediction: A*Σ exp(-d²/2l²) + b
   - High near cause points
   - Low far from cause points
   - Represents expected disturbance magnitude

2. EPISTEMIC UNCERTAINTY (Parameter Uncertainty)
   - Uncertainty in A and b parameters
   - Comes from Bayesian linear regression
   - Depends on data density at query point
   - High where we have less training data
   - Decreases as more observations are collected
   - σ_epistemic = sqrt(σ²_epistemic)

3. ALEATORIC UNCERTAINTY (Observation Noise)
   - Constant: σ_aleatoric = {np.sqrt(sigma2):.6f}
   - Irreducible measurement/process noise
   - Does not decrease with more data
   - σ_aleatoric = sqrt(σ²_noise)

4. PROBABILISTIC SAFETY BARRIER COST
   ═══════════════════════════════════════════════════════════════
   🛡️ Theoretical Foundation:
   - Based on Risk-Sensitive Control & Chance-Constrained Optimization
   - Treats uncertainty as virtual reduction in free space
   - Subtracts uncertainty from safety margin (not adds to cost)
   
   📐 Formula:
   cost_barrier = w_safe × exp(-γ × (d_obs - β × σ_total))
   
   where:
   - d_obs: distance to nearest obstacle
   - σ_total: total uncertainty (epistemic + aleatoric)
   - β = {args.beta}: safety factor (# of std devs, ~95% confidence)
   - γ = {args.gamma}: barrier decay rate (steepness of wall)
   - w_safe = {args.w_safe}: safety weight
   
   🎯 Key Insight:
   Effective Safety Margin = d_obs - β × σ_total
   
   This is the lower bound of free space with ~95% confidence.
   
   ✓ Far from obstacles (d_obs large):
     → Margin is positive even with high σ
     → Cost ≈ 0 (Correct! No ghost obstacles)
   
   ✗ Near obstacles (d_obs small):
     → Any σ makes margin negative
     → Cost explodes (Correct! Safety critical)
   
   📊 Visualization (log scale):
   - Purple/magenta: High cost (dangerous regions)
   - Light/white: Low cost (safe regions)
   - Obstacles appear as "force fields" that grow with uncertainty
   
   🔧 Tuning Parameters:
   - Increase β → More conservative (larger safety buffer)
   - Increase γ → Steeper wall (sharper cost gradient)
   - Increase w_safe → Prioritize safety over tracking

5. COMBINED FULL COST (The Complete Picture)
   ═══════════════════════════════════════════════════════════════
   💰 Total Cost Formula:
   
   cost_total = w_gp × mean_gp + w_unc × σ_total + barrier_cost
   
   where:
   - w_gp = {args.w_gp}: weight for GP mean disturbance
   - w_unc = {args.lambda_unc}: weight for total uncertainty
   - barrier_cost: from safety barrier (already weighted by w_safe)
   
   🎯 This is the ACTUAL cost used for MPPI planning!
   
   Three simultaneous effects:
   1. GP Mean → Repels from high disturbance zones
   2. Uncertainty → Penalizes unknown/risky regions
   3. Barrier → Creates force fields around obstacles
   
   🌟 Key Benefits:
   ✓ No ghost obstacles (uncertainty alone doesn't block)
   ✓ Safety scales with uncertainty near obstacles
   ✓ Exploration-exploitation tradeoff via weights
   ✓ Theoretically grounded (CVaR, risk-sensitive control)
   
   📈 Tuning the Full Cost:
   - Increase w_gp → Avoid disturbances more aggressively
   - Increase w_unc → Be more cautious in uncertain regions
   - Increase w_safe → Prioritize collision avoidance
   - Balance all three for desired behavior
    """)


if __name__ == "__main__":
    main()

