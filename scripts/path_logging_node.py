#!/usr/bin/env python3
"""
Path Tracking Metrics Logger

Standalone ROS 2 node that logs three key metrics from the time /global_path
becomes available until the node is killed. Uses topics from main_config.yaml.

Metrics tracked:
1. Arrival Time (T_arr): Time from mission start to goal arrival
2. Total Path Length (L_path): Cumulative Euclidean distance traveled
3. Normalized Cumulative Disturbance (D_cum): Cumulative deviation normalized by nominal path length
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from nav_msgs.msg import Path, Odometry
import time
import signal
import sys
import os
import yaml
from typing import Optional, Tuple, List, Dict
from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass
class DiscretizedPoint:
    """Discretized trajectory point"""
    position: np.ndarray  # 3D position [x, y, z]
    index: int
    distance_from_start: float


class TrajectoryDiscretizer:
    """Discretizes trajectory based on sampling length (same as path_manager)"""
    
    def __init__(self, sampling_length: float = 0.1):
        self.sampling_length = sampling_length
    
    def discretize_trajectory(self, points) -> list:
        """
        Discretize trajectory from points.
        Supports both numpy array and list of dicts (same as path_manager).
        """
        if len(points) == 0:
            return []
        
        # Convert to numpy array - handle both formats
        if isinstance(points, list) and len(points) > 0 and isinstance(points[0], dict):
            # Dict format (from path_manager)
            positions = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                                 for p in points])
        else:
            # Numpy array format (backward compatibility)
            positions = np.array(points) if not isinstance(points, np.ndarray) else points
        
        discretized = []
        current_distance = 0.0
        discretized.append(DiscretizedPoint(
            position=positions[0],
            index=0,
            distance_from_start=0.0
        ))
        
        # Walk along trajectory and sample at regular intervals
        for i in range(1, len(positions)):
            segment_length = np.linalg.norm(positions[i] - positions[i-1])
            current_distance += segment_length
            
            # Add points at sampling_length intervals
            while current_distance >= self.sampling_length:
                # Interpolate position along the segment
                alpha = self.sampling_length / segment_length
                interpolated_pos = positions[i-1] + alpha * (positions[i] - positions[i-1])
                
                discretized.append(DiscretizedPoint(
                    position=interpolated_pos,
                    index=len(discretized),
                    distance_from_start=len(discretized) * self.sampling_length
                ))
                
                current_distance -= self.sampling_length
                segment_length -= self.sampling_length
        
        return discretized


class PathLoggingNode(Node):
    """Logs path tracking metrics: T_arr, L_path, and D_cum."""
    
    def __init__(self):
        super().__init__('path_logging_node')
        
        # Load configuration from main_config.yaml
        self._load_config()
        
        # Discretization configuration (same as path_manager)
        discretization_config = self.path_config.get('discretization', {})
        self.sampling_distance = discretization_config.get('sampling_distance', 0.1)
        self.discretizer = TrajectoryDiscretizer(self.sampling_distance)
        
        # State
        self.lock = Lock()  # Thread safety for path updates
        self.nominal_path_points: List[Dict] = []  # Original path points in dict format (like path_manager)
        self.nominal_path_points_np: Optional[np.ndarray] = None  # Original path points as numpy (for backward compat)
        self.discretized_nominal: list = []  # List[DiscretizedPoint] - discretized path
        self.nominal_np: Optional[np.ndarray] = None  # Discretized path as numpy array (same as path_manager)
        self.path_received = False
        self.logging_active = False
        self.mission_start_time: Optional[float] = None
        self.arrival_time: Optional[float] = None
        self.goal_reached = False
        
        # Dynamic path merging state (same as path_manager)
        self.furthest_point_reached = -1  # Index of furthest point reached along discretized path
        self.enable_dynamic_merging = True  # Enable dynamic path merging (same as path_manager)
        self.last_path_signature = None  # For detecting redundant path updates
        self.path_update_count = 0
        
        # Robot pose tracking (observed trajectory τ_obs)
        self.observed_positions: deque = deque(maxlen=50000)  # Store all positions
        self.observed_timestamps: deque = deque(maxlen=50000)  # Store all timestamps (using message stamps)
        self.previous_pose: Optional[np.ndarray] = None
        self.previous_timestamp: Optional[float] = None  # Message stamp timestamp
        
        # Metrics
        self.total_path_length = 0.0  # L_path: cumulative distance
        self.cumulative_disturbance = 0.0  # Unnormalized sum for D_cum
        self.nominal_path_length: Optional[float] = None  # L_nom (calculated from discretized path)
        
        # Goal detection
        self.goal_position: Optional[np.ndarray] = None
        self.goal_threshold = 0.5  # meters - consider goal reached within this distance
        
        # Threshold for D_cum accumulation - FIXED at 0.1m to prevent noise accumulation
        # This is separate from soft_threshold used for breach detection
        self.d_cum_threshold = 0.1  # meters - only integrate when drift > 0.1m
        
        # Also store soft_threshold for reference (used for breach detection, not D_cum)
        self.soft_threshold = discretization_config.get('default_soft_threshold', 0.3)
        external_config = self.path_config.get('external_planner', {})
        thresholds_config = external_config.get('thresholds', {})
        if thresholds_config.get('soft_threshold'):
            self.soft_threshold = float(thresholds_config.get('soft_threshold'))
        
        # Maximum reasonable delta_t (seconds) - cap to prevent integration jumps
        # Typical odometry is 10-50Hz, so delta_t should be 0.02-0.1s
        # If delta_t > 1.0s, likely a timestamp jump - skip integration
        self.max_delta_t = 1.0  # seconds
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers (using topics from config)
        self.path_sub = self.create_subscription(
            Path, self.global_path_topic, self._path_callback, qos
        )
        self.pose_sub = self.create_subscription(
            Odometry, self.pose_topic, self._pose_callback, qos
        )
        
        self.get_logger().info("Path logging node initialized")
        self.get_logger().info(f"Subscribed to: {self.global_path_topic}, {self.pose_topic}")
        self.get_logger().info(f"Discretization: {self.sampling_distance:.3f}m sampling distance")
        self.get_logger().info(f"D_cum threshold: {self.d_cum_threshold:.3f}m (only integrate when drift > threshold)")
        self.get_logger().info(f"Max delta_t: {self.max_delta_t:.3f}s (skip integration if larger)")
        self.get_logger().info("Waiting for /global_path to become available...")
        
        # Periodic status update (every 10 seconds)
        self.create_timer(10.0, self._status_callback)
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self):
        """Load topic configuration from main_config.yaml."""
        # Initialize path_config to empty dict (will be set properly below)
        self.path_config = {}
        
        # Try to find main_config.yaml
        config_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'config', 'main_config.yaml'),
            os.path.expanduser('~/ros2_ws/src/resilience/config/main_config.yaml'),
            'config/main_config.yaml',
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            self.get_logger().warn("main_config.yaml not found, using defaults")
            self.global_path_topic = "/global_path"
            self.pose_topic = "/robot_1/odometry_conversion/odometry"
            # self.path_config already set to {} above
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract topics
            path_config = config.get('path_mode', {})
            self.global_path_topic = path_config.get('global_path_topic', '/global_path')
            self.path_config = path_config  # Store full path config for discretization params
            
            topics_config = config.get('topics', {})
            self.pose_topic = topics_config.get('pose_topic', '/robot_1/odometry_conversion/odometry')
            
            self.get_logger().info(f"Loaded config from: {config_path}")
        except Exception as e:
            self.get_logger().error(f"Error loading config: {e}, using defaults")
            self.global_path_topic = "/global_path"
            self.pose_topic = "/robot_1/odometry_conversion/odometry"
            self.path_config = {}  # Empty config, will use defaults
    
    def _find_current_position_in_new_path(self, new_path_points: List[dict], 
                                           current_position: np.ndarray) -> int:
        """
        Find where the current robot position is in the new path.
        Same as path_manager._find_current_position_in_new_path()
        
        Args:
            new_path_points: List of new path points (dict format)
            current_position: Current robot position (3D numpy array)
            
        Returns:
            Index in new path closest to current position
        """
        if len(new_path_points) == 0:
            return 0
        
        # Convert new path to numpy for comparison
        new_path_np = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                               for p in new_path_points])
        
        # Find closest point in new path to current position
        dists = np.linalg.norm(new_path_np - current_position, axis=1)
        closest_idx = int(np.argmin(dists))
        
        return closest_idx
    
    def _merge_paths(self, new_path_points: List[dict], current_path_points: List[dict], 
                    current_discretized: List[DiscretizedPoint], 
                    current_path_np: np.ndarray) -> Tuple[List[dict], List[DiscretizedPoint], np.ndarray]:
        """
        Merge new path with current path, preserving traversed portion.
        Same logic as path_manager._merge_paths()
        
        Args:
            new_path_points: New path points from planner
            current_path_points: Current path points
            current_discretized: Current discretized points
            current_path_np: Current path as numpy array
            
        Returns:
            Tuple (merged_points, merged_discretized, merged_np)
        """
        # Convert original path to numpy for mapping
        current_path_np_orig = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                                       for p in current_path_points]) if len(current_path_points) > 0 else np.array([])
        
        # Get current robot position from furthest point reached
        current_pos = None
        if self.enable_dynamic_merging and self.furthest_point_reached >= 0 and len(current_discretized) > 0:
            if self.furthest_point_reached < len(current_discretized):
                current_pos = current_discretized[self.furthest_point_reached].position
            else:
                # Fallback to last discretized point
                current_pos = current_discretized[-1].position
        elif len(current_path_np) > 0:
            # Fallback to last point in current path
            current_pos = current_path_np[-1]
        
        if current_pos is None:
            # No position available - use new path as-is
            self.get_logger().warn("No current position available for path merging, using new path")
            merged_points = new_path_points
            self.furthest_point_reached = -1
        else:
            # Map furthest_point_reached (discretized index) to original path index
            furthest_original_idx = -1
            if self.furthest_point_reached >= 0 and len(current_discretized) > 0:
                if self.furthest_point_reached < len(current_discretized):
                    furthest_pos = current_discretized[self.furthest_point_reached].position
                    
                    # Find closest original path point to this discretized position
                    if len(current_path_np_orig) > 0:
                        dists = np.linalg.norm(current_path_np_orig - furthest_pos, axis=1)
                        furthest_original_idx = int(np.argmin(dists))
            
            # Keep traversed portion (up to furthest point reached)
            keep_until_idx = furthest_original_idx if furthest_original_idx >= 0 else len(current_path_points) - 1
            keep_until_idx = min(keep_until_idx, len(current_path_points) - 1)
            keep_until_idx = max(0, keep_until_idx)
            
            # Find where current position is in the new path
            # Since new path is from start to T steps ahead, we should be near the start
            new_path_start_idx = self._find_current_position_in_new_path(new_path_points, current_pos)
            
            # Build merged path: keep traversed portion + new extension
            # The new path has 100% overlap with traversed portion, so we keep the traversed
            # portion and append everything from our current position forward in the new path
            merged_points = current_path_points[:keep_until_idx + 1]  # Keep traversed portion
            merged_points.extend(new_path_points[new_path_start_idx:])  # Append new extension
        
        # Discretize merged path
        merged_discretized = self.discretizer.discretize_trajectory(merged_points)
        
        # Convert to numpy
        if merged_discretized:
            merged_np = np.array([point.position for point in merged_discretized])
        else:
            merged_np = np.array([])
        
        return merged_points, merged_discretized, merged_np
    
    def _path_callback(self, msg: Path):
        """Process global path message (nominal/reference trajectory τ_ref) - handles continuous updates with merging."""
        try:
            # Validate message contains poses
            if msg is None or len(msg.poses) == 0:
                self.get_logger().warn("Received empty path; ignoring...")
                return
            
            # Track if this is first update
            is_first_update = not self.path_received
            
            # Build a simple geometric signature of the path (positions only)
            # This ignores header timestamps so that the same geometry is treated as identical
            coords = []
            for pose_stamped in msg.poses:
                p = pose_stamped.pose.position
                coords.append((float(p.x), float(p.y), float(p.z)))
            new_signature = hash(tuple(coords))
            
            # If path geometry hasn't changed, skip redundant processing
            if not is_first_update and self.last_path_signature == new_signature:
                return  # Skip redundant updates silently
            
            with self.lock:
                # Convert Path message to the internal dict format (same as path_manager)
                new_path_points = []
                for i, pose_stamped in enumerate(msg.poses):
                    point = {
                        'position': {
                            'x': float(pose_stamped.pose.position.x),
                            'y': float(pose_stamped.pose.position.y),
                            'z': float(pose_stamped.pose.position.z)
                        }
                    }
                    new_path_points.append(point)
                
                # Merge paths if dynamic merging is enabled and we have an existing path
                if self.enable_dynamic_merging and not is_first_update and len(self.nominal_path_points) > 0:
                    # Store current path length before merging
                    current_path_length = len(self.nominal_path_points)
                    
                    # Merge new path with current path
                    merged_points, merged_discretized, merged_np = self._merge_paths(
                        new_path_points, 
                        self.nominal_path_points,
                        self.discretized_nominal,
                        self.nominal_np
                    )
                    
                    self.nominal_path_points = merged_points
                    self.discretized_nominal = merged_discretized
                    self.nominal_np = merged_np
                    
                    # Also update numpy version for backward compatibility
                    self.nominal_path_points_np = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                                                           for p in self.nominal_path_points])
                else:
                    # First update or dynamic merging disabled - replace path
                    self.nominal_path_points = new_path_points
                    self.discretized_nominal = self.discretizer.discretize_trajectory(self.nominal_path_points)
                    
                    # Convert discretized points to numpy array (same as path_manager.nominal_np)
                    if self.discretized_nominal:
                        self.nominal_np = np.array([point.position for point in self.discretized_nominal])
                    else:
                        self.nominal_np = np.array([])
                    
                    # Also update numpy version for backward compatibility
                    self.nominal_path_points_np = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                                                           for p in self.nominal_path_points])
                    
                    # Reset furthest point tracking on first update
                    if is_first_update:
                        self.furthest_point_reached = -1
                
                # Calculate nominal path length L_nom from DISCRETIZED path (same as path_manager)
                self.nominal_path_length = 0.0
                if len(self.discretized_nominal) > 1:
                    for i in range(1, len(self.discretized_nominal)):
                        self.nominal_path_length += np.linalg.norm(
                            self.discretized_nominal[i].position - self.discretized_nominal[i-1].position
                        )
                
                # Goal is the last point of the discretized path
                if len(self.discretized_nominal) > 0:
                    self.goal_position = self.discretized_nominal[-1].position
                else:
                    self.goal_position = None
                
                self.path_received = True
                self.last_path_signature = new_signature
                self.path_update_count += 1
                
                # Use message timestamp for mission start (will be set from first pose message)
                if is_first_update:
                    self.mission_start_time = None  # Will be set from first pose message timestamp
                    self.logging_active = True
                    
                    self.get_logger().info(
                        f"✓ Path received: {len(self.nominal_path_points)} original points -> "
                        f"{len(self.discretized_nominal)} discretized points"
                    )
                    self.get_logger().info(
                        f"  L_nom (from discretized path) = {self.nominal_path_length:.3f}m, "
                        f"sampling = {self.sampling_distance:.3f}m"
                    )
                    if self.enable_dynamic_merging:
                        self.get_logger().info("Dynamic path merging: ENABLED (will merge future path updates)")
                    self.get_logger().info("Logging started. Press Ctrl+C to stop and view metrics.")
                else:
                    # Subsequent update - log merge info (throttled)
                    if self.path_update_count % 10 == 0:  # Log every 10th update
                        self.get_logger().info(
                            f"↻ Path merged (#{self.path_update_count}): {len(self.nominal_path_points)} total points "
                            f"-> {len(self.discretized_nominal)} discretized "
                            f"(furthest reached: {self.furthest_point_reached})"
                        )
        except Exception as e:
            self.get_logger().error(f"Error processing path: {e}")
            import traceback
            traceback.print_exc()
    
    def _pose_callback(self, msg: Odometry):
        """Process robot pose and update metrics."""
        if not self.logging_active or self.nominal_np is None or len(self.nominal_np) == 0:
            return
        
        # Extract position (observed trajectory point τ_obs(t_i)) - same format as main.py
        pos = msg.pose.pose.position
        current_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        
        # CRITICAL: Use message stamp for delta_t (not time.time())
        # This ensures integration matches physical simulation time
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Check if goal reached (for T_arr calculation)
        if not self.goal_reached and self.goal_position is not None:
            distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
            if distance_to_goal <= self.goal_threshold:
                self.goal_reached = True
                # Use message timestamp for arrival time
                if self.mission_start_time is not None:
                    self.arrival_time = current_timestamp - self.mission_start_time
                self.get_logger().info(
                    f"✓ Goal reached! T_arr = {self.arrival_time:.2f}s"
                )
        
        # Calculate time step Δt using message stamps
        delta_t = 0.0
        skip_integration = False
        if self.previous_timestamp is not None:
            delta_t = current_timestamp - self.previous_timestamp
            
            # Validate delta_t - skip integration if suspicious (timestamp jump or negative)
            if delta_t < 0:
                self.get_logger().warn(f"Negative delta_t detected: {delta_t:.6f}s, skipping integration")
                skip_integration = True
            elif delta_t > self.max_delta_t:
                self.get_logger().warn(
                    f"Large delta_t detected: {delta_t:.6f}s > {self.max_delta_t:.3f}s, "
                    f"likely timestamp jump - skipping integration"
                )
                skip_integration = True
        
        # Update metrics (will skip D_cum integration if skip_integration=True)
        self._update_metrics(current_pos, current_timestamp, delta_t, skip_integration=skip_integration)
        
        # Always update previous pose and timestamp (even if integration was skipped)
        # This prevents accumulation of errors from timestamp jumps
        self.previous_pose = current_pos
        self.previous_timestamp = current_timestamp
    
    def compute_drift(self, pos: np.ndarray) -> Tuple[float, int]:
        """
        Compute drift between current position and nearest discretized nominal point.
        EXACT SAME METHOD as path_manager.compute_drift() for consistency.
        Also updates furthest_point_reached for dynamic path merging.
        
        Returns:
            (drift, nearest_idx) where drift is the distance to nearest discretized point
        """
        with self.lock:
            if self.nominal_np is None or len(self.nominal_np) == 0:
                return 0.0, 0
            
            # EXACT SAME CALCULATION as path_manager.compute_drift()
            # Compute distances to all discretized points
            dists = np.linalg.norm(self.nominal_np - pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            drift = float(dists[nearest_idx])
            
            # Update furthest point reached for dynamic path merging (same as path_manager)
            if self.enable_dynamic_merging and nearest_idx > self.furthest_point_reached:
                self.furthest_point_reached = nearest_idx
            
            return drift, nearest_idx
    
    def _update_metrics(self, current_pos: np.ndarray, current_timestamp: float, delta_t: float, skip_integration: bool = False):
        """
        Update L_path and D_cum metrics.
        Uses EXACT SAME drift calculation as breach detection in main.py/path_manager.py
        
        Args:
            current_pos: Current robot position
            current_timestamp: Current message timestamp
            delta_t: Time step (seconds)
            skip_integration: If True, skip D_cum integration (e.g., due to timestamp issues)
        """
        # Set mission start time from first pose message timestamp
        if self.mission_start_time is None:
            self.mission_start_time = current_timestamp
        
        # 1. Total Path Length (L_path): L_path = sum ||x_{i+1} - x_i||_2
        if self.previous_pose is not None:
            segment_length = np.linalg.norm(current_pos - self.previous_pose)
            self.total_path_length += segment_length
        
        # Store observed position and timestamp (using message stamps)
        self.observed_positions.append(current_pos.copy())
        self.observed_timestamps.append(current_timestamp)
        
        # 2. Normalized Cumulative Disturbance (D_cum):
        # D_cum = (1/L_nom) * sum ||τ_obs(t_i) - τ_ref(t_i)||_2 * Δt
        # where ||τ_obs(t_i) - τ_ref(t_i)||_2 is the drift (same as breach detection)
        
        # Skip D_cum integration if requested (e.g., timestamp issues) or invalid conditions
        if skip_integration:
            # Still compute drift for logging, but don't integrate
            drift, nearest_idx = self.compute_drift(current_pos)
            d_cum_normalized = None
            if self.nominal_path_length is not None and self.nominal_path_length > 0:
                d_cum_normalized = self.cumulative_disturbance / self.nominal_path_length
            d_cum_str = f"{d_cum_normalized:.6f}" if d_cum_normalized is not None else "N/A"
            self.get_logger().info(
                f"[DRIFT] drift={drift:.4f}m | "
                f"delta_t={delta_t:.4f}s | "
                f"cumulative_disturbance={self.cumulative_disturbance:.6f} | "
                f"D_cum={d_cum_str} | "
                f"status=SKIPPED (timestamp issue)"
            )
            return
        
        if self.nominal_path_length is None or self.nominal_path_length == 0:
            # Still compute drift for logging
            drift, nearest_idx = self.compute_drift(current_pos)
            self.get_logger().info(
                f"[DRIFT] drift={drift:.4f}m | "
                f"delta_t={delta_t:.4f}s | "
                f"status=SKIPPED (L_nom not available)"
            )
            return
        
        # Skip if invalid time step
        if delta_t <= 0:
            # Still compute drift for logging
            drift, nearest_idx = self.compute_drift(current_pos)
            self.get_logger().info(
                f"[DRIFT] drift={drift:.4f}m | "
                f"delta_t={delta_t:.4f}s | "
                f"status=SKIPPED (invalid delta_t)"
            )
            return
        
        # CRITICAL: Use EXACT SAME drift calculation as path_manager.compute_drift()
        # This ensures D_cum matches the drift used for breach detection
        drift, nearest_idx = self.compute_drift(current_pos)
        
        # Calculate D_cum (normalized) for logging
        d_cum_normalized = None
        if self.nominal_path_length is not None and self.nominal_path_length > 0:
            d_cum_normalized = self.cumulative_disturbance / self.nominal_path_length
        
        # CRITICAL: Only accumulate D_cum if drift exceeds FIXED 0.1m threshold
        # This prevents "blow up" from sensor noise and jitter
        # Using fixed 0.1m threshold (not soft_threshold) to ensure consistent baseline
        integrated = False
        if drift > self.d_cum_threshold:
            # Add to cumulative disturbance: drift * Δt
            # This is the same drift value used for breach detection
            self.cumulative_disturbance += drift * delta_t
            integrated = True
            # Recalculate D_cum after integration
            if self.nominal_path_length is not None and self.nominal_path_length > 0:
                d_cum_normalized = self.cumulative_disturbance / self.nominal_path_length
        
        # Log drift and disturbance after every computation with detailed debug info
        status = "INTEGRATED" if integrated else "BELOW_THRESH"
        d_cum_str = f"{d_cum_normalized:.6f}" if d_cum_normalized is not None else "N/A"
        
        # Get nearest point for debugging
        nearest_point_str = "N/A"
        if self.nominal_np is not None and len(self.nominal_np) > 0 and nearest_idx < len(self.nominal_np):
            nearest_point = self.nominal_np[nearest_idx]
            nearest_point_str = f"[{nearest_point[0]:.3f}, {nearest_point[1]:.3f}, {nearest_point[2]:.3f}]"
        
        num_points = len(self.nominal_np) if self.nominal_np is not None else 0
        self.get_logger().info(
            f"[DRIFT] drift={drift:.4f}m | "
            f"nearest_idx={nearest_idx}/{num_points} | "
            f"pos=[{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] | "
            f"nearest={nearest_point_str} | "
            f"threshold={self.d_cum_threshold:.3f}m | "
            f"delta_t={delta_t:.4f}s | "
            f"cumulative_disturbance={self.cumulative_disturbance:.6f} | "
            f"D_cum={d_cum_str} | "
            f"status={status}"
        )
    
    def _calculate_metrics(self) -> dict:
        """Calculate final metrics: T_arr, L_path, D_cum."""
        metrics = {}
        
        # 1. Arrival Time (T_arr) - using message timestamps
        if self.goal_reached and self.arrival_time is not None:
            metrics['T_arr'] = self.arrival_time
            metrics['goal_reached'] = True
        else:
            # If not reached, use elapsed time from start to end (using message timestamps)
            if self.mission_start_time is not None and len(self.observed_timestamps) > 0:
                # Use last message timestamp, not system time
                metrics['T_arr'] = self.observed_timestamps[-1] - self.mission_start_time
            else:
                metrics['T_arr'] = None
            metrics['goal_reached'] = False
        
        # 2. Total Path Length (L_path)
        metrics['L_path'] = self.total_path_length
        
        # 3. Normalized Cumulative Disturbance (D_cum)
        if self.nominal_path_length is not None and self.nominal_path_length > 0:
            metrics['D_cum'] = self.cumulative_disturbance / self.nominal_path_length
        else:
            metrics['D_cum'] = None
        
        # Additional info
        metrics['L_nom'] = self.nominal_path_length
        metrics['num_samples'] = len(self.observed_positions)
        
        return metrics
    
    def _print_metrics(self):
        """Print all logged metrics."""
        metrics = self._calculate_metrics()
        
        if metrics['num_samples'] == 0:
            print("\n" + "="*70)
            print("PATH TRACKING METRICS")
            print("="*70)
            print("No data collected. Path may not have been received or robot did not move.")
            print("="*70 + "\n")
            return
        
        print("\n" + "="*70)
        print("PATH TRACKING METRICS")
        print("="*70)
        print()
        print("1. ARRIVAL TIME (T_arr):")
        if metrics['goal_reached']:
            print(f"   T_arr = {metrics['T_arr']:.2f} seconds")
            print(f"   Status: Goal reached successfully")
        else:
            print(f"   T_arr = {metrics['T_arr']:.2f} seconds (if available)")
            print(f"   Status: Goal not reached (within {self.goal_threshold}m threshold)")
        print()
        print("2. TOTAL PATH LENGTH (L_path):")
        print(f"   L_path = {metrics['L_path']:.3f} meters")
        print(f"   L_nom  = {metrics['L_nom']:.3f} meters (nominal path length)")
        if metrics['L_nom'] and metrics['L_nom'] > 0:
            efficiency = metrics['L_nom'] / metrics['L_path'] if metrics['L_path'] > 0 else 0.0
            print(f"   Path efficiency (L_nom/L_path) = {efficiency:.3f}")
        print()
        print("3. NORMALIZED CUMULATIVE DISTURBANCE (D_cum):")
        if metrics['D_cum'] is not None:
            print(f"   D_cum = {metrics['D_cum']:.6f}")
            print(f"   Formula: D_cum = (1/L_nom) * Σ drift(t_i) * Δt")
            print(f"   where drift(t_i) is computed using EXACT SAME method as breach detection")
            print(f"   (distance to nearest discretized point, sampling={self.sampling_distance:.3f}m)")
            print(f"   CRITICAL: Only drifts > {self.d_cum_threshold:.3f}m are integrated")
            print(f"   CRITICAL: Uses message stamps for Δt (not system time)")
            print(f"   CRITICAL: Skips integration if delta_t > {self.max_delta_t:.3f}s (timestamp jumps)")
        else:
            print("   D_cum = N/A (nominal path length not available)")
        print()
        print("Additional Info:")
        print(f"   Total samples: {metrics['num_samples']}")
        print(f"   D_cum threshold: {self.d_cum_threshold:.3f}m (fixed, prevents noise accumulation)")
        print(f"   Max delta_t: {self.max_delta_t:.3f}s (prevents integration jumps)")
        print(f"   Discretization: {self.sampling_distance:.3f}m sampling distance")
        print("="*70 + "\n")
    
    def _status_callback(self):
        """Periodic status update."""
        if not self.logging_active:
            if not self.path_received:
                self.get_logger().info("Still waiting for /global_path...")
            return
        
        metrics = self._calculate_metrics()
        goal_status = "✓" if metrics['goal_reached'] else "✗"
        d_cum_str = f"{metrics['D_cum']:.6f}" if metrics['D_cum'] else 'N/A'
        self.get_logger().info(
            f"Status: {metrics['num_samples']} samples | "
            f"L_path: {metrics['L_path']:.2f}m | "
            f"D_cum: {d_cum_str} | "
            f"Goal: {goal_status}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.get_logger().info("\nShutdown signal received. Calculating final metrics...")
        self.logging_active = False
        self._print_metrics()
        rclpy.shutdown()
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = PathLoggingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nKeyboard interrupt received.")
        node.logging_active = False
        node._print_metrics()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
