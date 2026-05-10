#!/usr/bin/env python3
"""
Path Manager Module

Handles unified path planning interface supporting two modes:
External planner mode: Listen to external planner's global path topic

Provides a consistent interface for drift detection regardless of path source.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from threading import Lock


@dataclass
class DiscretizedPoint:
    """Discretized trajectory point"""
    position: np.ndarray  # 3D position [x, y, z]
    index: int
    distance_from_start: float


class TrajectoryDiscretizer:
    """Discretizes trajectory based on sampling length"""
    
    def __init__(self, sampling_length: float = 0.1):
        self.sampling_length = sampling_length
    
    def discretize_trajectory(self, points: List[dict]) -> List[DiscretizedPoint]:
        """Discretize trajectory from JSON points"""
        if not points:
            return []
        
        # Convert to numpy array with coordinate convention: forward->X, left->Y, up->Z
        positions = np.array([[p['position']['x'], p['position']['y'], p['position']['z']] 
                             for p in points])
        
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
    
    def discretize_path_message(self, path_msg) -> List[DiscretizedPoint]:
        """Discretize trajectory from ROS Path message"""
        if not path_msg or len(path_msg.poses) == 0:
            return []
        
        # Convert Path message to discretized points
        points = []
        for i, pose_stamped in enumerate(path_msg.poses):
            point = {
                'position': {
                    'x': float(pose_stamped.pose.position.x),
                    'y': float(pose_stamped.pose.position.y),
                    'z': float(pose_stamped.pose.position.z)
                }
            }
            points.append(point)
        
        return self.discretize_trajectory(points)


class PathManager:
    """Unified path manager for resilience system."""
    
    def __init__(self, node: Node, config: Dict[str, Any]):
        """
        Initialize path manager.
        
        Args:
            node: ROS2 node instance
            config: Path configuration dictionary
        """
        self.node = node
        self.config = config
        self.lock = Lock()
        
        # Discretization configuration
        discretization_config = config.get('discretization', {})
        self.sampling_distance = discretization_config.get('sampling_distance', 0.1)
        self.lookback_window_size = discretization_config.get('lookback_window_size', 20)
        self.default_soft_threshold = discretization_config.get('default_soft_threshold', 0.17)
        
        # Initialize discretizer
        self.discretizer = TrajectoryDiscretizer(self.sampling_distance)
        
        # Path state
        self.nominal_points = []
        self.discretized_nominal = []  # List[DiscretizedPoint]
        self.nominal_np = None  # Initialize as None
        self.soft_threshold = 0.17  # Use default from config
        self.hard_threshold = 0.41
        self.initial_pose = np.array([0.0, 0.0, 0.0])
        self.path_ready = False
        self.last_path_update = 0.0
        self.last_path_signature = None  # For detecting redundant path updates
        
        # Mode configuration
        self.mode = config.get('mode', 'json_file')
        self.global_path_topic = '/global_path'
        self.json_config = config.get('json_file', {})
        self.external_config = config.get('external_planner', {})
        
        # Dynamic path merging state (read from external_planner config)
        self.furthest_point_reached = -1  # Index of furthest point reached along path
        self.enable_dynamic_merging = self.external_config.get('enable_dynamic_merging', True)  # Enable dynamic path merging
        self.path_overlap_threshold = self.external_config.get('path_overlap_threshold', 0.3)  # meters - threshold for path overlap detection
        self.path_publisher = None
        self.path_subscriber = None
        self._init_external_mode()

    
    def _init_external_mode(self):
        """Initialize external planner mode - continuous listener for path updates."""
        try:
            # Use thresholds from external config, fallback to defaults
            thresholds_config = self.external_config.get('thresholds', {})
            self.soft_threshold = 0.17
            self.hard_threshold = 0.41
            
            # Track path update count
            self.path_update_count = 0
            
            # Create subscriber for external global path (continuous updates)
            self.path_subscriber = self.node.create_subscription(
                Path,
                self.global_path_topic,
                self._external_path_callback,
                10
            )
            
            self.node.get_logger().info(f"Listening for external path on topic: {self.global_path_topic} (continuous updates)")
            self.node.get_logger().info(f"Using discretization: {self.sampling_distance}m sampling, {self.lookback_window_size} point lookback")
        except Exception as e:
            self.node.get_logger().error(f"Failed to initialize external mode: {e}")
            raise
    
    def _find_current_position_in_new_path(self, new_path_points: List[dict], 
                                           current_position: np.ndarray) -> int:
        """
        Find where the current robot position is in the new path.
        Since dynamic planner publishes from start to T steps ahead, 
        the new path will always start from the beginning and overlap 100% with traversed portion.
        
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
        
        Since dynamic planner publishes from start to T steps ahead, the new path
        will always have 100% overlap with the traversed portion. We simply:
        1. Keep the traversed portion (up to furthest_point_reached)
        2. Find where we are in the new path
        3. Append the new path extension from that point forward
        
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
            self.node.get_logger().warn("No current position available for path merging, using new path")
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
    
    def _external_path_callback(self, path_msg: Path):
        """Handle external path message (continuous updates with dynamic merging)."""
        try:
            # Validate message contains poses
            if path_msg is None or len(path_msg.poses) == 0:
                self.node.get_logger().warn("Received empty path; ignoring...")
                return
            
            # Track if this is first update
            is_first_update = not self.path_ready

            # Build a simple geometric signature of the path (positions only)
            # This ignores header timestamps so that the same geometry is treated as identical
            coords = []
            for pose_stamped in path_msg.poses:
                p = pose_stamped.pose.position
                coords.append((float(p.x), float(p.y), float(p.z)))
            new_signature = hash(tuple(coords))

            # If path geometry hasn't changed, skip redundant processing
            if not is_first_update and self.last_path_signature == new_signature:
                # Throttle log to first few redundant updates only
                if getattr(self, "_redundant_log_count", 0) < 5:
                    self._redundant_log_count = getattr(self, "_redundant_log_count", 0) + 1
                    self.node.get_logger().info(
                        "Received external path identical to current one; skipping redundant update."
                    )
                return

            with self.lock:
                # Convert Path message to the internal simple format
                new_path_points = []
                for i, pose_stamped in enumerate(path_msg.poses):
                    point = {
                        'position': {
                            'x': float(pose_stamped.pose.position.x),
                            'y': float(pose_stamped.pose.position.y),
                            'z': float(pose_stamped.pose.position.z)
                        }
                    }
                    new_path_points.append(point)
                
                # Merge paths if dynamic merging is enabled and we have an existing path
                if self.enable_dynamic_merging and not is_first_update and len(self.nominal_points) > 0:
                    # Store current path length before merging
                    current_path_length = len(self.nominal_points)
                    
                    # Merge new path with current path
                    merged_points, merged_discretized, merged_np = self._merge_paths(
                        new_path_points, 
                        self.nominal_points,
                        self.discretized_nominal,
                        self.nominal_np
                    )
                    
                    self.nominal_points = merged_points
                    self.discretized_nominal = merged_discretized
                    self.nominal_np = merged_np
                    
                    # Update initial pose if needed
                    if len(self.nominal_np) > 0:
                        self.initial_pose = self.nominal_np[0]
                    else:
                        self.initial_pose = np.array([0.0, 0.0, 0.0])
                    
                    # Calculate how many points were kept vs appended
                    appended_count = len(merged_points) - current_path_length
                    kept_count = current_path_length if appended_count >= 0 else len(merged_points)
                    
                    # self.node.get_logger().info(
                    #     f"↻ Path merged: {len(self.nominal_points)} points total "
                    #     f"(kept {kept_count} traversed, appended {appended_count} new, "
                    #     f"furthest reached idx: {self.furthest_point_reached})"
                    # )
                else:
                    # First update or dynamic merging disabled - replace path
                    self.nominal_points = new_path_points
                    self.discretized_nominal = self.discretizer.discretize_path_message(path_msg)
                    
                    # Convert discretized points to numpy array for fast distance checks
                    if self.discretized_nominal:
                        self.nominal_np = np.array([point.position for point in self.discretized_nominal])
                        self.initial_pose = self.nominal_np[0]
                    else:
                        self.nominal_np = np.array([])
                        self.initial_pose = np.array([0.0, 0.0, 0.0])
                    
                    # Reset furthest point tracking on first update
                    if is_first_update:
                        self.furthest_point_reached = -1
                
                self.path_ready = True
                self.last_path_update = time.time()
                self.path_update_count += 1
                self.last_path_signature = new_signature
            
            # Log based on whether it's first update or subsequent
            if is_first_update:
                self.node.get_logger().info(f"✓ External path received: {len(self.nominal_points)} points. Path ready.")
                self.node.get_logger().info(f"Discretized to {len(self.discretized_nominal)} points with {self.sampling_distance}m sampling")
                if self.enable_dynamic_merging:
                    self.node.get_logger().info(f"Dynamic path merging: ENABLED (overlap threshold: {self.path_overlap_threshold:.3f}m)")
                
                # Print detailed discretization status on first update
                print("=" * 60)
                print("EXTERNAL PATH DISCRETIZATION STATUS")
                print("=" * 60)
                print(f"Original path points: {len(self.nominal_points)}")
                print(f"Discretized points: {len(self.discretized_nominal)}")
                print(f"Sampling distance: {self.sampling_distance:.3f}m")
                print(f"Lookback window: {self.lookback_window_size} points")
                print(f"Soft threshold: {self.soft_threshold:.3f}m")
                print(f"Hard threshold: {self.hard_threshold:.3f}m")
                if self.enable_dynamic_merging:
                    print(f"Dynamic merging: ENABLED (threshold: {self.path_overlap_threshold:.3f}m)")
                print("=" * 60)
            else:
                if self.enable_dynamic_merging:
                    self.node.get_logger().info(
                        f"↻ Path merged (#{self.path_update_count}): {len(self.nominal_points)} total points "
                        f"-> {len(self.discretized_nominal)} discretized "
                        f"(furthest reached: {self.furthest_point_reached})"
                    )
                else:
                    self.node.get_logger().info(
                        f"↻ Path updated (#{self.path_update_count}): {len(self.nominal_points)} points "
                        f"-> {len(self.discretized_nominal)} discretized"
                    )
            
            # Notify the main node that path has been updated and enable processing
            try:
                setattr(self.node, 'path_ready', True)
                setattr(self.node, 'disable_drift_detection', False)
                
                if is_first_update:
                    self.node.get_logger().info("Notified main node: path_ready=TRUE, drift detection ENABLED")
                
                # Always update narration manager with new path
                if hasattr(self.node, 'narration_manager'):
                    nominal_points = self.get_nominal_points_as_numpy()
                    if len(nominal_points) > 0:
                        self.node.narration_manager.update_intended_trajectory(nominal_points)
                        if is_first_update:
                            self.node.get_logger().info("Updated narration manager with external path")
                        else:
                            self.node.get_logger().info("Updated narration manager with new path")
            except Exception as e:
                self.node.get_logger().warn(f"Could not update narration manager: {e}")
            
        except Exception as e:
            self.node.get_logger().error(f"Error processing external path: {e}")
            import traceback
            traceback.print_exc()
    
    def is_ready(self) -> bool:
        """Check if path manager is ready."""
        return self.path_ready and self.nominal_np is not None and len(self.nominal_points) > 0
    
    def wait_for_path(self, timeout_seconds: float = 30.0) -> bool:
        """
        Wait for path to be ready.
        
        Args:
            timeout_seconds: Maximum time to wait for path
            
        Returns:
            True if path is ready within timeout, False otherwise
        """
        start_time = time.time()
        while not self.is_ready() and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)
        
        if self.is_ready():
            self.node.get_logger().info(f"Path ready after {time.time() - start_time:.1f}s")
            return True
        else:
            self.node.get_logger().error(f"Path not ready after {timeout_seconds}s timeout")
            return False
    
    def get_discretized_nominal_points(self) -> List[DiscretizedPoint]:
        """Get discretized nominal trajectory points."""
        with self.lock:
            return self.discretized_nominal.copy()
    
    def get_discretized_nominal_as_numpy(self) -> np.ndarray:
        """Get discretized nominal trajectory as numpy array for narration manager."""
        with self.lock:
            if self.discretized_nominal and len(self.discretized_nominal) > 0:
                return np.array([point.position for point in self.discretized_nominal])
            else:
                return np.array([])
    
    def get_lookback_window_size(self) -> int:
        """Get lookback window size."""
        return self.lookback_window_size
    
    def get_sampling_distance(self) -> float:
        """Get sampling distance."""
        return self.sampling_distance

    def get_nominal_points(self) -> List[Dict]:
        """Get nominal trajectory points."""
        with self.lock:
            return self.nominal_points.copy()
    
    def get_nominal_points_as_numpy(self) -> np.ndarray:
        """Get nominal trajectory as numpy array for narration manager."""
        with self.lock:
            if self.nominal_np is not None and len(self.nominal_np) > 0:
                return self.nominal_np.copy()
            else:
                return np.array([])
    
    def get_nominal_np(self) -> np.ndarray:
        """Get nominal trajectory as numpy array."""
        with self.lock:
            if self.nominal_np is not None and len(self.nominal_np) > 0:
                return self.nominal_np.copy()
            else:
                return np.array([])
    
    def get_initial_pose(self) -> np.ndarray:
        """Get initial pose."""
        with self.lock:
            return self.initial_pose.copy()
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get drift thresholds."""
        return self.soft_threshold, self.hard_threshold
    
    def compute_drift(self, pos: np.ndarray) -> Tuple[float, int]:
        """Compute drift between current position and nearest nominal point."""
        with self.lock:
            if self.nominal_np is None or len(self.nominal_np) == 0:
                return 0.0, 0
            
            dists = np.linalg.norm(self.nominal_np - pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            drift = dists[nearest_idx]
            
            # Update furthest point reached for dynamic path merging
            if self.enable_dynamic_merging and nearest_idx > self.furthest_point_reached:
                self.furthest_point_reached = nearest_idx
            
            return drift, nearest_idx
    
    def is_breach(self, drift: float) -> bool:
        """Check if drift exceeds soft threshold."""
        return drift > self.soft_threshold
    
    def is_hard_breach(self, drift: float) -> bool:
        """Check if drift exceeds hard threshold."""
        return drift > self.hard_threshold
    
    def get_mode(self) -> str:
        """Get current path mode."""
        return self.mode
    
    def get_path_topic(self) -> str:
        """Get global path topic name."""
        return self.global_path_topic
    
    def update_thresholds(self, soft_threshold: float, hard_threshold: float):
        """Update drift thresholds dynamically (useful for external planner mode)."""
        with self.lock:
            self.soft_threshold = soft_threshold
            self.hard_threshold = hard_threshold
            self.node.get_logger().info(f"Updated thresholds - soft: {soft_threshold}, hard: {hard_threshold}")
    
    def get_threshold_source(self) -> str:
        """Get the source of current thresholds."""
        if self.mode == 'json_file':
            return "JSON file calibration"
        else:
            return "External planner config"
    
    def reset_furthest_point(self):
        """Reset furthest point reached (useful for testing or path resets)."""
        with self.lock:
            self.furthest_point_reached = -1
    
    def get_furthest_point_reached(self) -> int:
        """Get the index of the furthest point reached along the path."""
        with self.lock:
            return self.furthest_point_reached 