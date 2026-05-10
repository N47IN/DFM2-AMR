#!/usr/bin/env python3
"""
Offline Trajectory Debugger

Interactive debugging tool for trajectory narration analysis with:
1. JSON trajectory loading and discretization (0.3m sampling)
2. Spline-based modifiable actual trajectory with control points
3. 20-point lookback window analysis
4. Interactive GUI with robot position control
5. Real-time narration generation using XYSpatialDescriptor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy.interpolate import splprep, splev
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys

# Add the resilience package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from resilience.simple_descriptive_narration import XYSpatialDescriptor, TrajectoryPoint

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

class SplineTrajectory:
    """2D spline trajectory with control points for actual path modification"""
    
    def __init__(self, control_points: np.ndarray):
        self.control_points = np.array(control_points)  # Nx2 array (XY only)
        self.spline_params = None
        self.update_spline()
    
    def update_spline(self):
        """Recompute spline from current control points"""
        if len(self.control_points) < 3:
            return
        
        try:
            self.spline_params, _ = splprep([self.control_points[:, 0], 
                                            self.control_points[:, 1]], 
                                           s=0.1, k=min(3, len(self.control_points)-1))
        except:
            self.spline_params = None
    
    def sample_trajectory(self, num_points: int = 100) -> List[TrajectoryPoint]:
        """Sample trajectory points along the spline"""
        if self.spline_params is None:
            return []
        
        u_values = np.linspace(0, 1, num_points)
        try:
            positions = np.array(splev(u_values, self.spline_params)).T
        except:
            return []
        
        points = []
        for i, (u, pos) in enumerate(zip(u_values, positions)):
            points.append(TrajectoryPoint(
                position=pos,
                time=u * 10.0  # 10 second trajectory
            ))
        
        return points

class OfflineTrajectoryDebugger:
    """Interactive offline trajectory debugging tool"""
    
    def __init__(self, json_file_path: str, sampling_length: float = 0.1):
        self.sampling_length = sampling_length
        self.discretizer = TrajectoryDiscretizer(sampling_length)
        
        # Load and discretize nominal trajectory
        self.nominal_points = self.load_json_trajectory(json_file_path)
        self.discretized_nominal = self.discretizer.discretize_trajectory(self.nominal_points)
        
        print(f"Loaded {len(self.nominal_points)} nominal points")
        print(f"Discretized to {len(self.discretized_nominal)} points with {sampling_length}m sampling")
        print("Note: Discretized points used for lookback analysis, spline uses only 15 control points")
        
        # Initialize GUI
        self.fig = plt.figure(figsize=(14, 10))
        
        # Create layout
        self.ax_main = self.fig.add_subplot(221)      # Main trajectory view
        self.ax_lookback = self.fig.add_subplot(222) # Lookback window view
        self.ax_narration = self.fig.add_subplot(212) # Narration display
        
        # Initialize actual trajectory (spline-based, modifiable)
        self.setup_actual_trajectory()
        
        # Narration system
        self.descriptor = XYSpatialDescriptor(soft_threshold=0.1, hard_threshold=0.5)
        
        # Robot state
        self.robot_position = 0.0  # Position along trajectory (0-1)
        self.lookback_window_size = 20
        
        # Interaction state
        self.selected_control_point = None
        self.selected_trajectory = None
        
        self.setup_controls()
        self.setup_interaction()
        self.update_display()
        
    def load_json_trajectory(self, json_file_path: str) -> List[dict]:
        """Load trajectory from JSON file"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return data['points']
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    def setup_actual_trajectory(self):
        """Setup initial actual trajectory based on nominal path using only 15 control points"""
        if not self.discretized_nominal:
            # Fallback default trajectory with 15 control points
            default_points = np.array([
                [0, 0], [1, 0], [2, 0.1], [3, 0.2], [4, 0.3], 
                [5, 0.2], [6, 0.1], [7, 0], [8, -0.1], [9, -0.1],
                [10, 0], [11, 0.1], [12, 0.2], [13, 0.1], [14, 0]
            ])
        else:
            # Create 15 control points from discretized nominal trajectory
            # Using coordinate convention: forward->X, left->Y
            nominal_xy = np.array([[p.position[0], p.position[1]] for p in self.discretized_nominal])  # X=forward, Y=left
            
            # Sample 15 evenly spaced control points from the full trajectory
            num_control_points = 15
            if len(nominal_xy) >= num_control_points:
                # Use evenly spaced indices
                indices = np.linspace(0, len(nominal_xy) - 1, num_control_points, dtype=int)
                control_points = nominal_xy[indices]
            else:
                # If trajectory is shorter than 15 points, interpolate
                control_points = nominal_xy
            
            # Add some initial deviation to make it interesting
            actual_points = control_points.copy()
            # Add slight deviations to control points
            for i in range(len(actual_points)):
                if i > 0 and i < len(actual_points) - 1:
                    actual_points[i][1] += 0.1 * np.sin(i * 0.3)  # Slight sinusoidal deviation
        
        self.actual_trajectory = SplineTrajectory(actual_points)
    
    def setup_controls(self):
        """Setup control widgets"""
        # Robot position slider
        ax_robot = plt.axes((0.1, 0.02, 0.6, 0.03))
        self.robot_slider = widgets.Slider(ax_robot, 'Robot Position', 0.0, 1.0, 
                                          valinit=self.robot_position, valfmt='%.2f')
        self.robot_slider.on_changed(self.update_robot_position)
        
        # Lookback window size slider
        ax_lookback = plt.axes((0.1, 0.06, 0.3, 0.03))
        self.lookback_slider = widgets.Slider(ax_lookback, 'Lookback Window', 5, 50, 
                                             valinit=self.lookback_window_size, valfmt='%d')
        self.lookback_slider.on_changed(self.update_lookback_window)
        
        # Reset button
        ax_reset = plt.axes((0.75, 0.02, 0.15, 0.04))
        self.btn_reset = widgets.Button(ax_reset, 'Reset Actual Path')
        self.btn_reset.on_clicked(self.reset_actual_trajectory)
        
        # Threshold controls
        ax_soft = plt.axes((0.4, 0.06, 0.15, 0.03))
        self.soft_slider = widgets.Slider(ax_soft, 'Soft Threshold', 0.01, 1.0, 
                                        valinit=0.1, valfmt='%.2f')
        self.soft_slider.on_changed(self.update_thresholds)
        
        ax_hard = plt.axes((0.6, 0.06, 0.15, 0.03))
        self.hard_slider = widgets.Slider(ax_hard, 'Hard Threshold', 0.1, 2.0, 
                                        valinit=0.5, valfmt='%.2f')
        self.hard_slider.on_changed(self.update_thresholds)
        
        # Threshold band visibility controls
        ax_soft_band = plt.axes((0.4, 0.10, 0.15, 0.03))
        self.soft_band_slider = widgets.Slider(ax_soft_band, 'Soft Band Alpha', 0.0, 1.0, 
                                            valinit=0.3, valfmt='%.1f')
        self.soft_band_slider.on_changed(self.update_threshold_bands)
        
        ax_hard_band = plt.axes((0.6, 0.10, 0.15, 0.03))
        self.hard_band_slider = widgets.Slider(ax_hard_band, 'Hard Band Alpha', 0.0, 1.0, 
                                            valinit=0.2, valfmt='%.1f')
        self.hard_band_slider.on_changed(self.update_threshold_bands)
    
    def setup_interaction(self):
        """Setup mouse interaction for control point editing"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def update_robot_position(self, val):
        """Update robot position and regenerate analysis"""
        self.robot_position = val
        self.update_display()
    
    def update_lookback_window(self, val):
        """Update lookback window size"""
        self.lookback_window_size = int(val)
        self.update_display()
    
    def update_thresholds(self, val):
        """Update narration thresholds"""
        soft_thresh = self.soft_slider.val
        hard_thresh = self.hard_slider.val
        self.descriptor = XYSpatialDescriptor(soft_threshold=soft_thresh, hard_threshold=hard_thresh)
        self.update_display()
    
    def update_threshold_bands(self, val):
        """Update threshold band visibility"""
        self.update_display()
    
    def reset_actual_trajectory(self, event):
        """Reset actual trajectory to nominal using 15 control points"""
        if self.discretized_nominal:
            # Using coordinate convention: forward->X, left->Y
            nominal_xy = np.array([[p.position[0], p.position[1]] for p in self.discretized_nominal])  # X=forward, Y=left
            
            # Sample 15 evenly spaced control points from the full trajectory
            num_control_points = 15
            if len(nominal_xy) >= num_control_points:
                # Use evenly spaced indices
                indices = np.linspace(0, len(nominal_xy) - 1, num_control_points, dtype=int)
                control_points = nominal_xy[indices]
            else:
                # If trajectory is shorter than 15 points, use all points
                control_points = nominal_xy
            
            self.actual_trajectory = SplineTrajectory(control_points)
            self.update_display()
    
    def create_threshold_bands(self, trajectory_points: np.ndarray, soft_thresh: float, hard_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create threshold bands around trajectory points"""
        if len(trajectory_points) < 2:
            return np.array([]), np.array([])
        
        # Calculate perpendicular vectors for each point
        soft_band_points = []
        hard_band_points = []
        
        for i in range(len(trajectory_points)):
            # Get current point
            current_point = trajectory_points[i]
            
            # Calculate direction vector
            if i == 0:
                # First point: use direction to next point
                direction = trajectory_points[i + 1] - current_point
            elif i == len(trajectory_points) - 1:
                # Last point: use direction from previous point
                direction = current_point - trajectory_points[i - 1]
            else:
                # Middle point: use average direction
                direction = (trajectory_points[i + 1] - trajectory_points[i - 1]) / 2
            
            # Normalize direction
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            # Calculate perpendicular vector (rotate 90 degrees)
            perpendicular = np.array([-direction[1], direction[0]])
            
            # Create band points
            soft_band_points.extend([
                current_point + perpendicular * soft_thresh,
                current_point - perpendicular * soft_thresh
            ])
            
            hard_band_points.extend([
                current_point + perpendicular * hard_thresh,
                current_point - perpendicular * hard_thresh
            ])
        
        return np.array(soft_band_points), np.array(hard_band_points)

    def get_lookback_segments(self) -> Tuple[List[TrajectoryPoint], List[TrajectoryPoint]]:
        """Get intended and actual trajectory segments for lookback analysis"""
        # Sample trajectories
        intended_points = []
        actual_points = []
        
        # Convert discretized nominal to TrajectoryPoint format
        # Using coordinate convention: forward->X, left->Y (ignoring Z for 2D analysis)
        for i, point in enumerate(self.discretized_nominal):
            intended_points.append(TrajectoryPoint(
                position=np.array([point.position[0], point.position[1]]),  # X=forward, Y=left
                time=i * 0.1  # 0.1s intervals
            ))
        
        # Sample actual trajectory
        actual_points = self.actual_trajectory.sample_trajectory(len(intended_points))
        
        if not intended_points or not actual_points:
            return [], []
        
        # Calculate robot position index
        robot_idx = int(self.robot_position * (len(actual_points) - 1))
        
        # Get lookback window
        start_idx = max(0, robot_idx - self.lookback_window_size + 1)
        end_idx = min(len(intended_points), robot_idx + 1)
        
        intended_lookback = intended_points[start_idx:end_idx]
        actual_lookback = actual_points[start_idx:end_idx]
        
        return intended_lookback, actual_lookback
    
    def generate_narration(self) -> str:
        """Generate narration based on current robot position and lookback window"""
        intended_lookback, actual_lookback = self.get_lookback_segments()
        
        if not intended_lookback or not actual_lookback:
            return "No trajectory data available"
        
        # Calculate robot index within lookback window
        robot_idx = len(actual_lookback) - 1
        
        # Generate narration
        narration = self.descriptor.generate_description(
            intended_lookback, actual_lookback, robot_idx
        )
        
        return narration
    
    def update_display(self):
        """Update visual elements and analysis"""
        # Clear plots
        self.ax_main.clear()
        self.ax_lookback.clear()
        self.ax_narration.clear()
        
        # Get lookback segments
        intended_lookback, actual_lookback = self.get_lookback_segments()
        
        if not intended_lookback or not actual_lookback:
            self.ax_narration.text(0.5, 0.5, "Error: Unable to generate trajectory data", 
                                 ha='center', va='center', fontsize=14)
            self.fig.canvas.draw()
            return
        
        # Main trajectory view
        self.plot_main_trajectory()
        
        # Lookback window view
        self.plot_lookback_window(intended_lookback, actual_lookback)
        
        # Narration display
        self.plot_narration()
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def plot_main_trajectory(self):
        """Plot main trajectory view with threshold bands"""
        # Plot discretized nominal trajectory
        # Using coordinate convention: forward->X, left->Y
        if self.discretized_nominal:
            nominal_xy = np.array([[p.position[0], p.position[1]] for p in self.discretized_nominal])  # X=forward, Y=left
            
            # Create and plot threshold bands
            soft_thresh = self.soft_slider.val
            hard_thresh = self.hard_slider.val
            soft_band_points, hard_band_points = self.create_threshold_bands(nominal_xy, soft_thresh, hard_thresh)
            
            if len(soft_band_points) > 0 and len(hard_band_points) > 0:
                # Plot threshold bands
                soft_alpha = self.soft_band_slider.val
                hard_alpha = self.hard_band_slider.val
                
                # Plot hard threshold band (outer)
                if hard_alpha > 0:
                    self.ax_main.fill(hard_band_points[:, 0], hard_band_points[:, 1], 
                                    color='red', alpha=hard_alpha * 0.3, 
                                    label=f'Hard Threshold Band (±{hard_thresh:.2f}m)')
                
                # Plot soft threshold band (inner)
                if soft_alpha > 0:
                    self.ax_main.fill(soft_band_points[:, 0], soft_band_points[:, 1], 
                                    color='orange', alpha=soft_alpha * 0.4, 
                                    label=f'Soft Threshold Band (±{soft_thresh:.2f}m)')
            
            # Plot nominal trajectory
            self.ax_main.plot(nominal_xy[:, 0], nominal_xy[:, 1], 
                            'g--', linewidth=2, label='Nominal Path (Discretized)', alpha=0.8)
        
        # Plot actual trajectory
        actual_points = self.actual_trajectory.sample_trajectory(200)
        if actual_points:
            actual_positions = np.array([p.position for p in actual_points])
            self.ax_main.plot(actual_positions[:, 0], actual_positions[:, 1], 
                            'b-', linewidth=2, label='Actual Path')
        
        # Plot control points
        self.ax_main.scatter(self.actual_trajectory.control_points[:, 0],
                            self.actual_trajectory.control_points[:, 1],
                            c='blue', s=100, alpha=0.9, marker='o', 
                            label='Actual Control Points', zorder=15, edgecolor='darkblue')
        
        # Plot robot position
        robot_idx = int(self.robot_position * (len(actual_points) - 1))
        if actual_points and robot_idx < len(actual_points):
            robot_pos = actual_points[robot_idx].position
            self.ax_main.scatter([robot_pos[0]], [robot_pos[1]], 
                               c='red', s=120, label='Robot', zorder=10, edgecolor='darkred')
        
        self.ax_main.set_xlabel('X Position (m) - Forward')
        self.ax_main.set_ylabel('Y Position (m) - Left')
        self.ax_main.set_title('Main Trajectory View - Drag blue circles to modify actual path')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.axis('equal')
    
    def plot_lookback_window(self, intended_lookback, actual_lookback):
        """Plot lookback window analysis"""
        if not intended_lookback or not actual_lookback:
            return
        
        intended_positions = np.array([p.position for p in intended_lookback])
        actual_positions = np.array([p.position for p in actual_lookback])
        
        # Plot trajectories
        self.ax_lookback.plot(intended_positions[:, 0], intended_positions[:, 1], 
                            'g--', linewidth=2, label='Intended (Lookback)', alpha=0.8)
        self.ax_lookback.plot(actual_positions[:, 0], actual_positions[:, 1], 
                            'b-', linewidth=2, label='Actual (Lookback)')
        
        # Highlight robot position
        robot_idx = len(actual_lookback) - 1
        if robot_idx < len(actual_positions):
            robot_pos = actual_positions[robot_idx]
            self.ax_lookback.scatter([robot_pos[0]], [robot_pos[1]], 
                                   c='red', s=120, label='Robot', zorder=10, edgecolor='darkred')
        
        # Show deviation vectors
        for i in range(len(intended_positions)):
            if i < len(actual_positions):
                intended_pos = intended_positions[i]
                actual_pos = actual_positions[i]
                deviation = actual_pos - intended_pos
                deviation_magnitude = np.linalg.norm(deviation)
                
                if deviation_magnitude > self.descriptor.soft_threshold:
                    self.ax_lookback.arrow(intended_pos[0], intended_pos[1], 
                                         deviation[0], deviation[1],
                                         head_width=0.05, head_length=0.05, 
                                         fc='orange', ec='orange', alpha=0.7)
        
        self.ax_lookback.set_xlabel('X Position (m) - Forward')
        self.ax_lookback.set_ylabel('Y Position (m) - Left')
        self.ax_lookback.set_title(f'Lookback Window ({self.lookback_window_size} points)')
        self.ax_lookback.legend()
        self.ax_lookback.grid(True, alpha=0.3)
        self.ax_lookback.axis('equal')
    
    def plot_narration(self):
        """Plot narration display"""
        narration = self.generate_narration()
        
        self.ax_narration.text(0.5, 0.8, "Trajectory Narration Analysis:", 
                             ha='center', va='center', fontsize=16, weight='bold')
        
        if narration:
            self.ax_narration.text(0.5, 0.5, f'"{narration}"', 
                                 ha='center', va='center', fontsize=18,
                                 bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.8))
        else:
            self.ax_narration.text(0.5, 0.5, "Robot is on track - no deviation detected", 
                                 ha='center', va='center', fontsize=16, style='italic',
                                 bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.8))
        
        # Show analysis info
        intended_lookback, actual_lookback = self.get_lookback_segments()
        if intended_lookback and actual_lookback:
            robot_idx = len(actual_lookback) - 1
            if robot_idx < len(intended_lookback) and robot_idx < len(actual_lookback):
                intended_pos = intended_lookback[robot_idx].position
                actual_pos = actual_lookback[robot_idx].position
                deviation = actual_pos - intended_pos
                deviation_magnitude = np.linalg.norm(deviation)
                
                info_text = (f"Robot Position: {self.robot_position:.2f}\n"
                           f"Lookback Window: {self.lookback_window_size} points\n"
                           f"Current Deviation: {deviation_magnitude:.3f}m\n"
                           f"Soft Threshold: {self.descriptor.soft_threshold:.2f}m\n"
                           f"Hard Threshold: {self.descriptor.hard_threshold:.2f}m")
                
                self.ax_narration.text(0.5, 0.2, info_text, 
                                     ha='center', va='center', fontsize=12, 
                                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        self.ax_narration.set_xlim(0, 1)
        self.ax_narration.set_ylim(0, 1)
        self.ax_narration.axis('off')
    
    def on_click(self, event):
        """Handle control point selection"""
        if event.inaxes != self.ax_main:
            return
            
        # Find nearest control point
        min_dist = float('inf')
        selected_point = None
        
        for i, point in enumerate(self.actual_trajectory.control_points):
            dist = np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2)
            if dist < min_dist and dist < 0.5:  # Selection threshold
                min_dist = dist
                selected_point = i
        
        if selected_point is not None:
            self.selected_control_point = selected_point
    
    def on_drag(self, event):
        """Handle control point dragging"""
        if (self.selected_control_point is not None and
            event.inaxes == self.ax_main and
            event.xdata is not None and event.ydata is not None):
            
            # Update control point position
            self.actual_trajectory.control_points[self.selected_control_point] = [
                event.xdata, event.ydata
            ]
            self.actual_trajectory.update_spline()
            self.update_display()
    
    def on_release(self, event):
        """Handle mouse release"""
        self.selected_control_point = None

def main():
    """Launch the offline trajectory debugger"""
    print("🔧 Offline Trajectory Debugger")
    print("=" * 60)
    print("Interactive debugging tool for trajectory narration analysis")
    print()
    print("Features:")
    print("• JSON trajectory loading and 10cm discretization")
    print("• Spline-based modifiable actual trajectory (15 control points)")
    print("• Fine discretized points used for lookback analysis only")
    print("• Visual threshold bands around nominal trajectory")
    print("• 20-point lookback window analysis")
    print("• Real-time narration generation")
    print("• Interactive robot position control")
    print("• Coordinate convention: Forward->X, Left->Y, Up->Z")
    print()
    print("Controls:")
    print("• Drag blue circles: modify actual trajectory")
    print("• Robot Position slider: move robot along trajectory")
    print("• Lookback Window slider: adjust analysis window size")
    print("• Threshold sliders: adjust narration sensitivity")
    print("• Band Alpha sliders: control threshold band visibility")
    print("• Reset button: restore actual trajectory to nominal")
    print("=" * 60)
    
    # Find the JSON file
    json_file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'adjusted_nominal_spline.json')
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return
    
    try:
        debugger = OfflineTrajectoryDebugger(json_file_path, sampling_length=0.1)
        plt.show()
    except Exception as e:
        print(f"Error initializing debugger: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
