import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import matplotlib.widgets as widgets
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TrajectoryPoint:
    """Simple trajectory point representation"""
    position: np.ndarray  # 2D position [x, y] for top-down view
    time: float

class SplineTrajectory:
    """2D spline trajectory with control points for top-down view"""
    
    def __init__(self, control_points: np.ndarray):
        self.control_points = np.array(control_points)  # Nx2 array
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

class XYSpatialDescriptor:
    """Generates spatial descriptions for XY plane only (top-down view)"""
    
    def __init__(self, soft_threshold=0.2, hard_threshold=0.5):
        # Magnitude descriptors
        self.magnitudes = {
            'tiny': 'just slightly',
            'small': 'a bit', 
            'medium': 'quite a bit',
            'large': 'significantly',
            'huge': 'way too much'
        }
        
        # Motion types for intended behavior (horizontal only)
        self.motion_types = {
            'straight': 'straight',
            'curving_left': 'curving left',
            'curving_right': 'curving right',
            'hovering': 'staying in place'
        }
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
    
    def get_magnitude_descriptor(self, distance: float) -> str:
        """Convert distance to linguistic magnitude"""
        if distance < self.soft_threshold:
            return self.magnitudes['tiny']
        elif distance < 4 * self.soft_threshold:
            return self.magnitudes['small']
        elif distance < 10 * self.soft_threshold:
            return self.magnitudes['medium']
        elif distance < 20 * self.soft_threshold:
            return self.magnitudes['large']
        else:
            return self.magnitudes['huge']
    
    def get_recent_motion_adverb(self, points: List[TrajectoryPoint], center_idx: int, window_size: int = 8) -> str:
        """Return an adverb describing the manner of recent horizontal motion"""
        if center_idx < 2:
            return ""
        start_idx = max(0, center_idx - window_size)
        segment = points[start_idx:center_idx+1]
        if len(segment) < 3:
            return ""
        positions = np.array([p.position for p in segment])
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        speed_var = np.var(speeds)
        directions = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8)
        direction_changes = np.linalg.norm(np.diff(directions, axis=0), axis=1)
        avg_dir_change = np.mean(direction_changes)
        # Heuristics for adverb
        if avg_speed < 0.25 * self.soft_threshold:
            return "slowly"
        elif speed_var < 0.01 * self.soft_threshold and avg_dir_change < 0.05 * self.soft_threshold:
            return "steadily"
        elif speed_var > 0.05 * self.soft_threshold and avg_dir_change > 0.1 * self.soft_threshold:
            return "kept drifting and"
        elif avg_dir_change > 0.15 * self.soft_threshold:
            return "suddenly"
        elif speed_var > 0.03 * self.soft_threshold:
            return "gradually"
        else:
            return ""

    def get_xy_direction_description(self, deviation_vector: np.ndarray) -> str:
        """Convert 2D deviation to natural lateral direction description"""
        x_dev, y_dev = deviation_vector[0], deviation_vector[1]
        abs_x, abs_y = abs(x_dev), abs(y_dev)
        
        # If we're here, soft threshold was already breached
        # Always describe left/right movement, use lateral magnitude for word choice
        if abs_y > 0.02:  # Any detectable lateral movement
            # FIXED: Correct direction logic
            # Y_dev > 0 means robot is to the right of intended path = "drifted right"
            # Y_dev < 0 means robot is to the left of intended path = "drifted left"
            primary_dir = "drifted right" if y_dev > 0 else "drifted left"
            magnitude_desc = self.get_magnitude_descriptor(abs_y)
            
            if magnitude_desc in ["just slightly", "a bit"]:
                return f"{primary_dir}"
            else:
                return f"{primary_dir} {magnitude_desc}"
        else:
            # Very minimal lateral movement, but still indicate direction
            if y_dev > 0:
                return "drifted slightly right"
            elif y_dev < 0:
                return "drifted slightly left"
            else:
                return "stayed on course laterally"
    
    def get_intended_motion_descriptor(self, intended_points: List[TrajectoryPoint], 
                                     center_idx: int, window_size: int = 10) -> str:
        """Describe the intended motion from trajectory segment (horizontal only)"""
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(intended_points), center_idx + window_size // 2 + 1)
        
        if end_idx <= start_idx + 1:
            return self.motion_types['hovering']
        
        segment = intended_points[start_idx:end_idx]
        start_pos = segment[0].position
        end_pos = segment[-1].position
        
        movement_vector = end_pos - start_pos
        
        # Check for lateral curvature in XY plane
        if len(segment) > 2:
            positions = np.array([p.position for p in segment])
            if len(positions) >= 3:
                v1 = positions[len(positions)//2] - positions[0]
                v2 = positions[-1] - positions[len(positions)//2]
                # Cross product in 2D (z-component)
                cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                
                if abs(cross_z) > 0.1:  # Threshold for detecting curves
                    if cross_z > 0:
                        return self.motion_types['curving_left']
                    else:
                        return self.motion_types['curving_right']
        
        return self.motion_types['straight']
    
    def generate_description(self, intended_points: List[TrajectoryPoint], 
                           actual_points: List[TrajectoryPoint], 
                           robot_idx: int) -> str:
        """Generate natural XY spatial description - NO THRESHOLD CHECK HERE"""
        
        if (robot_idx >= len(intended_points) or 
            robot_idx >= len(actual_points) or 
            robot_idx < 0):
            print(f"NARRATION DEBUG: Position out of range - robot_idx={robot_idx}, intended_len={len(intended_points)}, actual_len={len(actual_points)}")
            return "Position out of range"
        
        # Get current positions (XY only)
        intended_pos = intended_points[robot_idx].position
        actual_pos = actual_points[robot_idx].position
        
        # Calculate deviation in XY plane
        deviation_vector = actual_pos - intended_pos
        deviation_magnitude = np.linalg.norm(deviation_vector)
        
        print(f"NARRATION DEBUG: intended_pos={intended_pos}, actual_pos={actual_pos}")
        print(f"NARRATION DEBUG: deviation_vector={deviation_vector}, magnitude={deviation_magnitude:.3f}")
        print(f"NARRATION DEBUG: Generating narration for deviation magnitude {deviation_magnitude:.3f}")
        
        # NO THRESHOLD CHECK HERE - main node already confirmed we're in breach
        # Just generate narration based on the deviation data
        
        # Get descriptors
        direction_desc = self.get_xy_direction_description(deviation_vector)
        intended_motion_desc = self.get_intended_motion_descriptor(intended_points, robot_idx)
        adverb = self.get_recent_motion_adverb(actual_points, robot_idx)
        
        print(f"NARRATION DEBUG: direction_desc='{direction_desc}', intended_motion_desc='{intended_motion_desc}', adverb='{adverb}'")
        
        # Compose natural human-like sentence
        if adverb:
            phrase = f"{adverb} {direction_desc}"
        else:
            phrase = direction_desc
        
        result = f"I was trying to go {intended_motion_desc}, but I {phrase}."
        print(f"NARRATION DEBUG: Final narration: '{result}'")
        return result

class XYNarrationApp:
    """Interactive XY trajectory editor with spatial narration"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create layout - only XY view
        self.ax_xy = self.fig.add_subplot(211)      # Top view
        self.ax_description = self.fig.add_subplot(212)  # Description
        
        # Initialize with simple default trajectories (XY only)
        default_intended = np.array([
            [0, 0], [2, 0], [4, 0.5], [6, 0], [8, 0], [10, 0]
        ])
        default_actual = np.array([
            [0, 0], [2, 0.3], [4, 1.2], [6, 0.8], [8, 0.5], [10, 0.2]
        ])
        
        self.intended_trajectory = SplineTrajectory(default_intended)
        self.actual_trajectory = SplineTrajectory(default_actual)
        
        self.descriptor = XYSpatialDescriptor()
        
        # Robot state
        self.robot_parameter = 0.3  # Position along trajectory (0-1)
        
        # Interaction state
        self.selected_control_point = None
        self.selected_trajectory = None
        
        self.setup_controls()
        self.setup_interaction()
        self.update_display()
        
    def setup_controls(self):
        """Setup control widgets"""
        # Robot position slider
        ax_robot = plt.axes((0.1, 0.02, 0.6, 0.03))
        self.robot_slider = widgets.Slider(ax_robot, 'Robot Position', 0.0, 1.0, 
                                          valinit=self.robot_parameter, valfmt='%.2f')
        self.robot_slider.on_changed(self.update_robot_position)
        
        # Reset button
        ax_reset = plt.axes((0.75, 0.02, 0.15, 0.04))
        self.btn_reset = widgets.Button(ax_reset, 'Reset Trajectories')
        self.btn_reset.on_clicked(self.reset_trajectories)
        
    def setup_interaction(self):
        """Setup mouse interaction for control point editing"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
    def update_robot_position(self, val):
        """Update robot position and regenerate description"""
        self.robot_parameter = val
        self.update_display()
        
    def reset_trajectories(self, event):
        """Reset to default trajectories"""
        default_intended = np.array([
            [0, 0], [2, 0], [4, 0.5], [6, 0], [8, 0], [10, 0]
        ])
        default_actual = np.array([
            [0, 0], [2, 0.3], [4, 1.2], [6, 0.8], [8, 0.5], [10, 0.2]
        ])
        
        self.intended_trajectory = SplineTrajectory(default_intended)
        self.actual_trajectory = SplineTrajectory(default_actual)
        self.robot_parameter = 0.3
        self.robot_slider.reset()
        self.update_display()
        
    def update_display(self):
        """Update visual elements and description"""
        # Clear plots
        self.ax_xy.clear()
        self.ax_description.clear()
        
        # Sample trajectories
        intended_points = self.intended_trajectory.sample_trajectory(100)
        actual_points = self.actual_trajectory.sample_trajectory(100)
        
        if not intended_points or not actual_points:
            self.ax_description.text(0.5, 0.5, "Error: Unable to generate trajectory", 
                                   ha='center', va='center', fontsize=14)
            self.fig.canvas.draw()
            return
            
        # Get robot position
        robot_idx = int(self.robot_parameter * (len(actual_points) - 1))
        robot_pos = actual_points[robot_idx].position
        
        # Generate description
        description = self.descriptor.generate_description(intended_points, actual_points, robot_idx)
        
        # Plot trajectories
        intended_positions = np.array([p.position for p in intended_points])
        actual_positions = np.array([p.position for p in actual_points])
        
        # X-Y view (top view only)
        self.ax_xy.plot(intended_positions[:, 0], intended_positions[:, 1], 
                       'g--', linewidth=2, label='Intended Path', alpha=0.8)
        self.ax_xy.plot(actual_positions[:, 0], actual_positions[:, 1], 
                       'b-', linewidth=2, label='Actual Path')
        self.ax_xy.scatter([robot_pos[0]], [robot_pos[1]], 
                          c='red', s=120, label='Robot', zorder=10, edgecolor='darkred')
        
        # Plot control points
        self.ax_xy.scatter(self.intended_trajectory.control_points[:, 0],
                          self.intended_trajectory.control_points[:, 1],
                          c='green', s=100, alpha=0.9, marker='s', 
                          label='Intended Control Points', zorder=15, edgecolor='darkgreen')
        
        self.ax_xy.scatter(self.actual_trajectory.control_points[:, 0],
                          self.actual_trajectory.control_points[:, 1],
                          c='blue', s=100, alpha=0.9, marker='o', 
                          label='Actual Control Points', zorder=15, edgecolor='darkblue')
        
        # Configure axes
        self.ax_xy.set_xlabel('X Position (m)')
        self.ax_xy.set_ylabel('Y Position (m)')
        self.ax_xy.set_title('Top-Down View (X-Y) - Drag control points to modify trajectories')
        self.ax_xy.legend()
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.axis('equal')
        
        # Display description prominently
        self.ax_description.text(0.5, 0.7, "Robot XY Spatial Description:", 
                               ha='center', va='center', fontsize=16, weight='bold')
        
        if description:
            self.ax_description.text(0.5, 0.4, f'"{description}"', 
                                   ha='center', va='center', fontsize=18,
                                   bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.8))
        else:
            self.ax_description.text(0.5, 0.4, "Robot is on track - no deviation detected", 
                                   ha='center', va='center', fontsize=16, style='italic',
                                   bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgreen", alpha=0.8))
        
        self.ax_description.text(0.5, 0.1, 
                               "Narration only starts when robot drifts beyond the threshold\n"
                               "Move the slider or drag control points to see different descriptions", 
                               ha='center', va='center', fontsize=11, style='italic')
        
        self.ax_description.set_xlim(0, 1)
        self.ax_description.set_ylim(0, 1)
        self.ax_description.axis('off')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle control point selection"""
        if event.inaxes != self.ax_xy:
            return
            
        # Find nearest control point
        min_dist = float('inf')
        selected_point = None
        selected_traj = None
        
        # Check intended trajectory control points
        for i, point in enumerate(self.intended_trajectory.control_points):
            dist = np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2)
            if dist < min_dist and dist < 0.5:  # Selection threshold
                min_dist = dist
                selected_point = i
                selected_traj = 'intended'
        
        # Check actual trajectory control points
        for i, point in enumerate(self.actual_trajectory.control_points):
            dist = np.sqrt((event.xdata - point[0])**2 + (event.ydata - point[1])**2)
            if dist < min_dist and dist < 0.5:
                min_dist = dist
                selected_point = i
                selected_traj = 'actual'
        
        if selected_point is not None:
            self.selected_control_point = selected_point
            self.selected_trajectory = selected_traj
    
    def on_drag(self, event):
        """Handle control point dragging"""
        if (self.selected_control_point is not None and 
            self.selected_trajectory is not None and
            event.inaxes == self.ax_xy and
            event.xdata is not None and event.ydata is not None):
            
            # Update control point position (XY only)
            if self.selected_trajectory == 'intended':
                self.intended_trajectory.control_points[self.selected_control_point] = [
                    event.xdata, event.ydata
                ]
                self.intended_trajectory.update_spline()
            else:
                self.actual_trajectory.control_points[self.selected_control_point] = [
                    event.xdata, event.ydata
                ]
                self.actual_trajectory.update_spline()
            
            self.update_display()
    
    def on_release(self, event):
        """Handle mouse release"""
        self.selected_control_point = None
        self.selected_trajectory = None

def main():
    """Launch the XY spatial description generator"""
    print("🤖 XY Spatial Description Generator")
    print("=" * 50)
    print("Generates human-like spatial descriptions only when robot drifts")
    print()
    print("Features:")
    print("• Smart threshold-based narration (only when deviation matters)")
    print("• Natural human-like language describing drift") 
    print("• Focus on direction and nature of deviation")
    print("• Real-time trajectory editing")
    print()
    print("Controls:")
    print("• Drag green squares: modify intended trajectory")
    print("• Drag blue circles: modify actual trajectory") 
    print("• Slider: move robot position")
    print("• Watch description update when robot starts drifting")
    print("=" * 50)
    
    app = XYNarrationApp()
    plt.show()

if __name__ == "__main__":
    main() 