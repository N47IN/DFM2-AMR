#!/usr/bin/env python3
"""
Trajectory Comparison Tool

Interactive tool for comparing two trajectory JSONs and analyzing where narrations would be generated.
Shows narration generation points and their content with adjustable thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy.interpolate import splprep, splev
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import argparse

# Add the resilience package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import only the specific modules we need, avoiding YOLO dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("simple_descriptive_narration", 
                                            os.path.join(os.path.dirname(__file__), '..', 'resilience', 'simple_descriptive_narration.py'))
narration_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(narration_module)
XYSpatialDescriptor = narration_module.XYSpatialDescriptor
TrajectoryPoint = narration_module.TrajectoryPoint

@dataclass
class DiscretizedPoint:
    """Discretized trajectory point"""
    position: np.ndarray  # 3D position [x, y, z]
    index: int
    distance_from_start: float

@dataclass
class NarrationEvent:
    """Narration event at a specific point"""
    position: np.ndarray
    index: int
    narration: str
    deviation_magnitude: float
    soft_threshold: float
    hard_threshold: float

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

class TrajectoryComparisonTool:
    """Interactive trajectory comparison tool with narration analysis"""
    
    def __init__(self, trajectory1_path: str, trajectory2_path: str, sampling_length: float = 0.1):
        self.sampling_length = sampling_length
        self.discretizer = TrajectoryDiscretizer(sampling_length)
        
        # Load and discretize both trajectories
        self.trajectory1_points = self.load_json_trajectory(trajectory1_path)
        self.trajectory2_points = self.load_json_trajectory(trajectory2_path)
        
        self.discretized_traj1 = self.discretizer.discretize_trajectory(self.trajectory1_points)
        self.discretized_traj2 = self.discretizer.discretize_trajectory(self.trajectory2_points)
        
        print(f"Loaded trajectory 1: {len(self.trajectory1_points)} points -> {len(self.discretized_traj1)} discretized")
        print(f"Loaded trajectory 2: {len(self.trajectory2_points)} points -> {len(self.discretized_traj2)} discretized")
        
        # Narration system
        self.descriptor = XYSpatialDescriptor(soft_threshold=0.1, hard_threshold=0.5)
        self.lookback_window_size = 20
        
        # Analysis results
        self.narration_events = []
        
        # Initialize GUI
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create layout
        self.ax_main = self.fig.add_subplot(221)      # Main trajectory view
        self.ax_analysis = self.fig.add_subplot(222) # Analysis view
        self.ax_narrations = self.fig.add_subplot(212) # Narration display
        
        self.setup_controls()
        self.perform_analysis()
        self.update_display()
        
    def load_json_trajectory(self, json_file_path: str) -> List[dict]:
        """Load trajectory from JSON file"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return data['points']
        except Exception as e:
            print(f"Error loading JSON file {json_file_path}: {e}")
            return []
    
    def setup_controls(self):
        """Setup control widgets"""
        # Threshold controls
        ax_soft = plt.axes((0.1, 0.02, 0.15, 0.03))
        self.soft_slider = widgets.Slider(ax_soft, 'Soft Threshold', 0.01, 1.0, 
                                        valinit=0.1, valfmt='%.2f')
        self.soft_slider.on_changed(self.update_thresholds)
        
        ax_hard = plt.axes((0.3, 0.02, 0.15, 0.03))
        self.hard_slider = widgets.Slider(ax_hard, 'Hard Threshold', 0.1, 2.0, 
                                        valinit=0.5, valfmt='%.2f')
        self.hard_slider.on_changed(self.update_thresholds)
        
        # Lookback window size slider
        ax_lookback = plt.axes((0.5, 0.02, 0.15, 0.03))
        self.lookback_slider = widgets.Slider(ax_lookback, 'Lookback Window', 5, 50, 
                                             valinit=self.lookback_window_size, valfmt='%d')
        self.lookback_slider.on_changed(self.update_lookback_window)
        
        # Analysis button
        ax_analyze = plt.axes((0.75, 0.02, 0.15, 0.04))
        self.btn_analyze = widgets.Button(ax_analyze, 'Re-analyze')
        self.btn_analyze.on_clicked(self.perform_analysis)
        
        # Sampling distance slider
        ax_sampling = plt.axes((0.1, 0.06, 0.15, 0.03))
        self.sampling_slider = widgets.Slider(ax_sampling, 'Sampling Distance', 0.01, 0.5, 
                                            valinit=self.sampling_length, valfmt='%.2f')
        self.sampling_slider.on_changed(self.update_sampling_distance)
        
        # Trajectory selection
        ax_traj1 = plt.axes((0.3, 0.06, 0.15, 0.03))
        self.traj1_checkbox = widgets.CheckButtons(ax_traj1, ['Traj 1'], [True])
        self.traj1_checkbox.on_clicked(self.update_display)
        
        ax_traj2 = plt.axes((0.5, 0.06, 0.15, 0.03))
        self.traj2_checkbox = widgets.CheckButtons(ax_traj2, ['Traj 2'], [True])
        self.traj2_checkbox.on_clicked(self.update_display)
    
    def update_thresholds(self, val):
        """Update narration thresholds"""
        soft_thresh = self.soft_slider.val
        hard_thresh = self.hard_slider.val
        self.descriptor = XYSpatialDescriptor(soft_threshold=soft_thresh, hard_threshold=hard_thresh)
        self.perform_analysis()
        self.update_display()
    
    def update_lookback_window(self, val):
        """Update lookback window size"""
        self.lookback_window_size = int(val)
        self.perform_analysis()
        self.update_display()
    
    def update_sampling_distance(self, val):
        """Update sampling distance and re-discretize"""
        self.sampling_length = val
        self.discretizer = TrajectoryDiscretizer(self.sampling_length)
        self.discretized_traj1 = self.discretizer.discretize_trajectory(self.trajectory1_points)
        self.discretized_traj2 = self.discretizer.discretize_trajectory(self.trajectory2_points)
        self.perform_analysis()
        self.update_display()
    
    def perform_analysis(self):
        """Perform narration analysis on both trajectories"""
        self.narration_events = []
        
        if not self.discretized_traj1 or not self.discretized_traj2:
            return
        
        # Convert to TrajectoryPoint format for narration
        traj1_points = []
        traj2_points = []
        
        for i, point in enumerate(self.discretized_traj1):
            traj1_points.append(TrajectoryPoint(
                position=np.array([point.position[0], point.position[1]]),  # X=forward, Y=left
                time=i * 0.1  # 0.1s intervals
            ))
        
        for i, point in enumerate(self.discretized_traj2):
            traj2_points.append(TrajectoryPoint(
                position=np.array([point.position[0], point.position[1]]),  # X=forward, Y=left
                time=i * 0.1  # 0.1s intervals
            ))
        
        # Analyze each point for potential narration
        min_len = min(len(traj1_points), len(traj2_points))
        
        for i in range(self.lookback_window_size, min_len):
            # Get lookback windows
            start_idx = max(0, i - self.lookback_window_size + 1)
            end_idx = i + 1
            
            intended_lookback = traj1_points[start_idx:end_idx]
            actual_lookback = traj2_points[start_idx:end_idx]
            
            if len(intended_lookback) == 0 or len(actual_lookback) == 0:
                continue
            
            # Calculate deviation at current point
            intended_pos = intended_lookback[-1].position
            actual_pos = actual_lookback[-1].position
            deviation_vector = actual_pos - intended_pos
            deviation_magnitude = np.linalg.norm(deviation_vector)
            
            # Check if deviation exceeds soft threshold
            if deviation_magnitude > self.descriptor.soft_threshold:
                # Generate narration
                robot_idx = len(actual_lookback) - 1
                narration = self.descriptor.generate_description(
                    intended_lookback, actual_lookback, robot_idx
                )
                
                # Store narration event
                event = NarrationEvent(
                    position=actual_pos,
                    index=i,
                    narration=narration,
                    deviation_magnitude=deviation_magnitude,
                    soft_threshold=self.descriptor.soft_threshold,
                    hard_threshold=self.descriptor.hard_threshold
                )
                self.narration_events.append(event)
        
        print(f"Analysis complete: Found {len(self.narration_events)} narration events")
    
    def update_display(self):
        """Update visual elements"""
        # Clear plots
        self.ax_main.clear()
        self.ax_analysis.clear()
        self.ax_narrations.clear()
        
        # Plot main trajectories
        self.plot_main_trajectories()
        
        # Plot analysis view
        self.plot_analysis_view()
        
        # Plot narrations
        self.plot_narrations()
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def plot_main_trajectories(self):
        """Plot main trajectory view"""
        show_traj1 = self.traj1_checkbox.get_status()[0]
        show_traj2 = self.traj2_checkbox.get_status()[0]
        
        if show_traj1 and self.discretized_traj1:
            traj1_xy = np.array([[p.position[0], p.position[1]] for p in self.discretized_traj1])
            self.ax_main.plot(traj1_xy[:, 0], traj1_xy[:, 1], 
                            'g-', linewidth=2, label='Trajectory 1 (Intended)', alpha=0.8)
        
        if show_traj2 and self.discretized_traj2:
            traj2_xy = np.array([[p.position[0], p.position[1]] for p in self.discretized_traj2])
            self.ax_main.plot(traj2_xy[:, 0], traj2_xy[:, 1], 
                            'b-', linewidth=2, label='Trajectory 2 (Actual)', alpha=0.8)
        
        # Plot narration events
        if self.narration_events:
            event_positions = np.array([event.position for event in self.narration_events])
            self.ax_main.scatter(event_positions[:, 0], event_positions[:, 1], 
                               c='red', s=50, alpha=0.7, marker='o', 
                               label=f'Narration Events ({len(self.narration_events)})', zorder=10)
        
        self.ax_main.set_xlabel('X Position (m) - Forward')
        self.ax_main.set_ylabel('Y Position (m) - Left')
        self.ax_main.set_title('Trajectory Comparison')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.axis('equal')
    
    def plot_analysis_view(self):
        """Plot analysis view showing deviation over time"""
        if not self.discretized_traj1 or not self.discretized_traj2:
            return
        
        min_len = min(len(self.discretized_traj1), len(self.discretized_traj2))
        
        distances = []
        deviations = []
        indices = []
        
        for i in range(min_len):
            pos1 = self.discretized_traj1[i].position
            pos2 = self.discretized_traj2[i].position
            deviation = np.linalg.norm(pos2 - pos1)
            
            distances.append(self.discretized_traj1[i].distance_from_start)
            deviations.append(deviation)
            indices.append(i)
        
        # Plot deviation over distance
        self.ax_analysis.plot(distances, deviations, 'b-', linewidth=1, alpha=0.7, label='Deviation')
        
        # Plot threshold lines
        soft_thresh = self.descriptor.soft_threshold
        hard_thresh = self.descriptor.hard_threshold
        
        self.ax_analysis.axhline(y=soft_thresh, color='orange', linestyle='--', alpha=0.7, 
                                label=f'Soft Threshold ({soft_thresh:.2f}m)')
        self.ax_analysis.axhline(y=hard_thresh, color='red', linestyle='--', alpha=0.7, 
                                label=f'Hard Threshold ({hard_thresh:.2f}m)')
        
        # Highlight narration events
        if self.narration_events:
            event_distances = []
            event_deviations = []
            for event in self.narration_events:
                if event.index < len(self.discretized_traj1):
                    event_distances.append(self.discretized_traj1[event.index].distance_from_start)
                    event_deviations.append(event.deviation_magnitude)
            
            self.ax_analysis.scatter(event_distances, event_deviations, 
                                   c='red', s=30, alpha=0.8, marker='o', 
                                   label=f'Narration Points ({len(event_distances)})', zorder=10)
        
        self.ax_analysis.set_xlabel('Distance from Start (m)')
        self.ax_analysis.set_ylabel('Deviation (m)')
        self.ax_analysis.set_title('Deviation Analysis')
        self.ax_analysis.legend()
        self.ax_analysis.grid(True, alpha=0.3)
    
    def plot_narrations(self):
        """Plot narration display"""
        self.ax_narrations.text(0.5, 0.9, "Narration Analysis Results:", 
                               ha='center', va='center', fontsize=16, weight='bold')
        
        if not self.narration_events:
            self.ax_narrations.text(0.5, 0.5, "No narration events found with current thresholds", 
                                   ha='center', va='center', fontsize=14, style='italic',
                                   bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.8))
        else:
            # Display narrations in a scrollable format
            y_pos = 0.8
            for i, event in enumerate(self.narration_events[:10]):  # Show first 10
                if y_pos < 0.1:
                    break
                
                narration_text = f"{i+1}. \"{event.narration}\""
                deviation_text = f"   Deviation: {event.deviation_magnitude:.3f}m at distance {self.discretized_traj1[event.index].distance_from_start:.1f}m"
                
                self.ax_narrations.text(0.05, y_pos, narration_text, 
                                       ha='left', va='center', fontsize=12,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.6))
                
                self.ax_narrations.text(0.05, y_pos - 0.05, deviation_text, 
                                       ha='left', va='center', fontsize=10, style='italic')
                
                y_pos -= 0.12
            
            if len(self.narration_events) > 10:
                self.ax_narrations.text(0.5, 0.05, f"... and {len(self.narration_events) - 10} more events", 
                                       ha='center', va='center', fontsize=10, style='italic')
        
        # Show analysis summary
        summary_text = (f"Analysis Summary:\n"
                       f"• Total narration events: {len(self.narration_events)}\n"
                       f"• Soft threshold: {self.descriptor.soft_threshold:.2f}m\n"
                       f"• Hard threshold: {self.descriptor.hard_threshold:.2f}m\n"
                       f"• Lookback window: {self.lookback_window_size} points\n"
                       f"• Sampling distance: {self.sampling_length:.2f}m")
        
        self.ax_narrations.text(0.95, 0.1, summary_text, 
                               ha='right', va='bottom', fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        self.ax_narrations.set_xlim(0, 1)
        self.ax_narrations.set_ylim(0, 1)
        self.ax_narrations.axis('off')

def main():
    """Launch the trajectory comparison tool"""
    parser = argparse.ArgumentParser(description='Compare two trajectory JSONs and analyze narration generation')
    parser.add_argument('trajectory1', help='Path to first trajectory JSON file (intended)')
    parser.add_argument('trajectory2', help='Path to second trajectory JSON file (actual)')
    parser.add_argument('--sampling', type=float, default=0.1, help='Sampling distance in meters (default: 0.1)')
    
    args = parser.parse_args()
    
    print("🔍 Trajectory Comparison Tool")
    print("=" * 60)
    print("Interactive tool for comparing two trajectory JSONs")
    print("Analyzes where narrations would be generated based on deviation thresholds")
    print()
    print("Features:")
    print("• Load two trajectory JSON files")
    print("• Discretize trajectories with configurable sampling distance")
    print("• Analyze narration generation points")
    print("• Adjustable thresholds and lookback window")
    print("• Visual deviation analysis")
    print("• Real-time narration preview")
    print()
    print("Controls:")
    print("• Threshold sliders: adjust narration sensitivity")
    print("• Lookback window: adjust analysis window size")
    print("• Sampling distance: adjust trajectory discretization")
    print("• Checkboxes: toggle trajectory visibility")
    print("• Re-analyze button: refresh analysis with new parameters")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(args.trajectory1):
        print(f"Error: Trajectory 1 file not found: {args.trajectory1}")
        return
    
    if not os.path.exists(args.trajectory2):
        print(f"Error: Trajectory 2 file not found: {args.trajectory2}")
        return
    
    try:
        tool = TrajectoryComparisonTool(args.trajectory1, args.trajectory2, args.sampling)
        plt.show()
    except Exception as e:
        print(f"Error initializing comparison tool: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
