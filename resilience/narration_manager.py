#!/usr/bin/env python3
"""
Narration Manager Module

Handles trajectory narration and breach event processing.
"""

import numpy as np
import threading
import time
from typing import List, Optional, Dict, Any, Tuple
from .simple_descriptive_narration import XYSpatialDescriptor, TrajectoryPoint


class NarrationManager:
    """Manages trajectory narration and breach event processing."""
    
    def __init__(self, soft_threshold: float, hard_threshold: float, lookback_window_size: int = 20, sampling_distance: float = 0.1):
        """Initialize narration manager."""
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.lookback_window_size = lookback_window_size
        self.sampling_distance = sampling_distance
        
        # Narration logic
        self.descriptor = XYSpatialDescriptor(soft_threshold=self.soft_threshold, hard_threshold=self.hard_threshold)
        self.intended_points = []  # List[TrajectoryPoint] - discretized points
        self.actual_points = []    # List[TrajectoryPoint]
        
        # MODIFIED: Limit to lookback window size for actual points
        self.max_actual_points = lookback_window_size
        
        # Thread management
        self.narration_thread = None
        self.narration_running = False
        self.narration_condition = threading.Condition()
        self.narration_queue = []  # Queue of breach events for narration processing
        
        # Narration state
        self.narration_sent_for_current_breach = False
        self.current_narration_text = None
        self.current_narration_timestamp = None
        
        # Start the persistent narration thread
        self.start_narration_thread()
    
    def start_narration_thread(self):
        """Start the persistent narration thread."""
        if self.narration_thread is None or not self.narration_thread.is_alive():
            self.narration_running = True
            self.narration_thread = threading.Thread(target=self.narration_worker, daemon=True)
            self.narration_thread.start()
            print(f"Narration thread started")
        else:
            print(f"Narration thread already running")
    
    def set_intended_trajectory(self, nominal_points: np.ndarray):
        """Set the intended trajectory points from discretized data."""
        # Convert nominal points to 2D for narration (X=forward, Y=left)
        nominal_2d = np.array([[p[0], p[1]] for p in nominal_points])  # Only X,Y
        self.intended_points = [TrajectoryPoint(position=pt, time=i) for i, pt in enumerate(nominal_2d)]
        print(f"[Narration] Initialized with {len(self.intended_points)} discretized intended points, soft_threshold={self.soft_threshold}")
    
    def update_intended_trajectory(self, nominal_points: np.ndarray):
        """Update the intended trajectory points (for when path becomes available later)."""
        if len(nominal_points) > 0:
            self.set_intended_trajectory(nominal_points)
        else:
            print("[Narration] Warning: No nominal points provided for trajectory update")
    
    def add_actual_point(self, pos: np.ndarray, timestamp: float, flip_y_axis: bool = False):
        """Add actual trajectory point, keeping only the last lookback_window_size points."""
        pos_2d = np.array([pos[0], pos[1]])  # X=forward, Y=left
        if flip_y_axis:
            pos_2d[1] = -pos_2d[1]
        self.actual_points.append(TrajectoryPoint(position=pos_2d, time=timestamp))
        
        # Keep only the last lookback_window_size points
        if len(self.actual_points) > self.max_actual_points:
            self.actual_points.pop(0)
    
    def check_for_narration(self, current_time: float, breach_idx: Optional[int] = None) -> Optional[str]:
        """Generate narration during active breach using discretized path with proper lookback."""
        try:
            # CRITICAL: Only generate ONE narration per breach
            if self.narration_sent_for_current_breach:
                return None
            
            if not self.actual_points:
                return None
            
            # Check if intended trajectory is available
            if not self.intended_points:
                print("[Narration] Warning: No intended trajectory available for narration")
                return None
            
            actual_len = len(self.actual_points)
            
            # Use the most recent actual point index
            robot_idx = actual_len - 1
            
            # Get lookback window from discretized intended trajectory
            # Use same logic as trajectory comparison tool
            intended_start = max(0, robot_idx - self.lookback_window_size + 1)
            intended_end = robot_idx + 1  # Include current point
            intended_lookback = self.intended_points[intended_start:intended_end]
            
            # Use same lookback window for actual points
            # Since actual_points is limited to lookback_window_size, use all available points
            actual_lookback = self.actual_points
            
            if len(intended_lookback) == 0 or len(actual_lookback) == 0:
                return None
            
            # Ensure we have matching lengths for comparison
            min_len = min(len(intended_lookback), len(actual_lookback))
            if min_len == 0:
                return None
            
            # Clip trajectories to matching length
            intended_clipped = intended_lookback[:min_len]
            actual_clipped = actual_lookback[:min_len]
            
            # Use the most recent point for narration generation (same as comparison tool)
            narration_idx = min_len - 1
            
            # Generate narration using clipped trajectories
            narration = self.descriptor.generate_description(intended_clipped, actual_clipped, narration_idx)
            
            if narration:  # Only return if there's actual narration
                print("=" * 50)
                print("NARRATION GENERATED (Discretized Path with Lookback)")
                print("=" * 50)
                print(f"Content: {narration}")
                print(f"Using {len(actual_clipped)} actual points (max {self.lookback_window_size})")
                print(f"Using {len(intended_clipped)} intended points from discretized trajectory")
                print(f"Sampling distance: {self.sampling_distance}m")
                print("=" * 50)
                
                # Mark that we've sent narration for this breach (ONLY ONE PER BREACH)
                self.narration_sent_for_current_breach = True
                
                # Store narration for later saving
                self.current_narration_text = narration
                self.current_narration_timestamp = current_time
                
                print(f"Narration sent")
                print(f"Waiting for VLM to analyze cause...")
                
                return narration
            
            return None
                
        except Exception as e:
            print(f"Error generating narration: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def queue_breach_event(self, event_type: str, timestamp: float):
        """Queue a breach event for narration processing."""
        with self.narration_condition:
            event = {
                'type': event_type,
                'timestamp': timestamp
            }
            self.narration_queue.append(event)
            self.narration_condition.notify()
    
    def process_breach_event(self, event: Dict[str, Any]):
        """Process a breach event (start or end)."""
        event_type = event.get('type')
        timestamp = event.get('timestamp')
        
        if event_type == 'start':
            self.handle_breach_start(timestamp)
        elif event_type == 'end':
            self.handle_breach_end(timestamp)
    
    def handle_breach_start(self, timestamp: float):
        """Handle breach start - reset narration flag."""
        self.narration_sent_for_current_breach = False
        self.current_narration_text = None
        self.current_narration_timestamp = None
    
    def handle_breach_end(self, timestamp: float):
        """Handle breach end - just log it."""
        pass  # No need to print anything here
    
    def narration_worker(self):
        """Persistent narration worker thread that processes breach events."""
        print(f"Narration worker started")
        
        while self.narration_running:
            try:
                # Wait for breach events
                with self.narration_condition:
                    while not self.narration_queue and self.narration_running:
                        self.narration_condition.wait(timeout=1.0)
                    
                    if not self.narration_running:
                        break
                    
                    # Process all queued breach events
                    while self.narration_queue:
                        event = self.narration_queue.pop(0)
                        self.process_breach_event(event)
                        
            except Exception as e:
                print(f"Error in narration worker: {e}")
                time.sleep(0.1)
        
        print(f"Narration worker ended")
    
    def reset_narration_state(self):
        """Reset narration state for new breach."""
        self.narration_sent_for_current_breach = False
        self.current_narration_text = None
        self.current_narration_timestamp = None
    
    def get_narration_sent(self) -> bool:
        """Check if narration was sent for current breach."""
        return self.narration_sent_for_current_breach
    
    def get_current_narration(self) -> Tuple[Optional[str], Optional[float]]:
        """Get current narration text and timestamp."""
        return self.current_narration_text, self.current_narration_timestamp
    
    def stop(self):
        """Stop the narration thread gracefully."""
        if self.narration_running:
            print("Stopping narration thread...")
            with self.narration_condition:
                self.narration_running = False
                self.narration_condition.notify_all()
            
            # Wait for thread to finish
            if self.narration_thread and self.narration_thread.is_alive():
                self.narration_thread.join(timeout=2.0)
                print("Narration thread stopped") 