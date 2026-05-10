#!/usr/bin/env python3
"""
Simple Risk Buffer Management System

Core Concept:
- ACTIVE: Currently collecting data during breach
- FROZEN: Breach ended, no more data, awaiting/has cause

Buffer Lifecycle:
1. Breach starts → Create new buffer in ACTIVE state
2. During breach → Buffer collects images, poses, depth data  
3. VLM answer arrives during breach → Assign cause to active buffer (stays ACTIVE)
4. Breach ends → Move buffer from ACTIVE to FROZEN state
5. VLM answer arrives after breach → Assign to oldest FROZEN buffer without cause
"""

import numpy as np
import json
import os
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import threading
import time
import cv2

class BufferState(Enum):
    """Simple buffer states - only 2 needed"""
    ACTIVE = "active"    # Collecting data during breach
    FROZEN = "frozen"    # Breach ended, no more data, awaiting/has cause

class RiskBuffer:
    """Simple risk buffer for collecting data during trajectory breaches"""
    
    # Class variable to track buffer count for sequential naming
    _buffer_count = 0
    
    def __init__(self, start_time: float):
        RiskBuffer._buffer_count += 1
        self.buffer_id = f"buffer{RiskBuffer._buffer_count}"
        self.start_time = start_time
        self.end_time = None
        self.state = BufferState.ACTIVE
        self.cause = None
        self.cause_location = None  # [x, y, z] in meters, world coordinates
        self.enhanced_cause_embedding = None  # Enhanced cause embedding tensor
        
        # Data storage
        self.images = []  # List of (timestamp, cv_image, ros_msg) tuples
        self.poses = []   # List of (timestamp, pose, drift) tuples  
        self.depth_msgs = {}  # Dict of timestamp -> depth_msg
        # self.drift
        
        # NEW: Narration image storage
        self.narration_image = None  # Store the narration image for easy retrieval
        self.narration_timestamp = None  # Timestamp when narration was generated
        self.narration_text = None  # Store the narration text
        
        # NEW: Store nominal path that was active when breach started
        # This is critical for proper GP computation - we compare observed poses
        # during breach against the predicted (nominal) path that was active at breach start
        self.nominal_path_xyz = None  # (N, 3) numpy array of nominal trajectory points
        
        print(f"[RiskBuffer] Created {self.buffer_id} (ACTIVE) at t={start_time:.2f}")

    def add_image(self, timestamp: float, cv_image, ros_msg) -> bool:
        """Add image data to buffer - DISABLED to reduce I/O bottleneck"""
        if self.state != BufferState.ACTIVE:
            return False
        
        # DISABLED: Image collection to reduce memory usage and I/O bottleneck
        # self.images.append((timestamp, cv_image, ros_msg))
        return True
    
    def add_pose(self, timestamp: float, pose, drift: float) -> bool:
        """Add pose data to buffer"""
        if self.state != BufferState.ACTIVE:
            return False
        
        self.poses.append((timestamp, pose, drift))
        return True
    
    def add_depth_msg(self, timestamp: float, depth_msg) -> bool:
        """Add depth message to buffer - DISABLED to reduce I/O bottleneck"""
        if self.state != BufferState.ACTIVE:
            return False

        # DISABLED: Depth message collection to reduce memory usage and I/O bottleneck
        # self.depth_msgs[timestamp] = depth_msg
        return True

    def assign_cause(self, cause: str) -> bool:
        """Assign cause to this buffer (can be active or frozen)"""
        if self.state not in [BufferState.ACTIVE, BufferState.FROZEN]:
            print(f"[RiskBuffer] Cannot assign cause to {self.buffer_id} - invalid state: {self.state}")
            return False
        
        self.cause = cause
        state_str = "ACTIVE" if self.state == BufferState.ACTIVE else "FROZEN"
        print(f"[RiskBuffer] Assigned cause '{cause}' to {self.buffer_id} ({state_str})")
        return True

    def assign_cause_location(self, location: List[float]) -> bool:
        """Assign 3D location of cause to this buffer"""
        if not self.cause:
            print(f"[RiskBuffer] Cannot assign location to {self.buffer_id} - no cause assigned")
            return False
        
        self.cause_location = location
        print(f"[RiskBuffer] Assigned cause location {location} to {self.buffer_id}")
        return True

    def assign_enhanced_cause_embedding(self, enhanced_embedding) -> bool:
        """Assign enhanced cause embedding to this buffer"""
        if not self.cause:
            print(f"[RiskBuffer] Cannot assign enhanced embedding to {self.buffer_id} - no cause assigned")
            return False
        
        self.enhanced_cause_embedding = enhanced_embedding
        print(f"[RiskBuffer] Assigned enhanced cause embedding to {self.buffer_id} (shape: {enhanced_embedding.shape if enhanced_embedding is not None else 'None'})")
        return True

    def has_enhanced_embedding(self) -> bool:
        """Check if buffer has enhanced cause embedding"""
        return self.enhanced_cause_embedding is not None
    
    def set_nominal_path(self, nominal_path_xyz: np.ndarray) -> bool:
        """Store the nominal path that was active when breach started.
        
        This is critical for proper GP computation - we need to compare observed
        poses during the breach against the predicted (nominal) path that was
        active at the time the breach started, not the current path which may
        have changed.
        
        Args:
            nominal_path_xyz: (N, 3) numpy array of nominal trajectory points
            
        Returns:
            True if successfully stored
        """
        try:
            if nominal_path_xyz is not None:
                self.nominal_path_xyz = nominal_path_xyz.copy()
                print(f"[RiskBuffer] Stored nominal path for {self.buffer_id} ({len(nominal_path_xyz)} points)")
            else:
                self.nominal_path_xyz = None
                print(f"[RiskBuffer] Warning: No nominal path provided for {self.buffer_id}")
            return True
        except Exception as e:
            print(f"[RiskBuffer] Error storing nominal path for {self.buffer_id}: {e}")
            return False
    
    def get_nominal_path(self) -> Optional[np.ndarray]:
        """Get the stored nominal path that was active when breach started."""
        return self.nominal_path_xyz

    def set_narration_data(self, narration_image, narration_text: str, timestamp: float) -> bool:
        """Store narration image and related data in the buffer."""
        try:
            self.narration_image = narration_image.copy() if narration_image is not None else None
            self.narration_text = narration_text
            self.narration_timestamp = timestamp
            print(f"[RiskBuffer] Stored narration data for {self.buffer_id} at t={timestamp:.2f}")
            return True
        except Exception as e:
            print(f"[RiskBuffer] Error storing narration data for {self.buffer_id}: {e}")
            return False
    
    def set_narration_data_with_timestamp(self, narration_image, narration_text: str, 
                                        narration_timestamp: float, original_image_timestamp: float) -> bool:
        """Store narration image and related data with original image timestamp for semantic mapping."""
        try:
            self.narration_image = narration_image.copy() if narration_image is not None else None
            self.narration_text = narration_text
            self.narration_timestamp = narration_timestamp
            self.original_image_timestamp = original_image_timestamp  # NEW: Store original image timestamp
            print(f"[RiskBuffer] Stored narration data for {self.buffer_id}")
            print(f"  Narration timestamp: {narration_timestamp:.6f}")
            print(f"  Original image timestamp: {original_image_timestamp:.6f}")
            return True
        except Exception as e:
            print(f"[RiskBuffer] Error storing narration data with timestamp for {self.buffer_id}: {e}")
            return False
    
    def get_original_image_timestamp(self) -> Optional[float]:
        """Get the original image timestamp for semantic mapping."""
        return getattr(self, 'original_image_timestamp', None)
    
    def has_narration_image(self) -> bool:
        """Check if buffer has narration image stored."""
        return self.narration_image is not None
    
    def get_narration_image(self):
        """Get the stored narration image."""
        return self.narration_image

    def save_immediately_if_ready(self, directory: str) -> bool:
        """Save buffer immediately if it's frozen and has a cause"""
        if self.state == BufferState.FROZEN and self.cause is not None:
            return self.save(directory)
        return False
    
    def save_poses_continuously(self, directory: str) -> bool:
        """Save poses continuously during breach so poses.npy is always available.
        Also saves nominal path if available."""
        try:
            if not self.poses:
                return False
                
            # Create buffer directory
            buffer_dir = os.path.join(directory, self.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            # Save poses array
            poses_array = np.array([(t, p[0], p[1], p[2], d) for t, p, d in self.poses])
            np.save(os.path.join(buffer_dir, 'poses.npy'), poses_array)
            
            # Also save nominal path if available (so it's ready for GP computation)
            if self.nominal_path_xyz is not None:
                nominal_path_path = os.path.join(buffer_dir, 'nominal_path.npy')
                np.save(nominal_path_path, self.nominal_path_xyz)
            
            return True
            
        except Exception as e:
            print(f"[RiskBuffer] Error saving poses for {self.buffer_id}: {e}")
            return False
    
    def freeze(self, end_time: float):
        """Freeze buffer when breach ends"""
        if self.state != BufferState.ACTIVE:
            print(f"[RiskBuffer] Warning: Attempting to freeze non-active buffer {self.buffer_id}")
            return

        self.end_time = end_time
        self.state = BufferState.FROZEN
        duration = end_time - self.start_time
        data_counts = self.get_data_counts()
        cause_info = f"cause: '{self.cause}'" if self.cause else "no cause"
        
        print(f"[RiskBuffer] {self.buffer_id} FROZEN at t={end_time:.2f} (duration: {duration:.2f}s, {cause_info}, images: {data_counts['images']}, poses: {data_counts['poses']})")
    
    def is_active(self) -> bool:
        """Check if buffer is active"""
        return self.state == BufferState.ACTIVE
    
    def is_frozen(self) -> bool:
        """Check if buffer is frozen"""
        return self.state == BufferState.FROZEN
    
    def has_cause(self) -> bool:
        """Check if buffer has cause assigned"""
        return self.cause is not None
    
    def needs_cause(self) -> bool:
        """Check if buffer needs cause assignment"""
        return self.cause is None
    
    def get_data_counts(self) -> Dict[str, int]:
        """Get counts of collected data"""
        return {
            'images': len(self.images),
            'poses': len(self.poses), 
            'depth_msgs': len(self.depth_msgs)
        }
    
    def get_duration(self) -> float:
        """Get buffer duration"""
        end_t = self.end_time if self.end_time else self.start_time
        return end_t - self.start_time

    def save(self, directory: str) -> bool:
        """Save buffer data to disk"""
        try:
            # Create buffer directory
            buffer_dir = os.path.join(directory, self.buffer_id)
            os.makedirs(buffer_dir, exist_ok=True)
            
            # Save metadata
            metadata = {
                'buffer_id': self.buffer_id,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'state': self.state.value,
                'cause': self.cause,
                'cause_location': self.cause_location,
                'has_enhanced_embedding': self.has_enhanced_embedding(),
                'enhanced_embedding_shape': list(self.enhanced_cause_embedding.shape) if self.enhanced_cause_embedding is not None else None,
                'has_nominal_path': self.nominal_path_xyz is not None,
                'nominal_path_points': len(self.nominal_path_xyz) if self.nominal_path_xyz is not None else 0,
                'data_counts': self.get_data_counts(),
                'duration': self.get_duration()
            }
            
            # Save nominal path if available
            if self.nominal_path_xyz is not None:
                nominal_path_path = os.path.join(buffer_dir, 'nominal_path.npy')
                np.save(nominal_path_path, self.nominal_path_xyz)
                print(f"[RiskBuffer] Saved nominal path to {nominal_path_path} ({len(self.nominal_path_xyz)} points)")
            
            with open(os.path.join(buffer_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save data arrays
            if self.poses:
                poses_array = np.array([(t, p[0], p[1], p[2], d) for t, p, d in self.poses])
                np.save(os.path.join(buffer_dir, 'poses.npy'), poses_array)
            
            if self.images:
                # Save first and last images as examples
                if len(self.images) >= 1:
                    first_img = self.images[0][1]  # cv_image
                    np.save(os.path.join(buffer_dir, 'first_image.npy'), first_img)
                
                if len(self.images) >= 2:
                    last_img = self.images[-1][1]  # cv_image  
                    np.save(os.path.join(buffer_dir, 'last_image.npy'), last_img)
            
            print(f"[RiskBuffer] Saved {self.buffer_id} ({self.state.value.upper()}) to {buffer_dir}")
            return True
            
        except Exception as e:
            print(f"[RiskBuffer] Error saving {self.buffer_id}: {e}")
            return False

class RiskBufferManager:
    """Simple buffer manager with active and frozen buffer lists"""
    
    def __init__(self, save_directory: str = None):
        self.active_buffers: List[RiskBuffer] = []
        self.frozen_buffers: List[RiskBuffer] = []
        self.save_directory = save_directory
        self.lock = threading.Lock()

        print(f"[RiskBufferManager] Initialized with empty active/frozen buffer lists (save_dir: {save_directory})")
    
    def start_buffer(self, start_time: float) -> RiskBuffer:
        """Start new buffer when breach begins"""
        with self.lock:
            # Create new buffer
            buffer = RiskBuffer(start_time)
            self.active_buffers.append(buffer)
            
            print(f"[RiskBufferManager] Started {buffer.buffer_id} (ACTIVE buffers: {len(self.active_buffers)})")
            return buffer
    
    def freeze_active_buffers(self, end_time: float):
        """Freeze all active buffers when breach ends"""
        with self.lock:
            if not self.active_buffers:
                print(f"[RiskBufferManager] No active buffers to freeze at t={end_time:.2f}")
                return
            
            print(f"[RiskBufferManager] Freezing {len(self.active_buffers)} active buffer(s) at t={end_time:.2f}")
            
            for buffer in self.active_buffers:
                data_counts = buffer.get_data_counts()
                duration = end_time - buffer.start_time
                cause_info = f"cause: '{buffer.cause}'" if buffer.cause else "no cause"
                
                print(f"[RiskBufferManager] Freezing {buffer.buffer_id}: {cause_info}, duration: {duration:.2f}s, images: {data_counts['images']}, poses: {data_counts['poses']}")
                buffer.freeze(end_time)
                self.frozen_buffers.append(buffer)
            
            num_frozen = len(self.active_buffers)
            self.active_buffers.clear()
            
            print(f"[RiskBufferManager] Successfully frozen {num_frozen} buffer(s) (FROZEN buffers: {len(self.frozen_buffers)})")
    
    def add_image(self, timestamp: float, cv_image, ros_msg) -> bool:
        """Add image to all active buffers"""
        with self.lock:
            if not self.active_buffers:
                return False

            added = False
            for buffer in self.active_buffers:
                if buffer.add_image(timestamp, cv_image, ros_msg):
                    added = True
            
            return added
    
    def add_pose(self, timestamp: float, pose, drift: float) -> bool:
        """Add pose to all active buffers and save poses continuously"""
        with self.lock:
            if not self.active_buffers:
                return False

            added = False
            for buffer in self.active_buffers:
                if buffer.add_pose(timestamp, pose, drift):
                    added = True
                    # Save poses continuously so poses.npy is always available
                    if self.save_directory:
                        buffer.save_poses_continuously(self.save_directory)
            
            return added
    
    def add_depth_msg(self, timestamp: float, depth_msg) -> bool:
        """Add depth message to all active buffers"""
        with self.lock:
            if not self.active_buffers:
                return False

            added = False
            for buffer in self.active_buffers:
                if buffer.add_depth_msg(timestamp, depth_msg):
                    added = True
            
            return added
    
    def store_narration_data(self, narration_image, narration_text: str, timestamp: float) -> bool:
        """Store narration data in all active buffers."""
        with self.lock:
            if not self.active_buffers:
                print("No active buffers to store narration data")
                return False

            stored = False
            for buffer in self.active_buffers:
                if buffer.set_narration_data(narration_image, narration_text, timestamp):
                    stored = True
            
            if stored:
                print(f"[RiskBufferManager] Stored narration data in {len(self.active_buffers)} active buffer(s)")
            
            return stored
    
    def store_narration_data_with_timestamp(self, narration_image, narration_text: str, 
                                          narration_timestamp: float, original_image_timestamp: float) -> bool:
        """Store narration data with original image timestamp in all active buffers (or most recent frozen buffer if no active)."""
        with self.lock:
            stored = False
            
            # First try active buffers
            if self.active_buffers:
                for buffer in self.active_buffers:
                    if buffer.set_narration_data_with_timestamp(narration_image, narration_text, 
                                                              narration_timestamp, original_image_timestamp):
                        stored = True
                
                if stored:
                    print(f"[RiskBufferManager] Stored narration data with timestamp in {len(self.active_buffers)} active buffer(s)")
                    print(f"  Narration timestamp: {narration_timestamp:.6f}")
                    print(f"  Original image timestamp: {original_image_timestamp:.6f}")
                    return stored
            
            # Fallback: try most recent frozen buffer if no active buffers
            if not stored and self.frozen_buffers:
                most_recent_buffer = self.frozen_buffers[-1]
                if most_recent_buffer.set_narration_data_with_timestamp(narration_image, narration_text, 
                                                                       narration_timestamp, original_image_timestamp):
                    stored = True
                    print(f"[RiskBufferManager] Stored narration data with timestamp in most recent frozen buffer: {most_recent_buffer.buffer_id}")
                    print(f"  Narration timestamp: {narration_timestamp:.6f}")
                    print(f"  Original image timestamp: {original_image_timestamp:.6f}")
                    return stored
            
            if not stored:
                print("No active or frozen buffers to store narration data with timestamp")
            
            return stored
    
    def assign_cause(self, cause: str) -> bool:
        """Intelligently assign cause to buffers - prioritize frozen buffers without causes, then active buffers"""
        with self.lock:
            print(f"[RiskBufferManager] Attempting to assign cause '{cause}' to buffers...")
            
            # STEP 1: Try to assign to frozen buffers that don't have causes yet
            frozen_needing_cause = [b for b in self.frozen_buffers if b.needs_cause()]
            if frozen_needing_cause:
                # Assign to the most recent frozen buffer that needs a cause
                buffer = frozen_needing_cause[-1]
                print(f"[RiskBufferManager] Found frozen buffer {buffer.buffer_id} needing cause")
                
                if buffer.assign_cause(cause):
                    print(f"[RiskBufferManager] Successfully assigned cause '{cause}' to frozen buffer {buffer.buffer_id}")
                    
                    # Save immediately if we have a save directory
                    if self.save_directory:
                        if buffer.save_immediately_if_ready(self.save_directory):
                            print(f"[RiskBufferManager] Immediately saved {buffer.buffer_id} with cause '{cause}'")
                        else:
                            print(f"[RiskBufferManager] Failed to immediately save {buffer.buffer_id}")
                    return True
                else:
                    print(f"[RiskBufferManager] Failed to assign cause to frozen buffer {buffer.buffer_id}")
            else:
                print(f"[RiskBufferManager] No frozen buffers need causes ({len(self.frozen_buffers)} frozen buffers)")
            
            # STEP 2: If no frozen buffers need causes, try active buffers
            if self.active_buffers:
                # Assign to most recent active buffer
                buffer = self.active_buffers[-1]
                print(f"[RiskBufferManager] No frozen buffers need causes, trying active buffer {buffer.buffer_id}")
                
                if buffer.assign_cause(cause):
                    print(f"[RiskBufferManager] Successfully assigned cause '{cause}' to active buffer {buffer.buffer_id}")
                    return True
                else:
                    print(f"[RiskBufferManager] Failed to assign cause to active buffer {buffer.buffer_id}")
            else:
                print(f"[RiskBufferManager] No active buffers available")
            
            # STEP 3: If we get here, no suitable buffers found
            print(f"[RiskBufferManager] No suitable buffers to assign cause '{cause}' to")
            print(f"[RiskBufferManager] Buffer status: {len(self.active_buffers)} active, {len(self.frozen_buffers)} frozen")
            print(f"[RiskBufferManager] Frozen needing causes: {len(frozen_needing_cause)}")
            return False

    def get_status(self) -> Dict[str, any]:
        """Get current buffer status"""
        with self.lock:
            return {
                'active_buffers': len(self.active_buffers),
                'frozen_buffers': len(self.frozen_buffers),
                'total_buffers': len(self.active_buffers) + len(self.frozen_buffers),
                'active_with_cause': sum(1 for b in self.active_buffers if b.has_cause()),
                'frozen_with_cause': sum(1 for b in self.frozen_buffers if b.has_cause()),
                'frozen_needing_cause': sum(1 for b in self.frozen_buffers if b.needs_cause())
            }
    
    def print_status(self):
        """Print current buffer status with detailed buffer information"""
        status = self.get_status()
        print(f"[RiskBufferManager] Status: {status['active_buffers']} ACTIVE, {status['frozen_buffers']} FROZEN")
        print(f"[RiskBufferManager] Causes: {status['active_with_cause']} active, {status['frozen_with_cause']} frozen, {status['frozen_needing_cause']} need cause")
        
        # Print active buffers
        if self.active_buffers:
            print(f"[RiskBufferManager] ACTIVE buffers:")
            for buffer in self.active_buffers:
                cause_info = f"cause: '{buffer.cause}'" if buffer.cause else "no cause"
                data_counts = buffer.get_data_counts()
                print(f"  - {buffer.buffer_id}: {cause_info}, images: {data_counts['images']}, poses: {data_counts['poses']}")
        else:
            print(f"[RiskBufferManager] No ACTIVE buffers")
        
        # Print frozen buffers
        if self.frozen_buffers:
            print(f"[RiskBufferManager] FROZEN buffers:")
            for buffer in self.frozen_buffers:
                cause_info = f"cause: '{buffer.cause}'" if buffer.cause else "needs cause"
                data_counts = buffer.get_data_counts()
                duration = buffer.get_duration()
                print(f"  - {buffer.buffer_id}: {cause_info}, duration: {duration:.2f}s, images: {data_counts['images']}, poses: {data_counts['poses']}")
        else:
            print(f"[RiskBufferManager] No FROZEN buffers")
    
    def get_cause_assignment_candidates(self) -> Dict[str, List[str]]:
        """Get list of buffer IDs that are candidates for cause assignment"""
        with self.lock:
            candidates = {
                'frozen_needing_cause': [b.buffer_id for b in self.frozen_buffers if b.needs_cause()],
                'active_available': [b.buffer_id for b in self.active_buffers],
                'frozen_with_cause': [b.buffer_id for b in self.frozen_buffers if b.has_cause()],
                'active_with_cause': [b.buffer_id for b in self.active_buffers if b.has_cause()]
            }
            return candidates
    
    def save_all_finalized(self, directory: str):
        """Save all frozen buffers to disk"""
        with self.lock:
            saved_count = 0
            for buffer in self.frozen_buffers:
                if buffer.save(directory):
                    saved_count += 1
            
            print(f"[RiskBufferManager] Saved {saved_count}/{len(self.frozen_buffers)} FROZEN buffers to {directory}")
    
    def cleanup_old_frozen(self, max_age_hours: float = 24.0):
        """Remove old frozen buffers to prevent memory buildup"""
        with self.lock:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            old_buffers = []
            for buffer in self.frozen_buffers:
                if buffer.end_time and (current_time - buffer.end_time) > max_age_seconds:
                    old_buffers.append(buffer)
            
            for buffer in old_buffers:
                self.frozen_buffers.remove(buffer)
                print(f"[RiskBufferManager] Removed old {buffer.buffer_id} (age: {(current_time - buffer.end_time)/3600:.1f}h)")
            
            if old_buffers:
                print(f"[RiskBufferManager] Cleaned up {len(old_buffers)} old buffers") 