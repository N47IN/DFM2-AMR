#!/usr/bin/env python3
"""
Clean Semantic Bridge with Binary Threshold Filtering

Simplified bridge that applies a binary threshold to similarity maps and
publishes only the hotspot mask and the original RGB image timestamp.
"""

import numpy as np
import json
import time
import cv2
import base64
from typing import Dict, List, Optional, Any, Tuple
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading


class SemanticHotspotPublisher:
    """Publisher that applies binary threshold to similarity maps and publishes results."""
    
    def __init__(self, node, config: Dict[str, Any] = None):
        self.node = node
        self.bridge = CvBridge()
        
        # Load configuration
        self.config = config.get('semantic_bridge', {}) if config else {}
        self.hotspot_threshold = self.config.get('hotspot_similarity_threshold', 0.6)
        self.min_area = self.config.get('min_hotspot_area', 100)
        self.publish_rate_limit = self.config.get('publish_rate_limit', 100.0)
        
        # Load topic configuration
        topics_config = self.config.get('topics', {})
        self.semantic_hotspots_topic = topics_config.get('semantic_hotspots_topic', '/semantic_hotspots')
        self.semantic_hotspot_mask_topic = topics_config.get('semantic_hotspot_mask_topic', '/semantic_hotspot_mask')
        self.semantic_hotspot_overlay_topic = topics_config.get('semantic_hotspot_overlay_topic', '/semantic_hotspot_overlay')
        
        # Rate limiting
        self.last_publish_time = 0.0
        
        # Color palette for VLM answers
        self.color_palette = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [128, 128, 0],  # Olive
            [0, 128, 128],  # Teal
            [128, 0, 128],  # Maroon
            [255, 165, 0],  # Orange Red
            [75, 0, 130],   # Indigo
            [240, 230, 140], # Khaki
            [255, 20, 147]  # Deep Pink
        ]
        self.vlm_color_map = {}  # vlm_answer -> color
        self.color_map_lock = threading.Lock()
        
        # Publishers
        self.hotspot_data_pub = self.node.create_publisher(String, self.semantic_hotspots_topic, 10)
        self.hotspot_mask_pub = self.node.create_publisher(Image, self.semantic_hotspot_mask_topic, 10)
        self.hotspot_overlay_pub = self.node.create_publisher(Image, self.semantic_hotspot_overlay_topic, 10)
        
        if hasattr(self.node, 'get_logger'):
            self.node.get_logger().info(f"Semantic bridge initialized - threshold: {self.hotspot_threshold}")
    
    def _get_color_for_vlm_answer(self, vlm_answer: str) -> List[int]:
        """Get unique color for VLM answer."""
        with self.color_map_lock:
            if vlm_answer not in self.vlm_color_map:
                color_idx = len(self.vlm_color_map) % len(self.color_palette)
                self.vlm_color_map[vlm_answer] = self.color_palette[color_idx]
            return self.vlm_color_map[vlm_answer]
    
    def _set_msg_stamp_from_float(self, ros_img_msg: Image, timestamp: float):
        """Set ROS2 header.stamp on an Image message from a float seconds timestamp."""
        try:
            sec = int(timestamp)
            nsec = int((timestamp - sec) * 1e9)
            ros_img_msg.header.stamp.sec = sec
            ros_img_msg.header.stamp.nanosec = nsec
        except Exception:
            ros_img_msg.header.stamp = self.node.get_clock().now().to_msg()
    
    def publish_merged_hotspots(self, vlm_hotspots: Dict[str, np.ndarray], 
                               timestamp: float, narration : Optional[bool] = False, original_image: Optional[np.ndarray] = None, buffer_id: Optional[str] = None) -> bool:
        """
        Publish merged hotspot mask with different colors for different VLM answers.
        Now sends mask as sensor_msgs/Image and a slim JSON for metadata.
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_publish_time < (1.0 / 30.0):
                return False
            
            if not vlm_hotspots:
                return False
            
            # Get dimensions from first mask
            first_mask = next(iter(vlm_hotspots.values()))
            h, w = first_mask.shape
            
            # Create merged colored mask
            merged_mask = np.zeros((h, w, 3), dtype=np.uint8)
            vlm_info = {}
            
            for vlm_answer, hotspot_mask in vlm_hotspots.items():
                if not np.any(hotspot_mask):
                    continue
                
                # Get unique color for this VLM answer
                color = self._get_color_for_vlm_answer(vlm_answer)
                
                # Ensure binary mask (0/1) for encoding
                hotspot_binary = (hotspot_mask > 0).astype(np.uint8)
                
                # Apply color to hotspot regions for visualization
                merged_mask[hotspot_binary > 0] = color
                
                # Count hotspots for this VLM answer
                hotspot_count = int(np.sum(hotspot_binary))
                
                # Encode binary mask as base64 PNG so consumers can reconstruct it
                mask_b64 = None
                try:
                    success, buf = cv2.imencode('.png', hotspot_binary * 255)
                    if success:
                        mask_b64 = base64.b64encode(buf).decode('ascii')
                except Exception:
                    mask_b64 = None
                
                info = {
                    'color': color,
                    'hotspot_pixels': hotspot_count,
                    'hotspot_threshold': float(self.hotspot_threshold)
                }
                if mask_b64 is not None:
                    info['mask_png'] = mask_b64
                
                vlm_info[vlm_answer] = info
            
            if not np.any(merged_mask):
                return False  # No hotspots
            
            # Publish merged mask image stamped with original timestamp
            self._publish_merged_hotspot_mask_image(merged_mask, timestamp)
            
            # Publish overlay image if original image provided (stamped with original timestamp)
            if original_image is not None:
                self._publish_merged_hotspot_overlay(original_image, merged_mask, timestamp)
            
            # Create slim structured data message (no flattened mask payload)
            hotspot_data = {
                'schema_version': 1,
                'type': 'merged_similarity_hotspots',
                'timestamp': float(timestamp),
                'vlm_info': vlm_info,
                'total_vlm_answers': len(vlm_info),
                'threshold_used': float(self.hotspot_threshold),
                'is_narration' : narration,
                'buffer_id': buffer_id
            }
            
            # Publish structured metadata
            msg = String(data=json.dumps(hotspot_data))
            self.hotspot_data_pub.publish(msg)
            
            self.last_publish_time = current_time
            
            # if hasattr(self.node, 'get_logger'):
            #     self.node.get_logger().info(f"Published merged hotspots for {len(vlm_info)} VLM answers @ {timestamp:.6f}")
            
            # NEW: Special logging for narration hotspot masks
            if len(vlm_info) == 1:  # Single VLM answer (likely narration)
                vlm_answer = list(vlm_info.keys())[0]
                hotspot_pixels = list(vlm_info.values())[0]['hotspot_pixels']
                # if hasattr(self.node, 'get_logger'):
                #     self.node.get_logger().info(f"NARRATION HOTSPOT: '{vlm_answer}' with {hotspot_pixels} pixels @ {timestamp:.6f}")
            
            return True
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing merged hotspots: {e}")
            return False
    
    def _publish_hotspot_mask_image(self, mask: np.ndarray, vlm_answer: str, timestamp: float):
        """Publish binary hotspot mask as ROS Image for RViz visualization."""
        try:
            # Convert binary mask to 3-channel image for better visibility
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask > 0] = [0, 255, 0]  # Green hotspots
            
            # Convert to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(mask_rgb, encoding='rgb8')
            self._set_msg_stamp_from_float(mask_msg, timestamp)
            mask_msg.header.frame_id = 'camera_link'
            
            self.hotspot_mask_pub.publish(mask_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing mask image: {e}")
    
    def _publish_merged_hotspot_mask_image(self, merged_mask: np.ndarray, timestamp: float):
        """Publish merged colored hotspot mask as ROS Image for RViz visualization."""
        try:
            # Convert to ROS Image message
            mask_msg = self.bridge.cv2_to_imgmsg(merged_mask, encoding='rgb8')
            self._set_msg_stamp_from_float(mask_msg, timestamp)
            mask_msg.header.frame_id = 'camera_link'
            
            self.hotspot_mask_pub.publish(mask_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing merged mask image: {e}")
    
    def _publish_hotspot_overlay(self, original_image: np.ndarray, mask: np.ndarray, 
                                vlm_answer: str, timestamp: float):
        """Publish overlay of hotspots on original image for RViz visualization."""
        try:
            # Create overlay
            overlay = original_image.copy()
            
            # Apply green overlay where hotspots are detected
            hotspot_pixels = mask > 0
            overlay[hotspot_pixels] = cv2.addWeighted(
                overlay[hotspot_pixels], 0.7,
                np.full_like(overlay[hotspot_pixels], [0, 255, 0]), 0.3,
                0
            )
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Hotspots: {vlm_answer}"
            cv2.putText(overlay, text, (10, 30), font, 0.7, (0, 255, 0), 2)
            
            # Convert to ROS Image message
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
            self._set_msg_stamp_from_float(overlay_msg, timestamp)
            overlay_msg.header.frame_id = 'camera_link'
            
            self.hotspot_overlay_pub.publish(overlay_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing overlay image: {e}")
    
    def _publish_merged_hotspot_overlay(self, original_image: np.ndarray, merged_mask: np.ndarray, 
                                      timestamp: float):
        """Publish overlay of merged colored hotspots on original image."""
        try:
            # Create overlay
            overlay = original_image.copy()
            
            # FIX: Resize mask to match original image dimensions if they don't match
            orig_h, orig_w = original_image.shape[:2]
            mask_h, mask_w = merged_mask.shape[:2]
            
            if (orig_h != mask_h) or (orig_w != mask_w):
                # Resize mask to match original image dimensions
                merged_mask = cv2.resize(merged_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Apply colored overlay where hotspots are detected
            hotspot_pixels = np.any(merged_mask > 0, axis=2)
            overlay[hotspot_pixels] = cv2.addWeighted(
                overlay[hotspot_pixels], 0.6,
                merged_mask[hotspot_pixels], 0.4,
                0
            )
            
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Merged Hotspots: {len(self.vlm_color_map)} VLM answers"
            cv2.putText(overlay, text, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Convert to ROS Image message
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
            self._set_msg_stamp_from_float(overlay_msg, timestamp)
            overlay_msg.header.frame_id = 'camera_link'
            
            self.hotspot_overlay_pub.publish(overlay_msg)
            
        except Exception as e:
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().error(f"Error publishing merged overlay image: {e}")


class SemanticHotspotSubscriber:
    """Subscriber for receiving binary hotspot masks in voxel mapper."""
    
    def __init__(self, node, voxel_helper, config: Dict[str, Any] = None):
        self.node = node
        self.voxel_helper = voxel_helper
        self.bridge = CvBridge()
        
        # Load configuration 
        self.config = config.get('semantic_bridge', {}) if config else {}
        self.enable_semantic = self.config.get('enable_semantic_mapping', True)
        
        # Subscribe to hotspots
        if self.enable_semantic:
            self.node.create_subscription(
                String, '/semantic_hotspots', 
                self.hotspot_callback, 10
            )
            
            if hasattr(self.node, 'get_logger'):
                self.node.get_logger().info("Subscribed to semantic hotspots")
    