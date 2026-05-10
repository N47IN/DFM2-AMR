#!/usr/bin/env python3
"""
Resilience Main Node - Clean NARadio Pipeline

Simplified node focused on drift detection, NARadio processing, and semantic mapping.
Removed YOLO/SAM and historical analysis components for lightweight operation.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import tf2_ros
from cv_bridge import CvBridge
import numpy as np
import json
import threading
import time
import cv2
import warnings
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
warnings.filterwarnings('ignore')

from resilience.path_manager import PathManager
from resilience.naradio_processor import NARadioProcessor
from resilience.narration_manager import NarrationManager
from resilience.risk_buffer import RiskBufferManager
from resilience.pointcloud_utils import depth_to_meters as pc_depth_to_meters, depth_mask_to_world_points, voxelize_pointcloud, create_cloud_xyz
from resilience.cause_registry import CauseRegistry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

# Optional RTAB-Map odom info (covariance trace) metric
from rtabmap_msgs.msg import OdomInfo  # type: ignore
RTABMAP_ODOMINFO_AVAILABLE = True


# Hardcoded narration image override (requested)
NARRATION_IMAGE_OVERRIDE_PATH = "/home/navin/ros2_ws/src/resilience/image(2).jpg"


class ResilienceNode(Node):
    """Resilience Node - Clean NARadio pipeline with drift detection and semantic mapping."""
    
    def __init__(self):
        super().__init__('resilience_node')

        # Professional startup message
        self.get_logger().info("=" * 60)
        self.get_logger().info("RESILIENCE SYSTEM INITIALIZING")
        self.get_logger().info("=" * 60)

        # Track recent VLM answers for smart semantic processing
        self.recent_vlm_answers = {}  # vlm_answer -> timestamp
        self.cause_registry = CauseRegistry()
        self.cause_registry_snapshot_path = None
        
        # OPTIMIZATION: Debounce registry snapshot saves to reduce I/O
        self.last_registry_save_time = 0.0
        self.registry_save_debounce_interval = 2.0  # Save at most once every 2 seconds
        
        # FIX: Track causes that have already had narration masks published (by vec_id for canonical identity)
        # Prevents double voxel publishing for the same or similar causes (similarity >0.8)
        # Since cause_registry merges similar causes to the same vec_id, this handles both exact and similar matches
        self.narration_published_vec_ids = set()  # Set of vec_ids that have narration masks published
        
        # Store RGB images with timestamps for hotspot publishing
        self.rgb_images_with_timestamps = []  # [(rgb_msg, timestamp)]
        self.max_rgb_buffer = 50  # Keep last 50 RGB images
        
        self.declare_parameters('', [
            ('flip_y_axis', False),
            ('use_tf', False),
            # If your physical camera is mounted upside-down, set this to 180.
            # Valid values: 0, 180.
            ('image_rotation_deg', 0),
            ('radio_model_version', 'radio_v2.5-b'),
            ('radio_lang_model', 'siglip'),
            ('radio_input_resolution', 512),
            ('enable_naradio_visualization', True),
            ('enable_combined_segmentation', True),
            ('main_config_path', ''),
            ('mapping_config_path', ''),
            ('enable_voxel_mapping', True),
            ('pose_is_base_link', True)
        ])

        param_values = self.get_parameters([
            'flip_y_axis', 'use_tf',
            'image_rotation_deg',
            'radio_model_version', 'radio_lang_model', 'radio_input_resolution',
            'pose_is_base_link',
            'enable_naradio_visualization', 'enable_combined_segmentation',
            'main_config_path', 'mapping_config_path', 'enable_voxel_mapping'
        ])
        
        (self.flip_y_axis, self.use_tf,
         self.image_rotation_deg,
         self.radio_model_version, self.radio_lang_model, self.radio_input_resolution,
         self.pose_is_base_link, self.enable_naradio_visualization, self.enable_combined_segmentation,
         self.main_config_path, self.mapping_config_path, self.enable_voxel_mapping
        ) = [p.value for p in param_values]

        try:
            self.image_rotation_deg = int(self.image_rotation_deg)
        except Exception:
            self.image_rotation_deg = 0
        if self.image_rotation_deg not in (0, 180):
            self.get_logger().warn(f"Invalid image_rotation_deg={self.image_rotation_deg}; using 0")
            self.image_rotation_deg = 0

        # Load topic configuration from main config
        self.load_topic_configuration()
        self.image_rotation_deg = 180

    def _rotate_image_if_needed(self, img: np.ndarray) -> np.ndarray:
        if self.image_rotation_deg == 180 and img is not None:
            return cv2.rotate(img, cv2.ROTATE_180)
        return img

    def load_topic_configuration(self):
        """Load topic configuration from main config file."""
        try:
            import yaml
            if self.main_config_path:
                config_path = self.main_config_path
            else:
                # Use default config path
                from ament_index_python.packages import get_package_share_directory
                package_dir = get_package_share_directory('resilience')
                config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract topic configuration
            topics = config.get('topics', {})
            
            # Input topics
            self.rgb_topic = topics.get('rgb_topic', '/robot_1/sensors/front_stereo/right/image')
            self.depth_topic = topics.get('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered')
            self.pose_topic = topics.get('pose_topic', '/robot_1/sensors/front_stereo/pose')
            self.camera_info_topic = topics.get('camera_info_topic', '/robot_1/sensors/front_stereo/right/camera_info')
            self.vlm_answer_topic = topics.get('vlm_answer_topic', '/vlm_answer')
            
            # Output topics
            self.drift_narration_topic = topics.get('drift_narration_topic', '/drift_narration')
            self.narration_text_topic = topics.get('narration_text_topic', '/narration_text')
            self.naradio_image_topic = topics.get('naradio_image_topic', '/naradio_image')
            self.narration_image_topic = topics.get('narration_image_topic', '/narration_image')
            self.vlm_similarity_map_topic = topics.get('vlm_similarity_map_topic', '/vlm_similarity_map')
            self.vlm_similarity_colored_topic = topics.get('vlm_similarity_colored_topic', '/vlm_similarity_colored')
            self.vlm_objects_legend_topic = topics.get('vlm_objects_legend_topic', '/vlm_objects_legend')

            # Optional image orientation configuration (used only if node parameter left at default)
            camera_cfg = config.get('camera', {}) or {}
            cfg_rot = camera_cfg.get('image_rotation_deg', None)
            if self.image_rotation_deg == 0 and cfg_rot is not None:
                try:
                    cfg_rot_int = int(cfg_rot)
                except Exception:
                    cfg_rot_int = 0
                if cfg_rot_int in (0, 180):
                    self.image_rotation_deg = cfg_rot_int
            
            # Extract path configuration
            self.path_config = config.get('path_mode', {})
            
            self.get_logger().info(f"Topic configuration loaded from: {config_path}")
            self.get_logger().info(f"Path mode: {self.path_config.get('mode', 'json_file')}")
            
        except Exception as e:
            pass
        #     self.get_logger().warn(f"Using default topic configuration: {e}")
        #     # Fallback to default topics
        #     self.rgb_topic = '/robot_1/sensors/front_stereo/right/image'
        #     self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
        #     self.pose_topic = '/robot_1/sensors/front_stereo/pose'
        #     self.camera_info_topic = '/robot_1/sensors/front_stereo/right/camera_info'
        #     self.vlm_answer_topic = '/vlm_answer'
        #     self.drift_narration_topic = '/drift_narration'
        #     self.narration_text_topic = '/narration_text'
        #     self.naradio_image_topic = '/naradio_image'
        #     self.narration_image_topic = '/narration_image'
        #     self.vlm_similarity_map_topic = '/vlm_similarity_map'
        #     self.vlm_similarity_colored_topic = '/vlm_similarity_colored'
        #     self.vlm_objects_legend_topic = '/vlm_objects_legend'
            
        #     # Default path configuration
        #     self.path_config = {'mode': 'json_file', 'global_path_topic': '/global_path'}
        #     self.get_logger().info("Using default topic configuration")

        self.init_components()
        
        self.last_breach_state = False
        self.current_breach_active = False
        self.breach_detection_disabled = False
        # If a breach persists too long without recovering or hitting hard-threshold,
        # end it and disable detection until the metric recovers below soft threshold.
        self.breach_timeout_seconds = 5.0
        self.current_breach_start_time = None  # pose_time when current breach started
        self.pending_narration_after_breach = False  # Flag to trigger narration after breach ends
        self.pending_narration_text = None  # Store narration text for VLM processing

        if self.use_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        else:
            self.tf_broadcaster = None
            self.tf_buffer = None
            self.tf_listener = None

        self.bridge = CvBridge()
        self.camera_intrinsics = [186.24478149414062, 186.24478149414062, 238.66322326660156, 141.6264190673828]
        self.camera_info_received = False

        self.init_publishers()
        self.init_subscriptions()
        self.init_services()

        self.last_rgb_msg = None
        self.last_depth_msg = None
        self.last_pose = None
        self.last_pose_time = None
        self.lock = threading.Lock()
        self.breach_idx = None
        
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_pose = None
        self.latest_pose_time = None
        
        self.naradio_processing_lock = threading.Lock()
        self.naradio_is_processing = False
        self.naradio_running = True
        
        self.detection_pose = None
        self.detection_pose_time = None
        
        # Metric for breach detection + GP fitting: trace(covariance) from /rtabmap/odom_info
        # This replaces drift-based breach triggers while keeping the breach lifecycle logic the same.
        self.odom_info_received = False
        self.latest_cov_trace = None  # float
        self.latest_cov_trace_time = None  # float (sec)
        self.cov_trace_soft_threshold = 2.0
        self.cov_trace_hard_threshold = 7.5
        self._last_cov_trace_log_time = 0.0
        self._last_cov_trace_logged = None

        self.image_buffer = []
        self.max_buffer_size = 250
        self.rolling_image_buffer = []
        self.rolling_buffer_duration = 1.0
        
        # Initial data reception flags for "ping"
        self.rgb_received = False
        self.depth_received = False
        self.pose_received = False
        
        self.transform_matrix_cache = None
        self.last_transform_time = 0
        self.transform_cache_duration = 0.1

        self.init_risk_buffer_manager()
        self.init_semantic_bridge()
        
        # Wait for path to be ready before starting functionality
        self.wait_for_path_ready()
        
        self.start_naradio_thread()
        
        self.print_initialization_status()

        # PointCloud worker thread state (only if direct_mapping is enabled)
    

    def wait_for_path_ready(self):
        """Wait for path to be ready before starting main functionality."""
        self.get_logger().info("Waiting for path to be ready...")
        
        timeout_seconds = 10.0  # Default timeout
        if self.path_manager.get_mode() == 'external_planner':
            timeout_seconds = self.path_config.get('external_planner', {}).get('timeout_seconds', 30.0)
        
        # For external planner mode, check periodically instead of blocking
        if self.path_manager.get_mode() == 'external_planner':
            self.get_logger().info(f"External planner mode: Waiting up to {timeout_seconds}s for path...")            
            
            if self.path_manager.is_ready():
                self.get_logger().info("External path received - starting main functionality")
                self.path_ready = True
                
                # Update narration manager with path points
                nominal_points = self.path_manager.get_nominal_points_as_numpy()
                if len(nominal_points) > 0:
                    self.narration_manager.update_intended_trajectory(nominal_points)
                    self.get_logger().info("Updated narration manager with external path points")
            else:
                self.get_logger().warn("External path not received within timeout")
                self.path_ready = False
                self.disable_drift_detection = True

    def can_proceed_with_drift_detection(self) -> bool:
        """Check if drift detection can proceed."""
        return (hasattr(self, 'path_ready') and 
                self.path_ready and 
                (not hasattr(self, 'disable_drift_detection') or 
                 not self.disable_drift_detection))

    def init_publishers(self):
        """Initialize publishers."""
        publishers = [
            (self.drift_narration_topic, String, 10),
            (self.narration_text_topic, String, 10),
            (self.naradio_image_topic, Image, 10),
            (self.narration_image_topic, Image, 10)
        ]
        
        self.narration_pub, self.narration_text_pub, self.naradio_image_pub, \
        self.narration_image_pub = [self.create_publisher(msg_type, topic, qos) 
                                   for topic, msg_type, qos in publishers]
        
        if self.enable_combined_segmentation:
            vlm_publishers = [
                (self.vlm_similarity_map_topic, Image, 10),
                (self.vlm_similarity_colored_topic, Image, 10),
                (self.vlm_objects_legend_topic, String, 10)
            ]
            self.original_mask_pub, self.refined_mask_pub, self.segmentation_legend_pub = \
                [self.create_publisher(msg_type, topic, qos) for topic, msg_type, qos in vlm_publishers]

    def init_subscriptions(self):
        """Initialize subscriptions."""
        subscriptions = [
            (self.rgb_topic, Image, self.rgb_callback, 1),
            (self.depth_topic, Image, self.depth_callback, 1),
            (self.pose_topic, Odometry, self.pose_callback, 10),
            (self.camera_info_topic, CameraInfo, self.camera_info_callback, 1)
            # ,
            # (self.vlm_answer_topic, String, self.vlm_answer_callback, 10)
        ]
        
        for topic, msg_type, callback, qos in subscriptions:
            self.create_subscription(msg_type, topic, callback, qos)

        # Covariance trace metric source (optional import)
        if RTABMAP_ODOMINFO_AVAILABLE:
            self.create_subscription(OdomInfo, '/rtabmap/odom_info', self.odom_info_callback, 10)
        else:
            self.get_logger().warn(
                "rtabmap_msgs/OdomInfo not available; cov_trace breach detection disabled "
                "(install rtabmap_msgs or source the appropriate workspace)."
            )
    
    def init_services(self):
        """Initialize ROS services for registry access (JSON over String messages)."""
        self.registry_query_sub = self.create_subscription(
            String,
            '/cause_registry/query',
            self._handle_registry_query,
            10
        )
        
        self.registry_response_pub = self.create_publisher(
            String,
            '/cause_registry/response',
            10
        )
        
        self.get_logger().info("Cause registry services initialized (topic-based)")

    def odom_info_callback(self, msg):
        """Track trace(covariance) from RTAB-Map for breach detection + GP fitting."""
        try:
            # Initial ping
            if not self.odom_info_received:
                self.odom_info_received = True
                self.get_logger().info("✓ PING: /rtabmap/odom_info received (cov_trace enabled)")

            cov = np.array(msg.covariance, dtype=float)
            if cov.size != 36:
                return
            cov_trace = float(np.trace(cov.reshape(6, 6)))
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            self.latest_cov_trace = cov_trace
            self.latest_cov_trace_time = t

        except Exception:
            # Best-effort; never break pipeline due to metric parsing
            return
    
    def _handle_registry_query(self, msg):
        """Handle registry query via JSON message."""
        try:
            query = json.loads(msg.data)
            query_type = query.get('type')
            query_id = query.get('query_id')  # For async callback routing
            response_data = {'success': False, 'message': 'Unknown query type'}
            
            if query_id:
                response_data['query_id'] = query_id
            
            if query_type == 'get_by_name':
                name = query.get('name')
                entry = self.cause_registry.get_entry_by_name(name) if name else None
                if entry:
                    response_data = {
                        'success': True,
                        'vec_id': entry.vec_id,
                        'names': entry.names,
                        'color_rgb': entry.color_rgb,
                        'embedding': entry.embedding.tolist(),
                        'enhanced_embedding': entry.enhanced_embedding.tolist() if entry.enhanced_embedding is not None else None,
                        'gp_params': self._gp_params_to_dict(entry.gp_params),
                        'metadata': entry.metadata,
                        'stats': entry.stats
                    }
                    if query_id:
                        response_data['query_id'] = query_id
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                    if query_id:
                        response_data['query_id'] = query_id
            
            elif query_type == 'get_by_vec_id':
                vec_id = query.get('vec_id')
                entry = self.cause_registry.get_entry_by_vec_id(vec_id) if vec_id else None
                if entry:
                    response_data = {
                        'success': True,
                        'vec_id': entry.vec_id,
                        'names': entry.names,
                        'color_rgb': entry.color_rgb,
                        'embedding': entry.embedding.tolist(),
                        'enhanced_embedding': entry.enhanced_embedding.tolist() if entry.enhanced_embedding is not None else None,
                        'gp_params': self._gp_params_to_dict(entry.gp_params),
                        'metadata': entry.metadata,
                        'stats': entry.stats
                    }
                    if query_id:
                        response_data['query_id'] = query_id
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                    if query_id:
                        response_data['query_id'] = query_id
            
            elif query_type == 'set_gp':
                name = query.get('name')
                vec_id = query.get('vec_id')
                gp_data = query.get('gp_params', {})
                
                entry = None
                if vec_id:
                    # Prefer vec_id (embedding-indexed) over name
                    entry = self.cause_registry.get_entry_by_vec_id(vec_id)
                elif name:
                    entry = self.cause_registry.get_entry_by_name(name)
                
                if entry:
                    from resilience.cause_registry import GPParams
                    gp_params = GPParams(
                        lxy=gp_data.get('lxy'),
                        lz=gp_data.get('lz'),
                        A=gp_data.get('A'),
                        b=gp_data.get('b'),
                        mse=gp_data.get('mse'),
                        rmse=gp_data.get('rmse'),
                        mae=gp_data.get('mae'),
                        r2_score=gp_data.get('r2_score'),
                        timestamp=gp_data.get('timestamp'),
                        buffer_id=gp_data.get('buffer_id')
                    )
                    # Use vec_id directly if available, otherwise fall back to name
                    if vec_id:
                        # Update via vec_id (embedding-indexed)
                        success = self.cause_registry.set_gp_params(
                            entry.names[0] if entry.names else '',
                            gp_params
                        )
                    else:
                        # Fallback to name lookup
                        success = self.cause_registry.set_gp_params(
                            name or (entry.names[0] if entry.names else ''),
                            gp_params
                        )
                    if success:
                        self.save_cause_registry_snapshot()
                        response_data = {'success': True, 'message': 'GP params updated'}
                    else:
                        response_data = {'success': False, 'message': 'Failed to update GP params'}
                else:
                    response_data = {'success': False, 'message': 'Entry not found'}
                
                if query_id:
                    response_data['query_id'] = query_id
            
            # Publish response
            response_msg = String(data=json.dumps(response_data))
            self.registry_response_pub.publish(response_msg)
            
        except Exception as e:
            error_response = {'success': False, 'message': f'Error: {str(e)}'}
            self.registry_response_pub.publish(String(data=json.dumps(error_response)))
    
    def _gp_params_to_dict(self, gp_params):
        """Convert GPParams to dict for JSON serialization."""
        if gp_params is None:
            return None
        return {
            'lxy': gp_params.lxy,
            'lz': gp_params.lz,
            'A': gp_params.A,
            'b': gp_params.b,
            'mse': gp_params.mse,
            'rmse': gp_params.rmse,
            'mae': gp_params.mae,
            'r2_score': gp_params.r2_score,
            'timestamp': gp_params.timestamp,
            'buffer_id': gp_params.buffer_id
        }

    def print_initialization_status(self):
        """Print initialization status."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("RESILIENCE SYSTEM READY")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"Path Configuration:")
        self.get_logger().info(f"   Mode: {self.path_manager.get_mode()}")
        self.get_logger().info(f"   Topic: {self.path_manager.get_path_topic()}")
        self.get_logger().info(f"   Status: {'READY' if hasattr(self, 'path_ready') and self.path_ready else 'NOT READY'}")
        
        if hasattr(self, 'disable_drift_detection'):
            self.get_logger().info(f"Drift Detection: {'ENABLED' if not self.disable_drift_detection else 'DISABLED'}")
        
        self.get_logger().info(
            f"Breach Metric: cov_trace from /rtabmap/odom_info "
            f"(soft={float(self.cov_trace_soft_threshold):.2f}, hard={float(self.cov_trace_hard_threshold):.2f})"
        )
        
        self.get_logger().info(f"NARadio Processing: {'READY' if self.naradio_processor.is_ready() else 'NOT READY'}")
        self.get_logger().info(f"Voxel Mapping: {'ENABLED' if self.enable_voxel_mapping else 'DISABLED'}")
        
        vlm_enabled = (self.enable_combined_segmentation and 
                      hasattr(self, 'naradio_processor') and 
                      self.naradio_processor.is_segmentation_ready())
        self.get_logger().info(f"VLM Similarity: {'ENABLED' if vlm_enabled else 'DISABLED'}")
        
        if vlm_enabled:
            all_objects = self.naradio_processor.get_all_objects()
            self.get_logger().info(f"   Objects loaded: {len(all_objects)}")
            
        config = self.naradio_processor.segmentation_config
        prefer_enhanced = config['segmentation'].get('prefer_enhanced_embeddings', True)
        self.get_logger().info(f"Embedding Method: {'ENHANCED' if prefer_enhanced else 'TEXT'}")
        
        self.get_logger().info("=" * 60)


    def init_components(self):
        """Initialize resilience components."""
        self.get_logger().info("Initializing system components...")
        
        # Initialize path manager with unified interface
        self.path_manager = PathManager(self, self.path_config)
        self.get_logger().info("Path Manager initialized")
        
        # Get thresholds from path manager
        soft_threshold, hard_threshold = self.path_manager.get_thresholds()
        
        # Wait for path to be ready and print discretization results
        if self.path_manager.wait_for_path(timeout_seconds=2.0):
            discretized_points = self.path_manager.get_discretized_nominal_points()
            self.get_logger().info(f"Path loaded: {len(discretized_points)} points, {self.path_manager.get_sampling_distance():.3f}m sampling")
        else:
            self.get_logger().warn("Path not ready within timeout - will retry during operation")
        
        try:
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None,
                cause_registry=self.cause_registry
            )
            
            # Read voxel mapping parameters from main config (non-blocking)
            self.enable_voxel_mapping = False  # Default value
            self.direct_mapping = False  # Default value
            if (self.naradio_processor.is_ready() and 
                hasattr(self.naradio_processor, 'segmentation_config')):
                try:
                    self.enable_voxel_mapping = self.naradio_processor.segmentation_config.get('enable_voxel_mapping', False)
                    self.direct_mapping = self.naradio_processor.segmentation_config.get('direct_mapping', False)
                except Exception as e:
                    self.get_logger().warn(f"Could not read voxel mapping parameters from config: {e}")
                    self.enable_voxel_mapping = False
                    self.direct_mapping = False
            else:
                self.get_logger().warn("NARadio processor not ready, using default voxel mapping: False, direct mapping: False")
                
        except Exception as e:
            self.get_logger().error(f"Error initializing NARadio processor: {e}")
            import traceback
            traceback.print_exc()
            self.naradio_processor = NARadioProcessor(
                radio_model_version=self.radio_model_version,
                radio_lang_model=self.radio_lang_model,
                radio_input_resolution=self.radio_input_resolution,
                enable_visualization=self.enable_naradio_visualization,
                enable_combined_segmentation=self.enable_combined_segmentation,
                segmentation_config_path=self.main_config_path if self.main_config_path else None,
                cause_registry=self.cause_registry
            )
        
        # Initialize narration manager with discretization parameters
        lookback_window_size = self.path_manager.get_lookback_window_size()
        sampling_distance = self.path_manager.get_sampling_distance()
        self.narration_manager = NarrationManager(
            soft_threshold, 
            hard_threshold, 
            lookback_window_size=lookback_window_size,
            sampling_distance=sampling_distance
        )
        self.get_logger().info("Narration Manager initialized")
        
        # Ensure voxel mapping parameters are always set (final fallback)
        if not hasattr(self, 'enable_voxel_mapping'):
            self.enable_voxel_mapping = False
            self.get_logger().info("Voxel mapping parameter not set, using default: False")
        if not hasattr(self, 'direct_mapping'):
            self.direct_mapping = False
            self.get_logger().info("Direct mapping parameter not set, using default: False")
        
        # Set nominal trajectory points if available (use discretized data)
        nominal_points = self.path_manager.get_discretized_nominal_as_numpy()
        if len(nominal_points) > 0:
            self.narration_manager.set_intended_trajectory(nominal_points)
            self.get_logger().info(f"Narration manager initialized with {len(nominal_points)} discretized points")
        else:
            self.get_logger().warn("No discretized nominal points available for narration manager")
        
        # Load pre-defined objects with GP parameters
        self.load_predefined_objects()
    
    def init_risk_buffer_manager(self):
        """Initialize risk buffer manager."""
        try:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            unique_id = str(uuid.uuid4())[:8]
            self.risk_buffer_save_dir = '/home/navin/ros2_ws/src/resilience/buffers'
            os.makedirs(self.risk_buffer_save_dir, exist_ok=True)
            
            self.current_run_dir = os.path.join(self.risk_buffer_save_dir, f"run_{run_timestamp}_{unique_id}")
            os.makedirs(self.current_run_dir, exist_ok=True)
            
            self.risk_buffer_manager = RiskBufferManager(save_directory=self.current_run_dir)
            print(f"Buffer directory: {self.current_run_dir}")
            
            self.node_id = f"resilience_{unique_id}"
            self.cause_registry_snapshot_path = os.path.join(self.current_run_dir, "cause_registry.json")
            self.save_cause_registry_snapshot(force=True)  # Force initial save
            
            # Directory and timing for periodic RGB snapshots (in RGB space)
            # self.rgb_snapshot_dir = os.path.join(self.current_run_dir, "rgb_snapshots")
            # os.makedirs(self.rgb_snapshot_dir, exist_ok=True)
            self.rgb_snapshot_interval = 4.0  # seconds
            self.last_rgb_snapshot_time = 0.0
            
        except Exception as e:
            print(f"Error initializing risk buffer manager: {e}")
            self.risk_buffer_manager = None
    
    def init_semantic_bridge(self):
        """Initialize semantic hotspot bridge for communication with octomap."""
        try:
            if self.enable_voxel_mapping:
                # Load semantic bridge config from main config
                main_config = getattr(self.naradio_processor, 'segmentation_config', {})
                
                from resilience.semantic_info_bridge import SemanticHotspotPublisher
                self.semantic_bridge = SemanticHotspotPublisher(self, main_config)
                print("Semantic bridge initialized")
            else:
                self.semantic_bridge = None
                print("Semantic bridge disabled")
        except Exception as e:
            print(f"Error initializing semantic bridge: {e}")
            self.semantic_bridge = None
    
    def start_naradio_thread(self):
        """Start the parallel NARadio processing thread."""
        if not hasattr(self, 'naradio_thread') or self.naradio_thread is None or not self.naradio_thread.is_alive():
            self.naradio_running = True
            self.naradio_thread = threading.Thread(target=self.naradio_processing_loop, daemon=True)
            self.naradio_thread.start()
            print("NARadio processing thread started")
        else:
            print("NARadio thread already running")

    def rgb_callback(self, msg):
        """Store RGB message with timestamp for hotspot publishing and sync buffers."""
        # Initial ping
        if not self.rgb_received:
            self.rgb_received = True
            self.get_logger().info("✓ PING: RGB image received")
        
        msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception:
            return
        cv_image = self._rotate_image_if_needed(cv_image)
        # Store RGB image with timestamp for hotspot publishing
        self.rgb_images_with_timestamps.append((msg, msg_timestamp))
        if len(self.rgb_images_with_timestamps) > self.max_rgb_buffer:
            self.rgb_images_with_timestamps.pop(0)

        
        with self.processing_lock:
            self.latest_rgb_msg = msg
            if self.latest_pose is not None:
                self.detection_pose = self.latest_pose.copy()
                self.detection_pose_time = self.latest_pose_time
        
        self.image_buffer.append((cv_image, msg_timestamp, msg))
        
        if len(self.image_buffer) > self.max_buffer_size:
            self.image_buffer.pop(0)
        
        current_system_time = time.time()
        self.rolling_image_buffer.append((cv_image, current_system_time, msg))
        
        while self.rolling_image_buffer and (current_system_time - self.rolling_image_buffer[0][1]) > self.rolling_buffer_duration:
            self.rolling_image_buffer.pop(0)
        
        # Periodically save an RGB snapshot for offline inspection
        # try:
        #     if hasattr(self, "rgb_snapshot_dir") and hasattr(self, "rgb_snapshot_interval"):
        #         if not hasattr(self, "last_rgb_snapshot_time"):
        #             self.last_rgb_snapshot_time = 0.0
        #         if current_system_time - self.last_rgb_snapshot_time >= self.rgb_snapshot_interval:
        #             self.last_rgb_snapshot_time = current_system_time
        #             ts_str = datetime.fromtimestamp(current_system_time).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        #             filename = f"rgb_snapshot_{ts_str}.png"
        #             save_path = os.path.join(self.rgb_snapshot_dir, filename)
        #             # cv_image is in RGB; we persist it as-is so on-disk representation is RGB.
        #             cv2.imwrite(save_path, cv_image)
        # except Exception:
        #     # Snapshot saving is best-effort and should never break the main pipeline
        #     pass
        
    def depth_callback(self, msg):
        """Store latest depth message and push into depth buffer."""
        # Initial ping
        if not self.depth_received:
            self.depth_received = True
            self.get_logger().info("✓ PING: Depth image received")
        
        with self.processing_lock:
            self.latest_depth_msg = msg
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_img = self._rotate_image_if_needed(depth_img)
            depth_m = pc_depth_to_meters(depth_img, msg.encoding)
        except Exception:
            return
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def pose_callback(self, msg: Odometry):
        """Process pose and trigger detection with consolidated pose updates."""
        # Initial ping
        if not self.pose_received:
            self.pose_received = True
            self.get_logger().info("✓ PING: Pose received")
        
        # Always compute path drift (used for indexing / visualization), even if breach detection is disabled
        pose = msg.pose.pose
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        # Build full 6D pose vector [x, y, z, qx, qy, qz, qw] for buffers / poses.npy
        orientation = pose.orientation
        pose_6d = np.array([
            pos[0], pos[1], pos[2],
            orientation.x, orientation.y, orientation.z, orientation.w
        ], dtype=float)
        pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        drift = 0.0
        nearest_idx = -1
        if self.path_manager.is_ready():
            drift, nearest_idx = self.path_manager.compute_drift(pos)
        
        # Check if path manager is ready
        if not self.path_manager.is_ready():
            print("Path manager not ready, skipping pose processing")
            return
        
        self.breach_idx = nearest_idx
        
        with self.lock:
            self.latest_pose = pos
            self.latest_pose_time = pose_time
            self.last_pose = pos
            self.last_pose_time = pose_time
            
            self.narration_manager.add_actual_point(pos, pose_time, self.flip_y_axis)

        # Post-breach narration must run even if cov_trace is temporarily unavailable
        # (same as pre-change behavior, where pose always carried drift).
        if self.pending_narration_after_breach and not self.current_breach_active:
            self.pending_narration_after_breach = False
            narration_text = self.pending_narration_text
            self.pending_narration_text = None

            self.get_logger().info("Triggering narration processing after breach end")
            if narration_text:
                self.vlm_answer_callback()
            else:
                self.get_logger().warn("No narration text available for VLM processing")

        metric = self.latest_cov_trace
        soft_threshold = float(self.cov_trace_soft_threshold)
        hard_threshold = float(self.cov_trace_hard_threshold)

        if metric is None:
            # No odom_info yet: do not run breach state machine (avoids false transitions).
            if (pose_time - float(self._last_cov_trace_log_time)) > 2.0:
                self._last_cov_trace_log_time = float(pose_time)
                self.get_logger().warn("Waiting for /rtabmap/odom_info cov_trace; breach detection not running yet.")
            return

        metric = float(metric)

        # Lightweight metric logging (rate-limited, plus extra signal during breaches)
        if (pose_time - float(self._last_cov_trace_log_time)) > 1.0:
            self._last_cov_trace_log_time = float(pose_time)
            if self._last_cov_trace_logged is None or abs(metric - float(self._last_cov_trace_logged)) > 0.05:
                self._last_cov_trace_logged = float(metric)
                self.get_logger().info(
                    f"cov_trace={metric:.3f} (soft={soft_threshold:.2f}, hard={hard_threshold:.2f})"
                )

        # Check for hard threshold breach (ends breach and disables detection)
        hard_breach_now = metric > hard_threshold
        
        # Re-enable breach detection if drift has recovered below soft threshold
        if self.breach_detection_disabled and metric < soft_threshold:
            self.breach_detection_disabled = False
            self.get_logger().info(
                f"BREACH DETECTION RE-ENABLED - cov_trace={metric:.3f} < soft threshold: {soft_threshold:.2f}"
            )
        
        # Check if we can proceed with drift detection
        can_proceed = self.can_proceed_with_drift_detection() and not self.breach_detection_disabled
        
        # Always compute breach state for ending detection, even if detection is disabled
        breach_now = (metric > soft_threshold) if can_proceed else False
        
        # Breach ends if:
        #  1) metric < soft_threshold (recovered), OR
        #  2) metric > hard_threshold (hard threshold), OR
        #  3) breach persists > breach_timeout_seconds without meeting (1) or (2)
        # Check ending even when detection is disabled (to handle existing breaches)
        breach_ended = False
        breach_ended_timeout = False
        if self.last_breach_state:
            if hard_breach_now:
                # Case 2: End due to hard threshold
                breach_ended = True
            elif metric < soft_threshold:
                # Case 1: End due to recovery (check metric directly, works even when detection disabled)
                breach_ended = True
            else:
                # Case 3: End due to timeout (still between soft and hard)
                try:
                    if self.current_breach_start_time is not None:
                        dt_breach = float(pose_time) - float(self.current_breach_start_time)
                        if dt_breach > float(self.breach_timeout_seconds):
                            breach_ended = True
                            breach_ended_timeout = True
                except Exception:
                    pass
        
        # Only detect new breaches if detection is enabled
        breach_started = can_proceed and not self.last_breach_state and breach_now
        
        if breach_started:
            self.last_breach_state = True
            self.current_breach_active = True
            self.current_breach_start_time = pose_time
            self.narration_manager.reset_narration_state()
            
            self.get_logger().warn(
                f"BREACH STARTED - cov_trace={metric:.3f} (> soft={soft_threshold:.2f}, hard={hard_threshold:.2f}); "
                f"drift={drift:.3f}m"
            )
            
            # Start new buffer when breach begins
            if self.risk_buffer_manager:
                buffer = self.risk_buffer_manager.start_buffer(pose_time)
                
                # CRITICAL: Store the nominal path that was active at breach start
                # This ensures GP computation compares observed poses during breach
                # against the correct predicted (nominal) path, not a path that may
                # have changed after the breach started
                if buffer and self.path_manager.is_ready():
                    try:
                        # Get the discretized nominal path (preferred for GP computation)
                        nominal_path_xyz = self.path_manager.get_discretized_nominal_as_numpy()
                        if nominal_path_xyz is not None and len(nominal_path_xyz) > 0:
                            buffer.set_nominal_path(nominal_path_xyz)
                            self.get_logger().info(f"Stored nominal path at breach start ({len(nominal_path_xyz)} points)")
                        else:
                            # Fallback to non-discretized path
                            nominal_path_xyz = self.path_manager.get_nominal_points_as_numpy()
                            if nominal_path_xyz is not None and len(nominal_path_xyz) > 0:
                                buffer.set_nominal_path(nominal_path_xyz)
                                self.get_logger().info(f"Stored nominal path at breach start (fallback, {len(nominal_path_xyz)} points)")
                            else:
                                self.get_logger().warn("No nominal path available to store at breach start")
                    except Exception as e:
                        self.get_logger().warn(f"Failed to store nominal path at breach start: {e}")
            
            self.narration_manager.queue_breach_event('start', pose_time)
            
        elif breach_ended:
            self.last_breach_state = False
            self.current_breach_active = False
            self.current_breach_start_time = None
            
            # Determine why breach ended
            if hard_breach_now:
                # Case 2: Breach ended due to hard threshold - disable detection
                self.breach_detection_disabled = True
                self.get_logger().warn(
                    f"BREACH ENDED (HARD THRESHOLD) - cov_trace={metric:.3f} > hard threshold: {hard_threshold:.2f}. "
                    f"Breach detection DISABLED until cov_trace < {soft_threshold:.2f} "
                    f"(drift={drift:.3f}m)"
                )
            elif breach_ended_timeout:
                # Case 3: Breach ended due to timeout - disable detection until recovery
                self.breach_detection_disabled = True
                self.get_logger().warn(
                    f"BREACH ENDED (TIMEOUT) - dt>{float(self.breach_timeout_seconds):.1f}s with cov_trace still between thresholds "
                    f"(cov_trace={metric:.3f}, soft={soft_threshold:.2f}, hard={hard_threshold:.2f}; drift={drift:.3f}m). "
                    f"Breach detection DISABLED until cov_trace < {soft_threshold:.2f}"
                )
            else:
                # Case 1: Breach ended due to drift < soft threshold
                self.get_logger().info(
                    f"BREACH ENDED (RECOVERED) - cov_trace={metric:.3f} < soft threshold: {soft_threshold:.2f} "
                    f"(drift={drift:.3f}m)"
                )
            
            # Get narration BEFORE freezing buffers (so we can save to active buffers)
            narration = None
            if self.narration_manager:
                narration = self.narration_manager.check_for_narration(pose_time, self.breach_idx)
            
            # Save narration image to buffer BEFORE freezing (buffers are still active)
            # Get breach start time from the active buffer
            breach_start_time = None
            if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0:
                breach_start_time = self.risk_buffer_manager.active_buffers[-1].start_time
            
            if narration and self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0:
                self.publish_narration_with_image(narration, breach_start_time=breach_start_time)
                self.get_logger().info(f"Saved narration image to buffer before freezing (breach started at {breach_start_time:.3f})")
            
            # Freeze buffers when breach ends
            if self.risk_buffer_manager:
                frozen_buffers = self.risk_buffer_manager.freeze_active_buffers(pose_time)
            
            self.narration_manager.queue_breach_event('end', pose_time)
            
            # Trigger narration processing after breach ends (for both recovery and hard threshold cases)
            if narration:
                self.pending_narration_after_breach = True
                self.pending_narration_text = narration  # Store narration text for VLM processing
                self.get_logger().info("Narration will be triggered after breach end")
            else:
                self.get_logger().warn("No narration generated for breach end")
            
        elif can_proceed and breach_now and not self.current_breach_active:
            self.last_breach_state = True
            self.current_breach_active = True
            self.current_breach_start_time = pose_time
            self.narration_manager.reset_narration_state()
            
            self.get_logger().warn(
                f"BREACH DETECTED - cov_trace={metric:.3f} (> soft={soft_threshold:.2f}); drift={drift:.3f}m"
            )
            
            # Start new buffer when breach is detected
            if self.risk_buffer_manager:
                buffer = self.risk_buffer_manager.start_buffer(pose_time)
                
                # CRITICAL: Store the nominal path that was active at breach start
                # This ensures GP computation compares observed poses during breach
                # against the correct predicted (nominal) path, not a path that may
                # have changed after the breach started
                if buffer and self.path_manager.is_ready():
                    try:
                        # Get the discretized nominal path (preferred for GP computation)
                        nominal_path_xyz = self.path_manager.get_discretized_nominal_as_numpy()
                        if nominal_path_xyz is not None and len(nominal_path_xyz) > 0:
                            buffer.set_nominal_path(nominal_path_xyz)
                            self.get_logger().info(f"Stored nominal path at breach start ({len(nominal_path_xyz)} points)")
                        else:
                            # Fallback to non-discretized path
                            nominal_path_xyz = self.path_manager.get_nominal_points_as_numpy()
                            if nominal_path_xyz is not None and len(nominal_path_xyz) > 0:
                                buffer.set_nominal_path(nominal_path_xyz)
                                self.get_logger().info(f"Stored nominal path at breach start (fallback, {len(nominal_path_xyz)} points)")
                            else:
                                self.get_logger().warn("No nominal path available to store at breach start")
                    except Exception as e:
                        self.get_logger().warn(f"Failed to store nominal path at breach start: {e}")
            
            self.narration_manager.queue_breach_event('start', pose_time)
        
        if not breach_started and not breach_ended and not (can_proceed and breach_now and not self.current_breach_active):
            if can_proceed:
                self.last_breach_state = breach_now
        
        with self.lock:
            if self.risk_buffer_manager and len(self.risk_buffer_manager.active_buffers) > 0 and self.current_breach_active:
                # Store full 6D pose (position + orientation) in buffers so poses.npy has proper pose
                # IMPORTANT: last column in poses.npy remains the "metric" consumed by downstream GP tooling.
                # We now store cov_trace here (not drift).
                self.risk_buffer_manager.add_pose(pose_time, pose_6d, metric)

        # Early return if detection is disabled (after processing breach ending and narration)
        if not can_proceed:
            return

    def publish_narration_with_image(self, narration_text, breach_start_time=None):
        """
        Publish both narration text and accompanying image together.
        
        Args:
            narration_text: The narration text to publish
            breach_start_time: The breach start time. If provided, image will be selected from 1s before breach start.
                              If None, falls back to 1s before current time.
        """
        if not self.image_buffer:
            self.narration_text_pub.publish(String(data=narration_text))
            return
        
        target_time_offset = 1
        
        # Use breach start time if provided, otherwise use current time
        if breach_start_time is not None:
            # Get image from 1 second before breach start
            target_time = breach_start_time - target_time_offset
            self.get_logger().info(f"Selecting image from 1s before breach start: target_time={target_time:.3f} (breach_start={breach_start_time:.3f})")
            
            # Check if we have images old enough (need to have images before breach start)
            oldest_timestamp = self.image_buffer[0][1] if self.image_buffer else None
            if oldest_timestamp is not None and oldest_timestamp > target_time:
                # Don't have images old enough, use oldest available
                target_time = oldest_timestamp
                actual_offset = breach_start_time - oldest_timestamp
                self.get_logger().warn(f"Not enough history: using oldest image at {oldest_timestamp:.3f} (offset: {actual_offset:.3f}s)")
            else:
                actual_offset = target_time_offset
        else:
            # Fallback: use current time (old behavior)
            newest_timestamp = self.image_buffer[-1][1]
            current_time = newest_timestamp
            target_time = current_time - target_time_offset
            
            oldest_timestamp = self.image_buffer[0][1] if self.image_buffer else current_time
            available_time_back = current_time - oldest_timestamp
            
            if available_time_back < target_time_offset:
                target_time = oldest_timestamp
                actual_offset = available_time_back
            else:
                actual_offset = target_time_offset
            self.get_logger().warn("No breach_start_time provided, using current time - 1s")
        
        if self.image_buffer:
            closest_image = None
            closest_msg = None
            min_time_diff = float('inf')
            
            for image, timestamp, msg in self.image_buffer:
                time_diff = abs(timestamp - target_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_image = image
                    closest_msg = msg
            
            if closest_image is not None and closest_msg is not None:
                # Get the original image timestamp from the message
                original_image_timestamp = closest_msg.header.stamp.sec + closest_msg.header.stamp.nanosec * 1e-9

                # Requested: whenever narration happens, use a fixed image for hotspot extraction and downstream.
                # We also store this overridden image into the buffer.
                # try:
                #     override_bgr = cv2.imread(NARRATION_IMAGE_OVERRIDE_PATH)
                #     if override_bgr is not None:
                #         closest_image = cv2.cvtColor(override_bgr, cv2.COLOR_BGR2RGB)
                #         self.get_logger().warn(
                #             f"[Narration] Overriding narration image with '{NARRATION_IMAGE_OVERRIDE_PATH}' "
                #             f"(shape={closest_image.shape})"
                #         )
                #     else:
                #         self.get_logger().warn(
                #             f"[Narration] Failed to read override image '{NARRATION_IMAGE_OVERRIDE_PATH}', "
                #             "falling back to buffered camera image"
                #         )
                # except Exception as e:
                #     self.get_logger().warn(
                #         f"[Narration] Override image load failed ({e}); falling back to buffered camera image"
                #     )
                
                # Use current time (breach end time) for narration timestamp
                # Get from latest_pose_time if available, otherwise use current system time
                narration_timestamp = time.time()
                if hasattr(self, 'latest_pose_time') and self.latest_pose_time is not None:
                    narration_timestamp = self.latest_pose_time
                
                # Save narration image (with trajectory overlay) to buffer
                self.save_narration_image_to_buffer(
                    closest_image, narration_text, narration_timestamp, original_image_timestamp
                )
                
                # Store narration data in active buffers for VLM processing
                # Include the original image timestamp for proper semantic mapping
                if self.risk_buffer_manager:
                    self.risk_buffer_manager.store_narration_data_with_timestamp(
                        closest_image, narration_text, narration_timestamp, original_image_timestamp
                    )
                
                image_msg = self.bridge.cv2_to_imgmsg(closest_image, encoding='rgb8')
                image_msg.header.stamp = closest_msg.header.stamp
                image_msg.header.frame_id = closest_msg.header.frame_id
                # self.narration_image_pub.publish(image_msg)
                
                # self.narration_text_pub.publish(String(data=narration_text))
        #     else:
        #         self.narration_text_pub.publish(String(data=narration_text))
        # else:
        #     self.narration_text_pub.publish(String(data=narration_text))

    def load_enhanced_embedding_from_buffer(self, buffer_dir: str, vlm_answer: str) -> Optional[np.ndarray]:
        """Load enhanced embedding from buffer directory."""
        try:
            embeddings_dir = os.path.join(buffer_dir, 'enhanced_embeddings')
            if not os.path.exists(embeddings_dir):
                return None
            
            # Look for enhanced embedding files for this VLM answer
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            embedding_files = [f for f in os.listdir(embeddings_dir) 
                             if f.startswith(f"enhanced_embedding_{safe_vlm_name}") and f.endswith('.npy')]
            
            if not embedding_files:
                return None
            
            # Get the most recent embedding file
            embedding_files.sort()
            latest_embedding_file = embedding_files[-1]
            embedding_path = os.path.join(embeddings_dir, latest_embedding_file)
            
            # Load the enhanced embedding
            enhanced_embedding = np.load(embedding_path)
            
            return enhanced_embedding
            
        except Exception as e:
            print(f"Error loading enhanced embedding: {e}")
            return None
    
    def _project_points_world_to_image(self, points_world: np.ndarray, camera_pos: np.ndarray,
                                       image_shape: tuple) -> Optional[tuple]:
        """
        Project 3D world points onto the narration RGB image using a simple pinhole model.
        
        Coordinate convention (path + odom):
          - World:  X = forward, Y = left, Z = up
          - Camera: X_cam = right, Y_cam = down, Z_cam = forward
        
        We assume the camera optical axis is aligned with the robot forward axis and
        use translation only (no roll/pitch/yaw), which is sufficient for a
        qualitative trajectory overlay.
        """
        try:
            if self.camera_intrinsics is None or len(self.camera_intrinsics) < 4:
                return None
            
            fx, fy, cx, cy = self.camera_intrinsics
            fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
            
            h, w = image_shape[:2]
            if points_world is None or len(points_world) == 0:
                return None
            
            pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
            cam_pos = np.asarray(camera_pos, dtype=np.float32).reshape(1, 3)
            
            rel = pts - cam_pos  # world → camera origin
            # Map world axes (forward=X, left=Y, up=Z) to camera axes (right, down, forward)
            X_cam = -rel[:, 1]          # right  (+) is world -Y
            Y_cam = -rel[:, 2]          # down   (+) is world -Z
            Z_cam = rel[:, 0]           # forward (+) is world +X
            
            # Keep only points in front of camera
            eps = 1e-3
            valid = Z_cam > eps
            if not np.any(valid):
                return None
            
            X_cam = X_cam[valid]
            Y_cam = Y_cam[valid]
            Z_cam = Z_cam[valid]
            
            u = fx * (X_cam / Z_cam) + cx
            v = fy * (Y_cam / Z_cam) + cy
            
            # In-frame filtering
            in_frame = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(in_frame):
                return None
            
            u = u[in_frame]
            v = v[in_frame]
            depths = Z_cam[in_frame]  # use forward distance as depth metric
            
            pts_2d = np.stack([u, v], axis=1)
            return pts_2d, depths
        except Exception:
            return None

    def _draw_depth_colored_polyline(self, image: np.ndarray, pts_2d: np.ndarray, depths: np.ndarray,
                                     color_near: tuple, color_far: tuple,
                                     thickness: int = 2, dashed: bool = False) -> np.ndarray:
        """
        Draw a polyline where segment color encodes relative depth (near vs far).
        
        Args:
            image: RGB image (H, W, 3)
            pts_2d: Nx2 array of pixel coordinates (float)
            depths: N array of depth values (float, same order as pts_2d)
            color_near: BGR color for nearest points
            color_far: BGR color for farthest points
            thickness: Line thickness in pixels
            dashed: If True, skip every other segment for a dashed look
        """
        if pts_2d is None or len(pts_2d) < 2:
            return image
        
        img = image
        pts = pts_2d.astype(np.int32)
        depths = depths.astype(np.float32)
        
        d_min = float(np.min(depths))
        d_max = float(np.max(depths))
        span = max(d_max - d_min, 1e-6)
        
        def _interp_color(t: float) -> tuple:
            t = float(max(0.0, min(1.0, t)))
            b = int(color_near[0] * (1 - t) + color_far[0] * t)
            g = int(color_near[1] * (1 - t) + color_far[1] * t)
            r = int(color_near[2] * (1 - t) + color_far[2] * t)
            return (b, g, r)
        
        num_segments = len(pts) - 1
        for i in range(num_segments):
            if dashed and (i % 2 == 1):
                continue
            p1 = tuple(pts[i])
            p2 = tuple(pts[i + 1])
            d_seg = 0.5 * (depths[i] + depths[i + 1])
            t = (d_seg - d_min) / span
            color = _interp_color(t)
            cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        
        return img

    def _overlay_breach_trajectories_on_image(self, image: np.ndarray,
                                              narration_timestamp: float,
                                              original_image_timestamp: Optional[float]) -> np.ndarray:
        """
        Overlay actual vs intended trajectory for the current breach on the narration image.
        
        - Actual trajectory: poses collected in the active risk buffer
        - Intended trajectory: nominal path segment the robot was supposed to follow
        - Depth is encoded by color gradient (near vs far) for easy visual parsing.
        """
        try:
            if not self.risk_buffer_manager:
                return image
            
            with self.lock:
                if len(self.risk_buffer_manager.active_buffers) == 0:
                    return image
                buffer = self.risk_buffer_manager.active_buffers[-1]
                poses = list(buffer.poses)
                nominal_xyz = buffer.get_nominal_path()
            
            if not poses or nominal_xyz is None or len(nominal_xyz) == 0:
                return image
            
            # Extract times and positions from poses
            times = np.array([p[0] for p in poses], dtype=np.float32)
            # Buffer poses may now be 6D (xyz + quaternion); use only xyz for trajectory overlay
            positions = np.stack([p[1][:3] for p in poses], axis=0).astype(np.float32)  # (N, 3)
            
            # Choose camera time: prefer original image timestamp, fall back to narration timestamp
            cam_time = float(narration_timestamp)
            if original_image_timestamp is not None:
                try:
                    cam_time = float(original_image_timestamp)
                except Exception:
                    pass
            
            # Use closest pose as camera center
            cam_idx = int(np.argmin(np.abs(times - cam_time)))
            camera_pos = positions[cam_idx]
            
            # Project actual trajectory (all poses in buffer)
            proj_actual = self._project_points_world_to_image(positions, camera_pos, image.shape)
            if proj_actual is None:
                return image
            actual_pts_2d, actual_depths = proj_actual
            
            # Derive intended (followed) nominal segment between first and last pose
            start_pos = positions[0]
            end_pos = positions[-1]
            d_start = np.linalg.norm(nominal_xyz - start_pos, axis=1)
            d_end = np.linalg.norm(nominal_xyz - end_pos, axis=1)
            idx_start = int(np.argmin(d_start))
            idx_end = int(np.argmin(d_end))
            if idx_end < idx_start:
                idx_start, idx_end = idx_end, idx_start
            nominal_segment = nominal_xyz[idx_start:idx_end + 1]
            
            proj_nominal = self._project_points_world_to_image(nominal_segment, camera_pos, image.shape)
            if proj_nominal is None:
                # Still draw actual path even if intended segment fails to project
                overlay = self._draw_depth_colored_polyline(
                    image.copy(), actual_pts_2d, actual_depths,
                    color_near=(0, 255, 0),   # bright green near
                    color_far=(0, 128, 0),    # darker green far
                    thickness=3,
                    dashed=False
                )
                return overlay
            
            nominal_pts_2d, nominal_depths = proj_nominal
            
            overlay = image.copy()
            
            # Actual trajectory: solid green, thicker line
            overlay = self._draw_depth_colored_polyline(
                overlay, actual_pts_2d, actual_depths,
                color_near=(0, 255, 0),       # near = bright green
                color_far=(0, 128, 0),        # far = darker green
                thickness=3,
                dashed=False
            )
            
            # Intended trajectory: magenta/blue, dashed and slightly thinner
            overlay = self._draw_depth_colored_polyline(
                overlay, nominal_pts_2d, nominal_depths,
                color_near=(255, 0, 255),     # near = bright magenta
                color_far=(128, 0, 255),      # far = bluish-magenta
                thickness=2,
                dashed=True
            )
            
            # Simple legend (bottom-left corner)
            try:
                h, w = overlay.shape[:2]
                legend_h = 48
                legend_w = 260
                margin = 10
                x0 = margin
                y0 = h - legend_h - margin
                x1 = x0 + legend_w
                y1 = y0 + legend_h
                
                # Semi-transparent background
                legend_bg = overlay.copy()
                cv2.rectangle(legend_bg, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
                alpha = 0.4
                overlay = cv2.addWeighted(legend_bg, alpha, overlay, 1 - alpha, 0)
                
                # Color swatches
                swatch_size = 12
                
                # Actual
                cv2.rectangle(overlay, (x0 + 10, y0 + 10),
                              (x0 + 10 + swatch_size, y0 + 10 + swatch_size),
                              (0, 255, 0), thickness=-1)
                cv2.putText(overlay, "Actual trajectory",
                            (x0 + 10 + swatch_size + 8, y0 + 10 + swatch_size),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Intended
                cv2.rectangle(overlay, (x0 + 10, y0 + 26),
                              (x0 + 10 + swatch_size, y0 + 26 + swatch_size),
                              (255, 0, 255), thickness=-1)
                cv2.putText(overlay, "Intended trajectory",
                            (x0 + 10 + swatch_size + 8, y0 + 26 + swatch_size),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                # Legend is non-critical; ignore failures
                pass
            
            return overlay
        except Exception as e:
            try:
                self.get_logger().warn(f"Failed to overlay trajectories on narration image: {e}")
            except Exception:
                pass
            return image

    def save_narration_image_to_buffer(self, image, narration_text, current_time,
                                       original_image_timestamp: Optional[float] = None):
        """
        Save narration image to the current buffer directory, with trajectory overlay.
        
        The overlay uses:
          - poses stored in the active risk buffer (actual trajectory)
          - nominal path snapshot stored at breach start (intended trajectory)
        """
        try:
            # Add trajectory overlay before writing to disk
            # image_with_overlay = self._overlay_breach_trajectories_on_image(
            #     image, current_time, original_image_timestamp
            # )
            image_with_overlay = image
            
            with self.lock:
                if len(self.risk_buffer_manager.active_buffers) == 0:
                    print("No active buffers to save narration image to")
                    return
            
            current_buffer = self.risk_buffer_manager.active_buffers[-1]
            
            buffer_dir = os.path.join(self.current_run_dir, current_buffer.buffer_id)
            narration_dir = os.path.join(buffer_dir, 'narration')
            os.makedirs(narration_dir, exist_ok=True)
            
            timestamp_str = f"{current_time:.3f}"
            image_filename = f"narration_image_{timestamp_str}.png"
            image_path = os.path.join(narration_dir, image_filename)
            cv2.imwrite(image_path, image_with_overlay)
            
            # Additionally save the pose corresponding to when the narration image was taken.
            # We choose the pose from the buffer with timestamp closest to the original image timestamp
            # (if available), otherwise closest to the narration timestamp.
            try:
                target_time = original_image_timestamp if original_image_timestamp is not None else current_time
                poses = list(current_buffer.poses)
                if poses:
                    times = np.array([p[0] for p in poses], dtype=float)
                    idx = int(np.argmin(np.abs(times - target_time)))
                    pose_t, pose_vec, pose_drift = poses[idx]
                    
                    pose_dict = {
                        "timestamp": float(pose_t),
                        "drift": float(pose_drift),
                        "position": {
                            "x": float(pose_vec[0]),
                            "y": float(pose_vec[1]),
                            "z": float(pose_vec[2])
                        }
                    }
                    # If we have orientation (6D pose), include quaternion
                    if len(pose_vec) >= 7:
                        pose_dict["orientation"] = {
                            "x": float(pose_vec[3]),
                            "y": float(pose_vec[4]),
                            "z": float(pose_vec[5]),
                            "w": float(pose_vec[6])
                        }
                    # Also record which time we matched against
                    pose_dict["matched_to_timestamp"] = float(target_time)
                    
                    pose_filename = f"narration_pose_{timestamp_str}.json"
                    pose_path = os.path.join(narration_dir, pose_filename)
                    with open(pose_path, "w") as f:
                        json.dump(pose_dict, f, indent=2)
            except Exception as e:
                print(f"Error saving narration pose to buffer: {e}")
            
        except Exception as e:
            print(f"Error saving narration image to buffer: {e}")
            import traceback
            traceback.print_exc()

    def publish_narration_hotspot_mask(self, narration_image: np.ndarray, vlm_answer: str, 
                                      original_image_timestamp: float, buffer_id: str) -> bool:
        """
        Publish narration image hotspot mask through semantic bridge for semantic voxel mapping.
        
        Args:
            narration_image: The narration image that was used for similarity processing
            vlm_answer: The VLM answer/cause that was identified
            original_image_timestamp: Timestamp when the original image was recorded (not narration time)
            buffer_id: Buffer ID for tracking
            
        Returns:
            True if successfully published, False otherwise
        """
        try:
            if not hasattr(self, 'semantic_bridge') or self.semantic_bridge is None:
                return False
            
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                return False
            
            # Narration stage: ALWAYS compute similarity using text-based path to bootstrap enhanced embedding
            similarity_result = self.naradio_processor.process_vlm_similarity_visualization_optimized(
                narration_image, vlm_answer, feat_map_np=None
            )
            
            # if not similarity_result or 'similarity_map' not in similarity_result:
            #     self.get_logger().error("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Failed to compute similarity for narration image")
                
            
            # Extract similarity map and apply binary threshold
            similarity_map = similarity_result['similarity_map']
            threshold = similarity_result.get('threshold_used', 0.6)

            # Always save similarity heatmap for narration, even if no hotspots are found.
            try:
                buffer_dir = os.path.join(self.current_run_dir, buffer_id)
                narration_hotspot_dir = os.path.join(buffer_dir, 'narration_hotspots')
                os.makedirs(narration_hotspot_dir, exist_ok=True)
                heatmap_filename = f"narration_similarity_heatmap_{time.time():.3f}.png"
                heatmap_path = os.path.join(narration_hotspot_dir, heatmap_filename)
                similarity_heatmap = (np.clip(similarity_map, 0.0, 1.0) * 255).astype(np.uint8)
                ok = cv2.imwrite(heatmap_path, similarity_heatmap)
                if ok:
                    self.get_logger().info(f"***********************************✓ Saved narration similarity heatmap: {heatmap_path}")
                else:
                    self.get_logger().warn(f"*********************************✗ Failed to write narration similarity heatmap (cv2.imwrite returned False): {heatmap_path}")
            except Exception as e:
                self.get_logger().warn(f"********************************Failed to save narration similarity heatmap: {e}")
            
            # Create binary hotspot mask
            hotspot_mask = (similarity_map > threshold).astype(np.uint8)
            
            if not np.any(hotspot_mask):
                self.get_logger().warn(f"No hotspots found in narration image for '{vlm_answer}'")
                return False
            
            # Create single VLM hotspot dictionary (same format as merged hotspots)
            vlm_hotspots = {vlm_answer: hotspot_mask}
            
            # Publish through semantic bridge with original image timestamp
            success = self.semantic_bridge.publish_merged_hotspots(
                vlm_hotspots=vlm_hotspots,
                timestamp=original_image_timestamp, narration=True,  # Use original image timestamp
                original_image=narration_image,
                buffer_id=buffer_id
            )
            
            if success:
                self.get_logger().info(f"✓ Published narration hotspot mask for '{vlm_answer}' through semantic bridge")
                self.get_logger().info(f"  Original timestamp: {original_image_timestamp:.6f}")
                self.get_logger().info(f"  Hotspot pixels: {int(np.sum(hotspot_mask))}")
                self.get_logger().info(f"  Threshold: {threshold:.3f}")
                return True
            else:
                self.get_logger().error("✗ Failed to publish narration hotspot mask through semantic bridge")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error publishing narration hotspot mask: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_ros_timestamp(self, msg):
        """Extract ROS timestamp as float from message header."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return time.time()


    def naradio_processing_loop(self):
        """OPTIMIZED: Parallel NARadio processing loop with frame skipping and reduced overhead."""
        print("NARadio processing loop started (OPTIMIZED)")
        
        last_memory_cleanup = time.time()
        memory_cleanup_interval = 60.0
        
        # Target loop rate (was effectively self-throttled by frame skipping + sleeps)
        # Keep this conservative to avoid starving ROS callbacks while still achieving ~8Hz.
        target_hz = 8.0
        target_period_s = 1.0 / max(target_hz, 1e-6)
        
        # OPTIMIZATION: Cached RGB conversion
        last_rgb_msg_id = None
        cached_rgb_image = None
        
        while rclpy.ok() and self.naradio_running:
            try:
                loop_start = time.time()
                current_time = time.time()
                
                # OPTIMIZATION: Less frequent memory cleanup
                if current_time - last_memory_cleanup > memory_cleanup_interval:
                    self.naradio_processor.cleanup_memory()
                    last_memory_cleanup = current_time
                
                # Check if processor is ready
                if not self.naradio_processor.is_ready():
                    time.sleep(0.05)
                    continue
                
                with self.processing_lock:
                    if self.latest_rgb_msg is None:
                        time.sleep(0.01)
                        continue
                    
                    rgb_msg = self.latest_rgb_msg
                    depth_msg = self.latest_depth_msg
                    pose_for_semantic = self.latest_pose.copy() if self.latest_pose is not None else None

                # If nothing to segment, don't burn compute — just rate-limit
                if not (self.enable_combined_segmentation and self.naradio_processor.is_segmentation_ready()):
                    # Maintain a gentle loop cadence
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.01, target_period_s - elapsed))
                    continue
                if not self.naradio_processor.dynamic_objects:
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.01, target_period_s - elapsed))
                    continue
                
                # OPTIMIZATION: Cache RGB conversion to avoid repeated cv_bridge calls
                rgb_msg_id = id(rgb_msg)
                if rgb_msg_id != last_rgb_msg_id:
                    try:
                        cached_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
                        cached_rgb_image = self._rotate_image_if_needed(cached_rgb_image)
                        last_rgb_msg_id = rgb_msg_id
                    except Exception as e:
                        time.sleep(0.01)
                        continue
                
                rgb_image = cached_rgb_image
                
                # OPTIMIZATION: Skip visualization completely
                feat_map_np, _ = self.naradio_processor.process_features_optimized(
                    rgb_image, 
                    need_visualization=False,
                    reuse_features=True
                )
                
                # OPTIMIZATION: Only process if we have dynamic objects and features
                if feat_map_np is not None:
                    
                    # Copy to avoid concurrent modification by callbacks
                    vlm_answers = list(self.naradio_processor.dynamic_objects)
                    
                    # OPTIMIZATION: Use batched similarity (fast version computes all masks in one pass)
                    vlm_hotspots = self.naradio_processor.create_merged_hotspot_masks_fast(
                        rgb_image, vlm_answers, feat_map_np=feat_map_np)
                    
                    if vlm_hotspots and len(vlm_hotspots) > 0:
                        rgb_timestamp = self._get_ros_timestamp(rgb_msg)
                        self.semantic_bridge.publish_merged_hotspots(
                            vlm_hotspots=vlm_hotspots,
                            timestamp=rgb_timestamp,
                            original_image=None  # OPTIMIZATION: Skip overlay for predictive
                        )
                
                # Rate-limit to target_hz (don't add extra sleeps beyond what's needed)
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, target_period_s - elapsed))
                            
            except Exception as e:
                time.sleep(0.05)  
    
    def camera_info_callback(self, msg):
        """Handle camera info to get intrinsics."""
        with self.lock:
            if not self.camera_info_received:
                fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
                if self.image_rotation_deg == 180 and getattr(msg, "width", 0) and getattr(msg, "height", 0):
                    # 180° rotation maps (u, v) -> (w-1-u, h-1-v)
                    cx = (float(msg.width) - 1.0) - float(cx)
                    cy = (float(msg.height) - 1.0) - float(cy)
                self.camera_intrinsics = [fx, fy, cx, cy]
                self.camera_info_received = True

    #NOTE demo code, replace with proper function later, also enable narration publishing whish
    #has been commented out for now
    def vlm_answer_callback(self):
        """Handle VLM answers for cause analysis and buffer association."""
        try:
            primary_cause = None
            # for obj_data in top_objects:
            #     vlm_answer = obj_data['name']
            #     score = float(obj_data.get('score', 0.0))
                
            #     # Track as recent
            #     self.recent_vlm_answers[vlm_answer] = time.time()
                
            #     # Store all in registry with confidence=score
            #     # Only add to processor (dynamic_objects) for predictive similarity if score > 0.8
                
            #     if score > 0.8:
            #         # Add to processor for predictive similarity (also adds to registry)
            #         success = self.naradio_processor.add_vlm_object(vlm_answer)
            #         if success:
            #             self.cause_registry.record_detection(vlm_answer, score)
            #             # Verify it's in dynamic_objects
            #             if vlm_answer in self.naradio_processor.dynamic_objects:
            #                 self.get_logger().info(f"✓ '{vlm_answer}' added for predictive similarity (score: {score:.4f} > 0.8)")
            #             else:
            #                 self.get_logger().warn(f"✗ '{vlm_answer}' not in dynamic_objects after add_vlm_object")
            #         else:
            #             self.get_logger().warn(f"Failed to add '{vlm_answer}' to processor")
            
            # Demo: Set both to the same value
            vlm_answer = "black sheet"
            primary_cause = vlm_answer  # Use the same value
            score = 1.0
            
            # Track as recent
            self.recent_vlm_answers[vlm_answer] = time.time()
            
            # Store all in registry with confidence=score
            # Only add to processor (dynamic_objects) for predictive similarity if score > 0.8
            
            if score > 0.8:
                # Add to processor for predictive similarity (also adds to registry)
                success = self.naradio_processor.add_vlm_object(vlm_answer)
                if success:
                    self.cause_registry.record_detection(vlm_answer, score)
                    # Verify it's in dynamic_objects
                    if vlm_answer in self.naradio_processor.dynamic_objects:
                        self.get_logger().info(f"✓ '{vlm_answer}' added for predictive similarity (score: {score:.4f} > 0.8)")
                    else:
                        self.get_logger().warn(f"✗ '{vlm_answer}' not in dynamic_objects after add_vlm_object")
                else:
                    self.get_logger().warn(f"Failed to add '{vlm_answer}' to processor")
            
            self.save_cause_registry_snapshot()
            if primary_cause:
                self.associate_vlm_answer_with_buffer(primary_cause)
                narration_success = self.process_narration_chain_for_vlm_answer(primary_cause)
                if narration_success:
                    self.get_logger().info(f"Narration processing completed for '{primary_cause}'")
            
        except Exception as e:
            print(f"Error processing VLM answer: {e}")
            import traceback
            traceback.print_exc()

    def process_narration_chain_for_vlm_answer(self, vlm_answer: str) -> bool:
        try:
            self.get_logger().info("=" * 80)
            self.get_logger().info(f" NARRATION HOTSPOT GENERATION STARTED for '{vlm_answer}'")
            self.get_logger().info("=" * 80)
            
            if not hasattr(self, 'naradio_processor') or not self.naradio_processor.is_segmentation_ready():
                self.get_logger().error(f" NARadio processor not ready for narration processing")
                return False
            
            self.get_logger().info(f"✓ NARadio processor ready")
            
            # OPTIMIZATION 1: Early duplicate check - exit before any expensive computation
            entry = self.cause_registry.get_entry_by_name(vlm_answer)
            if entry is not None and entry.vec_id in self.narration_published_vec_ids:
                self.get_logger().info(
                    f"  Skipping narration mask for '{vlm_answer}' (vec_id: {entry.vec_id}) - "
                    f"already published narration mask for this or similar cause (similarity >0.8)"
                )
                return True  # Return True since we intentionally skipped (not an error)
            
            self.get_logger().info(f" Searching for buffer with cause '{vlm_answer}'")
            
            # Find the buffer that was just assigned the cause
            target_buffer = None
            with self.lock:
                # Check frozen buffers first (most recent with this cause)
                self.get_logger().info(f"   Checking {len(self.risk_buffer_manager.frozen_buffers)} frozen buffers")
                for buffer in reversed(self.risk_buffer_manager.frozen_buffers):
                    if buffer.cause == vlm_answer:
                        target_buffer = buffer
                        self.get_logger().info(f"   ✓ Found in frozen buffer: {buffer.buffer_id}")
                        break
                
                # If not found, check active buffers
                if target_buffer is None:
                    self.get_logger().info(f"   Checking {len(self.risk_buffer_manager.active_buffers)} active buffers")
                    for buffer in reversed(self.risk_buffer_manager.active_buffers):
                        if buffer.cause == vlm_answer:
                            target_buffer = buffer
                            self.get_logger().info(f"   ✓ Found in active buffer: {buffer.buffer_id}")
                            break
            
            if not target_buffer:
                self.get_logger().error(f" No buffer found with cause '{vlm_answer}' for narration processing")
                return False
            
            buffer_dir = os.path.join(self.current_run_dir, target_buffer.buffer_id)
            self.get_logger().info(f" Buffer directory: {buffer_dir}")
            
            # Get narration image
            self.get_logger().info(f"  Retrieving narration image...")
            narration_image = None
            if target_buffer.has_narration_image():
                narration_image = target_buffer.get_narration_image() 
                self.get_logger().info(f"   ✓ Got narration image from buffer memory (shape: {narration_image.shape})")
            else:
                self.get_logger().info(f"   No narration image in buffer memory, checking disk...")
                # Fallback: look for narration images on disk
                narration_dir = os.path.join(buffer_dir, 'narration')
                if os.path.exists(narration_dir):
                    narration_files = [f for f in os.listdir(narration_dir) if f.endswith('.png')]
                    self.get_logger().info(f"   Found {len(narration_files)} narration files on disk")
                    if narration_files:
                        # Get the most recent narration image
                        narration_files.sort()
                        latest_narration_file = narration_files[-1]
                        narration_image_path = os.path.join(narration_dir, latest_narration_file)
                        self.get_logger().info(f"   Loading: {latest_narration_file}")
                        
                        # Load the narration image
                        narration_image = cv2.imread(narration_image_path)
                        if narration_image is not None:
                            # Convert BGR to RGB
                            narration_image = cv2.cvtColor(narration_image, cv2.COLOR_BGR2RGB)
                            self.get_logger().info(f"   ✓ Loaded narration image from disk (shape: {narration_image.shape})")
                else:
                    self.get_logger().warn(f"   Narration directory does not exist: {narration_dir}")
            
            if narration_image is None:
                self.get_logger().error(f"❌ No narration image found for buffer {target_buffer.buffer_id}")
                return False

            # Requested: force narration image to fixed on-disk image for hotspot extraction and downstream.
            # try:
            #     override_bgr = cv2.imread(NARRATION_IMAGE_OVERRIDE_PATH)
            #     if override_bgr is not None:
            #         narration_image = cv2.cvtColor(override_bgr, cv2.COLOR_BGR2RGB)
            #         self.get_logger().warn(
            #             f"[Narration] Using hardcoded image '{NARRATION_IMAGE_OVERRIDE_PATH}' "
            #             f"for narration processing (shape={narration_image.shape})"
            #         )
            #     else:
            #         self.get_logger().warn(
            #             f"[Narration] Failed to read hardcoded image '{NARRATION_IMAGE_OVERRIDE_PATH}', "
            #             "continuing with buffer narration image"
            #         )
            # except Exception as e:
            #     self.get_logger().warn(
            #         f"[Narration] Hardcoded image override failed ({e}); continuing with buffer narration image"
            #     )
            
            self.get_logger().info(f"✓ Processing narration image for '{vlm_answer}' from buffer {target_buffer.buffer_id}")

            # OPTIMIZATION 2: Extract features ONCE and reuse throughout
            # self.get_logger().info(f"🔬 Extracting features from narration image...")
            feat_map_np, _ = self.naradio_processor.process_features_optimized(
                narration_image, need_visualization=False, reuse_features=False
            )
            if feat_map_np is None:
                self.get_logger().error(f"❌ Failed to extract features for narration image")
                return False
            self.get_logger().info(f"   ✓ Features extracted (shape: {feat_map_np.shape})")

            # OPTIMIZATION 3: Compute similarity map ONCE using pre-computed features
            # self.get_logger().info(f"🎯 Computing similarity map for '{vlm_answer}'...")
            
            # DIAGNOSTIC: Check if VLM object is in the processor's object list
            all_objects = self.naradio_processor.get_all_objects()
            self.get_logger().info(f"   All objects in processor: {all_objects}")
            self.get_logger().info(f"   Is '{vlm_answer}' in objects? {vlm_answer in all_objects}")
            self.get_logger().info(f"   Number of objects: {len(all_objects)}")
            use_softmax = True
            if len(all_objects) == 1:
                use_softmax = False
                self.get_logger().warn("   Only 1 object in list; falling back to non-softmax similarity (normalized)")
            self.get_logger().info(f"   Using softmax: {use_softmax}")
            
            if vlm_answer not in all_objects:
                self.get_logger().error(f"   ❌ VLM answer '{vlm_answer}' NOT in processor object list!")
                self.get_logger().error(f"   Need to call add_vlm_object('{vlm_answer}') first")
                return False
            
            similarity_map = self.naradio_processor.compute_vlm_similarity_map_optimized(
                narration_image, vlm_answer, feat_map_np=feat_map_np, use_softmax=use_softmax, chunk_size=4000
            )
            if similarity_map is None:
                self.get_logger().error(f"❌ Failed to compute similarity for narration image")
                naradio_err = getattr(self.naradio_processor, "last_similarity_error", None)
                if naradio_err:
                    self.get_logger().error(f"   Similarity error: {naradio_err}")
                else:
                    self.get_logger().error("   Similarity error: (no details recorded)")
                return False
            self.get_logger().info(f"   ✓ Similarity map computed (shape: {similarity_map.shape}, max: {np.max(similarity_map):.4f})")

            # Get threshold from config
            threshold = 0.90
            self.get_logger().info(f"Using threshold: {threshold}")
            # if hasattr(self.naradio_processor, 'segmentation_config'):
            #     threshold = self.naradio_processor.segmentation_config.get('segmentation', {}).get('hotspot_threshold', 0.6)
            
            # CRITICAL FIX #2: Use the similarity map already computed (no redundant computation)
            # The similarity_map computed above is already text-based, which is what we want
            similarity_map_final = similarity_map

            # Always save similarity heatmap for every narration (even if thresholding yields no hotspots)
            try:
                narration_hotspot_dir = os.path.join(buffer_dir, 'narration_hotspots')
                os.makedirs(narration_hotspot_dir, exist_ok=True)
                heatmap_filename = f"narration_similarity_heatmap_{time.time():.3f}.png"
                heatmap_path = os.path.join(narration_hotspot_dir, heatmap_filename)
                similarity_heatmap = (np.clip(similarity_map_final, 0.0, 1.0) * 255).astype(np.uint8)
                ok = cv2.imwrite(heatmap_path, similarity_heatmap)
                if ok:
                    self.get_logger().info(f"   ✓ Saved narration similarity heatmap: {heatmap_path}")
                else:
                    self.get_logger().warn(f"   Failed to write narration similarity heatmap (cv2.imwrite returned False): {heatmap_path}")
            except Exception as e:
                self.get_logger().warn(f"   Failed to save narration similarity heatmap: {e} (buffer_dir={buffer_dir})")
            
            # Create hotspot mask using the computed similarity map
            # self.get_logger().info(f"🎭 Creating hotspot mask...")
            hotspot_mask_final = (similarity_map_final > threshold).astype(np.uint8)
            hotspot_pixels = int(np.sum(hotspot_mask_final))
            self.get_logger().info(f"   Hotspot pixels: {hotspot_pixels}")
            
            if not np.any(hotspot_mask_final):
                self.get_logger().warn(f"⚠️  Narration similarity produced no hotspots for '{vlm_answer}'")
                return False
            
            # Save narration hotspot mask to buffer directory
            # self.get_logger().info(f"💾 Saving narration hotspot mask to buffer...")
            try:
                # Save binary mask
                mask_filename = f"narration_hotspot_mask_{time.time():.3f}.png"
                mask_path = os.path.join(narration_hotspot_dir, mask_filename)
                cv2.imwrite(mask_path, hotspot_mask_final * 255)
                
                # Save metadata
                metadata = {
                    'vlm_answer': vlm_answer,
                    'threshold': float(threshold),
                    'hotspot_pixels': int(hotspot_pixels),
                    'max_similarity': float(np.max(similarity_map_final)),
                    'mean_similarity': float(np.mean(similarity_map_final)),
                    'buffer_id': target_buffer.buffer_id,
                    'timestamp': time.time()
                }
                metadata_filename = f"narration_hotspot_metadata_{time.time():.3f}.json"
                metadata_path = os.path.join(narration_hotspot_dir, metadata_filename)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.get_logger().info(f"   ✓ Saved mask: {mask_filename}")
                self.get_logger().info(f"   ✓ Saved metadata: {metadata_filename}")
            except Exception as e:
                self.get_logger().error(f"   ❌ Failed to save narration hotspot: {e}")
                import traceback
                traceback.print_exc()

            # Get original image timestamp (CRITICAL for proper synchronization)
            # self.get_logger().info(f"🕒 Retrieving original image timestamp...")
            original_image_timestamp = None
            if target_buffer.get_original_image_timestamp() is not None:
                original_image_timestamp = target_buffer.get_original_image_timestamp()
                self.get_logger().info(f"   ✓ Got from buffer: {original_image_timestamp:.6f}")
            elif target_buffer.narration_timestamp is not None:
                original_image_timestamp = target_buffer.start_time
                self.get_logger().info(f"   ✓ Using buffer start_time: {original_image_timestamp:.6f}")
            else:
                original_image_timestamp = time.time()
                self.get_logger().warn(f"   ⚠️  Using current time as fallback: {original_image_timestamp:.6f}")
            
            # Update registry (batch saves will be handled by debouncing)
            # self.get_logger().info(f"📝 Updating cause registry...")
            similarity_score = float(np.max(similarity_map_final))
            self.cause_registry.record_detection(vlm_answer, similarity_score)
            self.cause_registry.set_metadata(vlm_answer, {
                "last_buffer_id": target_buffer.buffer_id,
                "last_original_timestamp": original_image_timestamp,
                "last_similarity_threshold": threshold
            })
            self.get_logger().info(f"   ✓ Registry updated (similarity_score: {similarity_score:.4f})")
            # OPTIMIZATION 5: Debounce registry saves (will be saved periodically, not on every update)
            # Removed immediate save_cause_registry_snapshot() call here
            
            # Mark this cause as having narration mask published (by vec_id for canonical identity)
            if entry is not None:
                self.narration_published_vec_ids.add(entry.vec_id)
                self.get_logger().info(
                    f"✓ Marking '{vlm_answer}' (vec_id: {entry.vec_id}) as having narration mask published"
                )
            else:
                self.get_logger().warn(
                    f"⚠️  No cause registry entry found for '{vlm_answer}' - cannot track narration publication"
                )
            
            # Publish hotspot mask
            # self.get_logger().info(f"📡 Publishing narration hotspot mask through semantic bridge...")
            self.get_logger().info(f"   VLM answer: '{vlm_answer}'")
            self.get_logger().info(f"   Timestamp: {original_image_timestamp:.6f}")
            self.get_logger().info(f"   Buffer ID: {target_buffer.buffer_id}")
            self.get_logger().info(f"   Hotspot pixels: {hotspot_pixels}")
            
            success_pub = self.semantic_bridge.publish_merged_hotspots(
                vlm_hotspots={vlm_answer: hotspot_mask_final},
                timestamp=original_image_timestamp,
                narration=True,
                original_image=narration_image,
                buffer_id=target_buffer.buffer_id
            )
            if not success_pub:
                self.get_logger().error(f"❌ Failed to publish narration hotspot mask")
                return False
            
            self.get_logger().info(f"✅ Published narration hotspot mask for '{vlm_answer}'")
            
            # OPTIMIZATION 6: Compute enhanced embedding asynchronously (non-blocking)
            # Uses pre-computed features and similarity map to avoid redundant computation
            # self.get_logger().info(f"🧬 Computing enhanced embedding...")
            try:
                enhanced_embedding = self.naradio_processor.compute_enhanced_cause_embedding(
                    narration_image, vlm_answer, 
                    similarity_map=similarity_map_final,  # Use final similarity map
                    feat_map_np=feat_map_np  # Reuse pre-computed features
                )
                if enhanced_embedding is not None:
                    self.get_logger().info(f"   ✓ Enhanced embedding computed (shape: {enhanced_embedding.shape})")
                    # Save to buffer and register (async save could be added here)
                    self.naradio_processor._save_enhanced_embedding(vlm_answer, buffer_dir, enhanced_embedding)
                    self.get_logger().info(f"   ✓ Enhanced embedding saved to buffer")
                    target_buffer.assign_enhanced_cause_embedding(enhanced_embedding)
                    if self.naradio_processor.is_segmentation_ready():
                        self.naradio_processor.add_enhanced_embedding(vlm_answer, enhanced_embedding)
                        self.get_logger().info(f"   ✓ Enhanced embedding registered in processor")
                    self.get_logger().info(f"Enhanced embedding ready for '{vlm_answer}'")
                else:
                    self.get_logger().warn(f"Enhanced embedding computation returned None")
            except Exception as e:
                self.get_logger().warn(f"Enhanced embedding computation failed (non-critical): {e}")
            
            # self.get_logger().info("=" * 80)
            self.get_logger().info(f" NARRATION HOTSPOT GENERATION COMPLETED SUCCESSFULLY for '{vlm_answer}'")
            self.get_logger().info("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"Error in narration processing chain: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_cause_registry_snapshot(self, force: bool = False):
        """
        Persist the cause registry to the current run directory for other nodes.
        
        OPTIMIZATION: Debounced to reduce I/O - only saves if enough time has passed
        since last save, unless force=True.
        
        Args:
            force: If True, save immediately regardless of debounce interval
        """
        if not self.cause_registry_snapshot_path:
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_registry_save_time) < self.registry_save_debounce_interval:
            return  # Skip save due to debounce
        
        try:
            snapshot = self.cause_registry.snapshot()
            os.makedirs(os.path.dirname(self.cause_registry_snapshot_path), exist_ok=True)
            with open(self.cause_registry_snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2)
            self.last_registry_save_time = current_time
        except Exception as e:
            self.get_logger().warn(f"Failed to save cause registry snapshot: {e}")

    def load_predefined_objects(self):
        """Load pre-defined objects from config and register them with GP parameters."""
        try:
            if not hasattr(self.naradio_processor, 'segmentation_config'):
                self.get_logger().warn("No segmentation config available, skipping predefined objects")
                return
            
            preloaded_config = self.naradio_processor.segmentation_config.get('preloaded_objects', [])
            if not preloaded_config:
                self.get_logger().info("No predefined objects configured")
                return
            
            self.get_logger().info(f"Loading {len(preloaded_config)} predefined objects...")
            
            loaded_count = 0
            for obj_config in preloaded_config:
                if not isinstance(obj_config, dict):
                    continue
                
                enabled = obj_config.get('enabled', True)
                if not enabled:
                    continue
                
                obj_name = obj_config.get('name')
                gp_params_file = obj_config.get('gp_params_file')
                
                if not obj_name:
                    self.get_logger().warn(f"Skipping predefined object: missing 'name' field")
                    continue
                
                # Add object to dynamic_objects (registers in cause_registry)
                self.get_logger().info(f"Pre-loading object: '{obj_name}'")
                success = self.naradio_processor.add_vlm_object(obj_name)
                
                if not success:
                    self.get_logger().warn(f"Failed to add predefined object '{obj_name}' to processor")
                    continue
                
                # Load and set GP parameters if file is provided
                if gp_params_file:
                    gp_params = self._load_gp_params_from_file(gp_params_file)
                    if gp_params:
                        # Set GP params in cause registry
                        from resilience.cause_registry import GPParams
                        success = self.cause_registry.set_gp_params(obj_name, gp_params)
                        if success:
                            self.get_logger().info(f"✓ Loaded GP params for '{obj_name}' from {gp_params_file}")
                            loaded_count += 1
                        else:
                            self.get_logger().warn(f"Failed to set GP params for '{obj_name}' in registry")
                    else:
                        self.get_logger().warn(f"Failed to load GP params from {gp_params_file} for '{obj_name}'")
                else:
                    self.get_logger().info(f"✓ Pre-loaded object '{obj_name}' (no GP params file specified)")
                    loaded_count += 1
            
            self.get_logger().info(f"Successfully pre-loaded {loaded_count} objects with GP parameters")
            # Save registry snapshot after loading
            self.save_cause_registry_snapshot(force=True)
            
        except Exception as e:
            self.get_logger().error(f"Error loading predefined objects: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_gp_params_from_file(self, gp_params_file: str):
        """Load GP parameters from a voxel_gp_fit.json file."""
        try:
            # Handle wildcard paths (e.g., find latest buffer directory)
            if '*' in gp_params_file or '?' in gp_params_file:
                import glob
                matches = glob.glob(gp_params_file)
                if not matches:
                    self.get_logger().warn(f"No files found matching pattern: {gp_params_file}")
                    return None
                # Use most recently modified file
                gp_params_file = max(matches, key=os.path.getmtime)
                self.get_logger().info(f"Resolved wildcard to: {gp_params_file}")
            
            if not os.path.exists(gp_params_file):
                self.get_logger().warn(f"GP params file not found: {gp_params_file}")
                return None
            
            with open(gp_params_file, 'r') as f:
                data = json.load(f)
            
            # Extract fit_params from the JSON structure
            fit_params = data.get('fit_params', {})
            if not fit_params:
                self.get_logger().warn(f"No 'fit_params' found in {gp_params_file}")
                return None
            
            # Create GPParams object
            # Note: GPParams dataclass only stores basic params (lxy, lz, A, b, metrics)
            # Advanced params like sigma2, XtX_inv, hess_inv are not stored in registry
            # but are available in the JSON file for direct use if needed
            from resilience.cause_registry import GPParams
            gp_params = GPParams(
                lxy=fit_params.get('lxy'),
                lz=fit_params.get('lz'),
                A=fit_params.get('A'),
                b=fit_params.get('b'),
                mse=fit_params.get('mse'),
                rmse=fit_params.get('rmse'),
                mae=fit_params.get('mae'),
                r2_score=fit_params.get('r2_score'),
                timestamp=fit_params.get('timestamp', time.time()),
                buffer_id=fit_params.get('buffer_id')
            )
            
            return gp_params
            
        except Exception as e:
            self.get_logger().error(f"Error loading GP params from {gp_params_file}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def associate_vlm_answer_with_buffer(self, vlm_answer):
        """Associate VLM answer with buffer."""
        
        success = self.risk_buffer_manager.assign_cause(vlm_answer)
        
        if success:
            self.get_logger().warn(f"Associated '{vlm_answer}' with risk buffer")
        else:
            self.get_logger().warn(f"No suitable buffer found for '{vlm_answer}'")
            # print()
                    


    
    def _get_ros_timestamp(self, msg):
        """Extract ROS timestamp as float from message header."""
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            return time.time()



def main():
    rclpy.init()
    node = ResilienceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down Resilience Node...")
        node.narration_manager.stop()
        if hasattr(node, 'naradio_running') and node.naradio_running:
            node.naradio_running = False
            if hasattr(node, 'naradio_thread') and node.naradio_thread and node.naradio_thread.is_alive():
                node.naradio_thread.join(timeout=2.0)

        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 