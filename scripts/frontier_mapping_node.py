#!/usr/bin/env python3
"""
Semantic Depth VDB Mapping ROS2 Node

Simplified node that uses RayFronts SemanticRayFrontiersMap for efficient 3D mapping.
Subscribes to depth, pose, and semantic info to create semantic voxel maps using OpenVDB.
Maintains timestamped buffers for depth frames and poses to align with hotspot masks
received via the semantic bridge using original RGB timestamps.
"""

import rclpy
import matplotlib.cm as cm  
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.logging import LoggingSeverity

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import numpy as np
from collections import deque

# CRITICAL: Must be set BEFORE torch is imported.
# Converts asynchronous CUDA assertion failures (which kill the process)
# into synchronous Python RuntimeErrors that can be caught with try/except.
import os as _os
_os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

import torch
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
import time
import json
import math
from typing import Optional, List, Dict
import sensor_msgs_py.point_cloud2 as pc2
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import bisect
import base64
import glob

# Import RayFronts VDB mapping

import sys
sys.path.append('/home/navin/ros2_ws/src/resilience/RayFronts')
from rayfronts.mapping.semantic_ray_frontiers_map import SemanticRayFrontiersMap
from rayfronts import geometry3d as g3d
import rayfronts_cpp
VDB_AVAILABLE = True
RAYFRONTS_G3D_AVAILABLE = True

# Optional GP helper
from resilience.voxel_gp_helper import _sum_of_anisotropic_rbf_fast

try:
	from resilience.voxel_gp_helper import DisturbanceFieldHelper
	GP_HELPER_AVAILABLE = True
except ImportError:
	GP_HELPER_AVAILABLE = False

# Optional PathManager for global path access
try:
	from resilience.path_manager import PathManager
	PATH_MANAGER_AVAILABLE = True
except ImportError:
	PATH_MANAGER_AVAILABLE = False


class _ZeroImageEncoder:
	def __init__(self, embed_dim: int, device: str):
		self.embed_dim = embed_dim
		self.device = device

	def encode_image_to_vector(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch = rgb_img.shape[0]
		return torch.zeros(batch, self.embed_dim, device=rgb_img.device, dtype=rgb_img.dtype)

	def encode_image_to_feat_map(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch, _, h, w = rgb_img.shape
		return torch.zeros(batch, self.embed_dim, h, w, device=rgb_img.device, dtype=rgb_img.dtype)

	def align_spatial_features_with_language(self, feat: torch.Tensor) -> torch.Tensor:
		return feat


class SemanticDepthOctoMapNode(Node):
	"""Simplified semantic depth VDB mapping node using RayFronts SemanticRayFrontiersMap."""

	def __init__(self):
		super().__init__('semantic_depth_vdb_mapping_node')
		self.get_logger().set_level(LoggingSeverity.WARN)

		# Professional startup message
		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC VDB MAPPING SYSTEM INITIALIZING")
		self.get_logger().info("=" * 60)

		if not VDB_AVAILABLE:
			self.get_logger().error("RayFronts VDB not available! Please check installation.")
			return

		# Parameters
		self.declare_parameters('', [
			('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered'),
			('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info'),
			('pose_topic', '/robot_1/sensors/front_stereo/pose'),
			# If your physical camera is mounted upside-down, set this to 180.
			# Valid values: 0, 180.
			('image_rotation_deg', 0),
			('map_frame', 'map'),
			('semantic_colored_cloud_frame', 'fastlio'),
			('voxel_resolution', 0.2),
			('max_range', 1.5),
			('min_range', 0.1),
			('probability_hit', 0.7),
			('probability_miss', 0.4),
			('occupancy_threshold', 0.5),
			('publish_markers', True),
			('publish_stats', True),
			('publish_colored_cloud', True),
			('use_cube_list_markers', True),
			('max_markers', 30000),
			('marker_publish_rate', 20.0),
			('stats_publish_rate', 1.0),
			('pose_is_base_link', True),
			('apply_optical_frame_rotation', True),
			('cam_to_base_rpy_deg', [0.0, 0.0, 0.0]),
			('cam_to_base_xyz', [0.0, 0.0, 0.0]),
			('embedding_dim', 1152),
			('enable_semantic_mapping', True),
			('semantic_similarity_threshold', 0.6),
			('buffers_directory', os.path.expanduser('~/ros2_ws/src/buffers')),
			('enable_voxel_mapping', True),
			('sync_buffer_seconds', 2.0),
			('inactivity_threshold_seconds', 2.5),
			('semantic_export_directory', os.path.expanduser('~/ros2_ws/src/buffers')),
			('mapping_config_path', ''),
			('nominal_path', ''),  # Will be set from package share directory if not provided
			('main_config_path', ''),
			('hazard_sphere_voxel_resolution', 0.5),  # Secondary voxel resolution for sphere markers
			('hazard_sphere_radius', 1.5),  # Radius of spheres around semantic voxels
			('gp_combination_method', 'sum')  # How to combine multi-object GP predictions: 'sum' (additive) or 'max' (worst-case)
		])

		params = self.get_parameters([
			'depth_topic', 'camera_info_topic', 'pose_topic',
			'image_rotation_deg',
			'map_frame', 'semantic_colored_cloud_frame', 'voxel_resolution', 'max_range', 'min_range', 'probability_hit',
			'probability_miss', 'occupancy_threshold', 'publish_markers', 'publish_stats',
			'publish_colored_cloud', 'use_cube_list_markers', 'max_markers', 'marker_publish_rate', 'stats_publish_rate',
			'pose_is_base_link', 'apply_optical_frame_rotation', 'cam_to_base_rpy_deg', 'cam_to_base_xyz', 'embedding_dim',
			'enable_semantic_mapping', 'semantic_similarity_threshold', 'buffers_directory',
			'enable_voxel_mapping', 'sync_buffer_seconds', 'inactivity_threshold_seconds', 'semantic_export_directory', 'mapping_config_path', 'nominal_path', 'main_config_path',
			'hazard_sphere_voxel_resolution', 'hazard_sphere_radius', 'gp_combination_method'
		])

		# Extract parameter values
		(self.depth_topic, self.camera_info_topic, self.pose_topic,
		 self.image_rotation_deg,
		 self.map_frame, self.semantic_colored_cloud_frame, self.voxel_resolution, self.max_range, self.min_range, self.prob_hit,
		 self.prob_miss, self.occ_thresh, self.publish_markers, self.publish_stats, self.publish_colored_cloud,
		 self.use_cube_list_markers, self.max_markers, self.marker_publish_rate, self.stats_publish_rate,
		 self.pose_is_base_link, self.apply_optical_frame_rotation, self.cam_to_base_rpy_deg, self.cam_to_base_xyz,
			self.embedding_dim, self.enable_semantic_mapping, self.semantic_similarity_threshold,
			self.buffers_directory,
			self.enable_voxel_mapping, self.sync_buffer_seconds, self.inactivity_threshold_seconds,
		 self.semantic_export_directory, self.mapping_config_path, self.nominal_path, self.main_config_path,
		 self.hazard_sphere_voxel_resolution, self.hazard_sphere_radius, self.gp_combination_method) = [p.value for p in params]
		try:
			self.image_rotation_deg = int(self.image_rotation_deg)
		except Exception:
			self.image_rotation_deg = 0
		if self.image_rotation_deg not in (0, 180):
			self.get_logger().warn(f"Invalid image_rotation_deg={self.image_rotation_deg}; using 0")
			self.image_rotation_deg = 0

		# Hardcode for now to keep depth + hotspot mask aligned.
		# (This affects BOTH the depth path and the hotspot mask path via _rotate_image_if_needed.)
		self.image_rotation_deg = 180
		self.hazard_sphere_radius = 1.5
		# Read nominal path separately (optional for GP)
		self.nominal_path = self.get_parameter('nominal_path').value
		self.main_config_path = self.get_parameter('main_config_path').value		
		from collections import deque

		# Buffer sizes for ~10 seconds of data at high frame rates
		# Depth: 1000 frames (covers 10s at 100 Hz, or 33s at 30 Hz)
		# Pose: 2000 frames (covers 10s at 200 Hz, or 66s at 30 Hz)
		# Mask: 500 frames (covers 10s at 50 Hz)
		self.depth_buffer_data = deque(maxlen=1000)
		self.depth_buffer_ts = deque(maxlen=1000)

		self.pose_buffer_data = deque(maxlen=2000)
		self.pose_buffer_ts = deque(maxlen=2000)

		self.mask_buffer_data = deque(maxlen=500) # Increased for 10s buffer
		self.mask_buffer_ts = deque(maxlen=500)
		# Load topic configuration from mapping config
		self.load_topic_configuration()
		
		# Initialize state variables
		self.bridge = CvBridge()
		self.camera_intrinsics = None
		self._camera_width = None
		self._camera_height = None
		self.latest_pose = None
		self.latest_pose_frame_id: Optional[str] = None
		self.last_marker_pub = 0.0
		self.last_stats_pub = 0.0
		self.last_data_time = time.time()
		self.semantic_pcd_exported = False
		
		# Timestamped buffers for sync
		self.mask_buffer = []
		self.sync_buffer_duration = float(self.sync_buffer_seconds)
		self.sync_lock = threading.Lock()
		
		# Thread pool for processing semantic hotspots
		self.hotspot_executor = ThreadPoolExecutor(max_workers=3)
		
		# Cache for latest buffer subfolder (avoid repeated file system calls)
		self._cached_latest_subfolder = None
		self._cached_subfolder_time = 0.0
		self._subfolder_cache_ttl = 1.0  # Refresh cache every 1 second
		
		# GP fitting state - MULTI-OBJECT SUPPORT
		self.gp_fit_lock = threading.Lock()
		self.gp_fitting_active = False
		# MULTI-OBJECT: Dictionary mapping vlm_answer -> GP params
		# Replaces single global_gp_params to support multiple objects
		self.per_cause_gp_params = {}  # Dict[str, Dict[str, Any]] - vlm_answer -> GP params
		# Legacy: Keep global_gp_params for backward compatibility (most recent)
		self.global_gp_params = None
		# Note: nominal_points / disturbances no longer stored separately;
		# XtX_inv and hess_inv are precomputed in fit_params for uncertainty.
		self.last_gp_update_time = 0.0
		self.gp_update_interval = 0.75
		self.gp_computation_thread = None
		self.gp_thread_lock = threading.Lock()
		self.gp_thread_running = False
		self.min_radius = 0.5
		self.max_radius = 2.0
		self.base_radius = 1.0
		# Validate GP combination method
		if self.gp_combination_method not in ['sum', 'max']:
			self.get_logger().warn(f"Invalid gp_combination_method '{self.gp_combination_method}', using 'sum'")
			self.gp_combination_method = 'sum'
		self.get_logger().info(f"GP combination method: {self.gp_combination_method} (for multi-object GP costmap)")
		
		# Robot-centric 3D grid parameters (NEW)
		self.robot_grid_size_xy = 10.0  # 10m x 10m in XY plane
		self.robot_grid_size_z = 4.0    # 4m in Z axis
		self.robot_grid_resolution = 0.2  # 0.2m voxel resolution
		self.robot_position = None  # Current robot position (from latest_pose)
		
		# GPU tensor for mean and uncertainty fields (Channel=2, Depth, Height, Width)
		try:
			import torch
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			self.gp_grid_tensor = None  # Will be initialized on first update
			self.TORCH_AVAILABLE = True
			self.get_logger().info(f"PyTorch GPU tensor backend: {self.device}")
		except ImportError:
			self.TORCH_AVAILABLE = False
			self.get_logger().warn("PyTorch not available, using NumPy fallback")
		
		# PathManager initialization
		self.path_manager = None
		if PATH_MANAGER_AVAILABLE:
			try:
				path_config = None
				if isinstance(self.main_config_path, str) and len(self.main_config_path) > 0:
					import yaml
					with open(self.main_config_path, 'r') as f:
						cfg = yaml.safe_load(f)
					path_config = cfg.get('path_mode', {}) if isinstance(cfg, dict) else {}
				else:
					try:
						from ament_index_python.packages import get_package_share_directory
						package_dir = get_package_share_directory('resilience')
						default_main = os.path.join(package_dir, 'config', 'main_config.yaml')
						import yaml
						with open(default_main, 'r') as f:
							cfg = yaml.safe_load(f)
						path_config = cfg.get('path_mode', {}) if isinstance(cfg, dict) else {}
					except Exception:
						path_config = {}
				self.path_manager = PathManager(self, path_config)
				self.get_logger().info("PathManager initialized for nominal path access (non-blocking)")
			except Exception as e:
				self.get_logger().warn(f"Failed to initialize PathManager: {e}")
		
		# Simple event-driven processing - messages processed directly in callback
		self._latest_pose_rays = None  # (origin_world np.array(3,), dirs np.array(N,3))
		
		# Initialize unified VDB mapper (occupancy + frontiers + rays)
		self._initialize_vdb_mapper()
		
		# Create alias for backward compatibility
		self.rf_sem_map = self.vdb_mapper
		
		# Semantic voxel tracking with RayFronts-style confidence accumulation
		self.semantic_voxels = {}  # voxel_key -> {'vlm_answer': str, 'similarity': float, 'timestamp': float, 'position': np.array, 'confidence': float}
		self.semantic_voxels_lock = threading.Lock()
		
		# SPATIAL INDEX: KD-tree for fast neighbor queries (O(log N) instead of O(N))
		# Rebuilt whenever semantic voxels change significantly
		self.semantic_spatial_index = None  # scipy.spatial.cKDTree
		self.semantic_voxel_keys_indexed = []  # List of voxel keys aligned with spatial index
		self.spatial_index_dirty = True  # Flag to rebuild index when voxels change
		
		# Temporal confirmation: track observations for each voxel
		self.semantic_voxel_observations = {}  # voxel_key -> [{'vlm_answer': str, 'timestamp': float, 'frame_id': int}, ...]
		self.narration_confirmation_threshold = 1  # Narration: instant confirmation (1 frame)
		self.operational_confirmation_threshold = 3  # Operational: require 2 frames for noise rejection (non-blocking, incremental)
		self.semantic_observation_max_age = 5.0  # Keep observations for 5 seconds
		self.frame_counter = 0  # Track unique frames for operational hotspots
		
		# OPTIMIZED: Incremental spatial observation counts for fast threshold checks
		# Structure: (voxel_key, vlm_answer) -> {'count': int, 'unique_frames': set, 'last_update': float}
		# Updated incrementally when observations are added (O(1) threshold checks)
		self.spatial_observation_counts = {}  # (voxel_key, vlm_answer) -> {'count': int, 'unique_frames': set, 'last_update': float}
		# Accumulated pose-ray bins (match RayFronts behavior)
		self.pose_rays_orig_angles = None
		self.pose_rays_feats_cnt = None
		
		# Load existing embeddings
		if isinstance(self.buffers_directory, str) and len(self.buffers_directory) > 0:
			self.get_logger().info(f"Buffers directory: {self.buffers_directory}")
		
		# Simple GP visualization
		self.get_logger().info("Simple GP visualization system initialized")
		
		# Load pre-loaded GP parameters from registry at startup
		self._load_preloaded_gp_params_from_main_config()
		
		self._start_gp_computation_thread()
		
		# Precompute transforms
		self.R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
		self.R_cam_to_base_extra = self._rpy_deg_to_rot(self.cam_to_base_rpy_deg)
		self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)
		
		# QoS
		sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
		
		# Subscribers
		self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
		self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
		self.create_subscription(Odometry, '/Odometry', self.pose_callback, 10)
		if self.enable_semantic_mapping and self.enable_voxel_mapping:
			self.create_subscription(String, self.semantic_hotspots_topic, self.semantic_hotspot_callback, 10)
			mask_sub = self.create_subscription(Image, self.semantic_hotspot_mask_topic, self.semantic_hotspot_mask_callback, 10)
			self.get_logger().info(f"Created mask subscriber on topic: {self.semantic_hotspot_mask_topic}")
		
		# Publishers
		self.marker_pub = self.create_publisher(MarkerArray, self.semantic_octomap_markers_topic, 10) if self.publish_markers else None
		self.stats_pub = self.create_publisher(String, self.semantic_octomap_stats_topic, 10) if self.publish_stats else None
		self.cloud_pub = self.create_publisher(PointCloud2, self.semantic_octomap_colored_cloud_topic, 10) if self.publish_colored_cloud else None
		self.semantic_only_pub = self.create_publisher(PointCloud2, self.semantic_voxels_only_topic, 10) if self.publish_colored_cloud else None
		
		# Cause registry integration REMOVED:
		# We now load GP params directly from `config/main_config.yaml` (preloaded_objects[].gp_params_file)
		# and use that GP for all semantic voxels.
		self.gp_visualization_pub = self.create_publisher(PointCloud2, '/gp_field_visualization', 10)
		self.costmap_pub = self.create_publisher(PointCloud2, '/semantic_costmap', 10)
		self.gp_uncertainty_pub = self.create_publisher(PointCloud2, '/gp_uncertainty_field', 10)
		# New: frontiers and rays publishers
		self.frontiers_pub = self.create_publisher(PointCloud2, '/vdb_frontiers', 10)
		self.mask_frontiers_pub = self.create_publisher(PointCloud2, '/mask_frontiers', 10)
		
		# Sphere markers for semantic voxels (LaC-style)
		self.hazard_spheres_pub = self.create_publisher(MarkerArray, '/lac/hazard_spheres', 10)
		self.mask_rays_pub = self.create_publisher(MarkerArray, '/mask_rays', 10)
		# Raw GP grid publisher for control
		self.gp_grid_raw_pub = self.create_publisher(Float32MultiArray, '/gp_grid_raw', 10)
		self.voxel_resolution = 0.2

		# Mission metrics (must come after path_manager and voxel_resolution are set)
		self._init_mission_metrics()

		self.get_logger().info("=" * 60)
		self.get_logger().info("SEMANTIC VDB MAPPING SYSTEM READY")
		self.get_logger().info("=" * 60)
		self.get_logger().info(f"Mapping Configuration:")
		self.get_logger().info(f"   Mapper: SemanticRayFrontiersMap (OpenVDB)")
		self.get_logger().info(f"   Device: {self.vdb_mapper.device}")
		self.get_logger().info(f"   Voxel resolution: {self.voxel_resolution}m")
		self.get_logger().info(f"   Max range: {self.max_range}m")
		self.get_logger().info(f"   Min range: {self.min_range}m")
		self.get_logger().info(f"Feature Status:")
		self.get_logger().info(f"   VDB occupancy mapping: ENABLED")
		self.get_logger().info(f"   Semantic mapping: {'ENABLED' if self.enable_semantic_mapping else 'DISABLED'}")
		self.get_logger().info(f"   VDB mapper: {'READY' if hasattr(self, 'vdb_mapper') and self.vdb_mapper is not None else 'NOT READY'}")
		self.get_logger().info(f"Topics:")
		self.get_logger().info(f"   Depth: {self.depth_topic}")
		self.get_logger().info(f"   Pose: {self.pose_topic}")
		self.get_logger().info(f"   Semantic hotspots: {self.semantic_hotspots_topic}")
		self.get_logger().info("=" * 60)
		
	def load_topic_configuration(self):
		"""Load topic configuration from mapping config file."""
		try:
			import yaml
			if self.mapping_config_path:
				config_path = self.mapping_config_path
			else:
				# Use default config path
				from ament_index_python.packages import get_package_share_directory
				package_dir = get_package_share_directory('resilience')
				config_path = os.path.join(package_dir, 'config', 'mapping_config.yaml')
			
			with open(config_path, 'r') as f:
				config = yaml.safe_load(f)
			
			# Extract topic configuration
			topics = config.get('topics', {})
			
			# Input topics
			self.depth_topic = topics.get('depth_topic', '/robot_1/sensors/front_stereo/depth/depth_registered')
			self.camera_info_topic = topics.get('camera_info_topic', '/robot_1/sensors/front_stereo/left/camera_info')
			self.pose_topic = topics.get('pose_topic', '/robot_1/sensors/front_stereo/pose')
			self.semantic_hotspots_topic = topics.get('semantic_hotspots_topic', '/semantic_hotspots')
			self.semantic_hotspot_mask_topic = topics.get('semantic_hotspot_mask_topic', '/semantic_hotspot_mask')
			
			# Output topics
			self.semantic_octomap_markers_topic = topics.get('semantic_octomap_markers_topic', '/semantic_octomap_markers')
			self.semantic_octomap_stats_topic = topics.get('semantic_octomap_stats_topic', '/semantic_octomap_stats')
			self.semantic_octomap_colored_cloud_topic = topics.get('semantic_octomap_colored_cloud_topic', '/semantic_octomap_colored_cloud')
			self.semantic_voxels_only_topic = topics.get('semantic_voxels_only_topic', '/semantic_voxels_only')

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
			
			self.get_logger().info(f"Topic configuration loaded from: {config_path}")
			
		except Exception as e:
			self.get_logger().warn(f"Using default topic configuration: {e}")
			# Fallback to default topics
			self.depth_topic = '/robot_1/sensors/front_stereo/depth/depth_registered'
			self.camera_info_topic = '/robot_1/sensors/front_stereo/left/camera_info'
			self.pose_topic = '/robot_1/sensors/front_stereo/pose'
			self.semantic_hotspots_topic = '/semantic_hotspots'
			self.semantic_hotspot_mask_topic = '/semantic_hotspot_mask'
			self.semantic_octomap_markers_topic = '/semantic_octomap_markers'
			self.semantic_octomap_stats_topic = '/semantic_octomap_stats'
			self.semantic_octomap_colored_cloud_topic = '/semantic_octomap_colored_cloud'
			self.semantic_voxels_only_topic = '/semantic_voxels_only'
			self.get_logger().info("Using default topic configuration")

	def _initialize_vdb_mapper(self):
		"""Initialize the RayFronts VDB occupancy mapper with noise-robust parameters."""
		try:
			# Create dummy intrinsics (will be updated when camera info is received)
			dummy_intrinsics = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
			
			# Initialize SemanticRayFrontiersMap as the primary VDB mapper
			# This includes occupancy, frontiers, and rays all in one
			self.vdb_mapper = SemanticRayFrontiersMap(
				intrinsics_3x3=dummy_intrinsics,
				device=("cuda" if torch.cuda.is_available() else "cpu"),
				visualizer=None,
				clip_bbox=None,
				encoder=None,
				feat_compressor=None,
				interp_mode="bilinear",
				max_pts_per_frame=2000,  # Increased for better coverage
				vox_size=float(self.voxel_resolution),
				vox_accum_period=3,  # Accumulate over 2 frames for smoother updates
				max_empty_pts_per_frame=2000,  # Increased for better free space clearing
				max_rays_per_frame=2000,
				max_depth_sensing=2.5,  # 1.5m for voxelization and frontiers
				max_empty_cnt=8,  # Increased: require more evidence before removing voxels (reduces flicker)
				max_occ_cnt=7,  # Increased: require more confirmation before marking occupied (reduces noise)
				occ_observ_weight=3,  # Reduced: less aggressive updates per observation (smoother)
				occ_thickness=1,  # Increased: thicker occupied surface (more robust)
				occ_pruning_tolerance=2,  # Increased: more forgiving pruning (keeps stable voxels)
				occ_pruning_period=3,  # Increased: prune less frequently (more stable map)
				sem_pruning_thresh=0,
				sem_pruning_period=1,
				fronti_neighborhood_r=1,
				fronti_min_unobserved=9,
				fronti_min_empty=4,
				fronti_min_occupied=0,
				fronti_subsampling=4,
				fronti_subsampling_min_fronti=5,
				ray_accum_period=1,
				ray_accum_phase=0,
				angle_bin_size=30.0,
				ray_erosion=0,
				ray_tracing=False,
				global_encoding=True,
				zero_depth_mode=False,
				infer_direction=False,
			)
			
			# Set dummy encoder for SemanticRayFrontiersMap
			if self.vdb_mapper is not None:
				self.vdb_mapper.encoder = _ZeroImageEncoder(self.embedding_dim, self.vdb_mapper.device)
			
			self.get_logger().info(f"VDB SemanticRayFrontiersMap initialized (device: {self.vdb_mapper.device})")
			self.get_logger().info(f"Unified mapper settings:")
			self.get_logger().info(f"   Max depth sensing: 1.5m (voxelization and frontiers)")
			self.get_logger().info(f"   Empty count: 8 (stable free space)")
			self.get_logger().info(f"   Occupied count: 7 (confirmed occupancy)")
			self.get_logger().info(f"   Observation weight: 3 (smooth updates)")
			self.get_logger().info(f"   Surface thickness: 3 voxels (robust surfaces)")
			
		except Exception as e:
			self.get_logger().error(f"Failed to initialize VDB mapper: {e}")
			import traceback
			traceback.print_exc()
			raise

	def _rotate_image_if_needed(self, img: np.ndarray) -> np.ndarray:
		if self.image_rotation_deg == 180 and img is not None:
			return cv2.rotate(img, cv2.ROTATE_180)
		return img

	def camera_info_callback(self, msg: CameraInfo):
		if self.camera_intrinsics is None:
			self._camera_width = int(getattr(msg, "width", 0)) if getattr(msg, "width", 0) else None
			self._camera_height = int(getattr(msg, "height", 0)) if getattr(msg, "height", 0) else None

			fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
			if self.image_rotation_deg == 180 and self._camera_width and self._camera_height:
				cx = (float(self._camera_width) - 1.0) - float(cx)
				cy = (float(self._camera_height) - 1.0) - float(cy)

			# Update VDB mapper intrinsics
			intrinsics = torch.tensor([
				[fx, msg.k[1], cx],
				[msg.k[3], fy, cy],
				[msg.k[6], msg.k[7], msg.k[8]]
			], dtype=torch.float32)
			
			self.vdb_mapper.intrinsics_3x3 = intrinsics.to(self.vdb_mapper.device)
			self.camera_intrinsics = [fx, fy, cx, cy]
			self.get_logger().info(f"Camera intrinsics set: fx={fx:.2f}, fy={fy:.2f}")
		# Update activity
		self.last_data_time = time.time()

	def pose_callback(self, msg: Odometry):
		self.latest_pose = msg
		try:
			self.latest_pose_frame_id = str(getattr(msg.header, "frame_id", "") or "")
		except Exception:
			self.latest_pose_frame_id = None
		# Update robot position for robot-centric grid (NEW)
		pose = msg.pose.pose
		self.robot_position = np.array([
			pose.position.x,
			pose.position.y,
			pose.position.z
		], dtype=np.float32)
		# Push into pose buffer with timestamp
		
		pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.pose_buffer_data.append(msg)
		self.pose_buffer_ts.append(pose_time)
		self.last_data_time = time.time()
		# Update mission metrics on every odometry message
		self._update_mission_metrics(msg)

	# =========================================================================
	# MISSION METRICS
	# =========================================================================

	def _init_mission_metrics(self):
		"""
		Initialise all state for the three mission metrics.

		DESIGN CHOICES (document here for full reproducibility)
		────────────────────────────────────────────────────────
		START_MOVE_DIST_M  (default 0.5 m)
		  The robot must travel at least this distance from its position at
		  path-availability time before "mission start" is declared.  This
		  prevents pre-takeoff hovering or arming delays from inflating T_arr
		  and L_path.  Increase if the drone drifts while arming.

		GOAL_RADIUS_M  (default 1.5 m)
		  Goal arrival is declared when the robot enters a sphere of this
		  radius around the LAST waypoint of the nominal path.  Should match
		  (or be slightly larger than) the planner's own acceptance radius.

		MIN_POS_STEP_M  (default 0.01 m)
		  Only pose samples that differ by at least this distance from the
		  previous accepted sample are accumulated into L_path.  Removes
		  sub-centimetre odometry noise jitter without affecting real motion.

		REFERENCE TRAJECTORY τ_ref(t_i)
		  At every timestep the reference point is the NEAREST NEIGHBOUR on
		  the stored nominal-path array — no interpolation.  This is O(K)
		  per step (K = number of nominal waypoints, typically ≤ 500),
		  deterministic, and straightforward to reproduce independently.
		  Alternative: project observed position onto the nearest path
		  *segment* (gives slightly smaller disturbance values for curved
		  paths, but requires more code and is harder to explain).

		Δt COMPUTATION
		  Taken from consecutive ROS header stamps (msg.header.stamp) on the
		  odometry topic — NOT wall-clock time.  This is robust to system
		  load spikes and gives identical results when replaying a bag file.
		  A per-step sanity cap of 1.0 s is applied: gaps larger than 1 s
		  (e.g. from a topic outage) are skipped so they don't inflate D_cum.

		D_cum DISTURBANCE THRESHOLD
		  Only timesteps where ||τ_obs(t_i) - τ_ref(t_i)||_2 > 0.1 m are
		  accumulated into D_cum.  This filters out small tracking errors
		  (odometry noise, minor controller jitter) and focuses on significant
		  deviations from the nominal path.  The threshold of 0.1 m is chosen
		  to be above typical odometry noise (~0.01-0.05 m) but below meaningful
		  disturbance events.

		L_nom COMPUTATION
		  Computed once, as the sum of Euclidean distances between consecutive
		  nominal waypoints.  This matches the denominator in D̄_cum exactly
		  and is independent of how fast the robot actually moves.

		NOMINAL PATH SOURCE  (priority order)
		  1. self.path_manager.get_nominal_points_as_numpy()
		  2. np.load(self.nominal_path)  [file path parameter]
		  Whichever is found first with ≥ 2 waypoints is used and cached.
		"""
		# ── Tuneable constants ──────────────────────────────────────────────
		self._M_START_MOVE_DIST  = 0.50   # m  – start trigger distance
		self._M_GOAL_RADIUS      = 1.50   # m  – goal acceptance radius
		self._M_MIN_POS_STEP     = 0.01   # m  – L_path noise filter
		self._M_DT_SANITY_CAP    = 1.00   # s  – max plausible Δt per step
		self._M_DISTURBANCE_THRESHOLD = 0.10  # m  – D_cum only accumulates if ||τ_obs - τ_ref|| > this

		# ── Nominal-path cache ──────────────────────────────────────────────
		self._m_nominal_pts  = None   # np.ndarray (K, 3) or None
		self._m_l_nom        = None   # float: length of nominal path [m]

		# ── State machine ───────────────────────────────────────────────────
		self._m_path_ready        = False   # nominal path has been loaded
		self._m_pos_at_path_ready = None    # np.array(3,): position when path arrived
		self._m_mission_started   = False
		self._m_goal_reached      = False

		# ── Timestamps (seconds, from ROS header stamps) ────────────────────
		self._m_start_time = None   # float: stamp at mission start
		self._m_goal_time  = None   # float: stamp at goal arrival

		# ── L_path accumulation ─────────────────────────────────────────────
		self._m_path_length = 0.0       # metres
		self._m_last_pos    = None      # np.array(3,): last accepted pose

		# ── D_cum accumulation ──────────────────────────────────────────────
		self._m_d_cum        = 0.0    # Σ ||τ_obs - τ_ref|| · Δt  [m·s]
		self._m_last_stamp   = None   # float: previous ROS stamp

		# ── Optional full trajectory log ────────────────────────────────────
		# Stores every pose position & stamp after mission start for
		# offline re-analysis.  Memory cost: ~24 bytes per sample.
		self._m_traj_positions  = []   # list[np.array(3,)]
		self._m_traj_timestamps = []   # list[float]

	def _get_nominal_path_for_metrics(self) -> Optional[np.ndarray]:
		"""
		Return the cached nominal path (K×3 float32).
		Tries PathManager first, then the nominal_path file parameter.
		Computes and caches L_nom on first successful load.
		"""
		if self._m_nominal_pts is not None:
			return self._m_nominal_pts

		pts = None
		# Priority 1: PathManager (live path, updated during mission)
		if self.path_manager is not None:
			try:
				pts = self.path_manager.get_nominal_points_as_numpy()
				if pts is not None and len(pts) < 2:
					pts = None
			except Exception:
				pts = None

		# Priority 2: nominal_path file parameter
		if pts is None and isinstance(self.nominal_path, str) and self.nominal_path:
			try:
				raw = np.load(self.nominal_path)
				if raw.ndim == 2 and raw.shape[1] >= 3 and len(raw) >= 2:
					pts = raw[:, :3]
			except Exception:
				pts = None

		if pts is not None and len(pts) >= 2:
			self._m_nominal_pts = pts.astype(np.float32)
			diffs = np.linalg.norm(np.diff(self._m_nominal_pts, axis=0), axis=1)
			self._m_l_nom = float(np.sum(diffs))
			self.get_logger().warn(
				f"[Metrics] Nominal path loaded: {len(pts)} waypoints, "
				f"L_nom = {self._m_l_nom:.3f} m"
			)

		return self._m_nominal_pts

	def _nearest_nominal_point(self, pos: np.ndarray) -> Optional[np.ndarray]:
		"""
		Return the nearest point on the nominal path to *pos* using
		nearest-neighbour search (no interpolation).  O(K) per call.
		"""
		pts = self._m_nominal_pts   # already fetched & cached by caller
		if pts is None:
			return None
		dists = np.linalg.norm(pts - pos, axis=1)
		return pts[int(np.argmin(dists))]

	def _update_mission_metrics(self, msg) -> None:
		"""
		Called from pose_callback on every odometry message.
		Drives the mission-start state machine and accumulates all three metrics.
		All writes are from the single ROS callback thread → no locking needed.
		"""
		try:
			stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
			pose = msg.pose.pose
			pos = np.array([
				pose.position.x,
				pose.position.y,
				pose.position.z,
			], dtype=np.float32)

			# ── 1. Wait for nominal path ─────────────────────────────────────
			if not self._m_path_ready:
				pts = self._get_nominal_path_for_metrics()
				if pts is None:
					return   # nothing to do until path is available
				self._m_path_ready = True
				self._m_pos_at_path_ready = pos.copy()
				self.get_logger().warn(
					f"[Metrics] Nominal path available at pos="
					f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]. "
					f"Waiting for robot to move > {self._M_START_MOVE_DIST:.2f} m "
					f"to declare mission start."
				)
				return

			# ── 2. Declare mission start ──────────────────────────────────────
			if not self._m_mission_started:
				dist_moved = float(np.linalg.norm(pos - self._m_pos_at_path_ready))
				if dist_moved >= self._M_START_MOVE_DIST:
					self._m_mission_started = True
					self._m_start_time  = stamp
					self._m_last_pos    = pos.copy()
					self._m_last_stamp  = stamp
					self.get_logger().warn(
						f"[Metrics] *** MISSION STARTED *** "
						f"stamp={stamp:.3f} s, "
						f"dist_moved_from_path_ready={dist_moved:.2f} m"
					)
				return   # don't accumulate before mission start

			# ── 3. L_path: accumulate displacement ───────────────────────────
			if self._m_last_pos is not None:
				step = float(np.linalg.norm(pos - self._m_last_pos))
				if step >= self._M_MIN_POS_STEP:
					self._m_path_length += step
					self._m_last_pos = pos.copy()

			# ── 4. D_cum: accumulate ||τ_obs − τ_ref|| · Δt ──────────────────
			# Only accumulate if disturbance exceeds threshold (filters noise)
			if self._m_last_stamp is not None:
				dt = stamp - self._m_last_stamp
				if 0.0 < dt <= self._M_DT_SANITY_CAP:
					ref_pt = self._nearest_nominal_point(pos)
					if ref_pt is not None:
						disturbance = float(np.linalg.norm(pos - ref_pt))
						if disturbance > self._M_DISTURBANCE_THRESHOLD:
							self._m_d_cum += disturbance * dt
			self._m_last_stamp = stamp

			# ── 5. Store trajectory sample ───────────────────────────────────
			self._m_traj_positions.append(pos.copy())
			self._m_traj_timestamps.append(stamp)

			# ── 6. Goal arrival check ─────────────────────────────────────────
			if not self._m_goal_reached and self._m_nominal_pts is not None:
				dist_to_goal = float(np.linalg.norm(pos - self._m_nominal_pts[-1]))
				if dist_to_goal <= self._M_GOAL_RADIUS:
					self._m_goal_reached = True
					self._m_goal_time   = stamp
					t_arr = stamp - self._m_start_time
					self.get_logger().warn(
						f"[Metrics] *** GOAL REACHED *** "
						f"T_arr = {t_arr:.3f} s, "
						f"dist_to_last_waypoint = {dist_to_goal:.3f} m"
					)

		except Exception as e:
			self.get_logger().debug(f"[Metrics] _update_mission_metrics error (non-fatal): {e}")

	def log_mission_metrics(self) -> None:
		"""
		Print a full metrics report to both the ROS logger (WARN level) and
		directly to stdout so it is always visible when the user presses Ctrl+C.

		Called from the main() finally block on shutdown.
		"""
		SEP   = "=" * 68
		SEP_S = "-" * 68

		# ── Compute derived values ──────────────────────────────────────────
		if not self._m_mission_started:
			elapsed_note = "mission never started (robot did not move far enough while nominal path was available)"
			t_arr_str    = "N/A"
			t_arr_val    = None
			l_path_str   = "N/A"
			l_path_val   = None
			d_cum_str    = "N/A"
			d_cum_bar_str = "N/A"
			d_cum_bar_val = None
		else:
			elapsed = (
				(self._m_traj_timestamps[-1] - self._m_start_time)
				if self._m_traj_timestamps else 0.0
			)
			elapsed_note = f"elapsed since mission start = {elapsed:.2f} s"

			if self._m_goal_reached and self._m_goal_time is not None:
				t_arr_val = self._m_goal_time - self._m_start_time
				t_arr_str = f"{t_arr_val:.3f} s  [GOAL REACHED]"
			else:
				# T_arr = elapsed time from mission start to current time (even if goal not reached)
				t_arr_val = elapsed
				t_arr_str = f"{t_arr_val:.3f} s  (goal not reached; {elapsed_note})"

			l_path_val = self._m_path_length
			l_path_str = f"{l_path_val:.3f} m"

			if self._m_l_nom is not None and self._m_l_nom > 0.0:
				d_cum_bar_val = self._m_d_cum / self._m_l_nom
				d_cum_str     = f"{self._m_d_cum:.4f} m·s"
				d_cum_bar_str = (
					f"{d_cum_bar_val:.6f}"
					f"  (D_cum={self._m_d_cum:.4f} m·s,"
					f"  L_nom={self._m_l_nom:.3f} m)"
				)
			else:
				d_cum_bar_val = None
				d_cum_str     = f"{self._m_d_cum:.4f} m·s"
				d_cum_bar_str = f"N/A (nominal path unavailable; raw D_cum = {self._m_d_cum:.4f} m·s)"

		# ── Single-line summary for terminal (always printed first) ──────────
		if self._m_mission_started:
			t_arr_compact = f"{t_arr_val:.3f}" if t_arr_val is not None else "N/A"
			l_path_compact = f"{l_path_val:.3f}" if l_path_val is not None else "N/A"
			d_cum_bar_compact = f"{d_cum_bar_val:.6f}" if d_cum_bar_val is not None else "N/A"
			single_line = (
				f"[METRICS] T_arr={t_arr_compact}s | "
				f"L_path={l_path_compact}m | "
				f"D_cum_bar={d_cum_bar_compact}"
			)
		else:
			single_line = "[METRICS] Mission not started (robot did not move far enough)"

		# ── Format lines ────────────────────────────────────────────────────
		lines = [
			SEP,
			"  MISSION METRICS SUMMARY",
			SEP,
			f"  T_arr  (Arrival Time)                : {t_arr_str}",
			f"  L_path (Total Path Length)           : {l_path_str}",
			f"  D_cum  (Raw Cumulative Disturbance)  : {d_cum_str}",
			f"  D_cum_bar (Normalised Cum. Dist.)    : {d_cum_bar_str}",
			SEP_S,
			"  DESIGN CHOICES (for reproducibility):",
			f"    Mission-start trigger  : nominal path available"
			f" AND ||pos - pos_at_path_ready||_2 > {self._M_START_MOVE_DIST} m",
			f"    Goal-arrival criterion : ||pos - last_nominal_waypoint||_2 < {self._M_GOAL_RADIUS} m",
			f"    L_path noise filter    : step < {self._M_MIN_POS_STEP} m is discarded",
			f"    τ_ref matching         : nearest-neighbour on nominal-path array (no interpolation)",
			f"    Δt source              : ROS header stamps (msg.header.stamp), sanity cap = {self._M_DT_SANITY_CAP} s",
			f"    D_cum threshold        : only ||τ_obs - τ_ref|| > {self._M_DISTURBANCE_THRESHOLD} m is accumulated",
			f"    L_nom computation      : Σ ||waypoint_{{i+1}} - waypoint_i||_2 over nominal path",
			f"    Nominal path source    : PathManager (primary) → nominal_path file (fallback)",
			f"    Trajectory samples     : {len(self._m_traj_positions)} poses recorded after mission start",
			SEP,
		]

		# ── ROS logger (always visible at WARN level) ───────────────────────
		for line in lines:
			self.get_logger().warn(line)

		# ── stdout (guaranteed even if ROS logger is redirected) ─────────────
		# Print single-line summary first (always visible)
		print(single_line, flush=True)
		print("\n")
		for line in lines:
			print(line)
		print("\n", flush=True)

	# =========================================================================
	# END MISSION METRICS
	# =========================================================================

	def semantic_hotspot_mask_callback(self, msg: Image):
		"""Buffer the merged hotspot mask image keyed by its stamp time."""
		mask_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
		mask_rgb = self._rotate_image_if_needed(mask_rgb)
		mask_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.mask_buffer_data.append(mask_rgb)
		self.mask_buffer_ts.append(mask_time)
		self.last_data_time = time.time()
		# Helpful debug for sync issues (kept at debug to avoid log spam).
		self.get_logger().debug(
			f"Buffered hotspot mask image @ {mask_time:.6f} (mask_buffer={len(self.mask_buffer_ts)})"
		)


	def semantic_hotspot_callback(self, msg: String):
		"""Process incoming semantic hotspot metadata using an efficient thread pool."""
		if not self.enable_semantic_mapping or not self.enable_voxel_mapping:
			return

		# Submit the task to the pool instead of spawning a new thread
		self.hotspot_executor.submit(self._process_single_bridge_message, msg.data)
		
		self.last_data_time = time.time()

	def _process_single_bridge_message(self, msg_data: str) -> bool:
		"""Process a single bridge message and apply to voxel map by timestamp lookup."""
		
		# Parse the JSON message
		time_start = time.time()
		data = json.loads(msg_data)
		json_load_time = time.time() - time_start
		self.get_logger().warn(f"Time taken to load JSON: {json_load_time}")
		if data.get('type') == 'merged_similarity_hotspots':
			return self._process_merged_hotspot_message(data)
		else:
			return False

	
	def _precompute_color_indices(self, merged_mask: np.ndarray, vlm_info: dict) -> dict:
		"""Pre-compute pixel indices for each color once (fixes bottleneck #1).
		
		Returns dict mapping vlm_answer -> (v_coords, u_coords) numpy arrays.
		"""
		color_to_indices = {}
		h, w = merged_mask.shape[:2]
		
		# Vectorized approach: flatten and find matches
		mask_flat = merged_mask.reshape(-1, 3)  # (H*W, 3)
		
		for vlm_answer, info in vlm_info.items():
			color = np.array(info.get('color', [0, 0, 0]), dtype=np.uint8)
			# Vectorized comparison (much faster than per-pixel loop)
			matches = np.all(mask_flat == color, axis=1)
			if np.any(matches):
				indices = np.where(matches)[0]
				v_coords = indices // w
				u_coords = indices % w
				color_to_indices[vlm_answer] = (v_coords, u_coords)
			else:
				color_to_indices[vlm_answer] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
		
		return color_to_indices
	
	def _decode_hotspot_mask_png(self, mask_b64: str) -> Optional[np.ndarray]:
		"""Decode a binary hotspot mask from a base64-encoded PNG string."""
		try:
			if not mask_b64:
				return None
			mask_bytes = base64.b64decode(mask_b64.encode('ascii'))
			mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
			mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
			if mask_img is None:
				return None
			# Return a binary mask (0/1 as uint8)
			return (mask_img > 0).astype(np.uint8)
		except Exception:
			self.get_logger().warn("Failed to decode hotspot mask PNG from metadata")
			return None
	
	def _process_merged_hotspot_message(self, data: dict) -> bool:
		"""Process merged hotspot metadata; prefer embedded masks, fall back to image-based decode."""
		try:
			is_narration = data.get('is_narration')
			vlm_info = data.get('vlm_info', {})
			rgb_timestamp = float(data.get('timestamp', 0.0))
			buffer_id = data.get('buffer_id')  # Extract buffer_id

			hotspot_type = "NARRATION" if is_narration else "OPERATIONAL"
			# NOTE: This node runs at WARN level by default; use WARN so these show up.
			if is_narration:
				self.get_logger().warn(
					f"Hotspot metadata received ({hotspot_type}): "
					f"ts={rgb_timestamp:.6f}, answers={list(vlm_info.keys())}, buffer_id={buffer_id}"
				)
			else:
				self.get_logger().debug(
					f"Hotspot metadata received ({hotspot_type}): "
					f"ts={rgb_timestamp:.6f}, answers={list(vlm_info.keys())}, buffer_id={buffer_id}"
				)
			
			if rgb_timestamp <= 0.0:
				self.get_logger().warn(f"Incomplete hotspot data (no timestamp)")
				return False
			
			# Check if embedded masks are available in the metadata (preferred path)
			embedded_masks_available = any(
				isinstance(info, dict) and info.get('mask_png') is not None
				for info in vlm_info.values()
			)
			
			start = time.time()
			
			# Lookup closest depth frame and pose by timestamp (needed for both paths)
			depth_msg, pose_msg = self._lookup_depth_and_pose(rgb_timestamp)
			depth_lookup_time = time.time() - start
			self.get_logger().warn(f"Time taken to lookup depth: {depth_lookup_time}")
			if depth_msg is None or pose_msg is None:
				self.get_logger().warn(
					f"No matching depth/pose found for timestamp {rgb_timestamp:.6f} "
					f"(depth_buf={len(self.depth_buffer_ts)}, pose_buf={len(self.pose_buffer_ts)}, "
					f"mask_buf={len(self.mask_buffer_ts)}, sync_window_s={float(self.sync_buffer_duration):.3f})"
				)
				return False
			
			try:
				# Convert depth exactly like the regular mapping path (encoding-safe).
				depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
				depth_raw = self._rotate_image_if_needed(depth_raw)
				depth_image = self._depth_to_meters(depth_raw, getattr(depth_msg, "encoding", ""))
				if depth_image is None:
					self.get_logger().warn("Hotspot depth conversion produced None; skipping hotspot")
					return False

				# Quick sanity check for unit/encoding issues (visible at WARN by default).
				try:
					finite = np.isfinite(depth_image)
					if np.any(finite):
						dmin = float(np.min(depth_image[finite]))
						dmax = float(np.max(depth_image[finite]))
					else:
						dmin, dmax = float("nan"), float("nan")
					self.get_logger().warn(
						f"Hotspot depth stats: encoding='{getattr(depth_msg,'encoding','')}', "
						f"min={dmin:.4f}m, max={dmax:.4f}m"
					)
				except Exception:
					pass
			except Exception as e:
				self.get_logger().error(f"Failed to convert depth message: {e}")
				return False
			
			processed_count = 0
			
			if embedded_masks_available:
				# Directly decode and use per-answer masks from metadata (no image-based color decoding)
				for vlm_answer, info in vlm_info.items():
					mask_b64 = None
					if isinstance(info, dict):
						mask_b64 = info.get('mask_png')
					if not mask_b64:
						continue
					
					vlm_mask = self._decode_hotspot_mask_png(mask_b64)
					if vlm_mask is None or not np.any(vlm_mask):
						continue
					
					vlm_mask_time = time.time() - start - depth_lookup_time
					self.get_logger().debug(f"Time taken to decode vlm mask: {vlm_mask_time:.4f}s (embedded)")
					used_ts = 1.0
					success = self._process_hotspot_with_depth(
						vlm_mask, pose_msg, depth_image, vlm_answer,
						info.get('hotspot_threshold', 0.6) if isinstance(info, dict) else 0.6,
						{'hotspot_pixels': (info.get('hotspot_pixels', 0) if isinstance(info, dict) else 0)},
						rgb_timestamp, used_ts, is_narration, buffer_id
					)
					hotspot_processing_time = time.time() - start - depth_lookup_time - vlm_mask_time
					self.get_logger().debug(f"Time taken to process hotspot: {hotspot_processing_time:.4f}s (embedded)")
					if success:
						processed_count += 1
						if is_narration and len(vlm_info) == 1 and isinstance(info, dict):
							self.get_logger().info(
								f"NARRATION HOTSPOT PROCESSED (embedded): '{vlm_answer}' with {info.get('hotspot_pixels', 0)} pixels"
							)
				
				self.get_logger().info(f"Processed {processed_count}/{len(vlm_info)} VLM answers from embedded hotspot masks")
				total_time = time.time() - start
				self.get_logger().warn(f"Total time taken to process merged hotspot (embedded masks): {total_time}")
				return processed_count > 0
			
			# Legacy fallback: Lookup merged mask image by timestamp and split by color
			merged_mask = self._lookup_mask(rgb_timestamp)
			mask_lookup_time = time.time() - start
			self.get_logger().warn(f"Time taken to lookup mask: {mask_lookup_time}")
			if merged_mask is None:
				self.get_logger().warn(
					f"No matching hotspot mask found for timestamp {rgb_timestamp:.6f} "
					f"(mask_buf={len(self.mask_buffer_ts)}, sync_window_s={float(self.sync_buffer_duration):.3f})"
				)
				return False
			
			# OPTIMIZATION: Pre-compute color indices once (fixes bottleneck #1)
			color_to_indices = self._precompute_color_indices(merged_mask, vlm_info)
			
			# Process each VLM answer using pre-computed indices
			for vlm_answer, info in vlm_info.items():
				if vlm_answer not in color_to_indices:
					continue
				
				v_coords, u_coords = color_to_indices[vlm_answer]
				if len(v_coords) == 0:
					continue
				
				# Create sparse mask directly from indices (much faster than full image comparison)
				h, w = merged_mask.shape[:2]
				vlm_mask = np.zeros((h, w), dtype=bool)
				vlm_mask[v_coords, u_coords] = True
				
				vlm_mask_time = time.time() - start - mask_lookup_time - depth_lookup_time
				self.get_logger().debug(f"Time taken to create vlm mask: {vlm_mask_time:.4f}s (optimized, legacy)")
				used_ts = 1.0
				success = self._process_hotspot_with_depth(
					vlm_mask, pose_msg, depth_image, vlm_answer, 
					info.get('hotspot_threshold', 0.6) if isinstance(info, dict) else 0.6, 
					{'hotspot_pixels': (info.get('hotspot_pixels', 0) if isinstance(info, dict) else 0)}, 
					rgb_timestamp, used_ts, is_narration, buffer_id
				)
				hotspot_processing_time = time.time() - start - mask_lookup_time - depth_lookup_time - vlm_mask_time
				self.get_logger().debug(f"Time taken to process hotspot: {hotspot_processing_time:.4f}s (legacy)")
				if success:
					processed_count += 1
					if is_narration and len(vlm_info) == 1 and isinstance(info, dict):
						self.get_logger().info(
							f"NARRATION HOTSPOT PROCESSED: '{vlm_answer}' with {info.get('hotspot_pixels', 0)} pixels"
						)
			self.get_logger().info(f"Processed {processed_count}/{len(vlm_info)} VLM answers from merged hotspots (legacy)")
			total_time = time.time() - start
			self.get_logger().warn(f"Total time taken to process merged hotspot (legacy): {total_time}")
			return processed_count > 0
			
		except Exception as e:
			self.get_logger().error(f"Error processing merged hotspot message: {e}")
			return False
	
	def _lookup_depth_and_pose(self, target_ts):
		# Pass the dual deques for depth
		# //make the range for lookup larger, up to 2 seconds
		# //whats target_ts?
		# whre is the range for the lookup window can we chnage it?
		depth_data, depth_ts = self._binary_search_closest(
			self.depth_buffer_ts, self.depth_buffer_data, target_ts, 2
		)
	
		# Pass the dual deques for pose
		pose_data, pose_ts = self._binary_search_closest(
			self.pose_buffer_ts, self.pose_buffer_data, target_ts, 2
		)
	
		return depth_data, pose_data
	
	def _binary_search_closest(self, ts_deque: deque, data_deque: deque, target_ts: float, max_dt: float):
		"""Vectorized search across synchronized deques."""
		if not ts_deque:
			return None, None
		ts_array = np.array(ts_deque)
		idx = np.searchsorted(ts_array, target_ts)
		candidates = []
		if idx < len(ts_array): candidates.append(idx)
		if idx > 0: candidates.append(idx - 1)
	
		if not candidates:
			return None, None
	
		diffs = np.abs(ts_array[candidates] - target_ts)
		best_relative_idx = np.argmin(diffs)
		best_idx = candidates[best_relative_idx]
		if diffs[best_relative_idx] <= max_dt:
			return data_deque[best_idx], ts_array[best_idx]
	
		return None, None
	
	def _lookup_mask(self, target_ts: float) -> Optional[np.ndarray]:
		"""Find closest merged mask image using optimized dual-deque binary search."""
		with self.sync_lock:
			# Pass the separate timestamp and data deques
			best_mask, _ = self._binary_search_closest(
				self.mask_buffer_ts, 
				self.mask_buffer_data, 
				target_ts, 
				self.sync_buffer_duration
			)
			return best_mask
	
	def _process_hotspot_with_depth(self, mask: np.ndarray, pose: Odometry, depth_m: np.ndarray,
								   vlm_answer: str, threshold: float, stats: dict, rgb_ts: float, used_ts: tuple, is_narration: bool, buffer_id: str = None) -> bool:
		"""Project hotspot mask using matched depth and pose; update voxel map and semantics."""
		try:
			if self.camera_intrinsics is None:
				self.get_logger().warn("No camera intrinsics available for hotspot processing")
				return False
			
			# Get hotspot pixel coordinates
			v_coords, u_coords = np.where(mask > 0)
			if len(u_coords) == 0:
				self.get_logger().warn("No hotspot pixels found in mask")
				return False
			hotspot_pixel_count = int(len(u_coords))
			if is_narration:
				self.get_logger().warn(
					f"NARRATION hotspot pixels for '{vlm_answer}': {hotspot_pixel_count} "
					f"(mask_shape={mask.shape}, rgb_ts={rgb_ts:.6f}, buffer_id={buffer_id})"
				)
			else:
				# Keep this at WARN (node default) for time-profiling style visibility.
				self.get_logger().warn(
					f"Hotspot pixels for '{vlm_answer}': {hotspot_pixel_count} "
					f"(mask_shape={mask.shape}, rgb_ts={rgb_ts:.6f})"
				)
			
			# Extract only hotspot pixels from depth (no full array creation)
			h, w = mask.shape
			if depth_m.shape != (h, w):
				depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
				depth_values = depth_resized[v_coords, u_coords]
			else:
				depth_values = depth_m[v_coords, u_coords]
			
			# Filter valid depth values
			valid_mask = np.isfinite(depth_values) & (depth_values > 0.0)
			if not np.any(valid_mask):
				self.get_logger().warn(
					f"No valid depth values in hotspot (answer='{vlm_answer}', type={'NARRATION' if is_narration else 'OPERATIONAL'})"
				)
				return False
			valid_depth_count = int(np.count_nonzero(valid_mask))
			self.get_logger().warn(
				f"Hotspot depth-filter for '{vlm_answer}': valid_depth_pixels={valid_depth_count}/{hotspot_pixel_count}"
			)
			
			# Only process valid hotspot pixels directly (skip meshgrid)
			u_valid = u_coords[valid_mask]
			v_valid = v_coords[valid_mask]
			z_valid = depth_values[valid_mask]

			# Basic "nearest-object" filtering:
			# If the segmentation mask includes background, depth values often become multi-modal.
			# Pick the dominant near-depth cluster (1D histogram) to suppress far/background points.
			try:
				nz = int(z_valid.shape[0])
				if nz >= 80:
					z_min = float(np.min(z_valid))
					z_max = float(np.max(z_valid))
					z_span = z_max - z_min
					if np.isfinite(z_span) and z_span > 0.25:
						nbins = 30
						bins = np.linspace(z_min, z_max, nbins + 1, dtype=np.float32)
						bin_idx = np.clip(np.digitize(z_valid, bins) - 1, 0, nbins - 1)
						counts = np.bincount(bin_idx, minlength=nbins).astype(np.float32)
						centers = 0.5 * (bins[:-1] + bins[1:])
						# Bias toward nearer clusters when counts are similar.
						scores = counts / (1.0 + 0.6 * centers.astype(np.float32))
						best = int(np.argmax(scores))
						z_low = float(bins[best])
						z_high = float(bins[best + 1])
						keep = (z_valid >= z_low) & (z_valid < z_high)
						kept = int(np.count_nonzero(keep))
						# If the best bin is too thin (object split across bins), expand one bin on each side.
						if kept < max(50, int(0.15 * nz)):
							lo = max(best - 1, 0)
							hi = min(best + 2, nbins)  # bins index
							z_low = float(bins[lo])
							z_high = float(bins[hi])
							keep = (z_valid >= z_low) & (z_valid < z_high)
							kept = int(np.count_nonzero(keep))

						if kept >= 50 and kept < nz:
							u_valid = u_valid[keep]
							v_valid = v_valid[keep]
							z_valid = z_valid[keep]
							self.get_logger().warn(
								f"Hotspot depth clustering for '{vlm_answer}': kept={kept}/{nz} "
								f"in [{z_low:.3f}, {z_high:.3f})m (z_span={z_span:.3f}m)"
							)
						else:
							self.get_logger().warn(
								f"Hotspot depth clustering for '{vlm_answer}': skipped (kept={kept}/{nz}, z_span={z_span:.3f}m)"
							)
			except Exception as e:
				self.get_logger().warn(f"Hotspot depth clustering failed for '{vlm_answer}': {e}")
			
			# Convert to world points using only hotspot pixels
			t_project = time.time()
			points_world = self._depth_to_world_points_sparse(u_valid, v_valid, z_valid, self.camera_intrinsics, pose)
			project_dt = time.time() - t_project
			if points_world is None or len(points_world) == 0:
				self.get_logger().warn("Failed to project hotspot points to world coordinates")
				return False
			
			if points_world is None or len(points_world) == 0:
				self.get_logger().warn("Failed to project hotspot points to world coordinates")
				return False
			
			self.get_logger().warn(
				f"Hotspot projection for '{vlm_answer}': projected_world_points={int(points_world.shape[0])}, "
				f"dt={project_dt:.4f}s"
			)
			
			# Range filter using squared distance (faster than norm)
			origin = self._pose_position(pose)
			diff = points_world - origin
			dist_sq = np.sum(diff * diff, axis=1)
			min_range_sq = float(self.min_range) * float(self.min_range)
			max_range_sq = 10.0 * 10.0
			mask_range = (dist_sq >= min_range_sq) & (dist_sq <= max_range_sq)
			points_world_near = points_world[mask_range]
			if points_world_near.size == 0:
				self.get_logger().debug("Hotspot points beyond semantic max_range; skipping semantic voxel update but continuing with ray casting")
			if is_narration:
				self.get_logger().warn(
					f"NARRATION world points for '{vlm_answer}': total={int(points_world.shape[0])}, "
					f"within_range={int(points_world_near.shape[0])}"
				)
			else:
				self.get_logger().warn(
					f"Hotspot world points for '{vlm_answer}': total={int(points_world.shape[0])}, "
					f"within_range={int(points_world_near.shape[0])}"
				)
			
			# GP fitting for narration hotspots (background thread)
			
			buffer_dir, pcd_path = self.save_points_to_latest_nested_subfolder("/home/navin/ros2_ws/src/resilience/buffers", points_world_near)
			if is_narration:
				if buffer_dir is None or pcd_path is None:
					self.get_logger().warn(
						f"NARRATION points save skipped/failed for '{vlm_answer}' "
						f"(within_range={int(points_world_near.shape[0])}, buffer_id={buffer_id})"
					)
				else:
					self.get_logger().warn(
						f"NARRATION points saved for '{vlm_answer}' -> {pcd_path} (buffer_dir={buffer_dir})"
					)
			
			voxelized_points = self._voxelize_pointcloud(points_world_near, float(self.voxel_resolution), max_points=200)
			self._check_and_start_gp_fit(buffer_dir, voxelized_points, vlm_answer)
 				
			# Build depth image with only hotspot pixels (same as tmp.py) - prepare for threading
			h, w = mask.shape
			depth_hot = np.zeros((h, w), dtype=np.float32)
			if depth_m.shape != (h, w):
				depth_resized = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
				depth_hot[mask > 0] = depth_resized[mask > 0]
			else:
				depth_hot[mask > 0] = depth_m[mask > 0]
			
			# Update VDB map with semantic hotspot using masked depth - run in separate thread (optimized)
			mask_copy = mask.copy()
			depth_hot_copy = depth_hot.copy()
			pose_copy = Odometry()
			pose_copy.header = pose.header
			pose_copy.child_frame_id = getattr(pose, "child_frame_id", "")
			pose_copy.pose = pose.pose
			pose_copy.twist = pose.twist

			device = self.vdb_mapper.device
			h, w = mask.shape

			# Ensure minimum image size
			if h < 1 or w < 1:
				return True  # Return True so we don't silently drop the hotspot

			# Sanitize depth before sending to GPU (NaN/Inf → index OOB in CUDA kernels)
			depth_hot = np.nan_to_num(depth_hot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

			try:
				# Create tensors with batch size 1 (critical for indexing)
				depth_tensor = torch.from_numpy(depth_hot).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
				rgb_tensor = torch.zeros(1, 3, h, w, dtype=torch.float32, device=device)
				pose_4x4 = self._pose_to_4x4_matrix(pose)

				# Ensure pose_4x4 has correct batch dimension (1x4x4)
				if pose_4x4.dim() == 2:
					pose_4x4 = pose_4x4.unsqueeze(0)
				elif pose_4x4.shape[0] != 1:
					pose_4x4 = pose_4x4[:1]

				# Guard: pose must be finite
				if not torch.isfinite(pose_4x4).all():
					self.get_logger().warn("_process_hotspot_with_depth: non-finite pose matrix, skipping VDB update")
					pose_4x4 = None
			except (RuntimeError, Exception) as e:
				self.get_logger().warn(f"Tensor creation failed in hotspot processing: {e}")
				pose_4x4 = None

			# Process with VDB mapper for semantic occupancy

			# Prepare masked depth for rays-only beyond max_range
			depth_for_rays = np.zeros_like(depth_hot, dtype=np.float32)
			masked = (mask > 0)
			if self.camera_intrinsics is not None:
				# Use original depth_m if available, otherwise use depth_hot
				# For rays, we want pixels beyond max_range or missing depth
				masked_depth_vals = depth_hot[masked]
				threshold = 5.0
				beyond_or_missing = (masked_depth_vals <= 0.0) | (masked_depth_vals > threshold)
				dr = np.zeros_like(masked_depth_vals, dtype=np.float32)
				dr[beyond_or_missing] = np.inf
				depth_for_rays[masked] = dr
				mask_far = np.zeros_like(depth_for_rays, dtype=bool)
				mask_far[masked] = beyond_or_missing

				if np.any(mask_far):
					try:
						far_v, far_u = np.where(mask_far)
						fx, fy, cx, cy = self.camera_intrinsics
						fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
						u = far_u.astype(np.float32)
						v = far_v.astype(np.float32)
						dir_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
						dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True) + 1e-9
						pose_mat = self._pose_to_4x4_matrix(pose).detach().cpu().numpy()[0]
						R_world_cam = pose_mat[:3, :3]
						origin_world = pose_mat[:3, 3]
						dir_world = dir_cam @ R_world_cam.T
						dir_world /= np.linalg.norm(dir_world, axis=1, keepdims=True) + 1e-9
						self._latest_pose_rays = (origin_world, dir_world)
					except Exception:
						self._latest_pose_rays = None
				else:
					# Fallback: derive rays from all masked pixels (sampled)
					try:
						if np.any(masked):
							fx, fy, cx, cy = self.camera_intrinsics
							fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
							all_v, all_u = np.where(masked)
							max_samples = 800
							if all_u.shape[0] > max_samples:
								idx = np.random.choice(all_u.shape[0], size=max_samples, replace=False)
								all_u = all_u[idx]
								all_v = all_v[idx]
							u = all_u.astype(np.float32)
							v = all_v.astype(np.float32)
							dir_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
							dir_cam /= np.linalg.norm(dir_cam, axis=1, keepdims=True) + 1e-9
							pose_mat = self._pose_to_4x4_matrix(pose).detach().cpu().numpy()[0]
							R_world_cam = pose_mat[:3, :3]
							origin_world = pose_mat[:3, 3]
							dir_world = dir_cam @ R_world_cam.T
							dir_world /= np.linalg.norm(dir_world, axis=1, keepdims=True) + 1e-9
							self._latest_pose_rays = (origin_world, dir_world)
						else:
							self._latest_pose_rays = None
					except Exception:
						self._latest_pose_rays = None

				# Update intrinsics if available
				try:
					fx, fy, cx, cy = self.camera_intrinsics
					self.vdb_mapper.intrinsics_3x3 = torch.tensor(
						[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
						dtype=torch.float32, device=device)
				except (RuntimeError, Exception) as e:
					self.get_logger().warn(f"Failed to update VDB intrinsics: {e}")
				# Publish mask-specific frontiers and rays immediately (same as tmp.py)
				self._publish_mask_frontiers_and_rays()
			# Semantic label application - run in separate thread to avoid blocking
			if points_world_near.size > 0:
				points_copy = points_world_near.copy()
				threading.Thread(
					target=self._update_semantic_voxels,
					args=(points_copy, vlm_answer, threshold, stats, is_narration),
					daemon=True
				).start()
				near_count = points_world_near.shape[0]
			else:
				near_count = 0

			hotspot_type = "NARRATION" if is_narration else "OPERATIONAL"
			self.get_logger().info(
				f"Applied hotspot processing for '{vlm_answer}' (within_range={near_count}, rgb_ts={rgb_ts:.6f}, type={hotspot_type})"
			)
			return True

		except Exception as e:
			self.get_logger().error(f"Error processing hotspot with depth: {e}")
			import traceback
			traceback.print_exc()
			return False
	


	def _voxelize_pointcloud(self, points: np.ndarray, voxel_size: float, max_points: int = 200) -> np.ndarray:
		"""
		High-performance voxelization using Open3D (C++ backend).
		Reduces point density by averaging points within a spatial grid.
		"""
		if points.shape[0] == 0:
			return points

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
		voxelized_points = np.asarray(downsampled_pcd.points)
		num_voxelized = voxelized_points.shape[0]
		if num_voxelized > max_points:
			step = num_voxelized / max_points
			indices = np.arange(0, num_voxelized, step, dtype=np.int32)[:max_points]
			voxelized_points = voxelized_points[indices]
	
			self.get_logger().info(
				f"O3D Voxelized {points.shape[0]} -> {num_voxelized} points. "
				f"Sampled to {max_points} (voxel_size={voxel_size:.3f}m)"
			)
		return voxelized_points

	def save_points_to_latest_nested_subfolder(self, known_folder: str, 
										  points_world: np.ndarray, 
										  filename: str = "points.pcd"):
		"""
		Finds latest nested subfolders and saves points as a binary PCD using Open3D.
		"""
		if points_world.size == 0:
			return None, None
	
		voxelized_points = self._voxelize_pointcloud(points_world, float(self.voxel_resolution), max_points=200)
	
		current_time = time.time()
		if (self._cached_latest_subfolder and os.path.exists(self._cached_latest_subfolder) and 
			(current_time - self._cached_subfolder_time) < self._subfolder_cache_ttl):
			latest_subfolder2 = self._cached_latest_subfolder
		else:
			try:
				# Find latest subfolder1
				s1 = [os.path.join(known_folder, d) for d in os.listdir(known_folder) if os.path.isdir(os.path.join(known_folder, d))]
				if not s1: return None, None
				latest_s1 = max(s1, key=os.path.getmtime)
	
				# Find latest subfolder2
				s2 = [os.path.join(latest_s1, d) for d in os.listdir(latest_s1) if os.path.isdir(os.path.join(latest_s1, d))]
				if not s2: return None, None
				latest_subfolder2 = max(s2, key=os.path.getmtime)
	
				# Cache it
				self._cached_latest_subfolder = latest_subfolder2
				self._cached_subfolder_time = current_time
			except Exception as e:
				self.get_logger().error(f"Folder search failed: {e}")
				return None, None
	
		# 3. Save JSON Metadata (Mean position of the hazard)
		save_path = os.path.join(latest_subfolder2, filename)
		mean_pos = np.mean(voxelized_points, axis=0).tolist()
		with open(os.path.join(latest_subfolder2, "mean_cause.json"), "w") as f:
			json.dump(mean_pos, f)
	
		# 4. Save PCD using Open3D (Binary format is default and much faster)
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(voxelized_points)
		o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
	
		self.get_logger().info(f"O3D saved {len(voxelized_points)} points to {save_path}")
		return latest_subfolder2, save_path

	def _check_and_start_gp_fit(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Check if poses.npy is available and start GP fitting if ready."""
		try:
			poses_path = os.path.join(buffer_dir, 'poses.npy')
			if not os.path.exists(poses_path):
				self.get_logger().info(f"poses.npy not yet available in {buffer_dir}, skipping GP fit for now")
				return
		except Exception as e:
			self.get_logger().warn(f"Error checking poses.npy in {buffer_dir}: {e}")
			return
		try:
			poses_data = np.load(poses_path)
			if len(poses_data) == 0:
				self.get_logger().info(f"poses.npy is empty in {buffer_dir}, skipping GP fit for now")
				return
		except Exception as e:
			self.get_logger().warn(f"Error loading poses.npy from {buffer_dir}: {e}")
			return
		
		self.get_logger().info(f"Both PCD and poses.npy available in {buffer_dir}, starting GP fit")
		self._start_background_gp_fit(buffer_dir, pointcloud_xyz, cause_name)
			

	def _extract_safe_poses(self, buffer_dir: str, max_safe_points: int = 50,
						  lookback_seconds: float = 3.0) -> Optional[np.ndarray]:
		"""Extract pre-breach trajectory poses as safe evidence for GP fitting.
		
		The trajectory points recorded just BEFORE the breach/drift was detected
		were tracking the nominal path well, so their disturbance ≈ ambient.
		Including them as training data with low disturbance helps the GP:
		  - Anchor b ≈ ambient (baseline disturbance)
		  - Force A*phi to do the heavy lifting for spatial structure
		  - Sharpen the transition from safe → dangerous regions
		
		Uses timestamps: breach start from poses.npy col 0, pose buffer timestamps
		from self.pose_buffer_ts. Extracts poses in [breach_start - lookback, breach_start).
		
		Args:
			buffer_dir: Buffer directory containing poses.npy (breach poses)
			max_safe_points: Maximum number of safe points (subsample if more)
			lookback_seconds: How far back before breach to look (seconds)
		
		Returns:
			(K, 3) numpy array of safe XYZ positions, or None if unavailable
		"""
		try:
			# Load breach data to get the breach start timestamp
			poses_path = os.path.join(buffer_dir, 'poses.npy')
			if not os.path.exists(poses_path):
				return None
			breach_data = np.load(poses_path)
			if len(breach_data) == 0 or breach_data.shape[1] < 4:
				return None
			
			# Column 0 = timestamp in poses.npy
			breach_start_time = float(breach_data[0, 0])
			cutoff_time = breach_start_time - lookback_seconds
			
			# Snapshot deques for thread-safe iteration (called from bg thread)
			buffer_ts = list(self.pose_buffer_ts)
			buffer_data = list(self.pose_buffer_data)
			
			# Extract poses from the buffer that are BEFORE the breach
			safe_poses = []
			for i, ts in enumerate(buffer_ts):
				if cutoff_time <= ts < breach_start_time:
					pose_msg = buffer_data[i]
					pose = pose_msg.pose.pose if hasattr(pose_msg.pose, 'pose') else pose_msg.pose
					safe_poses.append([
						pose.position.x,
						pose.position.y,
						pose.position.z
					])
			
			if not safe_poses:
				self.get_logger().info("No pre-breach safe poses found in buffer")
				return None
			
			safe_xyz = np.array(safe_poses, dtype=np.float32)
			
			# Subsample uniformly if too many (keep temporal spread)
			if len(safe_xyz) > max_safe_points:
				indices = np.linspace(0, len(safe_xyz) - 1, max_safe_points, dtype=int)
				safe_xyz = safe_xyz[indices]
			
			self.get_logger().info(
				f"Extracted {len(safe_xyz)} pre-breach safe poses "
				f"(lookback={lookback_seconds}s before t={breach_start_time:.3f})"
			)
			return safe_xyz
			
		except Exception as e:
			self.get_logger().warn(f"Failed to extract safe poses: {e}")
			return None

	def _start_background_gp_fit(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Start GP fitting in a background thread if not already running."""
		try:
			with self.gp_fit_lock:
				if self.gp_fitting_active:
					self.get_logger().info("GP fit already running; skipping new request")
					return
				self.gp_fitting_active = True
			args = (buffer_dir, np.array(pointcloud_xyz, dtype=np.float32), cause_name)
			threading.Thread(target=self._run_gp_fit_task, args=args, daemon=True).start()
		except Exception as e:
			self.get_logger().warn(f"Failed to start GP fit thread: {e}")

	def _run_gp_fit_task(self, buffer_dir: str, pointcloud_xyz: np.ndarray, cause_name: Optional[str] = None):
		"""Run GP fitting and save parameters to buffer directory.
		
		CRITICAL: Uses the nominal path stored at breach start (nominal_path.npy in buffer_dir)
		rather than the current path_manager path. This ensures we compare observed poses
		during the breach against the correct predicted (nominal) path that was active when
		the breach started, not a path that may have changed after the breach started.
		"""
		try:
			self.get_logger().info(f"Starting GP fit for buffer: {buffer_dir}")
			helper = DisturbanceFieldHelper()
			
			# CRITICAL: First try to load the nominal path stored at breach start
			# This is the path that was active when the breach began, which is what we need
			# for proper GP computation (comparing predicted vs observed during breach)
			nominal_xyz = None
			nominal_source = "none"
			
			# Priority 1: Load stored nominal path from buffer (stored at breach start)
			stored_nominal_path = os.path.join(buffer_dir, 'nominal_path.npy')
			if os.path.exists(stored_nominal_path):
				try:
					nominal_xyz = np.load(stored_nominal_path)
					if nominal_xyz is not None and len(nominal_xyz) > 0:
						nominal_source = "stored_at_breach_start"
						self.get_logger().info(f"GP nominal source: STORED AT BREACH START (points={len(nominal_xyz)})")
					else:
						nominal_xyz = None
				except Exception as e:
					self.get_logger().warn(f"Failed to load stored nominal path: {e}")
					nominal_xyz = None
			
			# Priority 2: Fallback to current PathManager path (if stored path not available)
			if nominal_xyz is None:
				try:
					if self.path_manager is not None and hasattr(self.path_manager, 'get_nominal_points_as_numpy'):
						nominal_xyz = self.path_manager.get_nominal_points_as_numpy()
						if nominal_xyz is not None and len(nominal_xyz) == 0:
							nominal_xyz = None
						else:
							nominal_source = "current_path_manager"
							self.get_logger().info(f"GP nominal source: CURRENT PATH MANAGER (points={len(nominal_xyz)})")
							self.get_logger().warn("⚠️  Using current path manager instead of stored breach path - may be incorrect if path changed!")
				except Exception:
					pass
			
			# Priority 3: Fallback to file path
			if nominal_xyz is None:
				if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0:
					nominal_source = "file"
					self.get_logger().info(f"GP nominal source: FILE {self.nominal_path}")
				else:
					self.get_logger().warn("GP nominal source: NONE (using actual-only baseline)")
			
			# Extract pre-breach trajectory poses as safe evidence
			# These confirm disturbance ≈ ambient before the breach.
			# When available, we fix b = ambient in the GP fit (instead of fitting b)
			# to prevent b from absorbing spatial structure meant for A*phi.
			safe_poses = self._extract_safe_poses(buffer_dir)
			# ambient_disturbance: known nominal tracking error (0.05cm = 0.0005m)
			# Pass only when safe evidence validates this assumption
			ambient = 0.05
			
			result = helper.fit_from_pointcloud_and_buffer(
				pointcloud_xyz=pointcloud_xyz,
				buffer_dir=buffer_dir,
				nominal_path=(None if nominal_xyz is not None else (self.nominal_path if isinstance(self.nominal_path, str) and len(self.nominal_path) > 0 else None)),
				nominal_xyz=nominal_xyz,
				safe_points_xyz=safe_poses,
				ambient_disturbance=ambient
			)
			fit = result.get('fit', {})
			opt = fit.get('optimization_result') if isinstance(fit, dict) else None
			o = {
				'fit_params': {
					'lxy': fit.get('lxy'),
					'lz': fit.get('lz'),
					'A': fit.get('A'),
					'b': fit.get('b'),
					'mse': fit.get('mse'),
					'rmse': fit.get('rmse'),
					'mae': fit.get('mae'),
					'r2_score': fit.get('r2_score'),
					'sigma2': fit.get('sigma2'),  # Noise variance σ² (aleatoric)
					'nll': fit.get('nll'),  # Negative log-likelihood
					'XtX_inv': fit.get('XtX_inv'),  # (XᵀX)⁻¹ for BLR uncertainty
					'fixed_b': fit.get('fixed_b'),  # Whether b was fixed to ambient
					'hess_inv': fit.get('hess_inv'),  # Cov(lxy, lz) for delta method
				},
				'optimization': ({
					'nit': getattr(opt, 'nit', None),
					'nfev': getattr(opt, 'nfev', None),
					'success': getattr(opt, 'success', None),
					'message': getattr(opt, 'message', None)
				} if opt is not None else None),
				'metadata': {
					'timestamp': time.time(),
					'buffer_dir': buffer_dir,
					'nominal_path': self.nominal_path,
					'used_nominal_source': nominal_source,
					'has_stored_nominal_path': os.path.exists(stored_nominal_path) if 'stored_nominal_path' in locals() else False,
					'safe_evidence_points': len(safe_poses) if safe_poses is not None else 0
				}
			}
			out_path = os.path.join(buffer_dir, 'voxel_gp_fit.json')
			with open(out_path, 'w') as f:
				json.dump(o, f, indent=2)
			self.get_logger().info(f"Saved GP fit parameters to {out_path}")
			
			# Update cause registry with GP params if cause_name is available
			# NOTE: Cause registry integration removed. We persist GP params to buffer_dir only.
			# Loading for mapping is done from `main_config.yaml` preloaded_objects[].gp_params_file.
			
		except Exception as e:
			self.get_logger().error(f"GP fit task failed: {e}")
			import traceback
			traceback.print_exc()
		finally:
			with self.gp_fit_lock:
				self.gp_fitting_active = False
			
			# MULTI-OBJECT: Store GP parameters per cause
			if result and 'fit' in result and cause_name:
				fit_params = result['fit']
				with self.gp_fit_lock:
					# Store per object
					self.per_cause_gp_params[cause_name] = fit_params
					# Also update global_gp_params (most recent) for backward compatibility
					self.global_gp_params = fit_params
					
					# XtX_inv + hess_inv precomputed in fit_params → no need
					# to store nominal_points / disturbances separately
					try:
						status = fit_params.get('status', 'ok') if isinstance(fit_params, dict) else 'ok'
						reason = fit_params.get('reason', None) if isinstance(fit_params, dict) else None
						lxy = (fit_params.get('lxy') if isinstance(fit_params, dict) else None)
						lz = (fit_params.get('lz') if isinstance(fit_params, dict) else None)
						A = (fit_params.get('A') if isinstance(fit_params, dict) else None)
						if lxy is None or lz is None or A is None or status != 'ok':
							self.get_logger().info(f"Updated GP parameters for '{cause_name}': status={status}, reason={reason} (total objects: {len(self.per_cause_gp_params)})")
						else:
							self.get_logger().info(f"Updated GP parameters for '{cause_name}': lxy={float(lxy):.3f}, lz={float(lz):.3f}, A={float(A):.3f} (total objects: {len(self.per_cause_gp_params)})")
					except Exception:
						self.get_logger().info(f"Updated GP parameters for '{cause_name}' (total objects: {len(self.per_cause_gp_params)})")
			

	
	# NOTE: Cause registry integration removed (no `_update_registry_gp_params` / `_handle_registry_response`).
	

	def _load_pcd_points(self, pcd_path: str) -> np.ndarray:
		"""Load points from PCD file using Open3D for high compatibility and speed."""
		try:
			pcd = o3d.io.read_point_cloud(pcd_path)
			if pcd.is_empty():
				return np.array([], dtype=np.float32)
			return np.asarray(pcd.points, dtype=np.float32)
	
		except Exception as e:
			self.get_logger().error(f"Error loading PCD points with Open3D: {e}")
			return np.array([], dtype=np.float32)

	def _pointcloud_frame_id(self) -> str:
		"""
		Use the same frame selection as `/semantic_octomap_colored_cloud`.
		"""
		# Prefer the live frame from the odometry topic (e.g. FAST-LIO2 `camera_init` / `fastlio_base`).
		if isinstance(self.latest_pose_frame_id, str) and len(self.latest_pose_frame_id) > 0:
			return self.latest_pose_frame_id
		# Fallback to configured frame (historical behavior).
		if self.semantic_colored_cloud_frame:
			return str(self.semantic_colored_cloud_frame)
		return str(self.map_frame)

	def _create_gp_colored_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray) -> Optional[PointCloud2]:
		"""Create colored point cloud from GP field predictions using vectorized operations."""
		if len(grid_points) == 0 or len(gp_values) == 0:
			return None

		gp_min, gp_max = gp_values.min(), gp_values.max()
		if gp_max > gp_min:
			normalized_values = (gp_values - gp_min) / (gp_max - gp_min)
		else:
			normalized_values = np.zeros_like(gp_values)

		colors_rgba = cm.turbo(normalized_values)
		colors_uint8 = (colors_rgba[:, :3] * 255).astype(np.uint32)
		rgb_packed = (colors_uint8[:, 0] << 16) | (colors_uint8[:, 1] << 8) | colors_uint8[:, 2]

		cloud_data = np.empty(len(grid_points), dtype=[
			('x', np.float32), ('y', np.float32), ('z', np.float32), 
			('rgb', np.uint32)
		])

		cloud_data['x'] = grid_points[:, 0].astype(np.float32)
		cloud_data['y'] = grid_points[:, 1].astype(np.float32)
		cloud_data['z'] = grid_points[:, 2].astype(np.float32)
		cloud_data['rgb'] = rgb_packed

		# 5. Assemble PointCloud2 Message
		cloud_msg = PointCloud2()
		cloud_msg.header.stamp = self.get_clock().now().to_msg()
		cloud_msg.header.frame_id = self._pointcloud_frame_id()

		cloud_msg.fields = [
			pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
			pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
		]

		cloud_msg.point_step = 16
		cloud_msg.width = len(grid_points)
		cloud_msg.height = 1
		cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
		cloud_msg.is_dense = True
		cloud_msg.data = cloud_data.tobytes()
		return cloud_msg
	
	def _load_preloaded_gp_params_from_main_config(self) -> None:
		"""
		Load GP parameters directly from `config/main_config.yaml` (preloaded_objects[].gp_params_file).
		
		Simplified behavior (as requested):
		- No cause-registry dependency
		- Use the FIRST successfully loaded GP fit as `self.global_gp_params`
		- Use that GP for ALL semantic voxels (ignores per-cause GP)
		"""
		try:
			import yaml
			# Resolve config path
			if isinstance(self.main_config_path, str) and len(self.main_config_path) > 0:
				config_path = self.main_config_path
			else:
				from ament_index_python.packages import get_package_share_directory
				package_dir = get_package_share_directory('resilience')
				config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
			
			if not os.path.exists(config_path):
				self.get_logger().warn(f"Main config not found: {config_path} (skipping preloaded GP params)")
				return
			
			with open(config_path, 'r') as f:
				config = yaml.safe_load(f) or {}
			
			preloaded = config.get('preloaded_objects', [])
			if not isinstance(preloaded, list) or len(preloaded) == 0:
				self.get_logger().info("No preloaded_objects configured in main config; GP will stay unset until a narration GP fit runs.")
				return
			
			loaded = 0
			for obj in preloaded:
				if not isinstance(obj, dict):
					continue
				if not obj.get('enabled', True):
					continue
				name = obj.get('name', 'unknown')
				gp_file = obj.get('gp_params_file') or obj.get('gp_params_path') or obj.get('gp_params')
				if not gp_file:
					continue
				
				# Resolve wildcard → newest file
				resolved = gp_file
				if ('*' in gp_file) or ('?' in gp_file) or ('[' in gp_file):
					matches = glob.glob(gp_file)
					if not matches:
						self.get_logger().warn(f"[GP preload] No files match pattern for '{name}': {gp_file}")
						continue
					resolved = max(matches, key=os.path.getmtime)
				
				if not os.path.exists(resolved):
					self.get_logger().warn(f"[GP preload] File not found for '{name}': {resolved}")
					continue
				
				try:
					with open(resolved, 'r') as jf:
						data = json.load(jf) or {}
				except Exception as e:
					self.get_logger().warn(f"[GP preload] Failed to read JSON for '{name}' ({resolved}): {e}")
					continue
				
				# Support both:
				# - {"fit_params": {...}} (expected)
				# - {"fit": {"fit_params": {...}}} or other nesting (best-effort)
				fit_params = None
				if isinstance(data, dict):
					if isinstance(data.get('fit_params'), dict):
						fit_params = data.get('fit_params')
					elif isinstance(data.get('fit'), dict) and isinstance(data['fit'].get('fit_params'), dict):
						fit_params = data['fit']['fit_params']
				
				if not isinstance(fit_params, dict) or len(fit_params) == 0:
					self.get_logger().warn(f"[GP preload] Missing fit_params for '{name}' in {resolved}")
					continue
				
				# Minimal validation (required by _predict_gp_field_fast)
				lxy = fit_params.get('lxy')
				lz = fit_params.get('lz')
				A = fit_params.get('A')
				b = fit_params.get('b')
				if lxy is None or lz is None or A is None or b is None:
					self.get_logger().warn(
						f"[GP preload] fit_params incomplete for '{name}' in {resolved} "
						f"(need lxy,lz,A,b)."
					)
					continue
				
				gp_params_dict = dict(fit_params)
				gp_params_dict.setdefault('status', 'ok')
				gp_params_dict.setdefault('source', 'main_config_preload')
				gp_params_dict.setdefault('name', name)
				gp_params_dict.setdefault('file', resolved)
				
				with self.gp_fit_lock:
					# Store the first successfully loaded GP fit as the global GP
					if self.global_gp_params is None:
						self.global_gp_params = gp_params_dict
					# Keep per-cause dict empty by design (simplified single-GP mode)
					self.per_cause_gp_params = {}
				
				loaded += 1
				self.get_logger().warn(
					f"[GP preload] Loaded global GP params from main config: '{name}' "
					f"({resolved})"
				)
				# Simplification: first GP wins
				break
			
			if loaded == 0:
				self.get_logger().warn("[GP preload] No enabled preloaded_objects GP file could be loaded; GP will stay unset until runtime fit.")
			
		except Exception as e:
			self.get_logger().warn(f"Error loading preloaded GP params from main config: {e}")
			import traceback
			traceback.print_exc()
	
	def _start_gp_computation_thread(self):
		"""Start the GP computation thread."""
		try:
			with self.gp_thread_lock:
				if self.gp_thread_running:
					return
				self.gp_thread_running = True
			
			self.gp_computation_thread = threading.Thread(target=self._gp_computation_worker, daemon=True)
			self.gp_computation_thread.start()
			self.get_logger().info("GP computation thread started")
			
		except Exception as e:
			self.get_logger().error(f"Error starting GP computation thread: {e}")
			with self.gp_thread_lock:
				self.gp_thread_running = False
	
	def _gp_computation_worker(self):
		"""Background worker thread for GP computation, visualization, and semantic pruning."""
		semantic_prune_interval = 2.0  # Prune every 2 seconds
		last_prune_time = 0.0
		
		try:
			while self.gp_thread_running:
				current_time = time.time()
				
				# GP visualization update
				if (current_time - self.last_gp_update_time) >= self.gp_update_interval:
					self._update_semantic_gp_visualization()
					self.last_gp_update_time = current_time
				
				# Periodic semantic voxel pruning (sync with VDB occupancy)
				# if (current_time - last_prune_time) >= semantic_prune_interval:
				# 	pruned = self._prune_stale_semantic_voxels()
				# 	if pruned > 0:
				# 		self.get_logger().debug(f"Pruned {pruned} stale semantic voxels")
				# 	last_prune_time = current_time
				
				time.sleep(0.1)
				
		except Exception as e:
			self.get_logger().error(f"Error in GP computation worker: {e}")
			import traceback
			traceback.print_exc()
		finally:
			with self.gp_thread_lock:
				self.gp_thread_running = False
	
	def _update_semantic_gp_visualization(self):
		"""
		Update GP visualization using a ROBOT-CENTRIC 3D grid with MULTI-OBJECT support.
		
		For each object with GP params:
		  1. Get its semantic voxels (cause points)
		  2. Predict GP mean and uncertainty using its own GP params
		  3. Combine predictions from all objects (sum or max)
		
		Results are stored in a GPU tensor (Channel=2, Depth, Height, Width) where:
		  - Channel 0 = Combined GP Mean (sum or max of all objects)
		  - Channel 1 = Combined Epistemic Uncertainty
		
		Update rate: 2-5 Hz (asynchronous)
		"""
		try:
			# Simplified: use ONE global GP for ALL semantic voxels.
			with self.gp_fit_lock:
				global_gp = dict(self.global_gp_params) if isinstance(self.global_gp_params, dict) else None
			if global_gp is None:
				return
			
			# Check if robot position is available
			if self.robot_position is None:
				self.get_logger().warn("Robot position not available yet, skipping GP update")
				return
			
			# Get all semantic voxels (ignore per-cause grouping)
			semantic_points = np.array(self._get_all_semantic_voxels(), dtype=np.float32)
			if semantic_points.size == 0:
				return
			
			# ============================================================
			# ROBOT-CENTRIC 3D GRID GENERATION
			# ============================================================
			grid_points, grid_shape = self._create_robot_centric_3d_grid()
			if len(grid_points) == 0:
				return
			
			# ============================================================
			# MULTI-OBJECT GP PREDICTION
			# ============================================================
			# Predict GP mean + uncertainty using the global GP
			gp_mean = self._predict_gp_field_fast(grid_points, semantic_points, global_gp)
			uncertainty_std = self._compute_epistemic_uncertainty(grid_points, semantic_points, global_gp)
			if gp_mean is None or len(gp_mean) == 0:
				return
			
			# ============================================================
			# STORE IN GPU TENSOR (Channel=2, Depth, Height, Width)
			# ============================================================
			self._update_gp_gpu_tensor(gp_mean, uncertainty_std, grid_shape)
			

			colored_cloud = self._create_gp_colored_pointcloud(grid_points, gp_mean)
			if colored_cloud:
				self.gp_visualization_pub.publish(colored_cloud)
			
			# Publish costmap (mean disturbance values)
			costmap_cloud = self._create_costmap_pointcloud(grid_points, gp_mean)
			if costmap_cloud:
				self.costmap_pub.publish(costmap_cloud)
			
			# Publish epistemic uncertainty field
			if uncertainty_std is not None:
				uncertainty_cloud = self._create_uncertainty_pointcloud(grid_points, uncertainty_std)
				if uncertainty_cloud:
					self.gp_uncertainty_pub.publish(uncertainty_cloud)
			
			# Publish raw grid for control node
			self._publish_raw_gp_grid(gp_mean, uncertainty_std, grid_shape, grid_points)

			# Log with safe handling of None uncertainty
			uncertainty_str = (
				f"[{uncertainty_std.min():.3f}, {uncertainty_std.max():.3f}]"
				if uncertainty_std is not None else "N/A"
			)
			self.get_logger().info(
				f"Published GP fields (single-GP mode): {len(grid_points)} grid points, "
				f"mean_range=[{gp_mean.min():.3f}, {gp_mean.max():.3f}], "
				f"uncertainty_range={uncertainty_str}"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error updating multi-object GP visualization: {e}")
			import traceback
			traceback.print_exc()
	

	def _create_robot_centric_3d_grid(self):
		"""
		Optimized 3D grid generation using flattened coordinate arrays.
		Grid size: 10m × 10m (XY) × 4m (Z) centered on the robot.
		"""
		try:
			if self.robot_position is None:
				self.get_logger().warn("Robot position not available for grid generation")
				return np.array([], dtype=np.float32), (0, 0, 0)
	
			# 1. Define bounds
			rx, ry, rz = self.robot_position
			h_xy = self.robot_grid_size_xy / 2.0
			h_z = self.robot_grid_size_z / 2.0
			res = self.robot_grid_resolution
	
			# 2. Use linspace for stability or arange for exact resolution
			# np.arange can sometimes have 'off-by-one' errors with floating points
			x_c = np.arange(rx - h_xy, rx + h_xy, res, dtype=np.float32)
			y_c = np.arange(ry - h_xy, ry + h_xy, res, dtype=np.float32)
			z_c = np.arange(rz - h_z, rz + h_z, res, dtype=np.float32)
	
			# 3. Memory Efficient Meshgrid
			# Using indexing='ij' is correct for (D, H, W) mapping
			X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')
	
			# 4. Ravel is faster than flatten() as it returns a view when possible
			grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
	
			grid_shape = (len(x_c), len(y_c), len(z_c))
	
			return grid_points, grid_shape
	
		except Exception as e:
			self.get_logger().error(f"Error creating robot-centric 3D grid: {e}")
			return np.array([], dtype=np.float32), (0, 0, 0)
	
	def _update_gp_gpu_tensor(self, gp_mean, uncertainty_std, grid_shape):
		"""
		Update GPU tensor with GP mean and uncertainty fields.
		
		Tensor format: (Channel=2, Depth, Height, Width)
		  - Channel 0: GP Mean
		  - Channel 1: Epistemic Uncertainty
		
		Args:
			gp_mean: (N,) GP mean predictions
			uncertainty_std: (N,) Epistemic uncertainty predictions or None
			grid_shape: (D, H, W) grid dimensions
		"""
		try:
			if not self.TORCH_AVAILABLE:
				return
			
			import torch
			
			D, H, W = grid_shape
			expected_size = D * H * W
			
			# Validate gp_mean
			if gp_mean is None or len(gp_mean) != expected_size:
				self.get_logger().warn(f"Invalid gp_mean size: {len(gp_mean) if gp_mean is not None else 0} != {expected_size}")
				return
			
			# Handle uncertainty_std (can be None)
			if uncertainty_std is None or len(uncertainty_std) != expected_size:
				uncertainty_std = np.zeros_like(gp_mean)
			
			# Sanitize NaN/Inf values before GPU operations (critical for CUDA)
			gp_mean = np.nan_to_num(gp_mean, nan=0.0, posinf=0.0, neginf=0.0)
			uncertainty_std = np.nan_to_num(uncertainty_std, nan=0.0, posinf=0.0, neginf=0.0)
			
			# Reshape to 3D grids
			mean_grid = gp_mean.reshape(D, H, W).astype(np.float32)
			uncertainty_grid = uncertainty_std.reshape(D, H, W).astype(np.float32)
			
			# Stack into (2, D, H, W) tensor
			combined_grid = np.stack([mean_grid, uncertainty_grid], axis=0)
			
			# Convert to PyTorch tensor and move to device.
			# If CUDA fails, skip this update rather than silently falling back to CPU.
			try:
				tensor = torch.from_numpy(combined_grid).to(self.device)
				self.gp_grid_tensor = tensor
			except RuntimeError as cuda_err:
				self.get_logger().error(f"CUDA error updating GP tensor, skipping this frame: {cuda_err}")
				return
			
			self.get_logger().info(
				f"Updated GP tensor: shape={self.gp_grid_tensor.shape}, "
				f"device={self.gp_grid_tensor.device}, "
				f"mean=[{mean_grid.min():.3f}, {mean_grid.max():.3f}], "
				f"uncertainty=[{uncertainty_grid.min():.3f}, {uncertainty_grid.max():.3f}]"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error updating GP GPU tensor: {e}")
			import traceback
			traceback.print_exc()

	def _publish_raw_gp_grid(self, gp_mean, uncertainty_std, grid_shape, grid_points):
		"""Publish raw GP grid data for control node."""
		try:
			# grid_shape is (Nx, Ny, Nz) - corresponding to coords
			# gp_mean is flattened (N,)
			
			msg = Float32MultiArray()
			
			# Encode metadata in the layout using labels or dimensions
			# Dim 0: Meta [min_x, min_y, min_z, res, size_x, size_y, size_z]
			# We'll just put metadata as the first few elements of the data array, or use a structured approach
			# Let's pack metadata as a prefix to the data. 
			
			if self.robot_position is None:
				return
				
			# Recalculate bounds from robot position and fixed params
			half_size_xy = self.robot_grid_size_xy / 2.0
			half_size_z = self.robot_grid_size_z / 2.0
			min_x = self.robot_position[0] - half_size_xy
			min_y = self.robot_position[1] - half_size_xy
			min_z = self.robot_position[2] - half_size_z
			
			# Metadata header: 7 floats (ensure all are valid Python floats)
			metadata = [
				float(min_x), float(min_y), float(min_z), 
				float(self.robot_grid_resolution), 
				float(grid_shape[0]), float(grid_shape[1]), float(grid_shape[2])
			]
			
			# Sanitize metadata values (shouldn't be needed, but safety check)
			metadata = [float(np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)) for m in metadata]
			
			# Handle None uncertainty (fallback to zeros)
			if uncertainty_std is None:
				uncertainty_std = np.zeros_like(gp_mean)
			
			# CRITICAL: Sanitize NaN/Inf values before converting to list
			# ROS2 Float32MultiArray validation rejects NaN/Inf values
			gp_mean_clean = np.nan_to_num(gp_mean, nan=0.0, posinf=0.0, neginf=0.0)
			uncertainty_std_clean = np.nan_to_num(uncertainty_std, nan=0.0, posinf=0.0, neginf=0.0)
			
			# Ensure all values are valid float32 (clip to valid range)
			gp_mean_clean = np.clip(gp_mean_clean, -3.4e38, 3.4e38).astype(np.float32)
			uncertainty_std_clean = np.clip(uncertainty_std_clean, -3.4e38, 3.4e38).astype(np.float32)
			
			# Concatenate: Metadata + Mean + Uncertainty
			# Note: gp_mean and uncertainty_std are flattened
			# Convert numpy arrays to Python float lists explicitly
			gp_mean_list = [float(x) for x in gp_mean_clean.tolist()]
			uncertainty_list = [float(x) for x in uncertainty_std_clean.tolist()]
			data_list = metadata + gp_mean_list + uncertainty_list
			
			msg.data = data_list
			
			# Describe layout
			# Dim 0: Metadata (7)
			# Dim 1: Mean (N)
			# Dim 2: Uncertainty (N)
			# This isn't a standard multiarray layout, but the receiver will know how to parse it.
			# Or we can strictly use dimensions to describe the grid, but we need the origin offset.
			
			dim0 = MultiArrayDimension(label="metadata", size=7, stride=7)
			dim1 = MultiArrayDimension(label="mean", size=len(gp_mean), stride=len(gp_mean))
			dim2 = MultiArrayDimension(label="uncertainty", size=len(uncertainty_std), stride=len(uncertainty_std))
			msg.layout.dim = [dim0, dim1, dim2]
			
			self.gp_grid_raw_pub.publish(msg)
			
		except Exception as e:
			self.get_logger().error(f"Error publishing raw GP grid: {e}")
	
	
	def _create_fast_adaptive_gp_grid(self, semantic_points: np.ndarray, radius: float) -> np.ndarray:
		"""Create FAST, adaptive grid around semantic voxel clusters."""
		try:
			if len(semantic_points) == 0:
				return np.array([])
			
			# Use coarser resolution for speed (0.2m)
			resolution = 0.2
			
			# Find bounding box of all semantic voxels
			min_coords = np.min(semantic_points, axis=0)
			max_coords = np.max(semantic_points, axis=0)
			
			# Use adaptive radius for extension
			extent = np.max(max_coords - min_coords) + radius
			half_extent = extent / 2.0
			center = (min_coords + max_coords) / 2.0
			
			# Create FAST grid with coarser resolution
			x_range = np.arange(center[0] - half_extent, center[0] + half_extent, resolution)
			y_range = np.arange(center[1] - half_extent, center[1] + half_extent, resolution)
			z_range = np.arange(center[2] - half_extent, center[2] + half_extent, resolution)
			
			# Create meshgrid
			X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
			grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
			
			# FAST filtering using vectorized operations
			filtered_grid_points = self._filter_grid_points_fast(grid_points, semantic_points, radius)
			
			return filtered_grid_points
			
		except Exception as e:
			self.get_logger().error(f"Error creating fast adaptive GP grid: {e}")
			return np.array([])
	
	def _filter_grid_points_fast(self, grid_points: np.ndarray, voxel_positions: np.ndarray, max_distance: float) -> np.ndarray:
		"""FAST filtering using KD-tree (O(N log M) instead of O(N*M) full distance matrix)."""
		try:
			if len(grid_points) == 0 or len(voxel_positions) == 0:
				return grid_points
			
			# Use KD-tree for O(N log M) instead of O(N*M) full distance matrix
			try:
				from scipy.spatial import cKDTree
				# Build KD-tree once (O(M log M))
				tree = cKDTree(voxel_positions)
				
				# Query all grid points (O(N log M))
				distances, _ = tree.query(grid_points, k=1)
				
				# Filter points within max_distance
				mask = distances <= max_distance
				filtered_points = grid_points[mask]
				
				return filtered_points
				
			except ImportError:
				# Fallback: chunked computation to avoid large memory allocation
				chunk_size = 10000
				mask = np.zeros(len(grid_points), dtype=bool)
				
				for i in range(0, len(grid_points), chunk_size):
					chunk = grid_points[i:i+chunk_size]
					distances = np.linalg.norm(
						chunk[:, np.newaxis, :] - voxel_positions[np.newaxis, :, :], 
						axis=2
					)
					min_distances = np.min(distances, axis=1)
					mask[i:i+chunk_size] = min_distances <= max_distance
				
				return grid_points[mask]
			
		except Exception as e:
			self.get_logger().error(f"Error in fast grid filtering: {e}")
			return grid_points
	
	def _predict_gp_field_fast(self, grid_points: np.ndarray, cause_points: np.ndarray, fit_params: dict) -> np.ndarray:
		"""FAST GP field prediction using optimized anisotropic RBF.
		
		Uses the proper model from the paper (Eq. 3.2):
		  disturbance(x) = A * phi(x) + b
		where phi(x) = Σ_j exp(-0.5 * Q_ℓ(x - c_j)) is the voxel-centric
		superposed anisotropic RBF basis function.
		
		A >= 0 and b >= 0 are guaranteed by bounded least squares
		(scipy.optimize.lsq_linear) during GP fitting in voxel_gp_helper.py.
		This ensures:
		  - High disturbance near causes: A * phi_peak + b
		  - Low disturbance far from causes: ~b (ambient level)
		  - Always non-negative predictions
		"""
		try:
			# Extract GP parameters
			lxy = fit_params.get('lxy', 0.5)
			lz = fit_params.get('lz', 0.5)
			A = fit_params.get('A', 1.0)
			b = fit_params.get('b', 0.0)
			
			# Use OPTIMIZED anisotropic RBF computation
			phi = _sum_of_anisotropic_rbf_fast(grid_points, cause_points, lxy, lz)
			
			# Proper model: A * phi + b
			# A >= 0 guaranteed by bounded LS in fitting (no post-hoc abs needed)
			# b >= 0 ensures non-negative ambient disturbance
			predictions = A * phi + b
			
			# Sanitize NaN/Inf values (critical for downstream GPU operations)
			predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
			
			return predictions
			
		except Exception as e:
			self.get_logger().error(f"Error in fast GP prediction: {e}")
			return np.zeros(len(grid_points))
	
	
	def _compute_epistemic_uncertainty(self, grid_points: np.ndarray, cause_points: np.ndarray, 
									   fit_params: dict) -> Optional[np.ndarray]:
		"""
		Compute full predictive uncertainty via DisturbanceFieldHelper.
		
		Delegates to helper.predict_direct_field_3d which computes three components:
		  1. Aleatoric noise: σ²
		  2. Linear param (A, b) BLR: σ² · vᵀ (XᵀX)⁻¹ v
		  3. Length-scale (lxy, lz) delta method: A² · Jᵀ H⁻¹ J
		
		All required matrices (XtX_inv, hess_inv) are precomputed during fitting
		and stored in fit_params, so no training data is needed at prediction time.
		
		Args:
			grid_points: (N, 3) query points
			cause_points: (M, 3) cause points
			fit_params: Dictionary with lxy, lz, A, b, sigma2, XtX_inv, hess_inv
		
		Returns:
			(N,) predictive standard deviation (uncertainty)
		"""
		try:
			from resilience.voxel_gp_helper import DisturbanceFieldHelper
			helper = DisturbanceFieldHelper()
			_, std_pred = helper.predict_direct_field_3d(fit_params, grid_points, cause_points)
			# Sanitize for downstream GPU operations (critical for CUDA tensors)
			std_pred = np.nan_to_num(std_pred, nan=0.0, posinf=0.0, neginf=0.0)
			return std_pred
		except Exception as e:
			self.get_logger().error(f"Error computing epistemic uncertainty: {e}")
			import traceback
			traceback.print_exc()
			return None
	
	def _create_uncertainty_pointcloud(self, grid_points: np.ndarray, uncertainty_std: np.ndarray) -> Optional[PointCloud2]:
		"""
		Creates a PointCloud2 with a sharp 'Inferno' colormap and percentile normalization.
		"""
		try:
			if len(grid_points) == 0 or len(uncertainty_std) == 0:
				return None

			# 1. Percentile Normalization (The secret to the "Sharp" look)
			# Instead of min/max, use percentiles to ignore outliers and boost contrast
			u_min = np.percentile(uncertainty_std, 5)
			u_max = np.percentile(uncertainty_std, 95)
	
			# Avoid division by zero
			diff = u_max - u_min if u_max > u_min else 1.0
			normalized_values = np.clip((uncertainty_std - u_min) / diff, 0, 1)

			# 2. Apply Sharp Colormap (inferno or magma)
			# 'inferno' goes: Black -> Purple -> Red -> Bright Yellow
			colors_mapped = cm.inferno(normalized_values) 

			# Create structured array for PointCloud2
			cloud_data_combined = np.empty(len(grid_points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)
			])
	
			cloud_data_combined['x'] = grid_points[:, 0]
			cloud_data_combined['y'] = grid_points[:, 1]
			cloud_data_combined['z'] = grid_points[:, 2]

			# 3. Fast RGB Packing
			# Pack RGBA (normalized 0-1) into a single UINT32 for RViz
			r = (colors_mapped[:, 0] * 255).astype(np.uint32)
			g = (colors_mapped[:, 1] * 255).astype(np.uint32)
			b = (colors_mapped[:, 2] * 255).astype(np.uint32)
	
			# Bit-shift to pack into UINT32 (R << 16 | G << 8 | B)
			rgb_packed = (r << 16) | (g << 8) | b
			cloud_data_combined['rgb'] = rgb_packed

			# 4. Standard PointCloud2 Message Creation
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self._pointcloud_frame_id()
	
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
			]
			cloud_msg.point_step = 16
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			cloud_msg.data = cloud_data_combined.tobytes()

			return cloud_msg

		except Exception as e:
			self.get_logger().error(f"Error creating sharp uncertainty cloud: {e}")
			return None
	
	def _create_costmap_pointcloud(self, grid_points: np.ndarray, gp_values: np.ndarray) -> Optional[PointCloud2]:
		"""Create costmap point cloud with ACTUAL disturbance values for motion planning."""
		try:
			if len(grid_points) == 0 or len(gp_values) == 0:
				return None
			
			# Use ACTUAL GP disturbance values (not normalized) for motion planning
			# These are the real disturbance magnitudes that motion planning needs
			disturbance_values = gp_values.astype(np.float32)
			
			# Create PointCloud2 message with XYZ + disturbance values
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self._pointcloud_frame_id()
			
			# Create structured array with XYZ + disturbance value
			cloud_data_combined = np.empty(len(grid_points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), 
				('disturbance', np.float32)
			])
			
			# Fill in the data
			cloud_data_combined['x'] = grid_points[:, 0]
			cloud_data_combined['y'] = grid_points[:, 1]
			cloud_data_combined['z'] = grid_points[:, 2]
			cloud_data_combined['disturbance'] = disturbance_values
			
			# Create PointCloud2 message
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields - XYZ + disturbance value
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='disturbance', offset=12, datatype=pc2.PointField.FLOAT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, disturbance)
			cloud_msg.width = len(grid_points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			self.get_logger().info(f"Published costmap with ACTUAL disturbance values: min={disturbance_values.min():.3f}, max={disturbance_values.max():.3f}")
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating costmap point cloud: {e}")
			return None
	
	def _get_all_semantic_voxels(self) -> List[np.ndarray]:
		"""Return stored semantic voxel positions without further processing."""
		try:
			semantic_voxel_positions: List[np.ndarray] = []
			with self.semantic_voxels_lock:
				for semantic_info in self.semantic_voxels.values():
					voxel_position = semantic_info.get('position')
					if voxel_position is not None:
						semantic_voxel_positions.append(voxel_position)
			return semantic_voxel_positions
		except Exception as e:
			self.get_logger().error(f"Error getting semantic voxels: {e}")
			return []
	
	def _get_semantic_voxels_by_cause(self) -> Dict[str, List[np.ndarray]]:
		"""Return semantic voxel positions grouped by vlm_answer (cause).
		
		Returns:
			Dict mapping vlm_answer -> numpy array of voxel positions (Nx3)
		"""
		try:
			cause_to_positions: Dict[str, List[np.ndarray]] = {}
			with self.semantic_voxels_lock:
				for semantic_info in self.semantic_voxels.values():
					voxel_position = semantic_info.get('position')
					vlm_answer = semantic_info.get('vlm_answer', 'unknown')
					if voxel_position is not None:
						if vlm_answer not in cause_to_positions:
							cause_to_positions[vlm_answer] = []
						cause_to_positions[vlm_answer].append(voxel_position)
			
			# Convert lists to numpy arrays
			result = {}
			for cause, positions in cause_to_positions.items():
				if len(positions) > 0:
					result[cause] = np.array(positions, dtype=np.float32)
			return result
		except Exception as e:
			self.get_logger().error(f"Error getting semantic voxels by cause: {e}")
			return {}
	
	def _rebuild_semantic_spatial_index(self):
		"""Rebuild KD-tree spatial index for fast neighbor queries.
		
		Called when semantic voxels change significantly (batch updates).
		Uses lazy rebuilding (only when spatial_index_dirty flag is set).
		"""
		try:
			with self.semantic_voxels_lock:
				if not self.semantic_voxels:
					self.semantic_spatial_index = None
					self.semantic_voxel_keys_indexed = []
					self.spatial_index_dirty = False
					return
				
				# Build arrays of positions and keys
				positions = []
				keys = []
				for voxel_key, semantic_info in self.semantic_voxels.items():
					pos = semantic_info.get('position')
					if pos is not None:
						positions.append(pos)
						keys.append(voxel_key)
				
				if not positions:
					self.semantic_spatial_index = None
					self.semantic_voxel_keys_indexed = []
					self.spatial_index_dirty = False
					return
				
				# Build KD-tree for O(log N) spatial queries
				try:
					from scipy.spatial import cKDTree
					positions_array = np.array(positions)
					self.semantic_spatial_index = cKDTree(positions_array)
					self.semantic_voxel_keys_indexed = keys
					self.spatial_index_dirty = False
					self.get_logger().debug(f"Rebuilt semantic spatial index with {len(keys)} voxels")
				except ImportError:
					self.get_logger().warn("scipy not available, spatial index disabled")
					self.semantic_spatial_index = None
					self.spatial_index_dirty = False
					
		except Exception as e:
			self.get_logger().error(f"Error rebuilding spatial index: {e}")
			self.spatial_index_dirty = False
	
	def _get_semantic_neighbors_spatial(self, center: np.ndarray, radius: float, vlm_answer: str = None) -> List[tuple]:
		"""Fast neighbor query using KD-tree spatial index.
		
		Args:
			center: Center position (world coordinates)
			radius: Search radius in meters
			vlm_answer: Optional filter by VLM answer
			
		Returns:
			List of voxel keys within radius
		"""
		try:
			# Rebuild index if needed (lazy)
			if self.spatial_index_dirty or self.semantic_spatial_index is None:
				self._rebuild_semantic_spatial_index()
			
			if self.semantic_spatial_index is None:
				return []
			
			# Query KD-tree for neighbors (O(log N))
			indices = self.semantic_spatial_index.query_ball_point(center, radius)
			
			if not indices:
				return []
			
			# Get voxel keys from indices
			neighbor_keys = [self.semantic_voxel_keys_indexed[i] for i in indices]
			
			# Filter by VLM answer if specified
			if vlm_answer is not None:
				with self.semantic_voxels_lock:
					neighbor_keys = [
						key for key in neighbor_keys
						if key in self.semantic_voxels and 
						   self.semantic_voxels[key].get('vlm_answer') == vlm_answer
					]
			
			return neighbor_keys
			
		except Exception as e:
			self.get_logger().error(f"Error in spatial neighbor query: {e}")
			return []

	def _query_semantic_region(self, center: np.ndarray, radius: float) -> dict:
		"""Query all semantic voxels in a region and return their metadata.
		
		Args:
			center: Center position (world coordinates)
			radius: Search radius in meters
			
		Returns:
			Dict mapping voxel_key -> semantic_info for voxels in region
		"""
		try:
			neighbor_keys = self._get_semantic_neighbors_spatial(center, radius)
			
			if not neighbor_keys:
				return {}
			
			# Get metadata for all neighbors
			region_info = {}
			with self.semantic_voxels_lock:
				for key in neighbor_keys:
					if key in self.semantic_voxels:
						region_info[key] = self.semantic_voxels[key]
			
			return region_info
			
		except Exception as e:
			self.get_logger().error(f"Error querying semantic region: {e}")
			return {}
		
	def _get_neighboring_voxel_keys(self, voxel_key: tuple) -> List[tuple]:
		"""Get voxel key and its 26 neighbors (3x3x3 cube)."""
		vx, vy, vz = voxel_key
		neighbors = []
		for dx in [-1/2, 0, 1/2]:
			for dy in [-1/2, 0, 1/2]:
				for dz in [-1/2, 0, 1/2]:
					neighbors.append((vx + dx, vy + dy, vz + dz))
		return neighbors
	
	def _increment_spatial_observation_counts(self, voxel_key: tuple, vlm_answer: str, frame_id: int, timestamp: float):
		"""OPTIMIZED: Incrementally update spatial observation counts for all 27 neighbors (including self).
		
		This maintains pre-computed counts so threshold checks are O(1) instead of O(neighbors * observations).
		"""
		neighbors = self._get_neighboring_voxel_keys(voxel_key)
		current_time = time.time()
		
		for nkey in neighbors:
			key = (nkey, vlm_answer)
			if key not in self.spatial_observation_counts:
				self.spatial_observation_counts[key] = {
					'count': 0,
					'unique_frames': set(),
					'last_update': current_time
				}
			
			entry = self.spatial_observation_counts[key]
			
			# Increment count (for narration)
			entry['count'] += 1
			
			# Add unique frame (for operational)
			if frame_id is not None:
				entry['unique_frames'].add(frame_id)
			
			entry['last_update'] = current_time
	
	def _cleanup_old_spatial_counts(self, current_time: float):
		"""Periodically cleanup old entries from spatial_observation_counts."""
		# Only cleanup if dict is getting large (avoid overhead on every call)
		if len(self.spatial_observation_counts) < 1000:
			return
		
		# Remove entries older than max_age
		keys_to_remove = []
		for key, entry in self.spatial_observation_counts.items():
			if (current_time - entry['last_update']) > self.semantic_observation_max_age:
				keys_to_remove.append(key)
		
		for key in keys_to_remove:
			del self.spatial_observation_counts[key]
	
	def _get_observation_count_fast(self, voxel_key: tuple, vlm_answer: str) -> int:
		"""OPTIMIZED: Fast O(1) lookup for observation count with spatial support."""
		key = (voxel_key, vlm_answer)
		entry = self.spatial_observation_counts.get(key)
		if entry is None:
			return 0
		
		# Check if entry is still valid (not expired)
		current_time = time.time()
		if (current_time - entry['last_update']) > self.semantic_observation_max_age:
			return 0
		
		return entry['count']
	
	def _get_unique_frames_count_fast(self, voxel_key: tuple, vlm_answer: str) -> int:
		"""OPTIMIZED: Fast O(1) lookup for unique frames count with spatial support."""
		key = (voxel_key, vlm_answer)
		entry = self.spatial_observation_counts.get(key)
		if entry is None:
			return 0
		
		# Check if entry is still valid (not expired)
		current_time = time.time()
		if (current_time - entry['last_update']) > self.semantic_observation_max_age:
			return 0
		
		return len(entry['unique_frames'])

	def _prune_stale_semantic_voxels(self) -> int:
		"""Remove semantic voxels no longer occupied in VDB (syncs with occupancy map).
		
		Returns:
			Number of voxels pruned.
		"""
		if not self.semantic_voxels or self.vdb_mapper is None or self.vdb_mapper.is_empty():
			return 0
		
		try:
			with self.semantic_voxels_lock:
				if not self.semantic_voxels:
					return 0

				# Batch query: collect all positions
				keys = list(self.semantic_voxels.keys())
				positions = np.array([self.semantic_voxels[k]['position'] for k in keys], dtype=np.float32)

				# Guard: validate positions before sending to CUDA
				if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
					self.get_logger().warn(f"Pruning skipped: invalid positions shape {positions.shape}")
					return 0
				if not np.isfinite(positions).all():
					positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)

				# Query VDB occupancy in batch
				try:
					positions_tensor = torch.from_numpy(positions).float()
					occ_values = rayfronts_cpp.query_occ(self.vdb_mapper.occ_map_vdb, positions_tensor)
					occ_np = occ_values.numpy().squeeze()
					if occ_np.ndim == 0:
						occ_np = occ_np.reshape(1)
				except (RuntimeError, Exception) as cuda_err:
					self.get_logger().warn(f"query_occ CUDA error during pruning, skipping: {cuda_err}")
					return 0

				# Identify voxels to remove (occupancy <= 0 means free/unknown)
				keys_to_remove = [k for k, occ in zip(keys, occ_np) if occ <= 0]

				# Remove stale voxels
				for k in keys_to_remove:
					del self.semantic_voxels[k]
					# Also clean up observation tracking
					if k in self.semantic_voxel_observations:
						del self.semantic_voxel_observations[k]

				if keys_to_remove:
					self.spatial_index_dirty = True

				return len(keys_to_remove)

		except Exception as e:
			self.get_logger().warn(f"Semantic voxel pruning failed: {e}")
			return 0

	def _voxelize_points_fast(self, points_world: np.ndarray) -> np.ndarray:
		"""Voxelize points using RayFronts utilities (GPU-accelerated).
		
		Returns:
			Numpy array of unique voxel centers (Nx3, float32).
		"""
		if len(points_world) == 0:
			return np.array([], dtype=np.float32).reshape(0, 3)

		# Validate shape before sending to CUDA
		if points_world.ndim != 2 or points_world.shape[1] != 3:
			self.get_logger().warn(f"_voxelize_points_fast: unexpected shape {points_world.shape}, skipping GPU path")
		elif RAYFRONTS_G3D_AVAILABLE:
			try:
				points_tensor = torch.from_numpy(points_world.astype(np.float32)).to(self.device)
				if points_tensor.shape[0] == 0:
					return np.array([], dtype=np.float32).reshape(0, 3)
				vox_xyz = g3d.pointcloud_to_sparse_voxels(points_tensor, vox_size=float(self.voxel_resolution))
				return vox_xyz.cpu().numpy()
			except (RuntimeError, Exception) as e:
				self.get_logger().warn(f"GPU voxelization failed, falling back to numpy: {e}")

		# Fallback to numpy (still efficient)
		voxel_coords = np.floor(points_world / self.voxel_resolution).astype(np.int32)
		unique_coords = np.unique(voxel_coords, axis=0)
		return (unique_coords.astype(np.float32) * self.voxel_resolution)
	
	def _apply_semantic_labels_to_voxels(self, points_world: np.ndarray, vlm_answer: str,
									 threshold: float, stats: dict, is_narration: bool = False):
		"""Apply semantic labels with temporal+spatial confirmation.
		
		Uses RayFronts voxelization for efficiency and maintains observation counts
		for multi-frame confirmation (noise rejection).
		"""
		try:
			current_time = time.time()
			if not is_narration:
				self.frame_counter += 1
			frame_id = 0 if is_narration else self.frame_counter
			similarity_score = stats.get('avg_similarity', threshold + 0.1)
			confirmation_threshold = self.narration_confirmation_threshold if is_narration else self.operational_confirmation_threshold
			
			# Use RayFronts voxelization (GPU-accelerated when available)
			vox_xyz_np = self._voxelize_points_fast(points_world)
			if len(vox_xyz_np) == 0:
				return
			
			# Convert to voxel keys efficiently
			voxel_keys = [tuple(np.round(xyz / self.voxel_resolution).astype(np.int32)) for xyz in vox_xyz_np]
			
			with self.semantic_voxels_lock:
				semantic_voxels_before = int(len(self.semantic_voxels))
			
			# Batch update observations and spatial counts
			confirmed_count = 0
			new_voxels_added = False
			
			for i, voxel_key in enumerate(voxel_keys):
				# Update observation tracking
				if voxel_key not in self.semantic_voxel_observations:
					self.semantic_voxel_observations[voxel_key] = []
				
				self.semantic_voxel_observations[voxel_key].append({
					'vlm_answer': vlm_answer, 'timestamp': current_time,
					'frame_id': frame_id, 'similarity': similarity_score
				})
				
				# Incremental spatial counts update (O(1) threshold checks)
				self._increment_spatial_observation_counts(
					voxel_key, vlm_answer, 
					frame_id if not is_narration else None,
					current_time
				)
				
				# Lazy cleanup (only when list grows large)
				obs_list = self.semantic_voxel_observations[voxel_key]
				if len(obs_list) > 20:
					self.semantic_voxel_observations[voxel_key] = [
						o for o in obs_list if (current_time - o['timestamp']) <= self.semantic_observation_max_age
					]
				
				# Check confirmation threshold (O(1) lookup)
				if is_narration:
					meets_threshold = self._get_observation_count_fast(voxel_key, vlm_answer) >= confirmation_threshold
					confidence = self._get_observation_count_fast(voxel_key, vlm_answer)
				else:
					meets_threshold = self._get_unique_frames_count_fast(voxel_key, vlm_answer) >= confirmation_threshold
					confidence = self._get_unique_frames_count_fast(voxel_key, vlm_answer)
				
				if meets_threshold:
					with self.semantic_voxels_lock:
						self.semantic_voxels[voxel_key] = {
							'vlm_answer': vlm_answer,
							'similarity': similarity_score,
							'threshold_used': threshold,
							'timestamp': current_time,
							'position': vox_xyz_np[i].astype(np.float32),
							'confidence': confidence,
							'is_narration': is_narration
						}
					confirmed_count += 1
					new_voxels_added = True
			
			# Periodic cleanup
			self._cleanup_old_spatial_counts(current_time)
			
			if new_voxels_added:
				self.spatial_index_dirty = True
			
			with self.semantic_voxels_lock:
				semantic_voxels_after = int(len(self.semantic_voxels))
			semantic_added = semantic_voxels_after - semantic_voxels_before
			
			hotspot_type = "narration" if is_narration else "operational"
			self.get_logger().info(
				f"Semantic ({hotspot_type}): {len(voxel_keys)} voxels, {confirmed_count} confirmed for '{vlm_answer}'"
			)
			self.get_logger().warn(
				f"Semantic voxel store ({hotspot_type}) for '{vlm_answer}': "
				f"before={semantic_voxels_before}, after={semantic_voxels_after}, added={semantic_added}, "
				f"confirmed_this_update={confirmed_count}, candidate_voxels={int(len(voxel_keys))}"
			)
			
		except Exception as e:
			self.get_logger().error(f"Error applying semantic labels: {e}")
	
	def _get_voxel_key_from_point(self, point) -> tuple:
		"""Convert world point to voxel key. Handles both numpy arrays and torch tensors."""
		# Convert torch tensor to numpy if needed
		if torch.is_tensor(point):
			point = point.cpu().numpy()
		
		# Ensure it's a numpy array
		if not isinstance(point, np.ndarray):
			point = np.array(point)
		
		voxel_coords = np.floor(point / self.voxel_resolution).astype(np.int32)
		return tuple(voxel_coords)
		
	def depth_callback(self, msg: Image):
		if self.camera_intrinsics is None:
			self.get_logger().warn("No camera intrinsics received yet")
			return

		# Convert and store depth with timestamp in meters
		try:
			depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
			depth = self._rotate_image_if_needed(depth)
			depth_m = self._depth_to_meters(depth, msg.encoding)
			if depth_m is None:
				return
			
			depth_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
			self.depth_buffer_data.append(msg)
			self.depth_buffer_ts.append(depth_time)
			
			# Regular VDB occupancy mapping: run in separate thread to avoid blocking
			if self.latest_pose is not None:
				# Make a copy of depth and pose for thread safety
				depth_copy = depth_m.copy()
				pose_copy = Odometry()
				pose_copy.header = self.latest_pose.header
				pose_copy.child_frame_id = getattr(self.latest_pose, "child_frame_id", "")
				pose_copy.pose = self.latest_pose.pose
				pose_copy.twist = self.latest_pose.twist
				
				# Run regular mapping in background thread
				threading.Thread(
					target=self._update_regular_mapping,
					args=(depth_copy, pose_copy),
					daemon=True
				).start()
			
		except Exception as e:
			self.get_logger().error(f"Error storing depth frame: {e}")

		# Activity update
		self.last_data_time = time.time()

		# Periodic publishing (includes deferred frontier computation)
		self._periodic_publishing()
		
		# Compute regular frontiers periodically (not on every depth frame to reduce contention)
		now = time.time()
		if not hasattr(self, 'last_frontier_compute_time'):
			self.last_frontier_compute_time = 0.0


	def _update_regular_mapping(self, depth_m: np.ndarray, pose):
		"""Update regular VDB occupancy mapping in a separate thread. Handles both PoseStamped and Odometry."""
		try:
			# Guard: depth must be a valid 2-D finite array
			if depth_m is None or depth_m.ndim != 2:
				return
			h, w = depth_m.shape
			if h < 1 or w < 1:
				return

			# Sanitize depth before sending to CUDA (NaN/Inf can cause index OOB in kernels)
			depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

			# CRITICAL: Use no_grad to prevent gradient accumulation
			with torch.no_grad():
				device = self.vdb_mapper.device
				depth_tensor = torch.from_numpy(depth_m).float().unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHxW
				rgb_tensor = torch.zeros(1, 3, h, w, dtype=torch.float32, device=device)
				pose_4x4 = self._pose_to_4x4_matrix(pose)

				# Guard: pose matrix must be finite
				if not torch.isfinite(pose_4x4).all():
					return

				self.vdb_mapper.process_posed_rgbd(
					rgb_img=rgb_tensor,
					depth_img=depth_tensor,
					pose_4x4=pose_4x4
				)
		except (RuntimeError, Exception) as e:
			# Log only unexpected errors (RuntimeError covers CUDA OOB now that
			# CUDA_LAUNCH_BLOCKING=1 is set)
			if 'index out of bounds' in str(e) or 'CUDA' in str(e):
				self.get_logger().warn(f"_update_regular_mapping CUDA error (suppressed): {e}")
			# Silently discard other transient errors

	def _update_semantic_voxels(self, points_world: np.ndarray, vlm_answer: str, threshold: float, 
								 stats: dict, is_narration: bool):
		"""Update semantic voxel labels in a separate thread (optimized)."""
		try:
			if points_world.size > 0:
				self._apply_semantic_labels_to_voxels(points_world, vlm_answer, threshold, stats, is_narration)
		except Exception as e:
			self.get_logger().warn(f"Semantic voxel update error: {e}")

	def _depth_to_meters(self, depth, encoding: str):
		try:
			enc = (encoding or '').lower()
			if '16uc1' in enc or 'mono16' in enc:
				return depth.astype(np.float32) / 1000.0
			elif '32fc1' in enc or 'float32' in enc:
				return depth.astype(np.float32)
			else:
				return depth.astype(np.float32) / 1000.0
		except Exception:
			return None
	

	def _depth_to_world_points_sparse(self, u: np.ndarray, v: np.ndarray, z: np.ndarray, intrinsics, pose):
		"""Optimized version that only processes sparse hotspot pixels (no meshgrid). Handles both PoseStamped and Odometry."""
		try:
			fx, fy, cx, cy = intrinsics
			# Direct computation for sparse pixels
			x = (u - cx) * z / fx
			y = (v - cy) * z / fy
			pts_cam = np.stack([x, y, z], axis=1)

			# Transform to base if needed
			if bool(self.pose_is_base_link):
				pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
				pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra

			# World transform
			R_world = self._quat_to_rot(self._pose_quat(pose))
			p_world = self._pose_position(pose)
			pts_world = pts_cam @ R_world.T + p_world
			return pts_world
		except Exception:
			return None

	def _create_semantic_colored_cloud(self, max_points: int) -> Optional[PointCloud2]:
		"""Create a colored point cloud that shows both regular occupancy voxels and semantic voxels."""
		try:
			# Get occupancy voxels from VDB
			if self.vdb_mapper.is_empty():
				# If VDB is empty, only show semantic voxels
				with self.semantic_voxels_lock:
					if not self.semantic_voxels:
						return None
					
					points = []
					colors = []
					for voxel_key, semantic_info in self.semantic_voxels.items():
						voxel_center = semantic_info['position']
						if voxel_center is not None:
							points.append(voxel_center)
							vlm_answer = semantic_info.get('vlm_answer', 'unknown')
							color = self._get_vlm_answer_color(vlm_answer)
							colors.append(color)
			else:
				# Get occupancy data from VDB
				pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
				
				# Convert to numpy if it's a torch tensor
				if torch.is_tensor(pc_xyz_occ_size):
					pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
				
				# Filter occupied voxels
				occupied_mask = pc_xyz_occ_size[:, -2] > 0
				occupied_points_data = pc_xyz_occ_size[occupied_mask]
				
				# OPTIMIZED: Copy semantic voxels once with single lock acquisition
				# This avoids thousands of lock acquisitions inside the loop
				with self.semantic_voxels_lock:
					semantic_voxels_copy = dict(self.semantic_voxels)  # Fast shallow copy
				
				# Create point cloud data
				points = []
				colors = []
				semantic_count = 0
				regular_count = 0
				
				# Add regular occupancy voxels
				for point_data in occupied_points_data:
					point = point_data[:3]  # xyz
					voxel_key = self._get_voxel_key_from_point(point)
					
					# FAST: Check semantic voxels from copy (no lock needed)
					if voxel_key in semantic_voxels_copy:
						# Semantic voxel - use VLM answer color
						semantic_info = semantic_voxels_copy[voxel_key]
						vlm_answer = semantic_info.get('vlm_answer', 'unknown')
						color = self._get_vlm_answer_color(vlm_answer)
						semantic_count += 1
					else:
						# Regular occupancy voxel - use gray
						color = [128, 128, 128]
						regular_count += 1
					
					points.append(point)
					colors.append(color)
					
					# Limit points
					if len(points) >= max_points:
						break
				
				# Add any semantic voxels that aren't in VDB occupancy
				# Use the copy we already have (no lock needed)
				for voxel_key, semantic_info in semantic_voxels_copy.items():
					if len(points) >= max_points:
						break
					# Check if this semantic voxel is already added
					voxel_center = semantic_info['position']
					if voxel_center is not None:
						# Simple check: if voxel_key not in occupancy voxels
						# (This is approximate, but good enough for visualization)
						vlm_answer = semantic_info.get('vlm_answer', 'unknown')
						color = self._get_vlm_answer_color(vlm_answer)
						points.append(voxel_center)
						colors.append(color)
						semantic_count += 1
			
			if not points:
				return None
			
			# Log the coloring information
			if 'semantic_count' in locals() and semantic_count > 0:
				self.get_logger().info(f"Creating VDB colored cloud: {semantic_count} semantic voxels (colored by VLM answer), {regular_count if 'regular_count' in locals() else 0} regular voxels (GRAY)")
			else:
				self.get_logger().info(f"Creating VDB colored cloud: {len(points)} voxels")
			
			# Convert to numpy arrays
			points_array = np.array(points, dtype=np.float32)
			colors_array = np.array(colors, dtype=np.uint8)
			
			# Create PointCloud2 message with proper structure
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self._pointcloud_frame_id()
			
			# Create structured array with XYZ + RGB
			cloud_data_combined = np.empty(len(points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32), 
				('rgb', np.uint32)
			])
			
			# Fill in the data
			cloud_data_combined['x'] = points_array[:, 0]
			cloud_data_combined['y'] = points_array[:, 1]
			cloud_data_combined['z'] = points_array[:, 2]
			
			# Pack RGB values as UINT32 (standard for PointCloud2 RGB)
			rgb_packed = np.zeros(len(colors_array), dtype=np.uint32)
			for i, c in enumerate(colors_array):
				rgb_packed[i] = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
			cloud_data_combined['rgb'] = rgb_packed
			
			# Create PointCloud2 message with proper fields from the start
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields properly - use UINT32 for rgb to ensure RViz compatibility
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.UINT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 16  # 4 bytes per float * 4 fields (x, y, z, rgb)
			cloud_msg.width = len(points)  # Set correct width
			cloud_msg.height = 1  # Set height to 1 for organized point cloud
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data_combined.tobytes()
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating semantic colored cloud: {e}")
			import traceback
			traceback.print_exc()
			return None
	
	def _get_vlm_answer_color(self, vlm_answer: str) -> List[int]:
		"""Get consistent color for VLM answer (same as bridge)."""
		# Use same color palette as semantic bridge
		color_palette = [
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
		
		# Simple hash-based color assignment
		hash_val = hash(vlm_answer) % len(color_palette)
		return color_palette[hash_val]
	
	def _get_voxel_center_from_key(self, voxel_key: tuple) -> Optional[np.ndarray]:
		"""Get voxel center position from voxel key."""
		try:
			vx, vy, vz = voxel_key
			
			# Convert voxel coordinates to world coordinates
			world_x = vx * self.voxel_resolution
			world_y = vy * self.voxel_resolution
			world_z = vz * self.voxel_resolution
			
			return np.array([world_x, world_y, world_z], dtype=np.float32)
			
		except Exception as e:
			self.get_logger().warn(f"Error getting voxel center for key {voxel_key}: {e}")
			return None
	

	def _create_semantic_only_cloud(self) -> Optional[PointCloud2]:
		"""Create a point cloud containing all accumulated semantic voxels."""
		try:
			# Get all accumulated semantic voxels
			with self.semantic_voxels_lock:
				if not self.semantic_voxels:
					return None
				
				# Create point cloud data for all accumulated semantic voxels
				points = []
				for voxel_key, semantic_info in self.semantic_voxels.items():
					voxel_center = semantic_info['position']
					if voxel_center is not None:
						points.append(voxel_center)
			
			if not points:
				return None
			
			# Convert to numpy array
			points_array = np.array(points, dtype=np.float32)
			
			# Create PointCloud2 message with XYZ only (no RGB needed for semantic-only)
			header = Header()
			header.stamp = self.get_clock().now().to_msg()
			header.frame_id = self._pointcloud_frame_id()
			
			# Create structured array with just XYZ
			cloud_data = np.empty(len(points), dtype=[
				('x', np.float32), ('y', np.float32), ('z', np.float32)
			])
			
			# Fill in the data
			cloud_data['x'] = points_array[:, 0]
			cloud_data['y'] = points_array[:, 1]
			cloud_data['z'] = points_array[:, 2]
			
			# Create PointCloud2 message
			cloud_msg = PointCloud2()
			cloud_msg.header = header
			
			# Define the fields (XYZ only)
			cloud_msg.fields = [
				pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
				pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1)
			]
			
			# Set the message properties
			cloud_msg.point_step = 12  # 4 bytes per float * 3 fields (x, y, z)
			cloud_msg.width = len(points)
			cloud_msg.height = 1
			cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
			cloud_msg.is_dense = True
			
			# Set the data
			cloud_msg.data = cloud_data.tobytes()
			
			return cloud_msg
			
		except Exception as e:
			self.get_logger().error(f"Error creating semantic-only cloud: {e}")
			return None
	
	def _publish_semantic_voxel_spheres(self):
		"""
		Publish sphere markers only for semantic voxels that have at least 10 neighbors within 50cm.
		Uses efficient KD-tree for neighbor queries.
		"""
		try:
			# Get all semantic voxel positions
			with self.semantic_voxels_lock:
				if not self.semantic_voxels:
					# Publish empty marker array to clear previous markers
					marker_array = MarkerArray()
					marker_array.markers = []
					self.hazard_spheres_pub.publish(marker_array)
					return
				
				# Extract all voxel positions
				points = []
				for voxel_key, semantic_info in self.semantic_voxels.items():
					voxel_center = semantic_info.get('position')
					if voxel_center is not None:
						points.append(voxel_center)
			
			if not points:
				# Publish empty marker array to clear previous markers
				marker_array = MarkerArray()
				marker_array.markers = []
				self.hazard_spheres_pub.publish(marker_array)
				return
			
			# Convert to numpy array
			points_array = np.array(points, dtype=np.float32)
			
			# Filter voxels: only keep those with >= 10 neighbors within 50cm
			neighbor_radius = 0.5  # 50cm in meters
			min_neighbors = 20
			
			# Build KD-tree for efficient neighbor queries
			try:
				from scipy.spatial import cKDTree
				tree = cKDTree(points_array)
				
				# Query all points for neighbors within radius
				# query_ball_point returns list of lists, where each inner list contains
				# indices of neighbors (including self)
				neighbor_indices_list = tree.query_ball_point(points_array, neighbor_radius)
				
				# Filter points: keep only those with >= min_neighbors (including self)
				filtered_indices = []
				for i, neighbors in enumerate(neighbor_indices_list):
					# Count neighbors (excluding self for the count)
					neighbor_count = len(neighbors) - 1  # Subtract 1 to exclude self
					if neighbor_count >= min_neighbors:
						filtered_indices.append(i)
				
				# Extract filtered points
				if len(filtered_indices) > 0:
					filtered_points = points_array[filtered_indices]
				else:
					filtered_points = np.array([], dtype=np.float32).reshape(0, 3)
					
			except ImportError:
				# Fallback: use brute force if scipy not available (less efficient)
				self.get_logger().warn("scipy not available, using brute force neighbor search")
				filtered_indices = []
				for i, point in enumerate(points_array):
					# Compute distances to all other points
					distances = np.linalg.norm(points_array - point, axis=1)
					# Count neighbors within radius (excluding self)
					neighbor_count = np.sum((distances <= neighbor_radius) & (distances > 0))
					if neighbor_count >= min_neighbors:
						filtered_indices.append(i)
				
				if len(filtered_indices) > 0:
					filtered_points = points_array[filtered_indices]
				else:
					filtered_points = np.array([], dtype=np.float32).reshape(0, 3)
			
			if len(filtered_points) == 0:
				# Publish empty marker array to clear previous markers
				marker_array = MarkerArray()
				marker_array.markers = []
				self.hazard_spheres_pub.publish(marker_array)
				return
			
			# Create sphere markers for filtered points
			marker_array = MarkerArray()
			marker_id = 0
			
			hdr = Header()
			hdr.stamp = self.get_clock().now().to_msg()
			hdr.frame_id = self.map_frame
			
			# Default color for semantic voxels (red, similar to LaC)
			default_color = [255, 0, 0]  # BGR format
			sphere_radius = float(self.hazard_sphere_radius)
			
			# Create a sphere marker for each filtered point
			for point in filtered_points:
				marker = Marker()
				marker.header = hdr
				marker.ns = "semantic_voxel_sphere"
				marker.id = marker_id
				marker_id += 1
				marker.type = Marker.SPHERE
				marker.action = Marker.ADD
				
				# Set position (center of sphere)
				marker.pose.position.x = float(point[0])
				marker.pose.position.y = float(point[1])
				marker.pose.position.z = float(point[2])
				marker.pose.orientation.w = 1.0  # No rotation
				
				# Set scale (diameter = 2 * radius)
				marker.scale.x = float(sphere_radius * 2.0)
				marker.scale.y = float(sphere_radius * 2.0)
				marker.scale.z = float(sphere_radius * 2.0)
				
				# Set color (BGR to RGB, normalize to 0-1)
				marker.color.r = float(default_color[2]) / 255.0
				marker.color.g = float(default_color[1]) / 255.0
				marker.color.b = float(default_color[0]) / 255.0
				marker.color.a = 0.6  # Semi-transparent
				
				# Set lifetime (0 = infinite)
				marker.lifetime.sec = 0
				
				marker_array.markers.append(marker)
			
			if len(marker_array.markers) > 0:
				self.hazard_spheres_pub.publish(marker_array)
				self.get_logger().debug(
					f"Published {len(marker_array.markers)} sphere markers for semantic voxels "
					f"(filtered from {len(points_array)} to {len(filtered_points)} points with >= {min_neighbors} neighbors within {neighbor_radius}m, "
					f"radius={sphere_radius}m)"
				)
			else:
				# Publish empty marker array to clear previous markers
				marker_array.markers = []
				self.hazard_spheres_pub.publish(marker_array)
				
		except Exception as e:
			self.get_logger().warn(f"Error publishing semantic voxel spheres: {e}")
			import traceback
			self.get_logger().debug(traceback.format_exc())
	
	def _periodic_publishing(self):
		now = time.time()
		
		if self.cloud_pub:
			try:
				# Create VDB-based semantic-aware colored cloud
				semantic_cloud = self._create_semantic_colored_cloud(int(self.max_markers))
				if semantic_cloud:
					self.cloud_pub.publish(semantic_cloud)
					self.get_logger().debug(f"Published VDB semantic colored cloud with {len(semantic_cloud.data)//16} points")
				else:
					self.get_logger().debug("VDB cloud creation returned None (map may be empty)")
			except Exception as e:
				self.get_logger().warn(f"Failed to create VDB colored cloud: {e}")
		
		# Publish semantic-only point cloud (XYZ only, no RGB)
		if self.semantic_only_pub:
			try:
				t_sem_only = time.time()
				semantic_only_cloud = self._create_semantic_only_cloud()
				create_dt = time.time() - t_sem_only
				if semantic_only_cloud:
					with self.semantic_voxels_lock:
						semantic_voxel_count_now = int(len(self.semantic_voxels))
					point_count = int(getattr(semantic_only_cloud, "width", 0)) if getattr(semantic_only_cloud, "width", 0) else int(len(semantic_only_cloud.data) // 12)
					self.semantic_only_pub.publish(semantic_only_cloud)
					self.get_logger().debug(f"Published semantic-only cloud with {len(semantic_only_cloud.data)//12} points")
					self.get_logger().warn(
						f"/semantic_voxels_only publish: semantic_voxels={semantic_voxel_count_now}, "
						f"cloud_points={point_count}, frame_id='{semantic_only_cloud.header.frame_id}', "
						f"create_dt={create_dt:.4f}s"
					)
			except Exception as e:
				self.get_logger().warn(f"Failed to create semantic-only cloud: {e}")
		
		# Publish sphere markers for semantic voxels (revoxelized)
		# try:
		# 	self._publish_semantic_voxel_spheres()
		# except Exception as e:
		# 	self.get_logger().warn(f"Failed to publish semantic voxel spheres: {e}")
		
		# if self.stats_pub and (now - self.last_stats_pub) >= float(self.stats_publish_rate):
			# Get statistics from VDB mapper
			try:
				if not self.vdb_mapper.is_empty():
					pc_xyz_occ_size = rayfronts_cpp.occ_vdb2sizedpc(self.vdb_mapper.occ_map_vdb)
					
					# Convert to numpy if it's a torch tensor
					if torch.is_tensor(pc_xyz_occ_size):
						pc_xyz_occ_size = pc_xyz_occ_size.cpu().numpy()
					
					occupied_mask = pc_xyz_occ_size[:, -2] > 0
					total_voxels = int(np.sum(occupied_mask))
				else:
					total_voxels = 0
			except:
				total_voxels = 0
			
			# Add semantic mapping status and counts
			semantic_voxel_count = 0
			
			with self.semantic_voxels_lock:
				semantic_voxel_count = len(self.semantic_voxels)
			
			stats = {
				'mapper_type': 'VDB OccupancyMap',
				'total_voxels': total_voxels,
				'voxel_resolution': float(self.voxel_resolution),
				'semantic_mapping': {
					'enabled': self.enable_semantic_mapping,
					'status': 'active' if self.enable_semantic_mapping else 'disabled',
					'semantic_voxel_count': semantic_voxel_count
				}
			}
			
			self.stats_pub.publish(String(data=json.dumps(stats)))
			self.last_stats_pub = now

	def _pose_position(self, pose):
		"""Extract position from either PoseStamped or Odometry message."""
		if hasattr(pose.pose, 'pose'):  # Odometry message
			return np.array([pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z], dtype=np.float32)
		else:  # PoseStamped message
			return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)

	def _pose_quat(self, pose):
		"""Extract quaternion from either PoseStamped or Odometry message."""
		if hasattr(pose.pose, 'pose'):  # Odometry message
			q = pose.pose.pose.orientation
		else:  # PoseStamped message
			q = pose.pose.orientation
		return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

	def _quat_to_rot(self, q: np.ndarray):
		x, y, z, w = q
		n = x*x + y*y + z*z + w*w
		if n < 1e-8:
			return np.eye(3, dtype=np.float32)
		s = 2.0 / n
		xx, yy, zz = x*x*s, y*y*s, z*z*s
		xy, xz, yz = x*y*s, x*z*s, y*z*s
		wx, wy, wz = w*x*s, w*y*s, w*z*s
		return np.array([
			[1.0 - (yy + zz), xy - wz, xz + wy],
			[xy + wz, 1.0 - (xx + zz), yz - wx],
			[xz - wy, yz + wx, 1.0 - (xx + yy)]
		], dtype=np.float32)

	def _rpy_deg_to_rot(self, rpy_deg):
		try:
			roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
			cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
			Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
			Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
			Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
			return Rz @ Ry @ Rx
		except Exception:
			return np.eye(3, dtype=np.float32)

	def _pose_to_4x4_matrix(self, pose) -> torch.Tensor:
		"""Convert PoseStamped or Odometry to 4x4 transformation matrix with proper coordinate frame handling."""
		# Extract position and orientation (handle both PoseStamped and Odometry)
		if hasattr(pose.pose, 'pose'):  # Odometry message
			p = np.array([pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z], dtype=np.float32)
			q = np.array([pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, 
						 pose.pose.pose.orientation.z, pose.pose.pose.orientation.w], dtype=np.float32)
		else:  # PoseStamped message
			p = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)
			q = np.array([pose.pose.orientation.x, pose.pose.orientation.y, 
						 pose.pose.orientation.z, pose.pose.orientation.w], dtype=np.float32)
		R = self._quat_to_rot(q)
		
		# Apply coordinate frame transformation if pose is in base_link frame
		if bool(self.pose_is_base_link):
			# Transform from base_link to camera frame
			# This is the inverse of the transformation used in depth projection
			
			# Step 1: Transform pose from base_link to camera frame
			# Apply camera-to-base transformation (inverse)
			p = p - self.t_cam_to_base_extra
			R = R @ self.R_cam_to_base_extra
			
			# Step 2: Apply optical frame rotation (inverse)
			if bool(self.apply_optical_frame_rotation):
				R = R @ self.R_opt_to_base
		
		# Create 4x4 matrix
		T = np.eye(4, dtype=np.float32)
		T[:3, :3] = R
		T[:3, 3] = p
		
		# Convert to tensor and move to same device as mapper
		device = self.vdb_mapper.device
		return torch.from_numpy(T).unsqueeze(0).to(device)  # 1x4x4

	def _publish_mask_frontiers_and_rays(self):
		try:
			if self.vdb_mapper is None:
				return

			# Rays as arrows
			def _offset_origin(base: np.ndarray, direction: np.ndarray, idx: int) -> np.ndarray:
				offset_dir = np.cross(direction, np.array([0.0, 0.0, 1.0], dtype=np.float32))
				if np.linalg.norm(offset_dir) < 1e-6:
					offset_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
				offset_dir /= np.linalg.norm(offset_dir) + 1e-9
				offset_mag = float(self.voxel_resolution) * 0.3 * ((idx % 5) - 2)
				return base + offset_dir * offset_mag

			if self._latest_pose_rays is not None:
				origin_world, dir_world = self._latest_pose_rays
				now = self.get_clock().now().to_msg()
				# Use the same binning method as RayFronts for consistency
				try:
					# Guard: require valid numpy arrays with correct shapes
					if (dir_world is None or origin_world is None
							or dir_world.ndim != 2 or dir_world.shape[1] != 3
							or origin_world.ndim != 1 or origin_world.shape[0] != 3
							or dir_world.shape[0] == 0):
						raise ValueError(
							f"Invalid pose-ray shapes: origin={getattr(origin_world,'shape',None)}, "
							f"dirs={getattr(dir_world,'shape',None)}"
						)

					# Sanitize directions (NaN/Inf would cause OOB in CUDA kernels)
					dir_world = np.nan_to_num(dir_world, nan=0.0, posinf=0.0, neginf=0.0)
					norms = np.linalg.norm(dir_world, axis=1, keepdims=True)
					valid_dir_mask = (norms.squeeze() > 1e-6)
					if not np.any(valid_dir_mask):
						raise ValueError("All ray directions are zero-length after sanitization")
					dir_world = dir_world[valid_dir_mask]
					norms = norms[valid_dir_mask]
					dir_world = dir_world / (norms + 1e-9)

					# Convert numpy to torch tensors
					dirs_torch = torch.from_numpy(dir_world.astype(np.float32)).to(self.vdb_mapper.device)
					origin_torch = torch.from_numpy(origin_world.astype(np.float32)).to(self.vdb_mapper.device)

					# Convert cartesian to spherical coordinates (same as RayFronts)
					r, theta, phi = g3d.cartesian_to_spherical(
						dirs_torch[:, 0], dirs_torch[:, 1], dirs_torch[:, 2])

					# Create ray_orig_angle format: [x, y, z, theta_deg, phi_deg]
					ray_orig_angle = torch.cat([
						origin_torch.unsqueeze(0).expand(dir_world.shape[0], -1),
						torch.rad2deg(theta).unsqueeze(-1),
						torch.rad2deg(phi).unsqueeze(-1)
					], dim=-1)

					# Guard: ensure no NaN/Inf slipped into the angle tensor
					if not torch.isfinite(ray_orig_angle).all():
						ray_orig_angle = torch.nan_to_num(ray_orig_angle, nan=0.0, posinf=0.0, neginf=0.0)

					# Weights: shape Nx1
					weights_only = torch.ones(ray_orig_angle.shape[0], 1,
						device=self.vdb_mapper.device, dtype=torch.float32)

					if self.pose_rays_orig_angles is None:
						self.pose_rays_orig_angles, self.pose_rays_feats_cnt = g3d.bin_rays(
							ray_orig_angle,
							vox_size=float(self.voxel_resolution),
							bin_size=self.vdb_mapper.angle_bin_size,
							feat=weights_only,
							aggregation="weighted_mean"
						)
					else:
						# Guard: check accumulated bin tensor is still valid before accumulating
						if (not torch.isfinite(self.pose_rays_orig_angles).all()
								or self.pose_rays_orig_angles.shape[-1] != 5):
							# Reset stale / corrupted state
							self.pose_rays_orig_angles = None
							self.pose_rays_feats_cnt = None
							self.pose_rays_orig_angles, self.pose_rays_feats_cnt = g3d.bin_rays(
								ray_orig_angle,
								vox_size=float(self.voxel_resolution),
								bin_size=self.vdb_mapper.angle_bin_size,
								feat=weights_only,
								aggregation="weighted_mean"
							)
						else:
							self.pose_rays_orig_angles, self.pose_rays_feats_cnt = g3d.add_weighted_binned_rays(
								self.pose_rays_orig_angles,
								self.pose_rays_feats_cnt,
								ray_orig_angle,
								weights_only,
								vox_size=float(self.voxel_resolution),
								bin_size=self.vdb_mapper.angle_bin_size
							)

					# Convert accumulated bins back to numpy for visualization
					binned_rays_np = self.pose_rays_orig_angles.detach().cpu().numpy()
					if binned_rays_np.ndim != 2 or binned_rays_np.shape[1] < 5:
						raise ValueError(f"Unexpected binned_rays shape: {binned_rays_np.shape}")

					# Extract origins and angles
					origins_np = binned_rays_np[:, :3]
					theta_deg = binned_rays_np[:, 3]
					phi_deg = binned_rays_np[:, 4]

					# Convert spherical back to cartesian directions
					theta_rad = np.deg2rad(theta_deg)
					phi_rad = np.deg2rad(phi_deg)
					sin_phi = np.sin(phi_rad)
					dirs_np = np.stack([
						np.cos(theta_rad) * sin_phi,
						np.sin(theta_rad) * sin_phi,
						np.cos(phi_rad)
					], axis=1)

					# Normalize directions
					dirs_np = dirs_np / (np.linalg.norm(dirs_np, axis=1, keepdims=True) + 1e-9)

					# Create visualization
					length = 0.75
					msg_pose = MarkerArray()
					for i in range(len(binned_rays_np)):
						start = origins_np[i]
						end = start + dirs_np[i] * length
						m = Marker()
						m.header.frame_id = self._pointcloud_frame_id()
						m.header.stamp = now
						m.ns = "mask_rays_pose"
						m.id = i
						m.type = Marker.ARROW
						m.action = Marker.ADD
						m.scale.x = float(self.voxel_resolution) * 0.4
						m.scale.y = float(self.voxel_resolution) * 0.6
						m.scale.z = float(self.voxel_resolution) * 0.6
						m.color.r = 0.0
						m.color.g = 0.7
						m.color.b = 1.0
						m.color.a = 0.95
						m.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
							Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
						msg_pose.markers.append(m)

					if len(msg_pose.markers) > 0:
						self.mask_rays_pub.publish(msg_pose)

				except (RuntimeError, ValueError, Exception) as e:
					# Reset accumulated ray state so the next frame starts fresh
					self.pose_rays_orig_angles = None
					self.pose_rays_feats_cnt = None
					self.get_logger().warn(f"Pose ray binning failed (state reset): {e}")
		except Exception as e:
			self.get_logger().warn(f"Publishing mask rays/frontiers failed: {e}")


def main():
	rclpy.init()
	node = SemanticDepthOctoMapNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		# ── Print mission metrics before any teardown ──────────────────────
		if hasattr(node, 'log_mission_metrics'):
			try:
				node.log_mission_metrics()
			except Exception as e:
				print(f"[Metrics] Failed to log metrics on shutdown: {e}")

		# Cleanup GP computation thread
		if hasattr(node, 'gp_thread_running') and node.gp_thread_running:
			with node.gp_thread_lock:
				node.gp_thread_running = False
			if node.gp_computation_thread and node.gp_computation_thread.is_alive():
				node.gp_computation_thread.join(timeout=2.0)
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main() 