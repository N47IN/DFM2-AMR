#!/usr/bin/env python3
"""
Language as Cost (LaC) – Safe Navigation Baseline
===================================================
Based on: "Language as Cost: Proactive Hazard Mapping using VLM for Robot Navigation"
(Mintaek Oh et al., SNU)

Two-step VLM pipeline (Anthropic Claude):
  Step 1 – Scene hazard identification:
      Ask the VLM what objects / areas in the RGB frame pose potential risk.
  Step 2 – Per-object scoring:
      For each identified hazard ask the VLM for an anxiety score (0-10) and a
      safety avoidance radius in metres.  These drive a Gaussian cost field.

Segmentation: NARadioProcessor similarity maps (same pipeline as test_similarity.py).
Extracts masks via similarity thresholding, projects to 3D, and tracks hazards over time.
Each distinct object is rendered with a unique colour.

Runtime modes
─────────────
Standalone  →  python LaC.py <image_path>
              Queries VLM, segments hazards, shows annotated image.

ROS 2 node  →  python LaC.py          (no image argument)
              Subscribes to RGB / depth / pose / camera-info streams,
              projects hazard masks to 3D, publishes a Gaussian cost cloud
              and coloured hazard cloud.
"""

import sys, os, json, base64, io, time, threading, math, bisect, uuid
from collections import deque
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import cv2
from PIL import Image as PILImage
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity

# ── VLM ──────────────────────────────────────────────────────────────────────
from anthropic import Anthropic

# ── Detection / segmentation ─────────────────────────────────────────────────
import torch
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from resilience.naradio_processor import NARadioProcessor

# ── ROS 2 (optional) ─────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, CameraInfo, PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2
    from nav_msgs.msg import Odometry, OccupancyGrid
    from std_msgs.msg import Header
    from visualization_msgs.msg import MarkerArray, Marker
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from cv_bridge import CvBridge
    import open3d as o3d
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    o3d = None

# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_API_KEY  = os.getenv("ANTHROPIC_API_KEY",
    "")
CLAUDE_MODEL    = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

# Run directory name prefix (changeable in code, followed by timestamp and UUID)
LAC_RUN_DIR_PREFIX = os.getenv("LAC_RUN_DIR_PREFIX", "lac_run")

# NARadio similarity threshold for mask extraction
# NOTE: This constant is kept for backward compatibility but is no longer used.
# The actual threshold is now read from segmentation config (default 0.6) via
# create_merged_hotspot_masks_fast(). To use a different threshold, configure
# 'hotspot_threshold' in the segmentation config.
SIMILARITY_THRESHOLD = 0.85

# Visually distinct BGR colours for up to 10 objects
DISTINCT_COLORS_BGR = [
    (0,   0,   255),   # red
    (0,   255, 0),     # green
    (255, 0,   0),     # blue
    (0,   255, 255),   # yellow
    (255, 0,   255),   # magenta
    (255, 255, 0),     # cyan
    (0,   128, 255),   # orange
    (128, 0,   255),   # violet
    (0,   255, 128),   # spring-green
    (255, 128, 0),     # azure
]



# ─────────────────────────────────────────────────────────────────────────────
# VLM – Two-step Anthropic Claude query
# ─────────────────────────────────────────────────────────────────────────────

class LaCVLM:
    """Two-step VLM querier using Anthropic Claude.

    Step 1 – identify hazardous objects from the image.
    Step 2 – assign anxiety score (0-10) and avoidance radius (m) per object.

    Returns list of dicts:
        [{"object": str, "anxiety": float, "radius_m": float}, ...]
    """

    def __init__(self, api_key: str = CLAUDE_API_KEY, model: str = CLAUDE_MODEL):
        self.client = Anthropic(api_key=api_key)
        self.model  = model

    @staticmethod
    def _encode_image(bgr_img: np.ndarray, max_dim: int = 768) -> str:
        """Encode a BGR numpy array to base64 PNG."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        pil.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call(self, image_b64: str, prompt: str, max_tokens: int = 512) -> str:
        """Single Claude API call with an image + text prompt."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=(
                "You are a safety-aware robot navigation expert. "
                "You identify environmental hazards and assess risk for mobile robots."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": image_b64}},
                    {"type": "text", "text": prompt},
                ]
            }]
        )
        return response.content[0].text.strip()

    # ── Step 1 ────────────────────────────────────────────────────────────────

    _STEP1_PROMPT = """\
You are evaluating an RGB image captured by a drone robot camera flying through the air.

Identify all objects or areas in this scene that could pose a potential hazard
or risk to a drone robot navigating through this environment.Assume it already has basic collision avoidance.
Examples: doors that may open suddenly, people, staircases, wet floors,
sharp corners, fragile objects, narrow gaps, moving machinery, etc.

Return ONLY a JSON array of short descriptive object names.
Rules:
- Maximum 3 items.
- Use concise descriptions (e.g. "door", "person", "staircase", "wet floor").
- No markdown, no backticks, no extra explanation.
- If no hazards are visible return: []

Example: ["door", "person on left", "staircase"]"""

    def step1_identify_hazards(self, image_b64: str) -> List[str]:
        """Return a list of hazardous object names identified in the image."""
        raw = self._call(image_b64, self._STEP1_PROMPT)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            hazards = json.loads(raw)
            if isinstance(hazards, list):
                return [str(h) for h in hazards if h]
        except json.JSONDecodeError:
            pass
        # Fallback: try to find array substring
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            hazards = json.loads(raw[start:end])
            if isinstance(hazards, list):
                return [str(h) for h in hazards if h]
        except (ValueError, json.JSONDecodeError):
            pass
        return []

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _step2_prompt(self, hazards: List[str]) -> str:
        items = "\n".join(f"- {h}" for h in hazards)
        return f"""\
For each of the following hazardous objects / areas identified in the image,
assign an anxiety score and a safety avoidance radius:

{items}

Definitions:
- anxiety_score: integer 0-10 (0 = no hazard, 10 = extreme danger).
  Use a Weber-Fechner scale: 1=very minor, 5=moderate, 8=serious, 10=critical.
- radius_m: float, the minimum safe distance (metres) the robot should maintain.
  Typical values: 0.5 (small static object) to 3.0 (door/person/machinery).

Return ONLY a JSON array with one entry per hazard, in the same order.
Rules:
- No markdown, no backticks, no explanation.
- Each entry must have exactly: "object" (string), "anxiety" (int 0-10), "radius_m" (float).

Example:
[{{"object": "door", "anxiety": 7, "radius_m": 2.0}},
 {{"object": "person", "anxiety": 8, "radius_m": 2.5}}]"""

    def step2_score_hazards(
        self, image_b64: str, hazards: List[str]
    ) -> List[Dict]:
        """Return scored hazard list: [{"object", "anxiety", "radius_m"}, ...]"""
        if not hazards:
            return []
        raw = self._call(image_b64, self._step2_prompt(hazards))
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                out = []
                for item in result:
                    if isinstance(item, dict) and "object" in item:
                        out.append({
                            "object":   str(item["object"]),
                            "anxiety":  float(item.get("anxiety", 5)),
                            "radius_m": float(item.get("radius_m", 1.0)),
                        })
                return out
        except json.JSONDecodeError:
            pass
        # Fallback
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            result = json.loads(raw[start:end])
            if isinstance(result, list):
                out = []
                for item in result:
                    if isinstance(item, dict) and "object" in item:
                        out.append({
                            "object":   str(item["object"]),
                            "anxiety":  float(item.get("anxiety", 5)),
                            "radius_m": float(item.get("radius_m", 1.0)),
                        })
                return out
        except (ValueError, json.JSONDecodeError):
            pass
        return []

    # ── Combined public API ───────────────────────────────────────────────────

    def query(self, bgr_img: np.ndarray) -> List[Dict]:
        """Full two-step query.  Returns list of scored hazards."""
        image_b64 = self._encode_image(bgr_img)

        print("[LaC VLM] Step 1 – identifying hazards …")
        hazards = self.step1_identify_hazards(image_b64)
        print(f"[LaC VLM] Step 1 result: {hazards}")

        if not hazards:
            print("[LaC VLM] No hazards identified – scene appears safe.")
            return []

        print("[LaC VLM] Step 2 – scoring hazards …")
        scored = self.step2_score_hazards(image_b64, hazards)
        print(f"[LaC VLM] Step 2 result: {scored}")
        return scored


# ─────────────────────────────────────────────────────────────────────────────
# Segmentor – NARadioProcessor pipeline (same as test_similarity.py)
# ─────────────────────────────────────────────────────────────────────────────

class LaCSegmentor:
    """NARadioProcessor-based segmentor – extracts masks via similarity maps."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        print(f"[LaC Seg] Initializing NARadioProcessor...")
        self.processor = NARadioProcessor(
            radio_model_version='radio_v2.5-b',
            radio_lang_model='siglip',
            radio_input_resolution=512,
            enable_visualization=False,  # Skip visualization for speed
            enable_combined_segmentation=True,
            segmentation_config_path=None,
        )

        if not self.processor.is_ready():
            raise RuntimeError("NARadioProcessor failed to initialize")
        if not self.processor.is_segmentation_ready():
            raise RuntimeError("NARadioProcessor segmentation not ready")

        print(f"[LaC Seg] NARadioProcessor ready.")

    @torch.no_grad()
    def segment(
        self,
        bgr_img: np.ndarray,
        hazards: List[Dict],
    ) -> List[Dict]:
        """OPTIMIZED: Segment hazard objects using batched NARadioProcessor similarity maps.

        Args:
            bgr_img: OpenCV BGR image.
            hazards: List of dicts from LaCVLM.query() –
                     [{"object", "anxiety", "radius_m"}, ...]

        Returns:
            Augmented list; each entry gains:
                "mask"  – bool ndarray (H, W)  (or None if not detected)
                "box"   – float32 ndarray (4,) xyxy (or None, computed from mask)
                "color" – (B, G, R) tuple for visualisation
        """
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_img.shape[:2]

        # Step 1: Extract object names from hazards
        obj_names = [h["object"] for h in hazards]
        if not obj_names:
            return [dict(h, **{"mask": None, "box": None, "color": DISTINCT_COLORS_BGR[i % len(DISTINCT_COLORS_BGR)]})
                    for i, h in enumerate(hazards)]

        # Step 2: Extract features once (OPTIMIZED: reuse_features=True for better performance)
        feat_map_np, _ = self.processor.process_features_optimized(
            rgb_img,
            need_visualization=False,
            reuse_features=True,  # OPTIMIZATION: Reuse features for better performance
            return_tensor=False,
        )

        if feat_map_np is None:
            print("[LaC Seg] Error: Failed to extract features")
            return [dict(h, **{"mask": None, "box": None, "color": DISTINCT_COLORS_BGR[i % len(DISTINCT_COLORS_BGR)]})
                    for i, h in enumerate(hazards)]

        # Step 3: OPTIMIZATION: Batch process all objects at once using fast method
        # This computes all similarity maps in a single optimized pass instead of per-object loops
        vlm_hotspots = self.processor.create_merged_hotspot_masks_fast(
            rgb_img, obj_names, feat_map_np=feat_map_np
        )

        # Step 4: Convert batched results to individual hazard entries
        result = []
        for idx, hazard in enumerate(hazards):
            color = DISTINCT_COLORS_BGR[idx % len(DISTINCT_COLORS_BGR)]
            entry = dict(hazard)
            entry["color"] = color

            obj_name = hazard["object"]

            # Get mask from batched results (convert uint8 0/255 to bool)
            if vlm_hotspots and obj_name in vlm_hotspots:
                mask_u8 = vlm_hotspots[obj_name]
                mask = (mask_u8 > 0).astype(bool)  # Convert to bool mask
                
                if np.any(mask):
                    entry["mask"] = mask
                    # Compute bounding box from mask
                    v_coords, u_coords = np.where(mask)
                    if len(u_coords) > 0:
                        x1, y1 = float(u_coords.min()), float(v_coords.min())
                        x2, y2 = float(u_coords.max()), float(v_coords.max())
                        entry["box"] = np.array([x1, y1, x2, y2], dtype=np.float32)
                    else:
                        entry["box"] = None
                else:
                    entry["mask"] = None
                    entry["box"] = None
            else:
                entry["mask"] = None
                entry["box"] = None

            result.append(entry)

        # Final memory cleanup after all processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helper (standalone mode)
# ─────────────────────────────────────────────────────────────────────────────

def visualise(bgr_img: np.ndarray, segmented: List[Dict]) -> np.ndarray:
    """Overlay coloured masks and legend onto the image."""
    vis = bgr_img.copy()
    overlay = bgr_img.copy()

    for entry in segmented:
        mask  = entry.get("mask")
        box   = entry.get("box")
        color = entry["color"]
        label = entry["object"]
        anxiety  = entry["anxiety"]
        radius   = entry["radius_m"]

        if mask is not None and np.any(mask):
            overlay[mask] = color

        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis,
                        f"{label} a={anxiety:.0f} r={radius:.1f}m",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                        cv2.LINE_AA)

    # Blend mask overlay
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # Legend in top-right corner
    legend_x = vis.shape[1] - 260
    legend_y = 10
    for entry in segmented:
        color = entry["color"]
        text  = f"{entry['object'][:28]} (a={entry['anxiety']:.0f})"
        cv2.rectangle(vis,
                      (legend_x, legend_y),
                      (legend_x + 18, legend_y + 14),
                      color, -1)
        cv2.putText(vis, text,
                    (legend_x + 22, legend_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 20

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Utility – depth projection  (reused in ROS2 node)
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """quaternion [x,y,z,w] → 3×3 rotation matrix."""
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    s  = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1.0-(yy+zz), xy-wz,      xz+wy     ],
        [xy+wz,       1.0-(xx+zz), yz-wx     ],
        [xz-wy,       yz+wx,      1.0-(xx+yy)],
    ], dtype=np.float32)


def _rpy_deg_to_rot(rpy_deg):
    """Convert roll-pitch-yaw (degrees) to rotation matrix."""
    try:
        import math
        roll, pitch, yaw = [math.radians(float(x)) for x in rpy_deg]
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return Rz @ Ry @ Rx
    except Exception:
        return np.eye(3, dtype=np.float32)


def mask_to_world_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    position: np.ndarray,
    rotation: np.ndarray,
    max_range: float = 5.0,  # OPTIMIZED: Reduced from 10.0m to 2.5m to reduce processing load
    min_range: float = 0.1,
) -> np.ndarray:
    """Project mask pixels through depth to 3-D world points.

    Args:
        mask:       bool (H, W)
        depth_m:    float32 (H, W) – metres
        intrinsics: (fx, fy, cx, cy)
        position:   world position of camera  (3,)
        rotation:   world rotation matrix     (3×3)
        max_range / min_range: depth clipping (default max_range=2.5m for performance)

    Returns:
        (N, 3) world-frame XYZ points, may be empty.
        
    OPTIMIZATION: Only processes pixels with depth <= 2.5m to reduce computational load.
    """
    fx, fy, cx, cy = intrinsics
    v_coords, u_coords = np.where(mask)
    if len(u_coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    h, w = mask.shape
    if depth_m.shape != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    z = depth_m[v_coords, u_coords].astype(np.float32)
    # OPTIMIZATION: Filter out pixels beyond 2.5m early to reduce processing
    valid = np.isfinite(z) & (z > min_range) & (z <= max_range)
    u, v, z = u_coords[valid].astype(np.float32), v_coords[valid].astype(np.float32), z[valid]

    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam   = np.stack([x_cam, y_cam, z], axis=1)            # (N, 3)
    pts_world = pts_cam @ rotation.T + position                 # (N, 3)
    return pts_world


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────

if ROS_AVAILABLE:

    class LaCNode(Node):
        """ROS 2 node implementing the full LaC pipeline.

        Subscribes to:
            RGB image   – triggers async VLM + segmentation pipeline
            Depth image – buffered for timestamp-matched projection
            Odometry    – buffered for timestamp-matched pose
            CameraInfo  – reads intrinsics once

        Publishes:
            /lac/hazard_cloud   (PointCloud2)  – coloured 3-D hazard points
            /lac/costmap        (OccupancyGrid) – 2-D Gaussian cost projection
        """

        # ── init ──────────────────────────────────────────────────────────────
        def __init__(self):
            super().__init__("lac_node")

            # Read topic names from env or use mapping_config defaults
            rgb_topic    = os.getenv("LAC_RGB_TOPIC",
                "/robot_1/sensors/front_stereo/left/image_rect")
            depth_topic  = os.getenv("LAC_DEPTH_TOPIC",
                "/robot_1/sensors/front_stereo/left/depth_ground_truth")
            info_topic   = os.getenv("LAC_INFO_TOPIC",
                "/robot_1/sensors/front_stereo/left/camera_info")
            pose_topic   = os.getenv("LAC_POSE_TOPIC",
                "/robot_1/odometry_conversion/odometry")

            self.bridge = CvBridge()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Intrinsics (set from CameraInfo)
            self.intrinsics: Optional[Tuple] = None   # (fx, fy, cx, cy)
            
            # Coordinate frame transformation parameters (like frontier_mapping_node)
            self.pose_is_base_link = bool(os.getenv("LAC_POSE_IS_BASE_LINK", "True"))
            self.apply_optical_frame_rotation = bool(os.getenv("LAC_APPLY_OPTICAL_ROTATION", "True"))
            self.cam_to_base_rpy_deg = [
                float(os.getenv("LAC_CAM_TO_BASE_ROLL", "0.0")),
                float(os.getenv("LAC_CAM_TO_BASE_PITCH", "0.0")),
                float(os.getenv("LAC_CAM_TO_BASE_YAW", "0.0"))
            ]
            self.cam_to_base_xyz = [
                float(os.getenv("LAC_CAM_TO_BASE_X", "0.0")),
                float(os.getenv("LAC_CAM_TO_BASE_Y", "0.0")),
                float(os.getenv("LAC_CAM_TO_BASE_Z", "0.0"))
            ]
            
            # Precompute coordinate frame transformation matrices (like frontier_mapping_node)
            self.R_opt_to_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
            self.R_cam_to_base_extra = _rpy_deg_to_rot(self.cam_to_base_rpy_deg)
            self.t_cam_to_base_extra = np.array(self.cam_to_base_xyz, dtype=np.float32)

            # Timestamp-indexed buffers (deque for speed, reuse existing pattern)
            # Following frontier_mapping_node pattern
            self._buf_depth_ts:   deque = deque(maxlen=30)
            self._buf_depth_data: deque = deque(maxlen=30)
            self._buf_pose_ts:    deque = deque(maxlen=60)
            self._buf_pose_data:  deque = deque(maxlen=60)
            self._buf_rgb_ts:     deque = deque(maxlen=30)
            self._buf_rgb_data:   deque = deque(maxlen=30)
            self._buf_lock = threading.Lock()

            # VLM / segmentor (lazy init in background to not block __init__)
            self.vlm:  Optional[LaCVLM]      = None
            self.seg:  Optional[LaCSegmentor] = None
            self._models_ready = False
            threading.Thread(target=self._load_models, daemon=True).start()

            # 3D hazard tracking (thread-safe)
            # Maps object_name -> {points: np.ndarray (N,3), anxiety, radius_m, color, last_update}
            self._tracked_hazards: Dict[str, Dict] = {}
            self._hazard_lock = threading.Lock()
            # GPU MEMORY OPTIMIZATION: Reduce max points per hazard for small GPUs
            self._max_points_per_hazard = int(os.getenv("LAC_MAX_POINTS_PER_HAZARD", "2000"))  # Reduced from 5000
            self._hazard_max_age = float(os.getenv("LAC_HAZARD_MAX_AGE", "30.0"))  # seconds
            # GPU MEMORY OPTIMIZATION: Limit total tracked hazards
            self._max_tracked_hazards = int(os.getenv("LAC_MAX_TRACKED_HAZARDS", "10"))  # Max 10 hazards
            
            # Track which objects have been encoded and added to processor (like main.py)
            self._encoded_objects: set = set()  # Set of object names that have been encoded
            self._encoded_lock = threading.Lock()

            # RGB processing lock (ensure only one frame is processed at a time)
            self._rgb_processing_lock = threading.Lock()
            
            # Thread pool for async 3D projection (prevents blocking RGB callback)
            from concurrent.futures import ThreadPoolExecutor
            self._projection_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lac_projection")
            
            # Periodic VLM query (every 8 seconds when running as node)
            self._vlm_periodic_interval = 20.0  # seconds
            self._last_periodic_vlm_time = 0.0
            self._latest_rgb_image = None
            self._latest_rgb_timestamp = None
            self._latest_rgb_shape = None
            self._rgb_image_lock = threading.Lock() # For _latest_rgb_image, _latest_rgb_timestamp, _latest_rgb_shape
            
            # Robot-centric grid parameters (similar to frontier_mapping_node)
            self.robot_grid_size_xy = float(os.getenv("LAC_GRID_SIZE_XY", "10.0"))  # 10m x 10m
            self.robot_grid_size_z = float(os.getenv("LAC_GRID_SIZE_Z", "4.0"))     # 4m
            self.robot_grid_resolution = float(os.getenv("LAC_GRID_RESOLUTION", "0.2"))  # 0.2m
            self.robot_position = None  # Current robot position

            # QoS
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )

            # Subscribers
            self.create_subscription(Image,      rgb_topic,   self._rgb_cb,   qos)
            self.create_subscription(Image,      depth_topic, self._depth_cb, qos)
            self.create_subscription(CameraInfo, info_topic,  self._info_cb,  qos)
            self.create_subscription(Odometry,   pose_topic,  self._pose_cb,  qos)

            # Publishers
            self._pub_cloud = self.create_publisher(PointCloud2,   "/lac/hazard_cloud", 5)
            self._pub_cmap  = self.create_publisher(OccupancyGrid, "/lac/costmap",      5)
            # Semantic costmap (same format as frontier_mapping_node - disturbance values only)
            self._pub_semantic_costmap = self.create_publisher(PointCloud2, "/semantic_costmap", 10)
            # Separate visualization topic for RGB visualization of Gaussian cost field
            self._pub_costmap_visualization = self.create_publisher(PointCloud2, "/semantic_costmap_viz", 10)
            # Spherical markers around hazard points with VLM-provided radius
            self._pub_sphere_markers = self.create_publisher(MarkerArray, "/lac/hazard_spheres", 10)
            
            # Initialize semantic bridge for hotspot publishing (like main.py)
            self._init_semantic_bridge()
            
            # Initialize buffer directory for VLM answer storage (like main.py)
            self._init_buffer_directory()

            # Periodic publish timer (1 Hz)
            self.create_timer(1.0, self._publish_cb)
            
            # Periodic VLM query timer (every 8 seconds)
            self.create_timer(self._vlm_periodic_interval, self._periodic_vlm_cb)
            
            # Costmap generation timer (rate-limited like frontier_mapping_node)
            # OPTIMIZED: Only run when hazards exist, similar to frontier_mapping_node GP updates
            self._costmap_update_interval = 0.75  # seconds (same as GP update interval)
            self._last_costmap_update_time = 0.0
            self.create_timer(0.75, self._generate_and_publish_costmap_cb)  # Match update interval
            
            # CRITICAL: Periodic check to ensure objects never go missing from processor (every 5 seconds)
            self.create_timer(5.0, self._verify_objects_persist_cb)

            self.get_logger().info(
                f"LaC node started | RGB={rgb_topic} | "
                f"VLM interval={self._vlm_periodic_interval}s | device={self.device}"
            )

        # ── semantic bridge initialization ──────────────────────────────────────
        
        def _init_semantic_bridge(self):
            """Initialize semantic hotspot bridge for communication with octomap (like main.py)."""
            try:
                # Load config from main_config if available
                main_config = {}
                main_config_path = os.getenv("LAC_MAIN_CONFIG_PATH", "")
                if main_config_path and os.path.exists(main_config_path):
                    import yaml
                    with open(main_config_path, 'r') as f:
                        main_config = yaml.safe_load(f) or {}
                
                from resilience.semantic_info_bridge import SemanticHotspotPublisher
                self.semantic_bridge = SemanticHotspotPublisher(self, main_config)
                self.get_logger().info("Semantic bridge initialized for hotspot publishing")
            except Exception as e:
                self.get_logger().warn(f"Error initializing semantic bridge: {e}")
                self.semantic_bridge = None
        
        def _init_buffer_directory(self):
            """Initialize buffer directory for VLM answer storage (like main.py)."""
            try:
                run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                unique_id = str(uuid.uuid4())[:8]
                self.buffer_save_dir = '/root/AirStack/robot/ros_ws/src/autonomy/4_global/a_world_models/resilience/buffers'
                os.makedirs(self.buffer_save_dir, exist_ok=True)
                
                # Use configurable run directory name prefix (followed by timestamp and UUID)
                self.current_run_dir = os.path.join(self.buffer_save_dir, f"{LAC_RUN_DIR_PREFIX}_{run_timestamp}_{unique_id}")
                os.makedirs(self.current_run_dir, exist_ok=True)
                
                # Create vlm_answers subdirectory for storing VLM query results
                self.vlm_answers_dir = os.path.join(self.current_run_dir, "vlm_answers")
                os.makedirs(self.vlm_answers_dir, exist_ok=True)
                
                self.get_logger().info(f"Buffer directory initialized: {self.current_run_dir}")
                
            except Exception as e:
                self.get_logger().error(f"Error initializing buffer directory: {e}")
                self.buffer_save_dir = None
                self.current_run_dir = None
                self.vlm_answers_dir = None

        # ── model loading (background) ────────────────────────────────────────

        def _load_models(self):
            self.vlm = LaCVLM()
            self.seg = LaCSegmentor(device=self.device)
            self._models_ready = True
            self.get_logger().info("LaC models loaded – pipeline active.")
        

        # ── callbacks ────────────────────────────────────────────────────────

        def _info_cb(self, msg: CameraInfo):
            if self.intrinsics is None:
                self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
                self.get_logger().info(
                    f"Camera intrinsics set: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}")

        def _depth_cb(self, msg: Image):
            """Store depth message in buffer with timestamp."""
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self._buf_lock:
                self._buf_depth_ts.append(ts)
                self._buf_depth_data.append(msg)

        def _pose_cb(self, msg: Odometry):
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            with self._buf_lock:
                self._buf_pose_ts.append(ts)
                self._buf_pose_data.append(msg)
            
            # Update robot position for robot-centric grid
            self.robot_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ], dtype=np.float32)

        def _rgb_cb(self, msg: Image):
            """Process RGB frame: similarity extraction, hotspot mask publishing, and 3D projection."""
            if not self._models_ready:
                return
            
            # Acquire lock to ensure single-threaded processing of RGB frames
            if not self._rgb_processing_lock.acquire(blocking=False):
                # self.get_logger().debug("Skipping RGB frame, previous processing still running.")
                return
            
            try:
                rgb_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                try:
                    bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except Exception:
                    self.get_logger().error("Failed to convert RGB image message to CV2 format.")
                    return
                
                # Store latest RGB image for periodic VLM queries
                with self._rgb_image_lock:
                    self._latest_rgb_image = bgr.copy()
                    self._latest_rgb_timestamp = rgb_ts
                    self._latest_rgb_shape = bgr.shape[:2]  # (height, width)
                
                # Process the RGB frame (similarity, projection, hotspot publishing)
                threading.Thread(
                    target=self._process_rgb_frame,
                    args=(bgr, rgb_ts),
                    daemon=True,
                ).start()
            finally:
                self._rgb_processing_lock.release()

        # ── Core per-frame processing ─────────────────────────────────────────

        def _process_rgb_frame(self, bgr: np.ndarray, rgb_ts: float):
            """
            OPTIMIZED: Compute similarity maps for all VLM-tracked objects using batched processing.

            Flow:
              1. Snapshot current object list from _tracked_hazards.
              2. Extract feature map once via process_features_optimized (with reuse_features=True).
              3. Batch process all objects using create_merged_hotspot_masks_fast().
              4. Publish /semantic_hotspot_mask via semantic_bridge.
              5. If depth + pose available, project masks to 3D, accumulate in _tracked_hazards.
            """
            try:
                # 1. Snapshot current object list
                with self._hazard_lock:
                    obj_names = list(self._tracked_hazards.keys())

                if not obj_names:
                    return   # No VLM objects yet

                if (not self.seg or
                        not self.seg.processor.is_ready() or
                        not self.seg.processor.is_segmentation_ready()):
                    return

                # 2. Convert to RGB, extract feature map once (OPTIMIZED: reuse_features=True)
                rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w    = rgb_img.shape[:2]

                # GPU MEMORY OPTIMIZATION: Process features and immediately move to CPU
                with torch.no_grad():
                    feat_map_np, _ = self.seg.processor.process_features_optimized(
                        rgb_img,
                        need_visualization=False,
                        reuse_features=True,  # OPTIMIZATION: Reuse features for better performance
                        return_tensor=False,  # CRITICAL: Return numpy, not tensor (saves GPU memory)
                    )
                    
                    # GPU MEMORY OPTIMIZATION: Clear cache after feature extraction
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if feat_map_np is None:
                    return

                # 3. OPTIMIZATION: Batch process all objects at once using fast method
                # This computes all similarity maps in a single optimized pass
                # CRITICAL: All similarity computation must be in no_grad context
                # GPU MEMORY OPTIMIZATION: Limit batch size for small GPUs
                max_objects_per_batch = int(os.getenv("LAC_MAX_OBJECTS_PER_BATCH", "8"))  # Process max 8 objects at once
                
                # DEBUG: Check which objects are available in processor
                # CRITICAL: Ensure objects persist - if missing, add them immediately
                # CRITICAL: Also check for duplicates in processor (case-insensitive)
                all_processor_objects = self.seg.processor.get_all_objects()
                all_processor_objects_normalized = {obj.lower().strip() for obj in all_processor_objects}
                
                # Check for missing objects (case-insensitive comparison)
                missing_objects = []
                for obj_name in obj_names:
                    obj_normalized = obj_name.lower().strip()
                    # Check both exact match and normalized match
                    if obj_name not in all_processor_objects and obj_normalized not in all_processor_objects_normalized:
                        missing_objects.append(obj_name)
                
                if missing_objects:
                    self.get_logger().warn(
                        f"[LaC] Objects not in processor: {missing_objects}. "
                        f"Available: {all_processor_objects}. Adding missing objects..."
                    )
                    # Add missing objects immediately to ensure they persist
                    # GPU MEMORY OPTIMIZATION: Add one at a time with memory cleanup
                    for obj_name in missing_objects:
                        if obj_name not in self._encoded_objects:
                            with torch.no_grad():
                                success = self.seg.processor.add_vlm_object(obj_name)
                            if success:
                                self._encoded_objects.add(obj_name)
                                self.get_logger().info(f"[LaC] ✓ Re-added missing object '{obj_name}' to processor")
                            else:
                                self.get_logger().warn(f"[LaC] Failed to re-add '{obj_name}' to processor")
                            # GPU MEMORY OPTIMIZATION: Clear cache after each addition
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                
                # CRITICAL: Verify no duplicates in processor after adding
                all_processor_objects_after = self.seg.processor.get_all_objects()
                if len(all_processor_objects_after) != len(set(obj.lower().strip() for obj in all_processor_objects_after)):
                    self.get_logger().error(
                        f"[LaC] CRITICAL: Duplicates detected in processor! "
                        f"Total: {len(all_processor_objects_after)}, Unique: {len(set(all_processor_objects_after))}"
                    )
                
                # WORKAROUND: Temporarily set threshold in config to match LaC's SIMILARITY_THRESHOLD
                # The fast method uses config threshold (default 0.6), but LaC originally used 0.85
                # We'll use the config threshold but log it for debugging
                original_config = None
                try:
                    if hasattr(self.seg.processor, 'segmentation_config') and self.seg.processor.segmentation_config:
                        original_config = self.seg.processor.segmentation_config.get('segmentation', {}).copy()
                        # Set threshold to LaC's threshold if not already set
                        if 'hotspot_threshold' not in self.seg.processor.segmentation_config.get('segmentation', {}):
                            if 'segmentation' not in self.seg.processor.segmentation_config:
                                self.seg.processor.segmentation_config['segmentation'] = {}
                            self.seg.processor.segmentation_config['segmentation']['hotspot_threshold'] = SIMILARITY_THRESHOLD
                            self.get_logger().debug(f"[LaC] Set hotspot_threshold to {SIMILARITY_THRESHOLD}")
                except Exception as e:
                    self.get_logger().warn(f"[LaC] Could not adjust threshold config: {e}")
                
                # GPU MEMORY OPTIMIZATION: Process objects in smaller batches for small GPUs
                # CRITICAL: Also deduplicate obj_names before processing to avoid redundant computation
                obj_names_deduped = []
                seen_names = set()
                for obj_name in obj_names:
                    obj_normalized = obj_name.lower().strip()
                    if obj_normalized not in seen_names:
                        seen_names.add(obj_normalized)
                        obj_names_deduped.append(obj_name)
                
                if len(obj_names_deduped) < len(obj_names):
                    self.get_logger().warn(
                        f"[LaC] Deduplicated object names: {len(obj_names)} -> {len(obj_names_deduped)} "
                        f"before similarity computation"
                    )
                
                vlm_hotspots = {}
                if len(obj_names_deduped) <= max_objects_per_batch:
                    # Small batch - process all at once
                    with torch.no_grad():
                        batch_hotspots = self.seg.processor.create_merged_hotspot_masks_fast(
                            rgb_img, obj_names_deduped, feat_map_np=feat_map_np
                        )
                        if batch_hotspots:
                            vlm_hotspots.update(batch_hotspots)
                else:
                    # Large batch - process in chunks to avoid OOM
                    for i in range(0, len(obj_names_deduped), max_objects_per_batch):
                        batch = obj_names_deduped[i:i+max_objects_per_batch]
                        with torch.no_grad():
                            batch_hotspots = self.seg.processor.create_merged_hotspot_masks_fast(
                                rgb_img, batch, feat_map_np=feat_map_np
                            )
                            if batch_hotspots:
                                vlm_hotspots.update(batch_hotspots)
                        # GPU MEMORY OPTIMIZATION: Aggressive cleanup between batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()  # Ensure all operations complete
                            import gc
                            gc.collect()

                # DEBUG: Log what was returned
                if vlm_hotspots is None:
                    self.get_logger().debug(
                        f"[LaC] create_merged_hotspot_masks_fast returned None for {len(obj_names)} objects. "
                        f"Check: 1) Objects in processor? 2) Enhanced embeddings available? 3) Threshold too high?"
                    )
                elif len(vlm_hotspots) == 0:
                    self.get_logger().debug(
                        f"[LaC] create_merged_hotspot_masks_fast returned empty dict for {len(obj_names)} objects. "
                        f"All objects filtered out (threshold/min_area too strict?)"
                    )
                else:
                    self.get_logger().debug(
                        f"[LaC] ✓ create_merged_hotspot_masks_fast returned {len(vlm_hotspots)} hotspots: {list(vlm_hotspots.keys())}"
                    )

                # Convert from uint8 masks (0/255) to uint8 masks (0/1) for consistency
                # NOTE: Keep as uint8 (0/1) not bool, as semantic_bridge expects uint8
                # FIX: Proper None check before accessing items
                if vlm_hotspots and len(vlm_hotspots) > 0:
                    # Convert 0/255 to 0/1 (keep as uint8, not bool)
                    vlm_hotspots = {k: ((v > 0).astype(np.uint8)) for k, v in vlm_hotspots.items()}
                else:
                    vlm_hotspots = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 4. Publish /semantic_hotspot_mask
                if vlm_hotspots and len(vlm_hotspots) > 0 and self.semantic_bridge is not None:
                    self.semantic_bridge.publish_merged_hotspots(
                        vlm_hotspots=vlm_hotspots,
                        timestamp=rgb_ts,
                        narration=False,
                        original_image=rgb_img,
                        buffer_id=None,
                    )

                # 5. 3D projection (needs depth + pose)
                depth_msg, pose_msg = self._lookup_depth_pose(rgb_ts)
                if depth_msg is None or pose_msg is None or self.intrinsics is None:
                    return

                # Early return if no hotspots to project
                if not vlm_hotspots or len(vlm_hotspots) == 0:
                    return

                try:
                    depth_m = self.bridge.imgmsg_to_cv2(
                        depth_msg, desired_encoding="32FC1").astype(np.float32)
                except Exception:
                    return

                # Update robot position
                self.robot_position = np.array([
                    pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    pose_msg.pose.pose.position.z,
                ], dtype=np.float32)

                # OPTIMIZATION: Move projection to background thread to avoid blocking RGB callback
                # This significantly improves responsiveness when processing multiple objects
                if vlm_hotspots and len(vlm_hotspots) > 0:
                    # Make copies for thread safety
                    vlm_hotspots_copy = {k: v.copy() for k, v in vlm_hotspots.items()}
                    depth_m_copy = depth_m.copy()
                    pose_msg_copy = pose_msg  # Odometry messages are immutable
                    
                    # Submit to background thread pool for async processing
                    self._projection_executor.submit(
                        self._project_hotspots_to_3d_async,
                        vlm_hotspots_copy, depth_m_copy, pose_msg_copy, rgb_ts
                    )

            except Exception as e:
                import traceback
                self.get_logger().error(
                    f"[LaC] _process_rgb_frame error: {e}\n{traceback.format_exc()}")

        def _pose_position(self, pose):
            """Extract position from either PoseStamped or Odometry message (like frontier_mapping_node)."""
            if hasattr(pose.pose, 'pose'):  # Odometry message
                return np.array([pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z], dtype=np.float32)
            else:  # PoseStamped message
                return np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], dtype=np.float32)

        def _pose_quat(self, pose):
            """Extract quaternion from either PoseStamped or Odometry message (like frontier_mapping_node)."""
            if hasattr(pose.pose, 'pose'):  # Odometry message
                q = pose.pose.pose.orientation
            else:  # PoseStamped message
                q = pose.pose.orientation
            return np.array([q.x, q.y, q.z, q.w], dtype=np.float32)

        def _mask_to_world_points_sparse(
            self,
            mask: np.ndarray,
            depth_m: np.ndarray,
            intrinsics: Tuple[float, float, float, float],
            pose_msg,
            max_range: float = 5.0,  # OPTIMIZED: Reduced from 10.0m to 2.5m to reduce processing load
            min_range: float = 0.1,
        ) -> np.ndarray:
            """
            Project mask pixels through depth map to 3-D world coordinates.
            Uses same coordinate frame handling as frontier_mapping_node.
            
            OPTIMIZATION: Only processes pixels with depth <= 2.5m to reduce computational load.
            """
            try:
                v_coords, u_coords = np.where(mask > 0)
                if len(u_coords) == 0:
                    return np.zeros((0, 3), dtype=np.float32)
                h, w = mask.shape
                if depth_m.shape != (h, w):
                    depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
                z = depth_m[v_coords, u_coords].astype(np.float32)
                # OPTIMIZATION: Filter out pixels beyond 2.5m early to reduce processing
                valid = np.isfinite(z) & (z > min_range) & (z <= max_range)
                if not np.any(valid):
                    return np.zeros((0, 3), dtype=np.float32)
                u_v = u_coords[valid].astype(np.float32)
                v_v = v_coords[valid].astype(np.float32)
                z_v = z[valid]
                
                # Camera frame projection (same as frontier_mapping_node)
                fx, fy, cx, cy = intrinsics
                x = (u_v - cx) * z_v / fx
                y = (v_v - cy) * z_v / fy
                pts_cam = np.stack([x, y, z_v], axis=1)
                
                # Transform to base if needed (same as frontier_mapping_node)
                if bool(self.pose_is_base_link):
                    pts_cam = pts_cam @ (self.R_opt_to_base.T if bool(self.apply_optical_frame_rotation) else np.eye(3, dtype=np.float32))
                    pts_cam = pts_cam @ self.R_cam_to_base_extra.T + self.t_cam_to_base_extra
                
                # World transform (same as frontier_mapping_node)
                R_world = quat_to_rot(self._pose_quat(pose_msg))
                p_world = self._pose_position(pose_msg)
                pts_world = pts_cam @ R_world.T + p_world
                
                return pts_world
            except Exception as e:
                self.get_logger().warn(f"Sparse projection error: {e}")
                import traceback
                self.get_logger().warn(traceback.format_exc())
                return np.zeros((0, 3), dtype=np.float32)


        # ── buffer lookup (following frontier_mapping_node pattern) ────────────

        def _lookup_depth_pose(self, ts: float, max_dt: float = 1.0):
            """Lookup depth and pose using binary search (same as frontier_mapping_node)."""
            with self._buf_lock:
                depth_msg, _ = self._binary_search_closest(
                    self._buf_depth_ts, self._buf_depth_data, ts, max_dt)
                pose_msg, _ = self._binary_search_closest(
                    self._buf_pose_ts, self._buf_pose_data, ts, max_dt)
            return depth_msg, pose_msg

        def _binary_search_closest(self, ts_deque: deque, data_deque: deque, target_ts: float, max_dt: float):
            """Vectorized search across synchronized deques (same as frontier_mapping_node)."""
            if not ts_deque:
                return None, None
            ts_array = np.array(ts_deque)
            idx = np.searchsorted(ts_array, target_ts)
            candidates = []
            if idx < len(ts_array):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            
            if not candidates:
                return None, None
            
            diffs = np.abs(ts_array[candidates] - target_ts)
            best_relative_idx = np.argmin(diffs)
            best_idx = candidates[best_relative_idx]
            if diffs[best_relative_idx] <= max_dt:
                return data_deque[best_idx], ts_array[best_idx]
            
            return None, None

        # ── periodic publish ──────────────────────────────────────────────────

        def _publish_cb(self):
            with self._hazard_lock:
                # Convert tracked hazards dict to list format for publishing
                hazard_pts = [
                    {
                        "object":   obj_name,
                        "anxiety":  tracked["anxiety"],
                        "radius_m": tracked["radius_m"],
                        "color":    tracked["color"],
                        "points":   tracked["points"],
                    }
                    for obj_name, tracked in self._tracked_hazards.items()
                ]
            if not hazard_pts:
                return
            self._publish_cloud(hazard_pts)
            self._publish_costmap(hazard_pts)
            self._publish_sphere_markers(hazard_pts)

        def _voxelize_pointcloud(self, points: np.ndarray, voxel_size: float, max_points: int = 200) -> np.ndarray:
            """
            High-performance voxelization using Open3D (C++ backend) - same as frontier_mapping_node.
            Reduces point density by averaging points within a spatial grid.
            """
            if points.shape[0] == 0:
                return points
            
            if o3d is None:
                # Fallback to simple numpy-based voxelization if Open3D not available
                voxel_coords = np.floor(points / voxel_size).astype(np.int32)
                unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
                voxelized_points = []
                for i in range(len(unique_voxels)):
                    voxel_mask = inverse_indices == i
                    voxel_points = points[voxel_mask]
                    centroid = np.mean(voxel_points, axis=0)
                    voxelized_points.append(centroid)
                voxelized_points = np.array(voxelized_points)
                if len(voxelized_points) > max_points:
                    indices = np.random.choice(len(voxelized_points), size=max_points, replace=False)
                    voxelized_points = voxelized_points[indices]
                return voxelized_points
            
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                voxelized_points = np.asarray(downsampled_pcd.points)
                num_voxelized = voxelized_points.shape[0]
                if num_voxelized > max_points:
                    step = num_voxelized / max_points
                    indices = np.arange(0, num_voxelized, step, dtype=np.int32)[:max_points]
                    voxelized_points = voxelized_points[indices]
                
                return voxelized_points
            except Exception as e:
                self.get_logger().warn(f"Open3D voxelization failed, using fallback: {e}")
                # Fallback to simple numpy-based voxelization
                voxel_coords = np.floor(points / voxel_size).astype(np.int32)
                unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
                voxelized_points = []
                for i in range(len(unique_voxels)):
                    voxel_mask = inverse_indices == i
                    voxel_points = points[voxel_mask]
                    centroid = np.mean(voxel_points, axis=0)
                    voxelized_points.append(centroid)
                voxelized_points = np.array(voxelized_points)
                if len(voxelized_points) > max_points:
                    indices = np.random.choice(len(voxelized_points), size=max_points, replace=False)
                    voxelized_points = voxelized_points[indices]
                return voxelized_points

        def _project_hotspots_to_3d_async(self, vlm_hotspots: dict, depth_m: np.ndarray, pose_msg, rgb_ts: float):
            """
            OPTIMIZED: Async 3D projection of hotspots to world points.
            Runs in background thread to avoid blocking RGB callback.
            
            Key optimizations:
            1. Combine points first, voxelize once at end (not 3 times!)
            2. Batch process all objects
            3. Only update if there are new points
            4. Ensure objects persist once added
            """
            try:
                if self.intrinsics is None:
                    return
                
                current_time = time.time()
                voxel_size = 0.5  # 10cm default
                
                # Project all masks to points (batch operation)
                # OPTIMIZATION: Use 2.5m max depth to reduce processing load
                max_depth_mapping = 2.5  # Only process pixels within 2.5m
                # GPU MEMORY OPTIMIZATION: Limit max points per object during projection
                max_points_per_projection = int(os.getenv("LAC_MAX_POINTS_PER_PROJECTION", "1000"))  # Limit per object
                new_points_dict = {}
                for obj_name, mask_u8 in vlm_hotspots.items():
                    pts = self._mask_to_world_points_sparse(
                        mask_u8.astype(bool), depth_m, self.intrinsics, pose_msg,
                        max_range=max_depth_mapping  # Explicitly set max depth
                    )
                    if pts is not None and pts.shape[0] > 0:
                        # GPU MEMORY OPTIMIZATION: Limit points immediately after projection
                        if len(pts) > max_points_per_projection:
                            # Randomly sample to reduce memory
                            indices = np.random.choice(len(pts), size=max_points_per_projection, replace=False)
                            pts = pts[indices]
                        new_points_dict[obj_name] = pts
                
                if not new_points_dict:
                    return
                
                # Update tracked hazards (single lock acquisition for all updates)
                with self._hazard_lock:
                    # CRITICAL: Don't prune objects that are actively being tracked!
                    # Only prune if they haven't been updated in a very long time AND aren't in current hotspots
                    active_object_names = set(new_points_dict.keys())
                    current_time_check = time.time()
                    stale = []
                    for k, v in self._tracked_hazards.items():
                        age = current_time_check - v.get("last_update", 0)
                        # Only prune if very old AND not in current frame (2x max_age for safety)
                        if age > self._hazard_max_age * 2 and k not in active_object_names:
                            stale.append(k)
                    for k in stale:
                        del self._tracked_hazards[k]
                    
                    # GPU MEMORY OPTIMIZATION: Limit total number of tracked hazards
                    if len(self._tracked_hazards) >= self._max_tracked_hazards:
                        # Remove oldest hazards if we're at the limit
                        sorted_hazards = sorted(
                            self._tracked_hazards.items(),
                            key=lambda x: x[1].get("last_update", 0)
                        )
                        num_to_remove = len(self._tracked_hazards) - self._max_tracked_hazards + 1
                        for i in range(num_to_remove):
                            del self._tracked_hazards[sorted_hazards[i][0]]
                    
                    # Update each object with new points (OPTIMIZED: combine first, voxelize once)
                    for obj_name, new_pts in new_points_dict.items():
                        if obj_name in self._tracked_hazards:
                            existing = self._tracked_hazards[obj_name]["points"]
                            
                            # OPTIMIZATION: Combine first, then voxelize once (not 3 times!)
                            if len(existing) > 0:
                                combined = np.vstack([existing, new_pts])
                            else:
                                combined = new_pts
                            
                            # GPU MEMORY OPTIMIZATION: Aggressive voxelization to limit memory
                            if len(combined) > 0:
                                # Use smaller max_points to reduce memory footprint
                                combined = self._voxelize_pointcloud(
                                    combined, 
                                    voxel_size=voxel_size, 
                                    max_points=self._max_points_per_hazard
                                )
                            
                            self._tracked_hazards[obj_name]["points"] = combined
                            self._tracked_hazards[obj_name]["last_update"] = current_time
                        else:
                            # New object - initialize with voxelized points
                            if len(new_pts) > 0:
                                voxelized = self._voxelize_pointcloud(
                                    new_pts,
                                    voxel_size=voxel_size,
                                    max_points=self._max_points_per_hazard
                                )
                            else:
                                voxelized = new_pts
                            
                            # Initialize with default values (will be updated by VLM callback)
                            self._tracked_hazards[obj_name] = {
                                "points": voxelized,
                                "anxiety": 0.5,
                                "radius_m": 1.0,
                                "color": [255, 0, 0],  # Default red
                                "last_update": current_time
                            }
                            
            except Exception as e:
                self.get_logger().warn(f"[LaC] Async projection error: {e}")
                import traceback
                self.get_logger().debug(traceback.format_exc())

        def _publish_sphere_markers(self, hazard_pts):
            """
            Publish spherical markers around hazard points with VLM-provided radius.
            Each point gets a sphere marker centered at the point with radius from VLM.
            Topic: /lac/hazard_spheres (MarkerArray)
            """
            try:
                marker_array = MarkerArray()
                marker_id = 0
                
                hdr = Header()
                hdr.stamp = self.get_clock().now().to_msg()
                hdr.frame_id = "map"
                
                for hazard in hazard_pts:
                    obj_name = hazard["object"]
                    points = hazard["points"]
                    radius_m = hazard.get("radius_m", 1.0)  # Default 1.0m if not provided
                    color = hazard.get("color", [255, 0, 0])  # Default red
                    
                    if len(points) == 0:
                        continue
                    
                    # Create a sphere marker for each point
                    for point in points:
                        marker = Marker()
                        marker.header = hdr
                        marker.ns = f"hazard_{obj_name}"
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
                        marker.scale.x = float(radius_m * 2.0)
                        marker.scale.y = float(radius_m * 2.0)
                        marker.scale.z = float(radius_m * 2.0)
                        
                        # Set color (BGR to RGB, normalize to 0-1)
                        marker.color.r = float(color[2]) / 255.0
                        marker.color.g = float(color[1]) / 255.0
                        marker.color.b = float(color[0]) / 255.0
                        marker.color.a = 0.6  # Semi-transparent
                        
                        # Set lifetime (0 = infinite)
                        marker.lifetime.sec = 0
                        
                        marker_array.markers.append(marker)
                
                # if len(marker_array.markers) > 0:
                #     self._pub_sphere_markers.publish(marker_array)
                #     self.get_logger().debug(
                #         f"[LaC] Published {len(marker_array.markers)} sphere markers "
                #         f"for {len(hazard_pts)} hazards"
                #     )
                # else:
                #     # Publish empty marker array to clear previous markers
                #     marker_array.markers = []
                #     self._pub_sphere_markers.publish(marker_array)
                    
            except Exception as e:
                self.get_logger().warn(f"[LaC] Error publishing sphere markers: {e}")
                import traceback
                self.get_logger().debug(traceback.format_exc())

        def _publish_cloud(self, hazard_pts):
            """Publish coloured PointCloud2 of hazard 3-D points (voxelized)."""
            fields = [
                pc2.PointField(name="x",   offset=0,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="y",   offset=4,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="z",   offset=8,  datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32,  count=1),
            ]
            data_rows = []
            # Voxel size for publishing (can be different from storage voxel size)
            # GPU MEMORY OPTIMIZATION: Use larger voxel size and fewer points for publishing
            publish_voxel_size = 0.5  # 10cm for visualization (was 5cm)
            max_publish_points = int(os.getenv("LAC_MAX_PUBLISH_POINTS", "300"))  # Reduced from 500
            
            for h in hazard_pts:
                b, g, r = h["color"]
                # Pack BGR → RGB uint32 (RViz convention: 0x00RRGGBB)
                rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
                pts = h["points"]
                
                # NEW: Voxelize points before publishing (using Open3D like frontier_mapping_node)
                if len(pts) > 0:
                    pts_voxelized = self._voxelize_pointcloud(pts, voxel_size=publish_voxel_size, max_points=max_publish_points)
                else:
                    pts_voxelized = pts
                
                for p in pts_voxelized:
                    data_rows.append([float(p[0]), float(p[1]), float(p[2]), rgb_int])

            if not data_rows:
                return

            hdr = Header()
            hdr.stamp = self.get_clock().now().to_msg()
            hdr.frame_id = "map"
            cloud = pc2.create_cloud(hdr, fields, data_rows)
            self._pub_cloud.publish(cloud)

        # ── Periodic VLM query callback ────────────────────────────────────────
        
        def _verify_objects_persist_cb(self):
            """
            CRITICAL: Periodic verification that all encoded objects are still in processor.
            Re-adds any missing objects immediately to ensure they never go missing.
            Runs every 5 seconds.
            """
            if not self._models_ready or not self.seg or not self.seg.processor:
                return
            
            try:
                with self._encoded_lock:
                    if not self._encoded_objects:
                        return
                    
                    # Get current processor objects
                    processor_objects = self.seg.processor.get_all_objects()
                    processor_objects_normalized = {obj.lower().strip() for obj in processor_objects}
                    
                    # Check for missing objects
                    missing_objects = []
                    for encoded_obj in self._encoded_objects:
                        encoded_normalized = encoded_obj.lower().strip()
                        if encoded_obj not in processor_objects and encoded_normalized not in processor_objects_normalized:
                            missing_objects.append(encoded_obj)
                    
                    if missing_objects:
                        self.get_logger().error(
                            f"[LaC] CRITICAL: {len(missing_objects)} objects missing from processor! "
                            f"Missing: {missing_objects}. Re-adding immediately..."
                        )
                        
                        # Re-add missing objects
                        for missing_obj in missing_objects:
                            with torch.no_grad():
                                success = self.seg.processor.add_vlm_object(missing_obj)
                            if success:
                                self.get_logger().warn(f"[LaC] ✓ Re-added missing object '{missing_obj}' to processor")
                            else:
                                self.get_logger().error(f"[LaC] Failed to re-add missing object '{missing_obj}'")
                            # GPU MEMORY OPTIMIZATION: Clear cache after each re-addition
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                    
                    # Also check for duplicates in processor
                    if len(processor_objects) != len(processor_objects_normalized):
                        self.get_logger().error(
                            f"[LaC] CRITICAL: Duplicates detected in processor! "
                            f"Total: {len(processor_objects)}, Unique: {len(processor_objects_normalized)}"
                        )
                        
            except Exception as e:
                self.get_logger().error(f"[LaC] Error in object persistence verification: {e}")

        def _periodic_vlm_cb(self):
            """Periodically query VLM every 8 seconds to populate object list and radius list."""
            if not self._models_ready:
                return
            
            now = time.time()
            if (now - self._last_periodic_vlm_time) < self._vlm_periodic_interval:
                return
            
            # Get latest RGB image and timestamp
            with self._rgb_image_lock:
                if self._latest_rgb_image is None or self._latest_rgb_timestamp is None:
                    self.get_logger().debug("No latest RGB image for periodic VLM query.")
                    return
                bgr = self._latest_rgb_image.copy()
                rgb_ts = self._latest_rgb_timestamp
            
            self._last_periodic_vlm_time = now
            
            # Run VLM query in background thread
            threading.Thread(
                target=self._vlm_pipeline,
                args=(bgr, rgb_ts),
                daemon=True,
            ).start()
        
        def _filter_hazards_by_cosine_similarity(self, hazards: List[Dict], similarity_threshold: float = 0.5) -> List[Dict]:
            """
            Filter hazards by exact duplicates first, then cosine similarity of their text embeddings.
            If mutual cosine similarity > threshold, keep only the first one.
            
            CRITICAL: This is the PRIMARY deduplication point - all hazards must pass through here.
            
            Args:
                hazards: List of hazard dicts with "object" key
                similarity_threshold: Threshold for cosine similarity (default 0.5)
            
            Returns:
                Filtered list of hazards with duplicates removed
            """
            if len(hazards) <= 1:
                return hazards
            
            # STEP 1: Filter exact duplicates first (case-insensitive, trimmed)
            # This catches cases where VLM returns the same object name multiple times
            seen_objects = {}
            exact_deduped = []
            exact_duplicates_removed = 0
            
            for hazard in hazards:
                obj_name = hazard["object"]
                if not obj_name or not isinstance(obj_name, str):
                    continue  # Skip invalid objects
                    
                # Normalize: lowercase, strip whitespace
                obj_name_normalized = obj_name.lower().strip()
                
                if not obj_name_normalized:  # Skip empty strings
                    continue
                
                if obj_name_normalized not in seen_objects:
                    seen_objects[obj_name_normalized] = obj_name  # Store original case
                    exact_deduped.append(hazard)
                else:
                    exact_duplicates_removed += 1
                    self.get_logger().debug(
                        f"[LaC] Exact duplicate removed: '{obj_name}' (already seen as '{seen_objects[obj_name_normalized]}')"
                    )
            
            if exact_duplicates_removed > 0:
                self.get_logger().warn(
                    f"[LaC] Exact duplicate filter: {len(hazards)} -> {len(exact_deduped)} hazards "
                    f"(removed {exact_duplicates_removed} exact duplicates)"
                )
            
            # If only exact duplicates, return early
            if len(exact_deduped) <= 1:
                return exact_deduped
            
            # STEP 1.5: Also check against processor's existing objects to avoid re-adding
            # This prevents adding objects that are already in the processor
            existing_processor_objects = set()
            if self.seg and self.seg.processor:
                try:
                    existing_processor_objects = {obj.lower().strip() for obj in self.seg.processor.get_all_objects()}
                except Exception:
                    pass
            
            # Filter out objects that already exist in processor (case-insensitive)
            processor_filtered = []
            processor_duplicates_removed = 0
            for hazard in exact_deduped:
                obj_name = hazard["object"]
                obj_name_normalized = obj_name.lower().strip()
                if obj_name_normalized not in existing_processor_objects:
                    processor_filtered.append(hazard)
                else:
                    processor_duplicates_removed += 1
                    self.get_logger().debug(
                        f"[LaC] Processor duplicate removed: '{obj_name}' (already in processor)"
                    )
            
            if processor_duplicates_removed > 0:
                self.get_logger().warn(
                    f"[LaC] Processor duplicate filter: {len(exact_deduped)} -> {len(processor_filtered)} hazards "
                    f"(removed {processor_duplicates_removed} objects already in processor)"
                )
            
            if len(processor_filtered) <= 1:
                return processor_filtered
            
            exact_deduped = processor_filtered
            
            # STEP 2: Filter by cosine similarity for similar but not identical objects
            if not (self.seg and self.seg.processor and 
                    self.seg.processor.is_segmentation_ready() and
                    self.seg.processor.radio_encoder):
                self.get_logger().warn("[LaC] Cannot filter by cosine similarity - processor not ready")
                return exact_deduped
            
            try:
                # Extract object names from deduplicated list
                obj_names = [h["object"] for h in exact_deduped]
                
                # Encode all objects to get embeddings
                # GPU MEMORY OPTIMIZATION: Process in smaller batches to avoid OOM
                max_embedding_batch = 10  # Limit batch size for encoding
                all_embeddings = []
                
                for i in range(0, len(obj_names), max_embedding_batch):
                    batch_names = obj_names[i:i+max_embedding_batch]
                    with torch.no_grad():
                        batch_embeddings = self.seg.processor.radio_encoder.encode_labels(batch_names)
                        # GPU MEMORY OPTIMIZATION: Move to CPU immediately
                        if torch.is_tensor(batch_embeddings):
                            batch_embeddings = batch_embeddings.cpu()
                        # Convert to numpy
                        batch_embeddings_np = batch_embeddings.cpu().numpy() if torch.is_tensor(batch_embeddings) else batch_embeddings
                        all_embeddings.append(batch_embeddings_np)
                        # Clear GPU cache after each batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                
                # Concatenate all embeddings
                if not all_embeddings:
                    return exact_deduped  # Return deduplicated list if encoding failed
                
                embeddings_np = np.vstack(all_embeddings)
                
                # Normalize for cosine similarity
                norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # Avoid division by zero
                embeddings_norm = embeddings_np / norms
                
                # Compute pairwise cosine similarity matrix
                similarity_matrix = cosine_similarity(embeddings_norm)
                
                # Find groups of similar objects (mutual similarity > threshold)
                # Strategy: Keep first object in each group, discard others
                keep_indices = []
                discard_indices = set()
                
                for i in range(len(exact_deduped)):
                    if i in discard_indices:
                        continue
                    
                    # Find all objects similar to this one (similarity > threshold)
                    # Note: similarity matrix is symmetric, so we only need to check one direction
                    similar_indices = []
                    for j in range(i + 1, len(exact_deduped)):
                        if j in discard_indices:
                            continue
                        # Check similarity (matrix is symmetric, so [i,j] == [j,i])
                        if similarity_matrix[i, j] > similarity_threshold:
                            similar_indices.append(j)
                    
                    # Keep this object (first in group)
                    keep_indices.append(i)
                    # Discard all similar ones
                    discard_indices.update(similar_indices)
                    
                    if similar_indices:
                        similar_names = [obj_names[j] for j in similar_indices]
                        self.get_logger().info(
                            f"[LaC] Cosine similarity filter: '{obj_names[i]}' similar to {similar_names} "
                            f"(sim > {similarity_threshold:.2f}), keeping '{obj_names[i]}' only"
                        )
                
                # Create filtered hazards list
                filtered_hazards = [exact_deduped[i] for i in keep_indices]
                
                if len(filtered_hazards) < len(exact_deduped):
                    self.get_logger().info(
                        f"[LaC] Cosine similarity filter: {len(exact_deduped)} -> {len(filtered_hazards)} hazards "
                        f"(removed {len(exact_deduped) - len(filtered_hazards)} similar objects)"
                    )
                
                return filtered_hazards
                
            except Exception as e:
                self.get_logger().error(f"[LaC] Error in cosine similarity filtering: {e}")
                import traceback
                traceback.print_exc()
                # GPU MEMORY OPTIMIZATION: Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                return exact_deduped  # Return deduplicated list even on error (at least exact duplicates removed)
        
        def _save_vlm_answers(self, original_hazards: List[Dict], filtered_hazards: List[Dict], rgb_ts: float):
            """
            Save VLM answers (both filtered and unfiltered) to buffer directory as JSON.
            
            Args:
                original_hazards: List of hazards from VLM before filtering
                filtered_hazards: List of hazards after filtering
                rgb_ts: RGB image timestamp
            """
            if not hasattr(self, 'vlm_answers_dir') or self.vlm_answers_dir is None:
                return
            
            try:
                timestamp_str = f"{rgb_ts:.6f}"
                filename = f"vlm_answer_{timestamp_str}.json"
                filepath = os.path.join(self.vlm_answers_dir, filename)
                
                # Prepare data structure
                vlm_data = {
                    "timestamp": rgb_ts,
                    "timestamp_str": timestamp_str,
                    "original_count": len(original_hazards),
                    "filtered_count": len(filtered_hazards),
                    "original_hazards": [
                        {
                            "object": h.get("object", ""),
                            "anxiety": float(h.get("anxiety", 0.0)),
                            "radius_m": float(h.get("radius_m", 0.0)),
                            "filtered": False
                        }
                        for h in original_hazards
                    ],
                    "filtered_hazards": [
                        {
                            "object": h.get("object", ""),
                            "anxiety": float(h.get("anxiety", 0.0)),
                            "radius_m": float(h.get("radius_m", 0.0)),
                            "filtered": True
                        }
                        for h in filtered_hazards
                    ]
                }
                
                # Mark which original hazards were filtered out
                filtered_object_names = {h.get("object", "").lower().strip() for h in filtered_hazards}
                for hazard in vlm_data["original_hazards"]:
                    obj_normalized = hazard["object"].lower().strip()
                    if obj_normalized not in filtered_object_names:
                        hazard["filtered"] = True  # This one was filtered out
                
                # Save to JSON file
                with open(filepath, 'w') as f:
                    json.dump(vlm_data, f, indent=2)
                
                self.get_logger().debug(
                    f"[LaC] Saved VLM answers: {len(original_hazards)} original, "
                    f"{len(filtered_hazards)} filtered -> {filename}"
                )
                
            except Exception as e:
                self.get_logger().warn(f"[LaC] Error saving VLM answers: {e}")
                import traceback
                self.get_logger().debug(traceback.format_exc())
        
        def _vlm_pipeline(self, bgr: np.ndarray, rgb_ts: float):
            """Background thread for VLM query - encode new objects and update tracked hazards."""
            # This VLM pipeline is now only triggered by the periodic timer.
            # It should not be rate-limited by _vlm_running, as it's the sole VLM trigger.
            try:
                self.get_logger().info("[LaC] Periodic VLM query started...")
                with torch.no_grad():  # CRITICAL: No gradient computation
                    hazards = self.vlm.query(bgr)
                
                # Memory cleanup after VLM query
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Store original hazards (even if empty) for buffer storage
                original_hazards = []
                
                if not hazards:
                    self.get_logger().info("[LaC] Periodic VLM query: No hazards identified")
                    # Save empty result to buffer
                    self._save_vlm_answers(original_hazards, [], rgb_ts)
                    return
                
                # Clip radius_m to maximum 1.5m for all hazards
                MAX_RADIUS_M = 1.5
                for hazard in hazards:
                    if "radius_m" in hazard:
                        original_radius = hazard["radius_m"]
                        hazard["radius_m"] = min(float(hazard["radius_m"]), MAX_RADIUS_M)
                        if original_radius > MAX_RADIUS_M:
                            self.get_logger().debug(
                                f"[LaC] Clipped radius for '{hazard.get('object', 'unknown')}': "
                                f"{original_radius:.2f}m -> {hazard['radius_m']:.2f}m"
                            )
                
                # Store original hazards before filtering (for buffer storage)
                original_hazards = [dict(h) for h in hazards]  # Deep copy
                
                # NEW: Filter hazards by cosine similarity before processing
                # This removes duplicates where mutual cosine similarity > 0.85
                filtered_hazards = self._filter_hazards_by_cosine_similarity(hazards, similarity_threshold=0.7)
                
                # Save VLM answers (original and filtered) to buffer directory
                self._save_vlm_answers(original_hazards, filtered_hazards, rgb_ts)
                
                if not filtered_hazards:
                    self.get_logger().info("[LaC] All hazards filtered out by cosine similarity")
                    return
                
                # Use filtered hazards for rest of processing
                hazards = filtered_hazards
                
                # Encode and add NEW objects to processor (like main.py)
                # CRITICAL: Encoding should only happen once per object (tracked via _encoded_objects)
                # CRITICAL: Deduplication already happened in _filter_hazards_by_cosine_similarity
                new_objects_added = []
                current_time = time.time()
                
                # Additional safety: Check processor for existing objects before adding
                # CRITICAL: Use case-insensitive comparison to catch all duplicates
                all_processor_objects = []
                all_processor_objects_normalized = set()
                if self.seg and self.seg.processor:
                    try:
                        all_processor_objects = self.seg.processor.get_all_objects()
                        all_processor_objects_normalized = {obj.lower().strip() for obj in all_processor_objects}
                    except Exception:
                        pass
                
                with self._encoded_lock:
                    for hazard in hazards:
                        obj_name = hazard["object"]
                        if not obj_name or not isinstance(obj_name, str):
                            continue
                        
                        obj_name_normalized = obj_name.lower().strip()
                        
                        # CRITICAL: Check both _encoded_objects AND processor to avoid duplicates (case-insensitive)
                        # This prevents adding objects that were already added in previous VLM queries
                        already_encoded = (
                            obj_name in self._encoded_objects or 
                            obj_name_normalized in {o.lower().strip() for o in self._encoded_objects}
                        )
                        already_in_processor = (
                            obj_name in all_processor_objects or 
                            obj_name_normalized in all_processor_objects_normalized
                        )
                        
                        if already_encoded or already_in_processor:
                            self.get_logger().debug(
                                f"[LaC] Skipping '{obj_name}' - already encoded or in processor "
                                f"(encoded={already_encoded}, processor={already_in_processor})"
                            )
                            continue
                        
                        # Only encode and add if not already encoded (avoid duplicates)
                        if obj_name not in self._encoded_objects:
                            # Encode and add to processor (like main.py)
                            # CRITICAL: Wrap encoding in no_grad to ensure no gradient computation
                            if self.seg and self.seg.processor and self.seg.processor.is_segmentation_ready():
                                with torch.no_grad():
                                    success = self.seg.processor.add_vlm_object(obj_name)
                                if success:
                                    self._encoded_objects.add(obj_name)
                                    new_objects_added.append(obj_name)
                                    self.get_logger().info(
                                        f"[LaC] ✓ Encoded and added '{obj_name}' to processor (periodic)"
                                    )
                                    # GPU MEMORY OPTIMIZATION: Clear cache after each addition
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        import gc
                                        gc.collect()
                                else:
                                    self.get_logger().warn(f"[LaC] Failed to add '{obj_name}' to processor")
                
                # CRITICAL: Final verification - ensure no duplicates were added and objects persist
                if self.seg and self.seg.processor:
                    try:
                        final_processor_objects = self.seg.processor.get_all_objects()
                        final_normalized = {obj.lower().strip() for obj in final_processor_objects}
                        
                        # Check for duplicates
                        if len(final_processor_objects) != len(final_normalized):
                            self.get_logger().error(
                                f"[LaC] CRITICAL: Duplicates detected in processor after adding! "
                                f"Total: {len(final_processor_objects)}, Unique: {len(final_normalized)}"
                            )
                        
                        # CRITICAL: Verify all encoded objects are still in processor
                        missing_from_processor = []
                        for encoded_obj in self._encoded_objects:
                            encoded_normalized = encoded_obj.lower().strip()
                            if encoded_obj not in final_processor_objects and encoded_normalized not in final_normalized:
                                missing_from_processor.append(encoded_obj)
                        
                        if missing_from_processor:
                            self.get_logger().error(
                                f"[LaC] CRITICAL: Objects missing from processor after operations! "
                                f"Missing: {missing_from_processor}. Re-adding..."
                            )
                            # Re-add missing objects immediately
                            for missing_obj in missing_from_processor:
                                with torch.no_grad():
                                    success = self.seg.processor.add_vlm_object(missing_obj)
                                if success:
                                    self.get_logger().warn(f"[LaC] ✓ Re-added missing object '{missing_obj}' to processor")
                                else:
                                    self.get_logger().error(f"[LaC] Failed to re-add missing object '{missing_obj}'")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                    except Exception as e:
                        self.get_logger().error(f"[LaC] Error verifying processor state: {e}")
                
                # GPU MEMORY OPTIMIZATION: Final cleanup after all operations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                
                # Update tracked hazards with new VLM results
                # CRITICAL: Ensure objects persist once added - don't remove them!
                with self._hazard_lock:
                    for hazard in hazards:
                        obj_name = hazard["object"]
                        if obj_name in self._tracked_hazards:
                            # Update anxiety and radius from VLM (preserve existing points)
                            self._tracked_hazards[obj_name]["anxiety"] = hazard["anxiety"]
                            self._tracked_hazards[obj_name]["radius_m"] = hazard["radius_m"]
                            # Don't update last_update here - let projection thread handle it
                            # This ensures objects persist even if VLM query is slow
                        else:
                            # New hazard - initialize with empty points (will be populated by projection thread)
                            # CRITICAL: Initialize even if empty to ensure object persists
                            self._tracked_hazards[obj_name] = {
                                "points": np.zeros((0, 3), dtype=np.float32),
                                "anxiety": hazard["anxiety"],
                                "radius_m": hazard["radius_m"],
                                "color": DISTINCT_COLORS_BGR[len(self._tracked_hazards) % len(DISTINCT_COLORS_BGR)],
                                "last_update": current_time,
                            }
                    
                    self.get_logger().info(
                        f"[LaC] Periodic VLM: {len(new_objects_added)} new objects encoded, "
                        f"{len(hazards)} total hazards (tracked: {len(self._tracked_hazards)})"
                    )

            except Exception as e:
                import traceback
                self.get_logger().error(f"Periodic VLM pipeline error: {e}\n{traceback.format_exc()}")

        # ── Robot-centric grid generation ───────────────────────────────────────
        
        def _create_robot_centric_3d_grid(self):
            """
            Generate robot-centric 3D grid similar to frontier_mapping_node.
            Returns: (grid_points, grid_shape) where grid_points is (N, 3) and grid_shape is (D, H, W)
            """
            if self.robot_position is None:
                return np.array([], dtype=np.float32), (0, 0, 0)
            
            try:
                rx, ry, rz = self.robot_position
                h_xy = self.robot_grid_size_xy / 2.0
                h_z = self.robot_grid_size_z / 2.0
                res = self.robot_grid_resolution
                
                x_c = np.arange(rx - h_xy, rx + h_xy, res, dtype=np.float32)
                y_c = np.arange(ry - h_xy, ry + h_xy, res, dtype=np.float32)
                z_c = np.arange(rz - h_z, rz + h_z, res, dtype=np.float32)
                
                X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')
                grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                grid_shape = (len(x_c), len(y_c), len(z_c))
                
                return grid_points, grid_shape
            except Exception as e:
                self.get_logger().error(f"Error creating robot-centric grid: {e}")
                return np.array([], dtype=np.float32), (0, 0, 0)
        
        # ── Sphere generation and costmap ─────────────────────────────────────────
        
        def _generate_and_publish_costmap_cb(self):
            """Generate and publish 3D continuous Gaussian cost field centered at hazard voxels.
            
            OPTIMIZED: Rate-limited, early exits, and uses KD-tree for efficient distance queries (like frontier_mapping_node).
            """
            # CRITICAL: Early exit if models not ready (avoid any computation during initialization)
            if not self._models_ready:
                return
            
            # CRITICAL: Early exit if robot position not available (avoid any computation)
            if self.robot_position is None:
                return
            
            # CRITICAL: Early exit if no hazards exist (avoid expensive grid generation)
            # Quick check without full lock acquisition first
            if not hasattr(self, '_tracked_hazards') or not self._tracked_hazards:
                return
            
            # Quick check: count hazards with points (minimal lock time)
            with self._hazard_lock:
                if not self._tracked_hazards:
                    return
                hazards_with_points = sum(
                    1 for tracked in self._tracked_hazards.values()
                    if tracked["points"].shape[0] > 0
                )
                if hazards_with_points == 0:
                    return
            
            # Rate limiting (same as frontier_mapping_node GP update interval)
            current_time = time.time()
            if (current_time - self._last_costmap_update_time) < self._costmap_update_interval:
                return
            self._last_costmap_update_time = current_time
            
            # Get all tracked hazards (semantic voxels) - only after confirming they exist
            with self._hazard_lock:
                hazard_list = [
                    {
                        "object": obj_name,
                        "points": tracked["points"],
                        "radius_m": tracked["radius_m"],
                        "anxiety": tracked["anxiety"],
                        "color": tracked["color"],
                    }
                    for obj_name, tracked in self._tracked_hazards.items()
                    if tracked["points"].shape[0] > 0
                ]
            
            if not hazard_list:
                return
            
            # Generate robot-centric grid (only when hazards exist)
            grid_points, grid_shape = self._create_robot_centric_3d_grid()
            if len(grid_points) == 0:
                return
            
            # Initialize disturbance values (0 = no disturbance)
            disturbance_values = np.zeros(len(grid_points), dtype=np.float32)
            
            # OPTIMIZED: Use KD-tree for efficient nearest neighbor queries (O(N log M) instead of O(N*M))
            try:
                from scipy.spatial import cKDTree
                use_kdtree = True
            except ImportError:
                use_kdtree = False
                self.get_logger().debug("[LaC] scipy not available, using fallback distance computation")
            
            # For each hazard, compute continuous Gaussian cost field
            for hazard in hazard_list:
                voxel_points = hazard["points"]
                
                if len(voxel_points) == 0:
                    continue
                
                # VLM metrics: anxiety (peak cost) and radius (3σ boundary)
                alpha = hazard["anxiety"] / 10.0  # Normalize 0-10 to 0.0-1.0
                sigma = hazard["radius_m"] / 3.0  # Radius defines the 3-sigma spread
                
                # OPTIMIZATION: Early cutoff - only compute for points within 3*sigma (99.7% of Gaussian)
                max_search_radius = hazard["radius_m"]  # 3*sigma = radius_m
                
                if use_kdtree:
                    # OPTIMIZED: Use KD-tree for O(N log M) nearest neighbor queries
                    tree = cKDTree(voxel_points)
                    # Query all grid points for nearest neighbor within max_search_radius
                    min_distances, _ = tree.query(grid_points, k=1, distance_upper_bound=max_search_radius)
                    # Points beyond max_search_radius get inf distance
                    min_distances = np.nan_to_num(min_distances, nan=max_search_radius, posinf=max_search_radius)
                else:
                    # Fallback: chunked computation to avoid large memory allocation
                    chunk_size = 5000  # Process grid in chunks
                    min_distances = np.full(len(grid_points), max_search_radius, dtype=np.float32)
                    
                    for i in range(0, len(grid_points), chunk_size):
                        chunk = grid_points[i:i+chunk_size]
                        # Compute distances to all voxel points
                        distances_chunk = np.linalg.norm(
                            chunk[:, np.newaxis, :] - voxel_points[np.newaxis, :, :],
                            axis=2
                        )
                        # Find minimum distance for each grid point in chunk
                        min_distances[i:i+chunk_size] = np.min(distances_chunk, axis=1)
                        # Early cutoff: clip to max_search_radius
                        min_distances[i:i+chunk_size] = np.minimum(
                            min_distances[i:i+chunk_size],
                            max_search_radius
                        )
                
                # Continuous Gaussian cost: α * exp(-d² / 2σ²)
                # Only compute for points within reasonable distance (already filtered by KD-tree or chunking)
                gaussian_cost = alpha * np.exp(-0.5 * (min_distances**2) / (sigma**2))
                
                # Take maximum of all hazard fields (point-wise maximum)
                disturbance_values = np.maximum(disturbance_values, gaussian_cost)
            
            # Publish semantic costmap (same format as frontier_mapping_node - disturbance values only)
            self._publish_semantic_costmap(grid_points, disturbance_values)
            
            # Publish RGB visualization separately
            self._publish_costmap_visualization(grid_points, disturbance_values)
        
        def _publish_semantic_costmap(self, grid_points: np.ndarray, disturbance_values: np.ndarray):
            """Publish semantic costmap with disturbance values (exact same format as frontier_mapping_node)."""
            try:
                if len(grid_points) == 0 or len(disturbance_values) == 0:
                    return
                
                # Use ACTUAL disturbance values (not normalized) for motion planning
                # These are the real disturbance magnitudes that motion planning needs
                disturbance_vals = disturbance_values.astype(np.float32)
                
                # Create PointCloud2 message with XYZ + disturbance values
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                
                # Create structured array with XYZ + disturbance value
                cloud_data_combined = np.empty(len(grid_points), dtype=[
                    ('x', np.float32), ('y', np.float32), ('z', np.float32), 
                    ('disturbance', np.float32)
                ])
                
                # Fill in the data
                cloud_data_combined['x'] = grid_points[:, 0]
                cloud_data_combined['y'] = grid_points[:, 1]
                cloud_data_combined['z'] = grid_points[:, 2]
                cloud_data_combined['disturbance'] = disturbance_vals
                
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
                
                self._pub_semantic_costmap.publish(cloud_msg)
                
                self.get_logger().info(
                    f"Published semantic costmap with Gaussian cost values: "
                    f"min={disturbance_vals.min():.3f}, max={disturbance_vals.max():.3f}"
                )
                
            except Exception as e:
                import traceback
                self.get_logger().error(f"Error publishing semantic costmap: {e}\n{traceback.format_exc()}")
        
        def _publish_costmap_visualization(self, grid_points: np.ndarray, disturbance_values: np.ndarray):
            """Publish RGB visualization of avoidance spheres (separate from costmap)."""
            try:
                if len(grid_points) == 0 or len(disturbance_values) == 0:
                    return
                
                # Normalize disturbance values for colormap (0-1 range)
                d_min, d_max = disturbance_values.min(), disturbance_values.max()
                if d_max > d_min:
                    normalized = (disturbance_values - d_min) / (d_max - d_min)
                else:
                    normalized = np.zeros_like(disturbance_values)
                
                # Apply colormap (using turbo like frontier_mapping_node)
                colors_rgba = cm.turbo(normalized)
                colors_uint8 = (colors_rgba[:, :3] * 255).astype(np.uint32)
                rgb_packed = (colors_uint8[:, 0] << 16) | (colors_uint8[:, 1] << 8) | colors_uint8[:, 2]
                
                # Create structured array
                cloud_data = np.empty(len(grid_points), dtype=[
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('rgb', np.uint32)
                ])
                
                cloud_data['x'] = grid_points[:, 0].astype(np.float32)
                cloud_data['y'] = grid_points[:, 1].astype(np.float32)
                cloud_data['z'] = grid_points[:, 2].astype(np.float32)
                cloud_data['rgb'] = rgb_packed
                
                # Create PointCloud2 message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                
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
                cloud_msg.data = cloud_data.tobytes()
                
                self._pub_costmap_visualization.publish(cloud_msg)
                
                self.get_logger().debug(
                    f"Published semantic costmap visualization: {len(grid_points)} points, "
                    f"Gaussian cost range=[{d_min:.3f}, {d_max:.3f}]"
                )
                
            except Exception as e:
                import traceback
                self.get_logger().error(f"Error publishing costmap visualization: {e}\n{traceback.format_exc()}")
        

        def _publish_costmap(self, hazard_pts):
            """Publish 2-D OccupancyGrid with Gaussian cost fields per hazard."""
            # Filter out hazards with no points
            hazards_with_points = [h for h in hazard_pts if h["points"].shape[0] > 0]
            
            if not hazards_with_points:
                return  # No points to publish
            
            # Determine map bounds from points
            all_pts = np.vstack([h["points"][:, :2] for h in hazards_with_points])
            
            # Check if all_pts is empty (shouldn't happen after filtering, but be safe)
            if all_pts.shape[0] == 0:
                return
            
            resolution = 0.2   # m/cell
            margin     = 3.0   # extra metres around points

            min_xy = all_pts.min(axis=0) - margin
            max_xy = all_pts.max(axis=0) + margin
            width  = int((max_xy[0] - min_xy[0]) / resolution) + 1
            height = int((max_xy[1] - min_xy[1]) / resolution) + 1

            cost = np.zeros((height, width), dtype=np.float32)

            for h in hazards_with_points:
                # Weber-Fechner amplitude: anxiety on 0-10 scale → 0-1
                amp    = min(h["anxiety"] / 10.0, 1.0)
                sigma  = max(h["radius_m"] / resolution, 1.0)  # cells
                # 2-D Gaussian centered at mean hazard XY
                center = h["points"][:, :2].mean(axis=0)
                cx_cell = int((center[0] - min_xy[0]) / resolution)
                cy_cell = int((center[1] - min_xy[1]) / resolution)
                # Vectorised Gaussian evaluation
                gx = np.arange(width,  dtype=np.float32) - cx_cell
                gy = np.arange(height, dtype=np.float32) - cy_cell
                GX, GY = np.meshgrid(gx, gy)
                gauss  = amp * np.exp(-(GX**2 + GY**2) / (2.0 * sigma**2))
                cost   = np.maximum(cost, gauss)

            # Clip and scale to 0-100 for OccupancyGrid
            cost_int = (np.clip(cost, 0.0, 1.0) * 100).astype(np.int8)

            msg            = OccupancyGrid()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.info.resolution = resolution
            msg.info.width      = width
            msg.info.height     = height
            msg.info.origin.position.x = float(min_xy[0])
            msg.info.origin.position.y = float(min_xy[1])
            msg.data = cost_int.flatten().tolist()
            self._pub_cmap.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def standalone_demo(image_path: str):
    """Run LaC on a single image and display segmented hazards."""
    if not os.path.isfile(image_path):
        print(f"[LaC] Image not found: {image_path}")
        sys.exit(1)

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[LaC] Failed to read image: {image_path}")
        sys.exit(1)

    print(f"[LaC] Image: {image_path}  ({bgr.shape[1]}×{bgr.shape[0]})")

    # --- VLM ---
    vlm  = LaCVLM()
    hazards = vlm.query(bgr)

    if not hazards:
        print("[LaC] No hazards identified – displaying original image.")
        cv2.imshow("LaC – No Hazards", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("\n[LaC] Hazard summary:")
    for h in hazards:
        print(f"  • {h['object']:30s}  anxiety={h['anxiety']:4.1f}  radius={h['radius_m']:.1f} m")

    # --- Segmentation ---
    seg = LaCSegmentor()
    segmented = seg.segment(bgr, hazards)

    # --- Visualise ---
    vis = visualise(bgr, segmented)

    # Save result
    out_path = os.path.splitext(image_path)[0] + "_lac.png"
    cv2.imwrite(out_path, vis)
    print(f"\n[LaC] Saved annotated image to: {out_path}")

    # Show
    cv2.imshow("LaC – Hazard Segmentation", vis)
    print("[LaC] Press any key to close …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1:
        # Standalone demo: python LaC.py <image>
        standalone_demo(sys.argv[1])
    else:
        # ROS 2 node
        if not ROS_AVAILABLE:
            print("[LaC] ROS 2 not available. Pass an image path for standalone mode.")
            print("Usage:  python LaC.py <image_path>")
            sys.exit(1)
        rclpy.init()
        node = LaCNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
