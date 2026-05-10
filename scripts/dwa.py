#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
import math
import numpy as np
import matplotlib.pyplot as plt
import tf2_ros
import tf2_geometry_msgs
from threading import Lock
from matplotlib.widgets import CheckButtons  

from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import tf_transformations as tf_trans
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import sensor_msgs_py.point_cloud2 as pc2

# --- Configuration ---
ROBOT_RADIUS = 0.5      
VIEW_RADIUS = 3      
WAYPOINT_TOLERANCE = 1.5
SCAN_TOPIC = '/scan'
ODOM_TOPIC = '/Odometry'
PATH_TOPIC = '/global_path'
CMD_VEL_TOPIC = '/cmd_vel'
SEMANTIC_VOXELS_TOPIC = '/semantic_voxels_only'
GP_RAW_TOPIC = '/gp_grid_raw'   # frontier_mapping_node publishes here
GP_SLICE_Z = 0.25              # meters
GLOBAL_FRAME = 'odom'    

# --- DWA Parameters ---
MAX_SPEED = 0.7         
MAX_YAW_RATE = 1.5     
DT = 0.2                
PREDICT_TIME = 3.0      
V_RES = 0.1             
W_RES = 0.1             

# Constraints
TURN_LIMIT_W = 0.1      
TURN_LIMIT_V = 0.3      

# Weights
W_HEADING = 2.0
W_DIST = 1.0
W_SPEED = 0.5

class RailViz(Node):
    def __init__(self):
        super().__init__('rail_visualizer')
        
        self.lock = Lock()
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        self.full_path_x = []
        self.full_path_y = []
        self.target_idx = 0
        
        self.scan_x = np.array([])
        self.scan_y = np.array([])
        
        # Accumulated semantic "hotspot" voxels (projected to 2D in GLOBAL_FRAME)
        self.hotspot_x = np.array([], dtype=np.float32)
        self.hotspot_y = np.array([], dtype=np.float32)
        self._hotspot_total_received = 0
        self._hotspot_last_log_t = 0.0

        # GP raw grid overlay (2D slice at z=GP_SLICE_Z, in GLOBAL_FRAME coordinates)
        self._gp_overlay_img = None          # 2D float32 array (ny, nx) or (nx, ny) depending on mesh
        self._gp_overlay_extent = None       # [xmin, xmax, ymin, ymax]
        self._gp_overlay_vmin = None
        self._gp_overlay_vmax = None
        self._gp_last_log_t = 0.0

        self.cmd_v = 0.0
        self.cmd_w = 0.0
        
        self.viz_trajs_valid = [] 
        self.viz_trajs_invalid = [] 
        self.best_traj = ([], [])  
        
        self.has_path = False
        self.path_finished = False

        self.ignore_obstacles = True 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.pub_vel = self.create_publisher(Twist, CMD_VEL_TOPIC, 1)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odom_cb, 10)
        self.create_subscription(Path, PATH_TOPIC, self.path_cb, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, sensor_qos)
        self.create_subscription(PointCloud2, SEMANTIC_VOXELS_TOPIC, self.semantic_voxels_cb, sensor_qos)
        self.create_subscription(Float32MultiArray, GP_RAW_TOPIC, self.gp_raw_cb, 10)

        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        plt.subplots_adjust(bottom=0.2) 
        
        self.setup_plot()
        
        self.get_logger().info("Rail Viz & DWA Started (Toggle Added)...")

    def setup_plot(self):
        self.ax.set_aspect('equal')
        self.ax.grid(True, color='#dddddd', linestyle='--', alpha=0.5)
        self.ax.set_title(f"DWA Planner ({GLOBAL_FRAME})")
        
        self.robot_patch = Circle((0, 0), ROBOT_RADIUS, color='orange', fill=True, alpha=0.4)
        self.ax.add_patch(self.robot_patch)
        self.robot_arrow = self.ax.arrow(0, 0, 0.5, 0, head_width=0.2, color='red')
        
        self.line_passed, = self.ax.plot([], [], color='#A0A0A0', lw=2, label='Passed')
        self.line_current, = self.ax.plot([], [], color='blue', lw=3, label='Current')
        self.line_future, = self.ax.plot([], [], color='black', lw=2, label='Future')
        self.target_dot, = self.ax.plot([], [], 'ro', markersize=8, zorder=10)
        self.scan_scatter, = self.ax.plot([], [], 'k.', markersize=2, alpha=0.5)
        self.hotspot_scatter, = self.ax.plot([], [], 'r.', markersize=3, alpha=0.9, label='Hotspot voxels')

        self.dwa_lines_valid = LineCollection([], colors='green', linewidths=0.5, alpha=0.5)
        self.ax.add_collection(self.dwa_lines_valid)
        self.dwa_lines_invalid = LineCollection([], colors='red', linewidths=0.5, alpha=0.3)
        self.ax.add_collection(self.dwa_lines_invalid)

        # GP 2D overlay (drawn behind everything)
        # NOTE: we update the data/extent dynamically once GP raw grid arrives.
        self.gp_im = self.ax.imshow(
            np.zeros((2, 2), dtype=np.float32),
            extent=[0.0, 1.0, 0.0, 1.0],
            origin='lower',
            cmap='turbo',
            interpolation='bilinear',
            alpha=0.35,
            zorder=0,
        )
        self.gp_im.set_visible(False)

        self.best_line, = self.ax.plot([], [], color='lime', linewidth=2.5, alpha=0.9, label='Best Traj')
        self.info_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                      fontsize=9, verticalalignment='top',
                                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        ax_check = plt.axes([0.05, 0.05, 0.3, 0.08])
        self.check = CheckButtons(ax_check, ['Ignore Obstacles'], [False])
        self.check.on_clicked(self.toggle_obstacles)

        self.ax.legend(loc='lower right', fontsize='small')

    def gp_raw_cb(self, msg: Float32MultiArray):
        """
        Decode frontier_mapping_node raw GP grid and prepare a z=GP_SLICE_Z 2D slice overlay.

        Format (see frontier_mapping_node._publish_raw_gp_grid):
        - metadata (7 floats): [min_x, min_y, min_z, res, nx, ny, nz]
        - mean (N floats), uncertainty (N floats), where N = nx*ny*nz
        """
        try:
            data = np.asarray(msg.data, dtype=np.float32)
            if data.size < 7:
                return

            min_x, min_y, min_z, res, nx_f, ny_f, nz_f = [float(x) for x in data[:7]]
            nx = int(round(nx_f))
            ny = int(round(ny_f))
            nz = int(round(nz_f))
            if nx <= 1 or ny <= 1 or nz <= 0 or res <= 0.0:
                return

            n = nx * ny * nz
            if data.size < 7 + 2 * n:
                return

            mean_flat = data[7:7 + n]

            # Reconstruct (nx, ny, nz) grid; meshgrid was created with indexing='ij' then flattened.
            mean_grid = mean_flat.reshape((nx, ny, nz), order='C')

            # Slice at z=GP_SLICE_Z: pick the nearest z-plane (no cross-plane blending).
            z_rel = (float(GP_SLICE_Z) - float(min_z)) / float(res)
            if not np.isfinite(z_rel):
                return
            k = int(np.clip(int(np.round(z_rel)), 0, max(0, nz - 1)))
            sl = mean_grid[:, :, k]

            # Convert to (ny, nx) image for imshow (imshow expects [rows(y), cols(x)]).
            sl_img = sl.T.astype(np.float32, copy=False)  # (ny, nx)

            # Interpolate within this z-plane along x/y only (discard other planes).
            # We upsample onto a finer regular grid using separable 1D linear interpolation.
            up = 4  # upsample factor (visual smoothness)
            if up > 1 and sl_img.shape[0] >= 2 and sl_img.shape[1] >= 2:
                ny0, nx0 = sl_img.shape
                x0 = np.arange(nx0, dtype=np.float32)
                y0 = np.arange(ny0, dtype=np.float32)
                x1 = np.linspace(0.0, float(nx0 - 1), nx0 * up, dtype=np.float32)
                y1 = np.linspace(0.0, float(ny0 - 1), ny0 * up, dtype=np.float32)

                # Interp along x for each y row
                tmp = np.empty((ny0, x1.size), dtype=np.float32)
                for j in range(ny0):
                    tmp[j, :] = np.interp(x1, x0, sl_img[j, :]).astype(np.float32, copy=False)

                # Interp along y for each x column
                sl_up = np.empty((y1.size, x1.size), dtype=np.float32)
                for i in range(x1.size):
                    sl_up[:, i] = np.interp(y1, y0, tmp[:, i]).astype(np.float32, copy=False)

                sl_img = sl_up

            # Extent in world coords (GLOBAL_FRAME). We treat min_x/min_y as world-aligned origin.
            # Use cell-centered coordinates: [min_x, min_x + res*(nx-1)] etc.
            x_max = float(min_x + res * float(nx - 1))
            y_max = float(min_y + res * float(ny - 1))
            extent = [float(min_x), x_max, float(min_y), y_max]

            # Match /gp_field_visualization coloring: normalize by min/max of the field.
            vmin = float(np.nanmin(sl_img))
            vmax = float(np.nanmax(sl_img))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                return

            now_s = self.get_clock().now().nanoseconds * 1e-9
            if now_s - self._gp_last_log_t > 1.0:
                self._gp_last_log_t = now_s
                self.get_logger().info(
                    f"/gp_grid_raw: decoded nx={nx}, ny={ny}, nz={nz}, res={res:.3f}, "
                    f"slice_z={GP_SLICE_Z:.2f} (k={k}), "
                    f"mean_range=[{vmin:.3f},{vmax:.3f}]"
                )

            with self.lock:
                self._gp_overlay_img = sl_img
                self._gp_overlay_extent = extent
                self._gp_overlay_vmin = vmin
                self._gp_overlay_vmax = vmax

        except Exception:
            pass

    def _pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        """Extract Nx3 float32 XYZ points from a PointCloud2."""
        try:
            # Fast path if available (avoids Python tuple lists)
            if hasattr(pc2, "read_points_numpy"):
                arr = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
                if arr is None:
                    return np.zeros((0, 3), dtype=np.float32)
                arr = arr.astype(np.float32, copy=False)
                # Filter NaNs
                if arr.size == 0:
                    return np.zeros((0, 3), dtype=np.float32)
                mask = np.isfinite(arr).all(axis=1)
                return arr[mask]

            pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            arr = np.fromiter((c for p in pts for c in p), dtype=np.float32)
            if arr.size == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return arr.reshape((-1, 3))
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

    def _lookup_transform_for_cloud(self, cloud_msg: PointCloud2):
        src = str(getattr(cloud_msg.header, "frame_id", "") or "")
        if not src:
            return None, "empty frame_id"

        try:
            stamp_msg = getattr(cloud_msg.header, "stamp", None)
            if stamp_msg is not None and (stamp_msg.sec != 0 or stamp_msg.nanosec != 0):
                t = rclpy.time.Time.from_msg(stamp_msg)
            else:
                t = rclpy.time.Time()

            if not self.tf_buffer.can_transform(GLOBAL_FRAME, src, t, timeout=Duration(seconds=0.2)):
                # Try latest available as fallback
                t_latest = rclpy.time.Time()
                if not self.tf_buffer.can_transform(GLOBAL_FRAME, src, t_latest, timeout=Duration(seconds=0.2)):
                    return None, f"no TF {src} -> {GLOBAL_FRAME}"
                trans = self.tf_buffer.lookup_transform(GLOBAL_FRAME, src, t_latest)
                return trans, f"latest TF used ({src}->{GLOBAL_FRAME})"

            trans = self.tf_buffer.lookup_transform(GLOBAL_FRAME, src, t)
            return trans, f"stamped TF used ({src}->{GLOBAL_FRAME})"
        except Exception as e:
            return None, f"TF exception: {e}"

    def _transform_points_xyz(self, pts_xyz: np.ndarray, transform) -> np.ndarray:
        """Transform Nx3 XYZ points using a geometry_msgs TransformStamped."""
        if pts_xyz.size == 0:
            return pts_xyz
        t = transform.transform.translation
        q = transform.transform.rotation
        T = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(np.float32)
        T[0, 3] = float(t.x)
        T[1, 3] = float(t.y)
        T[2, 3] = float(t.z)
        ones = np.ones((pts_xyz.shape[0], 1), dtype=np.float32)
        pts_h = np.hstack([pts_xyz.astype(np.float32), ones])
        out = (T @ pts_h.T).T
        return out[:, :3]

    def semantic_voxels_cb(self, msg: PointCloud2):
        """Accumulate semantic voxels as 2D hotspot points in the DWA window."""
        try:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            pts_xyz = self._pointcloud2_to_xyz(msg)
            if pts_xyz.size == 0:
                if now_s - self._hotspot_last_log_t > 1.0:
                    self._hotspot_last_log_t = now_s
                    self.get_logger().warn(
                        f"/semantic_voxels_only: received but extracted 0 points "
                        f"(frame='{msg.header.frame_id}', w={int(getattr(msg,'width',0))}, "
                        f"h={int(getattr(msg,'height',0))}, point_step={int(getattr(msg,'point_step',0))})"
                    )
                return

            trans, tf_info = self._lookup_transform_for_cloud(msg)
            if trans is None:
                if now_s - self._hotspot_last_log_t > 1.0:
                    self._hotspot_last_log_t = now_s
                    self.get_logger().warn(
                        f"/semantic_voxels_only: TF failed ({tf_info}); "
                        f"cloud_frame='{msg.header.frame_id}', target='{GLOBAL_FRAME}'"
                    )
                return
            pts_g = self._transform_points_xyz(pts_xyz, trans)

            gx = pts_g[:, 0]
            gy = pts_g[:, 1]

            with self.lock:
                self.hotspot_x = np.concatenate([self.hotspot_x, gx.astype(np.float32)], axis=0)
                self.hotspot_y = np.concatenate([self.hotspot_y, gy.astype(np.float32)], axis=0)
                self._hotspot_total_received += int(gx.shape[0])

                # Compute how many are currently in-view for quick sanity.
                dx = self.hotspot_x - self.robot_x
                dy = self.hotspot_y - self.robot_y
                in_view = int(np.sum((dx * dx + dy * dy) <= (VIEW_RADIUS * VIEW_RADIUS)))

            if now_s - self._hotspot_last_log_t > 1.0:
                self._hotspot_last_log_t = now_s
                self.get_logger().warn(
                    f"/semantic_voxels_only: cloud ok; extracted={int(pts_xyz.shape[0])}, "
                    f"added={int(gx.shape[0])}, total_accum={int(self._hotspot_total_received)}, "
                    f"in_view≈{in_view}, cloud_frame='{msg.header.frame_id}', {tf_info}"
                )
        except Exception:
            pass

    def toggle_obstacles(self, label):
        self.ignore_obstacles = not self.ignore_obstacles
        state = "IGNORING" if self.ignore_obstacles else "ACTIVE"
        self.get_logger().info(f"Obstacle Avoidance: {state}")

    def odom_cb(self, msg):
        with self.lock:
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            o = msg.pose.pose.orientation
            _, _, self.robot_yaw = tf_trans.euler_from_quaternion([o.x, o.y, o.z, o.w])

    def path_cb(self, msg):
        if not msg.poses: return
        px, py = [], []
        try:
            target_frame = GLOBAL_FRAME
            source_frame = msg.header.frame_id
            if self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
                for pose in msg.poses:
                    p = tf2_geometry_msgs.do_transform_pose(pose.pose, transform)
                    px.append(p.position.x)
                    py.append(p.position.y)
            else:
                return
        except Exception:
            return
        
        with self.lock:
            self.full_path_x = px
            self.full_path_y = py
            self.target_idx = 0
            self.has_path = True
            self.path_finished = False

    def scan_cb(self, msg):
        try:
            trans = self.tf_buffer.lookup_transform(GLOBAL_FRAME, msg.header.frame_id, rclpy.time.Time())
            tx, ty = trans.transform.translation.x, trans.transform.translation.y
            q = trans.transform.rotation
            _, _, t_yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            ranges = np.array(msg.ranges)
            valid = (ranges >= msg.range_min) & (ranges <= msg.range_max)
            ranges, angles = ranges[valid], angles[valid]
            
            lx = ranges * np.cos(angles)
            ly = ranges * np.sin(angles)
            
            gx = lx * math.cos(t_yaw) - ly * math.sin(t_yaw) + tx
            gy = lx * math.sin(t_yaw) + ly * math.cos(t_yaw) + ty
            
            with self.lock:
                self.scan_x = gx
                self.scan_y = gy
        except Exception:
            pass

    def predict_trajectory(self, v, w):
        x = [self.robot_x]
        y = [self.robot_y]
        yaw = self.robot_yaw
        steps = int(PREDICT_TIME / DT)
        for _ in range(steps):
            yaw += w * DT
            nx = x[-1] + v * math.cos(yaw) * DT
            ny = y[-1] + v * math.sin(yaw) * DT
            x.append(nx)
            y.append(ny)
        return np.array(x), np.array(y)

    def dwa_control_loop(self):
        if not self.has_path or self.path_finished:
            self.cmd_v, self.cmd_w = 0.0, 0.0
            return

        while self.target_idx < len(self.full_path_x):
            tx = self.full_path_x[self.target_idx]
            ty = self.full_path_y[self.target_idx]
            dist = math.hypot(tx - self.robot_x, ty - self.robot_y)
            if dist < WAYPOINT_TOLERANCE:
                self.target_idx += 1
            else:
                break
        
        if self.target_idx >= len(self.full_path_x):
            self.path_finished = True
            self.cmd_v, self.cmd_w = 0.0, 0.0
            return

        best_score = -float('inf')
        best_u = (0.0, 0.0)
        best_traj_viz = ([], [])
        
        valid_trajs = []
        invalid_trajs = []

        gx = self.full_path_x[self.target_idx]
        gy = self.full_path_y[self.target_idx]

        obs_pts = None
        
        if not self.ignore_obstacles and len(self.scan_x) > 0:
            dx_scan = self.scan_x - self.robot_x
            dy_scan = self.scan_y - self.robot_y
            mask = (dx_scan**2 + dy_scan**2) < (VIEW_RADIUS + 1.0)**2
            if np.any(mask):
                obs_pts = np.vstack((self.scan_x[mask], self.scan_y[mask])).T

        vs = np.arange(0.0, MAX_SPEED + 0.01, V_RES)
        ws = np.arange(-MAX_YAW_RATE, MAX_YAW_RATE + 0.01, W_RES)
        robot_rad_sq = ROBOT_RADIUS**2

        for v in vs:
            for w in ws:
                if v == 0 and w == 0: continue 

                tx, ty = self.predict_trajectory(v, w)
                is_collision = False
                min_dist_sq = 100.0

                if abs(w) > TURN_LIMIT_W and v > TURN_LIMIT_V:
                    is_collision = True

                if not is_collision and obs_pts is not None:
                    end_pt = np.array([tx[-1], ty[-1]])
                    dists_sq = np.sum((obs_pts - end_pt)**2, axis=1)
                    min_dist_sq = np.min(dists_sq)
                    
                    if min_dist_sq < robot_rad_sq:
                        is_collision = True
                    else:
                        for i in range(0, len(tx), 2): 
                            pt = np.array([tx[i], ty[i]])
                            d_sq = np.min(np.sum((obs_pts - pt)**2, axis=1))
                            if d_sq < robot_rad_sq:
                                is_collision = True
                                break

                traj_points = list(zip(tx, ty))
                
                if is_collision:
                    invalid_trajs.append(traj_points)
                    score = -float('inf')
                else:
                    valid_trajs.append(traj_points)
                    
                    end_x, end_y = tx[-1], ty[-1]
                    target_yaw = math.atan2(gy - end_y, gx - end_x)
                    end_yaw = self.robot_yaw + w * PREDICT_TIME 
                    
                    yaw_err = abs(target_yaw - end_yaw)
                    while yaw_err > math.pi: yaw_err -= 2*math.pi
                    while yaw_err < -math.pi: yaw_err += 2*math.pi
                    yaw_err = abs(yaw_err)

                    heading_score = (math.pi - yaw_err)
                    
                    if obs_pts is not None:
                        min_clearance = math.sqrt(min_dist_sq)
                    else:
                        min_clearance = 2.0 
                    clearance_score = min(min_clearance, 2.0)
                    
                    score = (W_HEADING * heading_score) + \
                            (W_DIST * clearance_score) + \
                            (W_SPEED * v)

                    if score > best_score:
                        best_score = score
                        best_u = (v, w)
                        best_traj_viz = (tx, ty)

        self.cmd_v, self.cmd_w = best_u
        self.viz_trajs_valid = valid_trajs
        self.viz_trajs_invalid = invalid_trajs
        self.best_traj = best_traj_viz

    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            with self.lock:
                self.dwa_control_loop()
                twist = Twist()
                twist.linear.x = self.cmd_v
                twist.angular.z = self.cmd_w
                self.pub_vel.publish(twist)

                rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
                px, py = self.full_path_x, self.full_path_y
                sx, sy = self.scan_x, self.scan_y
                hx, hy = self.hotspot_x, self.hotspot_y
                idx = self.target_idx
                
                valid_segs = self.viz_trajs_valid
                invalid_segs = self.viz_trajs_invalid
                btx, bty = self.best_traj
                v_disp, w_disp = self.cmd_v, self.cmd_w

                gp_img = self._gp_overlay_img
                gp_extent = self._gp_overlay_extent
                gp_vmin = self._gp_overlay_vmin
                gp_vmax = self._gp_overlay_vmax

            self.robot_patch.center = (rx, ry)
            self.robot_arrow.remove()
            self.robot_arrow = self.ax.arrow(rx, ry, math.cos(ryaw)*ROBOT_RADIUS, math.sin(ryaw)*ROBOT_RADIUS, 
                                             head_width=0.2, color='red')

            # Update GP overlay (z=GP_SLICE_Z) if available
            if gp_img is not None and gp_extent is not None and gp_vmin is not None and gp_vmax is not None:
                self.gp_im.set_data(gp_img)
                self.gp_im.set_extent(gp_extent)
                self.gp_im.set_clim(gp_vmin, gp_vmax)
                self.gp_im.set_visible(True)
            else:
                self.gp_im.set_visible(False)

            self.line_passed.set_data(px[:idx], py[:idx])
            s, e = max(0, idx-1), min(len(px), idx+1)
            self.line_current.set_data(px[s:e], py[s:e])
            self.line_future.set_data(px[idx:], py[idx:])
            
            if self.has_path and not self.path_finished and idx < len(px):
                self.target_dot.set_data([px[idx]], [py[idx]])
            else:
                self.target_dot.set_data([], [])

            self.scan_scatter.set_data(sx, sy)
            self.hotspot_scatter.set_data(hx, hy)
            self.dwa_lines_valid.set_segments(valid_segs)
            self.dwa_lines_invalid.set_segments(invalid_segs)
            self.best_line.set_data(btx, bty)

            self.ax.set_xlim(rx - VIEW_RADIUS, rx + VIEW_RADIUS)
            self.ax.set_ylim(ry - VIEW_RADIUS, ry + VIEW_RADIUS)
            
            status = "MOVING" if not self.path_finished else "FINISHED"
            
            safety_mode = "BYPASSED" if self.ignore_obstacles else "ACTIVE"
            info = f"STATUS: {status}\nCmd V: {v_disp:.2f}\nCmd W: {w_disp:.2f}\nSafety: {safety_mode}"
            
            self.info_text.set_text(info)
            if self.ignore_obstacles:
                self.info_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="#ffcccc", ec="red", alpha=0.8))
            else:
                self.info_text.set_bbox(dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    rv = RailViz()
    try:
        rv.run()
    except KeyboardInterrupt:
        pass
    rv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()