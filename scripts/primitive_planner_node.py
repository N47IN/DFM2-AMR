#!/usr/bin/env python3
"""
Motion Primitive Planner Node
Path planning using a library of motion primitives evaluated against a Robot-Centric GP Grid.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, ColorRGBA
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point, Vector3, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.cm as cm  
import numpy as np
import torch
import torch.nn.functional as F
import math
import sys
import os
import json

# ============================================================================
# GRID-BASED GP MODEL (COPIED FROM MPPI NODE)
# ============================================================================

class GridDisturbanceGP(torch.nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.grid_tensor = None  
        self.min_bound = None    
        self.max_bound = None    
        self.resolution = 0.1
        self.grid_size = None    
        
    def update_grid(self, mean_data, uncert_data, metadata):
        min_x, min_y, min_z = metadata[0:3]
        res = metadata[3]
        nx, ny, nz = int(metadata[4]), int(metadata[5]), int(metadata[6])
        self.resolution = res
        self.min_bound = torch.tensor([min_x, min_y, min_z], device=self.device, dtype=self.dtype)
        self.grid_size = torch.tensor([nx, ny, nz], device=self.device, dtype=self.dtype)
        self.max_bound = self.min_bound + self.grid_size * res
        try:
            mean_grid = torch.as_tensor(mean_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            uncert_grid = torch.as_tensor(uncert_data, device=self.device, dtype=self.dtype).reshape(nx, ny, nz)
            self.grid_tensor = torch.stack([mean_grid, uncert_grid], dim=0).unsqueeze(0) 
        except Exception as e:
            print(f"Error reshaping grid: {e}")

    def normalize_coords(self, pos):
        grid_range = torch.clamp(self.max_bound - self.min_bound, min=1e-6)
        norm_pos = (pos - self.min_bound) / grid_range * 2.0 - 1.0
        return torch.stack([norm_pos[:, 2], norm_pos[:, 1], norm_pos[:, 0]], dim=1)

    def forward_with_uncertainty(self, pos):
        mean = self._sample(pos, channel=0)
        std = self._sample(pos, channel=1)
        return mean, std

    def _sample(self, pos, channel=0):
        if self.grid_tensor is None:
            return torch.zeros(pos.shape[0], device=self.device, dtype=self.dtype)
        N = pos.shape[0]
        norm_coords = self.normalize_coords(pos)
        grid_coords = norm_coords.view(1, 1, 1, N, 3)
        # grid_sample expects input in range [-1, 1]
        sampled = F.grid_sample(self.grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        return sampled[0, channel, 0, 0, :]

# ============================================================================
# MOTION PRIMITIVE LIBRARY
# ============================================================================

class MotionPrimitiveLibrary:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # Generator params
        self.speed = 1.0        # m/s
        self.horizon = 2.0      # seconds
        self.dt = 0.1          # time step
        self.num_steps = int(self.horizon / self.dt)
        
        # Define ranges for primitives
        # Curvature (yaw rate) - extending left and right
        self.yaw_rates = torch.linspace(-0.8, 0.8, 15, device=device, dtype=dtype)
        # Vertical angle (pitch) - extending up and down
        self.pitch_angles = torch.linspace(-0.4, 0.4, 7, device=device, dtype=dtype)
        
        # Create library parameters
        yy, pp = torch.meshgrid(self.yaw_rates, self.pitch_angles, indexing='ij')
        self.prim_ws = yy.flatten()       # Yaw rates
        self.prim_gammas = pp.flatten()   # Pitch angles
        self.num_prims = self.prim_ws.shape[0]
        
    def generate_primitives(self, start_pose):
        """
        Generate motion primitives from the current pose.
        start_pose: [x, y, z, yaw] (Tensor)
        Returns: Tensor of shape (NumPrims, NumSteps, 3) representing (x,y,z) trajectories
        """
        x0, y0, z0, yaw0 = start_pose
        
        # Initialize trajectories
        # Use simple kinematic model:
        # x_dot = v * cos(pitch) * cos(yaw)
        # y_dot = v * cos(pitch) * sin(yaw)
        # z_dot = v * sin(pitch)
        # yaw_dot = w
        
        trajectories = torch.zeros((self.num_prims, self.num_steps, 3), device=self.device, dtype=self.dtype)
        
        # Initial state for all primitives
        curr_x = x0.repeat(self.num_prims)
        curr_y = y0.repeat(self.num_prims)
        curr_z = z0.repeat(self.num_prims)
        curr_yaw = yaw0.repeat(self.num_prims)
        
        v_xy = self.speed * torch.cos(self.prim_gammas)
        v_z = self.speed * torch.sin(self.prim_gammas)
        
        for t in range(self.num_steps):
            # Update position
            curr_x = curr_x + v_xy * torch.cos(curr_yaw) * self.dt
            curr_y = curr_y + v_xy * torch.sin(curr_yaw) * self.dt
            curr_z = curr_z + v_z * self.dt # Constant vertical vel
            
            # Update yaw
            curr_yaw = curr_yaw + self.prim_ws * self.dt
            
            # Store
            trajectories[:, t, 0] = curr_x
            trajectories[:, t, 1] = curr_y
            trajectories[:, t, 2] = curr_z
            
        return trajectories

# ============================================================================
# MOTION PLANNER NODE
# ============================================================================

class MotionPrimitivePlannerNode(Node):
    def __init__(self):
        super().__init__('motion_primitive_planner_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Motion Primitive Planner Node on {self.device}")
        self.latest_obstacles_indices = None
        self.obstacle_grid_tensor = None
        # ADD THIS LINE:
        self.prev_best_idx = None
        self.nominal_path_file = '/root/AirStack/robot/ros_ws/src/autonomy/4_global/a_world_models/resilience/assets/adjusted_nominal_spline.json'
        
        self.gp_model = GridDisturbanceGP(device=self.device)
        self.primitive_lib = MotionPrimitiveLibrary(device=self.device)
        
        self.robot_pose = None
        self.nominal_path_points = None
        self.latest_obstacles_indices = None
        self.obstacle_grid_tensor = None
        
        self.load_nominal_path()
        
        # Publishers & Subscribers
        self.path_pub = self.create_publisher(Path, '/mppi_path', 10)
        self.primitives_pub = self.create_publisher(MarkerArray, '/motion_primitives/candidates', 10)
        
        self.create_subscription(Float32MultiArray, '/gp_grid_raw', self.grid_callback, 10)
        self.create_subscription(PoseStamped, '/robot_1/sensors/front_stereo/pose', self.pose_callback, 10)
        self.create_subscription(PointCloud2, '/semantic_octomap_colored_cloud', self.obstacle_callback, 10)
        
        self.create_timer(0.2, self.run_planner_loop) # Run at 5Hz

    def load_nominal_path(self):
        if os.path.exists(self.nominal_path_file):
            try:
                with open(self.nominal_path_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.nominal_path_points = torch.tensor(data, device=self.device, dtype=torch.float32)
            except Exception as e:
                self.get_logger().error(f"Path load error: {e}")

    def pose_callback(self, msg): 
        self.robot_pose = msg

    def obstacle_callback(self, msg):
        if self.gp_model.min_bound is None: return
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        if not points: return
        pts = np.array(points, dtype=np.float32)
        min_b = self.gp_model.min_bound.cpu().numpy()
        res = self.gp_model.resolution
        # Convert to grid indices
        occupied_indices = np.floor((pts - min_b) / res).astype(np.int32)
        self.update_obstacle_grid(occupied_indices)

    def update_obstacle_grid(self, occupied_indices):
        if len(occupied_indices) == 0:
            self.obstacle_grid_tensor = None
            return
        grid_shape = self.gp_model.grid_size.cpu().numpy()
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        
        grid = torch.zeros((nx, ny, nz), device=self.device, dtype=torch.float32)
        inds = torch.as_tensor(occupied_indices, device=self.device, dtype=torch.long)
        mask = (inds[:,0]>=0) & (inds[:,0]<nx) & (inds[:,1]>=0) & (inds[:,1]<ny) & (inds[:,2]>=0) & (inds[:,2]<nz)
        valid_inds = inds[mask]
        if len(valid_inds) > 0:
            grid[valid_inds[:,0], valid_inds[:,1], valid_inds[:,2]] = 1.0
        self.obstacle_grid_tensor = grid.unsqueeze(0).unsqueeze(0)

    def grid_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        meta = data[0:7]
        nx, ny, nz = int(meta[4]), int(meta[5]), int(meta[6])
        n_elements = nx * ny * nz
        self.gp_model.update_grid(data[7 : 7+n_elements], data[7+n_elements : 7+2*n_elements], meta)

    def get_yaw_from_pose(self, pose):
        # Extract yaw from quaternion
        q = pose.orientation
        # sin(y) = 2(wz + xy)
        # cos(y) = 1 - 2(y^2 + z^2)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def compute_costs(self, trajectories, goal):
        num_prims, num_steps, _ = trajectories.shape
        flat_traj = trajectories.reshape(-1, 3) # [NumPrims*NumSteps, 3]
        costs = torch.zeros(num_prims, device=self.device)
        
        # 1. Goal Cost (Distance to lookahead point)
        end_points = trajectories[:, -1, :]
        costs += 5.0 * torch.norm(end_points - goal, dim=1)
        
        # 2. Reference Path Cost (Whole-Trajectory)
        if self.nominal_path_points is not None:
            # Check every point in every primitive against the nominal path
            # Distances: [NumPrims*NumSteps, NumRefPoints]
            dists = torch.cdist(flat_traj, self.nominal_path_points)
            min_dists_per_pt, _ = torch.min(dists, dim=1)
            # Average distance for each primitive
            path_err = min_dists_per_pt.view(num_prims, num_steps).mean(dim=1)
            costs += 8.0 * path_err # Increased weight for tighter tracking
        
        # 3. Obstacle Cost (Safety)
        if self.obstacle_grid_tensor is not None:
            norm_coords = self.gp_model.normalize_coords(flat_traj)
            grid_coords = norm_coords.view(1, 1, 1, -1, 3)
            samp = F.grid_sample(self.obstacle_grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
            obs_vals = samp.view(num_prims, num_steps)
            costs += 50.0 * torch.max(obs_vals, dim=1)[0] # Penalize ANY collision in the path

        # 4. GP Risk Cost
        if self.gp_model.grid_tensor is not None:
            mean, std = self.gp_model.forward_with_uncertainty(flat_traj)
            risk_val = mean*std
            risk_cost_steps = torch.exp(2.5 * risk_val).view(num_prims, num_steps)
            costs += 15.0 * torch.mean(risk_cost_steps, dim=1)

        # 5. NEW: Smoothness Cost (Temporal Consistency)
        if self.prev_best_idx is not None:
            prev_w = self.primitive_lib.prim_ws[self.prev_best_idx]
            prev_gamma = self.primitive_lib.prim_gammas[self.prev_best_idx]
            
            # Penalize large changes in yaw rate and pitch
            w_diff = (self.primitive_lib.prim_ws - prev_w)**2
            g_diff = (self.primitive_lib.prim_gammas - prev_gamma)**2
            costs += 2.0 * (w_diff + g_diff)

        return costs

    def find_local_goal(self, current_pos):
        if self.nominal_path_points is None:
            # Default forward if no path
            # Assuming robot faces +x roughly or just extend forward
            return torch.tensor([current_pos.x + 5.0, current_pos.y, current_pos.z], device=self.device)
        
        curr_vec = torch.tensor([current_pos.x, current_pos.y, current_pos.z], device=self.device)
        dists = torch.norm(self.nominal_path_points - curr_vec, dim=1)
        min_idx = torch.argmin(dists)
        
        # Lookahead
        lookahead_dist = 6.0
        target_idx = min_idx
        for i in range(min_idx, len(self.nominal_path_points)):
            if torch.norm(self.nominal_path_points[i] - curr_vec) > lookahead_dist:
                target_idx = i
                break
        return self.nominal_path_points[target_idx]

    def run_planner_loop(self):
        if self.robot_pose is None: return
        
        with torch.no_grad():
            pos = self.robot_pose.pose.position
            yaw = self.get_yaw_from_pose(self.robot_pose.pose)
            start_pose_t = torch.tensor([pos.x, pos.y, pos.z, yaw], device=self.device)
            
            # Generate primitives
            trajectories = self.primitive_lib.generate_primitives(start_pose_t)
            
            # Get goal
            goal = self.find_local_goal(pos)
            
            # Compute costs
            costs = self.compute_costs(trajectories, goal)
            
            # Find best
            best_idx = torch.argmin(costs)
            self.prev_best_idx = best_idx.item()
            best_traj = trajectories[best_idx]
            
            # Visualize / Publish
            self.publish_best_path(best_traj)
            self.publish_primitives(trajectories, costs)

    def publish_best_path(self, traj):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        scaler_cpu = traj.cpu().numpy()
        for i in range(scaler_cpu.shape[0]):
            p = PoseStamped()
            p.pose.position.x = float(scaler_cpu[i, 0])
            p.pose.position.y = float(scaler_cpu[i, 1])
            p.pose.position.z = float(scaler_cpu[i, 2])
            # Orientation? identity for now
            path_msg.poses.append(p)
        
        self.path_pub.publish(path_msg)

    def publish_primitives(self, trajectories, costs):
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        
        # --- SHARP NORMALIZATION & COLORMAP ---
        # 1. Percentile Normalization: Ignore the top 10% of "horrible" paths 
        # to preserve contrast for the viable candidates.
        c_min = torch.min(costs)
        c_90 = torch.quantile(costs, 0.9)
        
        # Avoid division by zero
        denom = c_90 - c_min if c_90 > c_min else 1.0
        
        # Normalize to [0, 1] and clip outliers
        norm_costs = torch.clamp((costs - c_min) / denom, 0.0, 1.0)
        
        # 2. Use a high-contrast colormap (Turbo is great for path costs)
        # Low cost = Blue/Green, High cost = Red
        colors_mapped = cm.turbo(norm_costs.cpu().numpy()) 

        traj_cpu = trajectories.cpu().numpy()
        best_idx = torch.argmin(costs).item()

        # Clear old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i in range(traj_cpu.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = timestamp
            marker.ns = "motion_primitives"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            
            # Use the mapped colors
            r, g, b, _ = colors_mapped[i]

            if i == best_idx:
                marker.scale.x = 0.08  # Thick highlight
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0 # Cyan for best
                marker.color.a = 1.0
                marker.pose.position.z += 0.02 # Lift slightly
            else:
                marker.scale.x = 0.02
                marker.color.r = float(r)
                marker.color.g = float(g)
                marker.color.b = float(b)
                # Alpha based on cost: high cost paths fade out
                marker.color.a = float(0.7 - (0.5 * norm_costs[i].item()))

            for t in range(traj_cpu.shape[1]):
                p = Point()
                p.x = float(traj_cpu[i, t, 0])
                p.y = float(traj_cpu[i, t, 1])
                p.z = float(traj_cpu[i, t, 2])
                marker.points.append(p)

            marker_array.markers.append(marker)

        self.primitives_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPrimitivePlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()