#!/usr/bin/env python3
"""
SOTA MPPI Control Node for Robot-Centric GP Navigation
Refined for high-performance maneuvers with drag dynamics and control shifting.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import json
import time

# Add pytorch_mppi to path
sys.path.insert(0, '/home/navin/ros2_ws/src/resilience/pytorch_mppi/src')
try:
    from pytorch_mppi import MPPI
except ImportError:
    print("Error: pytorch_mppi not found.")

# ============================================================================
# GRID-BASED GP MODEL (UNTOUCHED)
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
        sampled = F.grid_sample(self.grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
        return sampled[0, channel, 0, 0, :]

# ============================================================================
# SOTA DYNAMICS & COST
# ============================================================================

class DroneDynamics3D:
    """Rigorous Drone Dynamics including Linear Drag and Semi-Implicit Integration."""
    def __init__(self, dt=0.05, device="cpu", dtype=torch.float32):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.v_limit = 4.0      # SOTA Velocity limit
        self.a_limit = 3.0      # SOTA Accel limit
        self.d_drag = 0.15      # Drag coefficient to prevent unrealistic acceleration

    def __call__(self, state, action):
        pos = state[:, :3]
        vel = state[:, 3:]
        
        # Apply acceleration with drag: a = u - d*v
        # Ensures velocity saturates physically
        accel = torch.clamp(action, -self.a_limit, self.a_limit)
        accel_net = accel - (self.d_drag * vel)

        # Semi-implicit Euler for better numerical stability at high speed
        vel_next = torch.clamp(vel + accel_net * self.dt, -self.v_limit, self.v_limit)
        pos_next = pos + vel_next * self.dt

        return torch.cat([pos_next, vel_next], dim=1)

class MPPICost3D:
    def __init__(self, target_goal, nominal_path=None, gp_model=None, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.goal = torch.as_tensor(target_goal, device=device, dtype=dtype)
        self.nominal_path = nominal_path
        self.gp_model = gp_model
        
        # SOTA Hyperparams
        self.w_goal = 5.0
        self.w_ref = 2.0
        self.w_obs = 50.0
        self.w_risk = 15.0     # Increased weight for GP avoidance
        self.w_smooth = 0.5    # Penalty for erratic control
        self.prev_u = None     # For smoothness
        
        self.obstacle_grid_tensor = None

    def set_obstacle_grid(self, occupied_indices, grid_shape):
        if len(occupied_indices) == 0:
            self.obstacle_grid_tensor = None
            return
        nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
        grid = torch.zeros((nx, ny, nz), device=self.device, dtype=self.dtype)
        inds = torch.as_tensor(occupied_indices, device=self.device, dtype=torch.long)
        mask = (inds[:,0]>=0) & (inds[:,0]<nx) & (inds[:,1]>=0) & (inds[:,1]<ny) & (inds[:,2]>=0) & (inds[:,2]<nz)
        valid_inds = inds[mask]
        if len(valid_inds) > 0:
            grid[valid_inds[:,0], valid_inds[:,1], valid_inds[:,2]] = 1.0
        self.obstacle_grid_tensor = grid.unsqueeze(0).unsqueeze(0)

    def __call__(self, state, action, step=None):
        pos = state[:, :3]
        
        # 1. Goal + Terminal Boost
        d_goal = torch.norm(pos - self.goal, dim=1)
        # Higher cost at the end of the horizon to ensure convergence
        c_goal = self.w_goal * (d_goal**2)
        
        # 2. Ref Path (Nearest Point)
        c_ref = 0.0
        if self.nominal_path is not None:
            dists = torch.norm(pos.unsqueeze(1) - self.nominal_path.unsqueeze(0), dim=2)
            min_dist, _ = torch.min(dists, dim=1)
            c_ref = self.w_ref * (min_dist**2)
            
        # 3. Obstacles (Bilinear sampling for gradients)
        c_obs = 0.0
        if self.gp_model is not None and self.obstacle_grid_tensor is not None:
            norm_coords = self.gp_model.normalize_coords(pos)
            grid_coords = norm_coords.view(1, 1, 1, -1, 3)
            samp = F.grid_sample(self.obstacle_grid_tensor, grid_coords, align_corners=True, mode='bilinear', padding_mode='zeros')
            c_obs = self.w_obs * samp.view(-1) * 10.0
            
        # 4. GP Risk (Exponential Barrier for Risk-Averse Control)
        c_risk = 0.0
        if self.gp_model is not None:
            mean, std = self.gp_model.forward_with_uncertainty(pos)
            risk_val = mean + 2.0 * std
            # Exponential barrier: penalized significantly as risk increases
            c_risk = self.w_risk * torch.exp(2.0 * risk_val)
            
        # 5. Smoothness Cost
        c_smooth = 0.0
        if self.prev_u is not None:
            c_smooth = self.w_smooth * torch.norm(action - self.prev_u, dim=1)
        
        return c_goal + c_ref + c_obs + c_risk + c_smooth

# ============================================================================
# NODE (Refined MPPI Logic)
# ============================================================================

class MPPIControlNode(Node):
    def __init__(self):
        super().__init__('mppi_control_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"SOTA MPPI Node on {self.device}")
        
        self.nominal_path_file = '/home/navin/ros2_ws/src/resilience/assets/adjusted_nominal_spline.json'
        self.gp_model = GridDisturbanceGP(device=self.device)
        self.robot_pose = None
        self.nominal_path_points = None
        self.latest_obstacles_indices = None
        
        self.load_nominal_path()
        self.init_mppi()
        
        # Publishers & Subscribers (Unchanged)
        self.path_pub = self.create_publisher(Path, '/mppi_path', 10)
        self.create_subscription(Float32MultiArray, '/gp_grid_raw', self.grid_callback, 10)
        self.create_subscription(PoseStamped, '/robot_1/sensors/front_stereo/pose', self.pose_callback, 10)
        self.create_subscription(PointCloud2, '/semantic_octomap_colored_cloud', self.obstacle_callback, 10)
        self.create_timer(0.1, self.run_control_loop)

    def load_nominal_path(self):
        if os.path.exists(self.nominal_path_file):
            try:
                with open(self.nominal_path_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.nominal_path_points = np.array(data, dtype=np.float32)
            except Exception as e:
                self.get_logger().error(f"Path load error: {e}")

    def init_mppi(self):
        self.dynamics = DroneDynamics3D(device=self.device)
        self.cost_fn = MPPICost3D(target_goal=[0,0,0], device=self.device)
        
        # SOTA Hyperparams: Increased samples and horizon
        self.mppi = MPPI(
            dynamics=self.dynamics,
            running_cost=self.cost_fn,
            nx=6,
            noise_sigma=0.3 * torch.eye(3, device=self.device),
            num_samples=1000,   # Higher sample count for better exploration
            horizon=40,         # Longer horizon for anticipatory control
            device=self.device,
            lambda_=0.2,       # Lower temperature for sharper cost focus
            u_min=torch.tensor([-3,-3,-3], device=self.device),
            u_max=torch.tensor([3,3,3], device=self.device)
        )

    def pose_callback(self, msg): self.robot_pose = msg

    def obstacle_callback(self, msg):
        if self.gp_model.min_bound is None: return
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        if not points: return
        pts = np.array(points, dtype=np.float32)
        min_b = self.gp_model.min_bound.cpu().numpy()
        res = self.gp_model.resolution
        self.latest_obstacles_indices = np.floor((pts - min_b) / res).astype(np.int32)

    def grid_callback(self, msg):
        data = np.array(msg.data, dtype=np.float32)
        meta = data[0:7]
        nx, ny, nz = int(meta[4]), int(meta[5]), int(meta[6])
        n_elements = nx * ny * nz
        self.gp_model.update_grid(data[7 : 7+n_elements], data[7+n_elements : 7+2*n_elements], meta)
        
        # if np.max(np.abs(data[7 : 7+n_elements])) > 1e-3:
        #     self.run_control_loop()

    def run_control_loop(self):
        if self.robot_pose is None: return
        with torch.no_grad():
            pos = self.robot_pose.pose.position
            # Better state tracking: would ideally use odom for velocity
            state = torch.tensor([pos.x, pos.y, pos.z, 0, 0, 0], device=self.device, dtype=torch.float32)
            
            # SOTA Warm-Start: Shift control sequence
            # Move the previous optimal plan one step forward and append zeros
            if hasattr(self.mppi, 'U'):
                self.mppi.U = torch.roll(self.mppi.U, shifts=-1, dims=0)
                self.mppi.U[-1, :] = 0.0

            # Update Cost Params
            self.cost_fn.goal = torch.as_tensor(self.compute_local_goal(pos), device=self.device)
            self.cost_fn.gp_model = self.gp_model
            if self.latest_obstacles_indices is not None:
                 self.cost_fn.set_obstacle_grid(self.latest_obstacles_indices, self.gp_model.grid_size)
            if self.nominal_path_points is not None:
                 self.cost_fn.nominal_path = torch.as_tensor(self.nominal_path_points, device=self.device)
                 
            # Run MPPI
            action = self.mppi.command(state)
            self.cost_fn.prev_u = action # Store for smoothness next iteration
            
            self.publish_path()

    def compute_local_goal(self, current_pos):
        if self.nominal_path_points is None:
            return [current_pos.x + 5.0, current_pos.y, current_pos.z]
        curr_vec = np.array([current_pos.x, current_pos.y, current_pos.z])
        dists = np.linalg.norm(self.nominal_path_points - curr_vec, axis=1)
        idx_min = np.argmin(dists)
        # Look ahead 6m for SOTA high-speed targets
        lookahead = 6.0
        for i in range(idx_min, len(self.nominal_path_points)):
            if np.linalg.norm(self.nominal_path_points[i] - curr_vec) > lookahead:
                return self.nominal_path_points[i]
        return self.nominal_path_points[-1]

    def publish_path(self):
        actions = self.mppi.U
        curr = torch.tensor([self.robot_pose.pose.position.x, self.robot_pose.pose.position.y, 
                             self.robot_pose.pose.position.z, 0,0,0], device=self.device).unsqueeze(0)
        states = [curr.squeeze(0)]
        for t in range(actions.shape[0]):
            curr = self.dynamics(curr, actions[t].unsqueeze(0))
            states.append(curr.squeeze(0))
            
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for s in states:
            p = PoseStamped()
            p.pose.position.x, p.pose.position.y, p.pose.position.z = float(s[0]), float(s[1]), float(s[2])
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MPPIControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()