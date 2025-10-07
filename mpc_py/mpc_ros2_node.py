#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry

from mpc_app import MobileRobotMPC

import numpy as np
from numpy.typing import NDArray
import time
import rerun as rr

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.publisher_ = self.create_publisher(PoseStamped, 'mpc_pose', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        timer_period = 0.1  # seconds (10 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.mpc = MobileRobotMPC(dt=0.1, horizon=10)
        self.mpc.set_weights(position_weight=50.0, heading_weight=2.0, control_weight=0.05)
        self.mpc.set_physical_constraints(v_min=0.0, v_max=0.5, omega_max=1.0)

        self.twist_msg = Twist()

        self.goal = np.array([5.0, 5.0, 0.0])  # Example goal position
        self.current_state = np.array([0.0, 0.0, 0.0])  # x, y, theta
        
        # Initialize Rerun
        self.setup_rerun()
        
        # Data storage for trajectory and time series
        self.trajectory_points = []
        self.time_step = 0
        self.max_history = 1000  # Keep last 1000 points
        
        # Current control values
        self.current_velocity = 0.0
        self.current_omega = 0.0

    def setup_rerun(self):
        """Initialize Rerun logging and set up the visualization entities."""
        rr.init("MPC_Robot_Controller")
        rr.spawn()
        
        # Set up static styling for the plots based on the Rerun documentation
        # Control plots styling (for time series)
        rr.log("controls/velocity", rr.SeriesLines(), static=True)
        rr.log("controls/omega", rr.SeriesLines(), static=True)
        
        # State plots styling (for time series)
        rr.log("state/heading", rr.SeriesLines(), static=True)
        
        # Log goal position once
        rr.log("trajectory/goal", rr.Points2D([[self.goal[0], self.goal[1]]]))
        
        self.get_logger().info("Rerun visualization initialized")

    def log_trajectory(self):
        """Log the robot's current position to the trajectory plot."""
        # Add current position to trajectory
        self.trajectory_points.append([self.current_state[0], self.current_state[1]])
        
        # Keep only the last max_history points
        if len(self.trajectory_points) > self.max_history:
            self.trajectory_points.pop(0)
        
        # Log the trajectory as a line strip
        if len(self.trajectory_points) > 1:
            rr.log("trajectory/robot_path", rr.LineStrips2D([self.trajectory_points]))

    def log_time_series(self):
        """Log time series data for velocity, omega, and heading."""
        rr.set_time("frame_nr", sequence=self.time_step)
        
        # Log control values
        rr.log("controls/velocity", rr.Scalars(self.current_velocity))
        rr.log("controls/omega", rr.Scalars(self.current_omega))
        
        # Log state values
        rr.log("state/heading", rr.Scalars(self.current_state[2]))
        
        self.time_step += 1

    def odom_callback(self, msg: Odometry):
        """
        Callback function to update the robot's current state from odometry data.
        """
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = msg.pose.pose.position.y
        # Yaw extraction from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_state[2] = np.arctan2(siny_cosp, cosy_cosp)

    def robot_dynamics(self, state: NDArray, control: NDArray):
        # Store current control values for logging
        self.current_velocity = float(control[0])
        self.current_omega = float(control[1])
        
        # Publish control commands
        self.twist_msg.linear.x = self.current_velocity
        self.twist_msg.angular.z = self.current_omega
        self.cmd_vel_publisher.publish(self.twist_msg)
        
        return self.current_state

    def timer_callback(self):
        dist_to_goal = np.linalg.norm(self.current_state[:2] - self.goal[:2])
        print(f"goal distance {dist_to_goal}")
        if dist_to_goal < 0.2:
            self.get_logger().info('Goal reached!')
            self.twist_msg.linear.x = 0.0
            self.twist_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(self.twist_msg)
            
            # Final logging
            self.current_velocity = 0.0
            self.current_omega = 0.0
            self.log_trajectory()
            self.log_time_series()
            
            time.sleep(1)
            self.goal = np.array([10.0, 5.0, 0.0])  # New goal
            # exit(0)
            # return
            
        # Solve MPC optimization
        control, pred_traj = self.mpc.solve(self.current_state, self.goal)
        
        # Apply control and update robot
        self.robot_dynamics(self.current_state, control)
        
        # Log data to Rerun
        self.log_trajectory()
        self.log_time_series()
        
        # Optional: Log predicted trajectory
        if len(pred_traj) > 1:
            pred_points = [[float(point[0]), float(point[1])] for point in pred_traj]
            rr.log("trajectory/predicted_path", rr.LineStrips2D([pred_points]))

def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPCNode()
    rclpy.spin(mpc_node)
    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
