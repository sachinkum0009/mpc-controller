"""
Complete MPC Implementation for Mobile Robots
==============================================

This code demonstrates three practical scenarios:
1. Path following (following a predefined trajectory)
2. Point-to-point navigation with obstacle avoidance
3. Dynamic target tracking (following a moving target)

Usage Examples at the bottom of the file!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from scipy.optimize import minimize


class MobileRobotMPC:
    """
    Model Predictive Controller for differential drive mobile robots.
    
    Robot Model (Kinematic Unicycle Model):
    - State: [x, y, theta]
        x, y: position (meters)
        theta: heading angle (radians)
    
    - Control: [v, omega]
        v: linear velocity (m/s)
        omega: angular velocity (rad/s)
    """
    
    def __init__(self, dt=0.1, horizon=15):
        """
        Initialize MPC controller.
        
        Args:
            dt: Time step (seconds)
            horizon: Prediction horizon (number of steps)
        """
        self.dt = dt
        self.horizon = horizon
        
        # Cost function weights
        self.Q = np.diag([20.0, 20.0, 5.0])       # State: [x, y, theta]
        self.R = np.diag([1.0, 0.5])               # Control: [v, omega]
        self.Qf = self.Q * 5                       # Terminal cost
        
        # Physical constraints
        self.v_max = 1.5        # Max linear velocity (m/s)
        self.v_min = -0.5       # Min linear velocity (m/s, allow small backward)
        self.omega_max = 1.5    # Max angular velocity (rad/s)
        
        # Obstacles
        self.obstacles = []
        self.robot_radius = 0.25  # Robot radius (m)
        self.safety_distance = 0.3  # Additional safety margin (m)
        
        # Previous control sequence (for warm start)
        self.u_prev = None

    def set_physical_constraints(self, v_min: float, v_max: float, omega_max: float):
        """Set physical constraints for the robot."""
        self.v_min = v_min
        self.v_max = v_max
        self.omega_max = omega_max

    def set_weights(self, position_weight=20.0, heading_weight=5.0, control_weight=0.1):
        """Adjust cost function weights for different behaviors."""
        self.Q = np.diag([position_weight, position_weight, heading_weight])
        self.R = np.diag([control_weight, control_weight * 5])
        self.Qf = self.Q * 5
        
    def add_circular_obstacle(self, x, y, radius):
        """Add a circular obstacle at (x, y) with given radius."""
        self.obstacles.append({'type': 'circle', 'x': x, 'y': y, 'r': radius})
        
    def add_rectangular_obstacle(self, x_min, x_max, y_min, y_max):
        """Add a rectangular obstacle."""
        self.obstacles.append({
            'type': 'rect', 
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        })
        
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()
        
    def robot_dynamics(self, state, control):
        """
        Compute next state using kinematic unicycle model.
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
        Returns:
            next_state: [x_new, y_new, theta_new]
        """
        x, y, theta = state
        v, omega = control
        
        # Clip control inputs to physical limits
        v = np.clip(v, self.v_min, self.v_max)
        omega = np.clip(omega, -self.omega_max, self.omega_max)
        
        # Update position and heading using kinematic model
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        
        # Normalize angle to [-pi, pi]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return np.array([x_new, y_new, theta_new])
    
    def predict_trajectory(self, state, controls):
        """Predict future trajectory given control sequence."""
        controls = controls.reshape(self.horizon, 2)
        trajectory = np.zeros((self.horizon + 1, 3))
        trajectory[0] = state
        
        for i in range(self.horizon):
            trajectory[i + 1] = self.robot_dynamics(trajectory[i], controls[i])
            
        return trajectory
    
    def compute_collision_cost(self, trajectory):
        """Compute cost for potential collisions with obstacles."""
        cost = 0.0
        
        for i in range(len(trajectory)):
            x, y = trajectory[i, 0], trajectory[i, 1]
            
            for obs in self.obstacles:
                if obs['type'] == 'circle':
                    # Distance to obstacle center
                    dist = np.sqrt((x - obs['x'])**2 + (y - obs['y'])**2)
                    min_dist = obs['r'] + self.robot_radius + self.safety_distance
                    
                    if dist < min_dist:
                        # Exponential penalty for being too close
                        cost += 500 * np.exp(-2 * (dist - min_dist))
                        
                elif obs['type'] == 'rect':
                    # Check if inside expanded rectangle
                    margin = self.robot_radius + self.safety_distance
                    if (obs['x_min'] - margin <= x <= obs['x_max'] + margin and
                        obs['y_min'] - margin <= y <= obs['y_max'] + margin):
                        cost += 1000
                        
        return cost
    
    def cost_function(self, controls, current_state, reference):
        """
        MPC cost function.
        
        Args:
            controls: Flattened control sequence
            current_state: Current robot state
            reference: Reference trajectory or target state
        """
        trajectory = self.predict_trajectory(current_state, controls)
        controls = controls.reshape(self.horizon, 2)
        
        cost = 0.0
        
        # Stage costs
        for i in range(self.horizon):
            # State tracking error
            if reference.ndim == 1:  # Single target point
                ref = reference
            else:  # Reference trajectory
                ref = reference[min(i, len(reference) - 1)]
            
            error = trajectory[i] - ref
            # Wrap angle error to [-pi, pi]
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
            
            cost += error @ self.Q @ error
            
            # Control effort
            cost += controls[i] @ self.R @ controls[i]
            
        # Terminal cost
        if reference.ndim == 1:
            final_error = trajectory[-1] - reference
        else:
            final_error = trajectory[-1] - reference[-1]
        final_error[2] = np.arctan2(np.sin(final_error[2]), np.cos(final_error[2]))
        cost += final_error @ self.Qf @ final_error
        
        # Obstacle avoidance cost
        cost += self.compute_collision_cost(trajectory)
        
        return cost
    
    def solve(self, current_state, reference):
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current robot state [x, y, theta, v]
            reference: Target state or reference trajectory
            
        Returns:
            optimal_control: [a, omega] to apply
            predicted_trajectory: Predicted states over horizon
        """
        # Initial guess (warm start with previous solution)
        if self.u_prev is not None:
            # Shift previous solution and repeat last control
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1:]])
        else:
            u0 = np.zeros((self.horizon, 2))
        u0 = u0.flatten()
        
        # Control bounds
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.v_min, self.v_max))
            bounds.append((-self.omega_max, self.omega_max))
        
        # Solve optimization
        result = minimize(
            fun=lambda u: self.cost_function(u, current_state, reference),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-3}
        )
        
        # Extract optimal control sequence
        optimal_controls = result.x.reshape(self.horizon, 2)
        self.u_prev = optimal_controls
        
        # Predict trajectory
        predicted_traj = self.predict_trajectory(current_state, result.x)
        
        return optimal_controls[0], predicted_traj


# ============================================================================
# SCENARIO 1: Point-to-Point Navigation with Obstacle Avoidance
# ============================================================================

def scenario_point_to_point():
    """Navigate robot from start to goal while avoiding obstacles."""
    print("\n" + "="*70)
    print("SCENARIO 1: Point-to-Point Navigation with Obstacle Avoidance")
    print("="*70)
    
    # Create MPC controller
    mpc = MobileRobotMPC(dt=0.1, horizon=20)
    mpc.set_weights(position_weight=50.0, heading_weight=2.0, 
                    control_weight=0.05)
    
    # Add obstacles
    mpc.add_circular_obstacle(3.0, 2.0, 0.5)
    mpc.add_circular_obstacle(5.5, 3.5, 0.6)
    mpc.add_circular_obstacle(6.0, 2.0, 0.4)
    mpc.add_circular_obstacle(7.0, 1.0, 0.4)
    # mpc.add_rectangular_obstacle(4.0, 4.5, 0.0, 1.5)
    
    # Initial state: [x, y, theta]
    state = np.array([0.0, 0.0, 0.0])
    
    # Goal state
    goal = np.array([9.0, 3.0, 0.0])
    
    # Simulation
    max_time = 25.0
    states = [state.copy()]
    controls = []
    
    print(f"Start: ({state[0]:.1f}, {state[1]:.1f})")
    print(f"Goal:  ({goal[0]:.1f}, {goal[1]:.1f})")
    print("\nSimulating...")
    
    t = 0
    while t < max_time:
        # Check if goal reached
        dist_to_goal = np.linalg.norm(state[:2] - goal[:2])
        if dist_to_goal < 0.3:
            print(f"\n✓ Goal reached at t={t:.1f}s!")
            break
        
        # Compute optimal control
        control, pred_traj = mpc.solve(state, goal)
        
        # Apply control
        state = mpc.robot_dynamics(state, control)
        
        # Store
        states.append(state.copy())
        controls.append(control.copy())
        
        t += mpc.dt
        
        if len(states) % 30 == 0:
            print(f"  t={t:.1f}s: pos=({state[0]:.2f}, {state[1]:.2f}), "
                  f"heading={state[2]:.2f} rad, dist={dist_to_goal:.2f}m")
    
    # Convert to arrays
    states = np.array(states)
    controls = np.array(controls)
    
    # Plot results
    plot_scenario(states, controls, mpc, goal, "Point-to-Point Navigation")
    
    return states, controls


# ============================================================================
# SCENARIO 2: Path Following
# ============================================================================

def scenario_path_following():
    """Follow a predefined path (e.g., figure-8 or circular path)."""
    print("\n" + "="*70)
    print("SCENARIO 2: Path Following (Figure-8 Trajectory)")
    print("="*70)
    
    # Create MPC controller with higher position tracking weight
    mpc = MobileRobotMPC(dt=0.1, horizon=15)
    mpc.set_weights(position_weight=50.0, heading_weight=2.0, 
                    control_weight=0.05)
    
    # Generate figure-8 reference trajectory
    t_ref = np.linspace(0, 4*np.pi, 200)
    x_ref = 4 * np.sin(t_ref)
    y_ref = 2 * np.sin(2 * t_ref)
    
    # Compute heading angles (tangent to path)
    dx = np.gradient(x_ref)
    dy = np.gradient(y_ref)
    theta_ref = np.arctan2(dy, dx)
    
    # Reference trajectory
    reference_path = np.column_stack([x_ref, y_ref, theta_ref])
    
    # Initial state (start slightly off the path)
    state = np.array([0.5, 0.5, 0.0])
    
    # Simulation
    states = [state.copy()]
    controls = []
    
    print("Following figure-8 path...")
    
    for i in range(len(reference_path) - mpc.horizon):
        # Get reference trajectory segment
        ref_segment = reference_path[i:i+mpc.horizon+1]
        
        # Compute control
        control, _ = mpc.solve(state, ref_segment)
        
        # Apply control
        state = mpc.robot_dynamics(state, control)
        
        states.append(state.copy())
        controls.append(control.copy())
        
        if i % 40 == 0:
            tracking_error = np.linalg.norm(state[:2] - reference_path[i, :2])
            print(f"  Progress: {100*i/len(reference_path):.0f}%, "
                  f"tracking error: {tracking_error:.3f}m")
    
    states = np.array(states)
    controls = np.array(controls)
    
    # Plot
    plot_path_following(states, controls, reference_path, mpc)
    
    return states, controls


# ============================================================================
# SCENARIO 3: Moving Target Tracking
# ============================================================================

def scenario_moving_target():
    """Track a moving target (e.g., following another robot)."""
    print("\n" + "="*70)
    print("SCENARIO 3: Moving Target Tracking")
    print("="*70)
    
    # Create MPC controller
    mpc = MobileRobotMPC(dt=0.1, horizon=20)
    mpc.set_weights(position_weight=30.0, heading_weight=5.0)
    
    # Add some obstacles
    mpc.add_circular_obstacle(5.0, 5.0, 0.6)
    
    # Initial robot state
    state = np.array([0.0, 0.0, 0.0])
    
    # Target moves in a circular path
    target_states = []
    
    max_time = 20.0
    states = [state.copy()]
    controls = []
    
    print("Tracking moving target...")
    
    t = 0
    step = 0
    while t < max_time:
        # Update target position (circular motion)
        target_x = 5 + 3 * np.cos(0.3 * t)
        target_y = 5 + 3 * np.sin(0.3 * t)
        target_theta = 0.3 * t + np.pi/2
        target = np.array([target_x, target_y, target_theta])
        target_states.append(target.copy())
        
        # Compute control to track target
        control, _ = mpc.solve(state, target)
        
        # Apply control
        state = mpc.robot_dynamics(state, control)
        
        states.append(state.copy())
        controls.append(control.copy())
        
        t += mpc.dt
        step += 1
        
        if step % 30 == 0:
            dist_to_target = np.linalg.norm(state[:2] - target[:2])
            print(f"  t={t:.1f}s: distance to target: {dist_to_target:.2f}m")
    
    states = np.array(states)
    controls = np.array(controls)
    target_states = np.array(target_states)
    
    # Plot
    plot_moving_target(states, controls, target_states, mpc)
    
    return states, controls


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_scenario(states, controls, mpc: MobileRobotMPC, goal, title):
    """Plot results for point-to-point navigation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.plot(states[0, 0], states[0, 1], 'go', markersize=12, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    # Draw obstacles
    for obs in mpc.obstacles:
        if obs['type'] == 'circle':
            circle = Circle((obs['x'], obs['y']), obs['r'], 
                          color='red', alpha=0.3)
            ax.add_patch(circle)
        elif obs['type'] == 'rect':
            rect = Rectangle((obs['x_min'], obs['y_min']),
                                obs['x_max'] - obs['x_min'],
                                obs['y_max'] - obs['y_min'],
                                color='red', alpha=0.3)
            ax.add_patch(rect)
    
    # Draw robot orientation
    for i in range(0, len(states), 15):
        arrow = FancyArrow(states[i, 0], states[i, 1],
                          0.3*np.cos(states[i, 2]), 0.3*np.sin(states[i, 2]),
                          width=0.1, color='blue', alpha=0.5)
        ax.add_patch(arrow)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Linear velocity control
    time_ctrl = np.arange(len(controls)) * mpc.dt
    axes[0, 1].plot(time_ctrl, controls[:, 0], 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Linear Velocity [m/s]')
    axes[0, 1].set_title('Linear Velocity Control')
    axes[0, 1].grid(True)
    
    # Heading angle
    time = np.arange(len(states)) * mpc.dt
    axes[1, 0].plot(time, states[:, 2], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Heading [rad]')
    axes[1, 0].set_title('Robot Heading')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time_ctrl, controls[:, 1], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angular Velocity [rad/s]')
    axes[1, 1].set_title('Angular Velocity Control')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_path_following(states, controls, reference, mpc: MobileRobotMPC):
    """Plot results for path following."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], 'r--', 
            linewidth=2, alpha=0.5, label='Reference Path')
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Actual Path')
    ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Path Following: Figure-8 Trajectory')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Tracking error
    time = np.arange(len(states)) * mpc.dt
    tracking_error = np.linalg.norm(states[:len(reference), :2] - 
                                   reference[:len(states), :2], axis=1)
    axes[0, 1].plot(time[:len(tracking_error)], tracking_error, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Tracking Error [m]')
    axes[0, 1].set_title('Position Tracking Error')
    axes[0, 1].grid(True)
    
    # Heading angle
    axes[1, 0].plot(time, states[:, 2], 'b-', linewidth=2, label='Actual')
    axes[1, 0].plot(time[:len(reference)], reference[:len(states), 2], 
                    'r--', linewidth=2, label='Reference')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Heading [rad]')
    axes[1, 0].set_title('Heading Tracking')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Control inputs
    time_ctrl = np.arange(len(controls)) * mpc.dt
    axes[1, 1].plot(time_ctrl, controls[:, 0], 'g-', linewidth=2, label='Linear v')
    axes[1, 1].plot(time_ctrl, controls[:, 1], 'm-', linewidth=2, label='Angular ω')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Control')
    axes[1, 1].set_title('Control Inputs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_moving_target(states, controls, targets, mpc):
    """Plot results for moving target tracking."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(targets[:, 0], targets[:, 1], 'r--', 
            linewidth=2, alpha=0.7, label='Target Path')
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    
    # Draw obstacles
    for obs in mpc.obstacles:
        if obs['type'] == 'circle':
            circle = Circle((obs['x'], obs['y']), obs['r'], color='red', alpha=0.3)
            ax.add_patch(circle)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Moving Target Tracking')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Distance to target over time
    time = np.arange(len(states)) * mpc.dt
    # Ensure both arrays have same length by taking minimum length
    min_len = min(len(states), len(targets))
    dist_to_target = np.linalg.norm(states[:min_len, :2] - targets[:min_len, :2], axis=1)
    axes[0, 1].plot(time[:min_len], dist_to_target, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Distance to Target [m]')
    axes[0, 1].set_title('Tracking Performance')
    axes[0, 1].grid(True)
    
    # Heading
    axes[1, 0].plot(time, states[:, 2], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Heading [rad]')
    axes[1, 0].set_title('Robot Heading')
    axes[1, 0].grid(True)
    
    # Controls
    time_ctrl = np.arange(len(controls)) * mpc.dt
    axes[1, 1].plot(time_ctrl, controls[:, 0], 'g-', linewidth=2, label='Linear v')
    axes[1, 1].plot(time_ctrl, controls[:, 1], 'm-', linewidth=2, label='Angular ω')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Control')
    axes[1, 1].set_title('Control Inputs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN: Run All Scenarios
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  MPC FOR MOBILE ROBOTS - PRACTICAL EXAMPLES")
    print("="*70)
    print("\nThis demo shows three real-world scenarios:")
    print("  1. Point-to-point navigation with obstacle avoidance")
    print("  2. Path following (figure-8 trajectory)")
    print("  3. Moving target tracking")
    print("\nPress Ctrl+C to skip a scenario.")
    print("="*70)
    
    # Run scenarios
    try:
        # Scenario 1
        states1, controls1 = scenario_point_to_point()
        
        # Scenario 2
        # states2, controls2 = scenario_path_following()
        
        # # Scenario 3
        # states3, controls3 = scenario_moving_target()
        
        print("\n" + "="*70)
        print("All scenarios completed successfully!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
