//
// MPC Controller for Mobile Robots in Rust
//
// Author: Sachin Kumar
// Date: 17-08-2025
// Converted from Python implementation
//

use nalgebra::{DVector, DMatrix, Vector3, Vector2};
use std::f64::consts::PI;
use rerun::{RecordingStream, external::glam};

/// Obstacle types
#[derive(Debug, Clone)]
enum Obstacle {
    Circle { x: f64, y: f64, radius: f64 },
    Rectangle { x_min: f64, x_max: f64, y_min: f64, y_max: f64 },
}

/// Mobile Robot MPC Controller
/// 
/// Robot Model (Kinematic Unicycle Model):
/// - State: [x, y, theta]
///   x, y: position (meters)
///   theta: heading angle (radians)
/// 
/// - Control: [v, omega]
///   v: linear velocity (m/s)
///   omega: angular velocity (rad/s)
pub struct MobileRobotMPC {
    dt: f64,
    horizon: usize,
    
    // Cost function weights
    q: Vector3<f64>,           // State weights: [x, y, theta]
    r: Vector2<f64>,           // Control weights: [v, omega]
    qf: Vector3<f64>,          // Terminal cost weights
    
    // Physical constraints
    v_max: f64,
    v_min: f64,
    omega_max: f64,
    
    // Obstacles and safety
    obstacles: Vec<Obstacle>,
    robot_radius: f64,
    safety_distance: f64,
    
    // Previous control sequence for warm start
    u_prev: Option<DVector<f64>>,
    
    // Rerun logging
    rec: Option<RecordingStream>,
}

impl MobileRobotMPC {
    /// Create new MPC controller
    pub fn new(dt: f64, horizon: usize) -> Self {
        Self {
            dt,
            horizon,
            q: Vector3::new(20.0, 20.0, 5.0),
            r: Vector2::new(1.0, 0.5),
            qf: Vector3::new(100.0, 100.0, 25.0),
            v_max: 1.5,
            v_min: -0.5,
            omega_max: 1.5,
            obstacles: Vec::new(),
            robot_radius: 0.25,
            safety_distance: 0.3,
            u_prev: None,
            rec: None,
        }
    }
    
    /// Initialize Rerun logging
    pub fn init_logging(&mut self, app_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let rec = rerun::RecordingStreamBuilder::new(app_id).spawn()?;
        self.rec = Some(rec);
        Ok(())
    }
    
    /// Set physical constraints
    pub fn set_physical_constraints(&mut self, v_min: f64, v_max: f64, omega_max: f64) {
        self.v_min = v_min;
        self.v_max = v_max;
        self.omega_max = omega_max;
    }
    
    /// Set cost function weights
    pub fn set_weights(&mut self, position_weight: f64, heading_weight: f64, control_weight: f64) {
        self.q = Vector3::new(position_weight, position_weight, heading_weight);
        self.r = Vector2::new(control_weight, control_weight * 5.0);
        self.qf = self.q * 5.0;
    }
    
    /// Add circular obstacle
    pub fn add_circular_obstacle(&mut self, x: f64, y: f64, radius: f64) {
        self.obstacles.push(Obstacle::Circle { x, y, radius });
    }
    
    /// Add rectangular obstacle
    pub fn add_rectangular_obstacle(&mut self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) {
        self.obstacles.push(Obstacle::Rectangle { x_min, x_max, y_min, y_max });
    }
    
    /// Clear all obstacles
    pub fn clear_obstacles(&mut self) {
        self.obstacles.clear();
    }
    
    /// Log robot trajectory to Rerun
    pub fn log_trajectory(&self, entity_path: &str, trajectory: &DMatrix<f64>, color: [u8; 3]) {
        if let Some(ref rec) = self.rec {
            let mut points = Vec::new();
            for i in 0..trajectory.nrows() {
                points.push(glam::Vec3::new(
                    trajectory[(i, 0)] as f32,
                    trajectory[(i, 1)] as f32,
                    0.0, // Z coordinate for 2D visualization
                ));
            }
            
            let _ = rec.log(
                entity_path,
                &rerun::LineStrips3D::new([points])
                    .with_colors([rerun::Color::from_rgb(color[0], color[1], color[2])]),
            );
        }
    }
    
    /// Log robot position and heading
    pub fn log_robot_state(&self, entity_path: &str, state: &Vector3<f64>, time_seq: i64) {
        if let Some(ref rec) = self.rec {
            // Log position as a point
            let _ = rec.log(
                format!("{}/position", entity_path),
                &rerun::Points3D::new([(state[0] as f32, state[1] as f32, 0.0f32)])
                    .with_colors([rerun::Color::from_rgb(255, 0, 0)])
                    .with_radii([0.1]),
            );
            
            // Log heading as an arrow
            let arrow_length = 0.5f32;
            let arrow_end = glam::Vec3::new(
                (state[0] + arrow_length as f64 * state[2].cos()) as f32,
                (state[1] + arrow_length as f64 * state[2].sin()) as f32,
                0.0,
            );
            let arrow_start = glam::Vec3::new(state[0] as f32, state[1] as f32, 0.0);
            
            let _ = rec.log(
                format!("{}/heading", entity_path),
                &rerun::Arrows3D::from_vectors([arrow_end - arrow_start])
                    .with_origins([arrow_start])
                    .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
            );
            
            // Log time series data
            rec.set_time_sequence("frame", time_seq);
            let _ = rec.log(
                "robot/state/x",
                &rerun::Scalars::new([state[0] as f64]),
            );
            let _ = rec.log(
                "robot/state/y", 
                &rerun::Scalars::new([state[1] as f64]),
            );
            let _ = rec.log(
                "robot/state/theta",
                &rerun::Scalars::new([state[2] as f64]),
            );
        }
    }
    
    /// Log control inputs
    pub fn log_control(&self, control: &Vector2<f64>, time_seq: i64) {
        if let Some(ref rec) = self.rec {
            rec.set_time_sequence("frame", time_seq);
            let _ = rec.log(
                "robot/control/linear_velocity",
                &rerun::Scalars::new([control[0] as f64]),
            );
            let _ = rec.log(
                "robot/control/angular_velocity",
                &rerun::Scalars::new([control[1] as f64]),
            );
        }
    }
    
    /// Log obstacles
    pub fn log_obstacles(&self) {
        if let Some(ref rec) = self.rec {
            for (i, obstacle) in self.obstacles.iter().enumerate() {
                match obstacle {
                    Obstacle::Circle { x, y, radius } => {
                        // Create circle points
                        let mut circle_points = Vec::new();
                        let num_points = 32;
                        for j in 0..=num_points {
                            let angle = 2.0 * PI * (j as f64) / (num_points as f64);
                            circle_points.push(glam::Vec3::new(
                                (*x + radius * angle.cos()) as f32,
                                (*y + radius * angle.sin()) as f32,
                                0.0,
                            ));
                        }
                        
                        let _ = rec.log(
                            format!("obstacles/circle_{}", i),
                            &rerun::LineStrips3D::new([circle_points])
                                .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                        );
                    }
                    Obstacle::Rectangle { x_min, x_max, y_min, y_max } => {
                        let rect_points = vec![
                            glam::Vec3::new(*x_min as f32, *y_min as f32, 0.0),
                            glam::Vec3::new(*x_max as f32, *y_min as f32, 0.0),
                            glam::Vec3::new(*x_max as f32, *y_max as f32, 0.0),
                            glam::Vec3::new(*x_min as f32, *y_max as f32, 0.0),
                            glam::Vec3::new(*x_min as f32, *y_min as f32, 0.0), // Close the rectangle
                        ];
                        
                        let _ = rec.log(
                            format!("obstacles/rect_{}", i),
                            &rerun::LineStrips3D::new([rect_points])
                                .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                        );
                    }
                }
            }
        }
    }
    
    /// Log goal point
    pub fn log_goal(&self, goal: &Vector3<f64>) {
        if let Some(ref rec) = self.rec {
            let _ = rec.log(
                "goal",
                &rerun::Points3D::new([(goal[0] as f32, goal[1] as f32, 0.0f32)])
                    .with_colors([rerun::Color::from_rgb(0, 255, 0)])
                    .with_radii([0.15]),
            );
        }
    }
    
    /// Robot dynamics - compute next state
    pub fn robot_dynamics(&self, state: &Vector3<f64>, control: &Vector2<f64>) -> Vector3<f64> {
        let x = state[0];
        let y = state[1];
        let theta = state[2];
        
        // Clip control inputs to physical limits
        let v = control[0].clamp(self.v_min, self.v_max);
        let omega = control[1].clamp(-self.omega_max, self.omega_max);
        
        // Update position and heading using kinematic model
        let x_new = x + v * theta.cos() * self.dt;
        let y_new = y + v * theta.sin() * self.dt;
        let theta_new = Self::normalize_angle(theta + omega * self.dt);
        
        Vector3::new(x_new, y_new, theta_new)
    }
    
    /// Normalize angle to [-pi, pi]
    fn normalize_angle(angle: f64) -> f64 {
        angle.sin().atan2(angle.cos())
    }
    
    /// Predict future trajectory given control sequence
    pub fn predict_trajectory(&self, state: &Vector3<f64>, controls: &DVector<f64>) -> DMatrix<f64> {
        let mut trajectory = DMatrix::zeros(self.horizon + 1, 3);
        trajectory.set_row(0, &state.transpose());
        
        let mut current_state = *state;
        
        for i in 0..self.horizon {
            let control_idx = i * 2;
            let control = Vector2::new(controls[control_idx], controls[control_idx + 1]);
            current_state = self.robot_dynamics(&current_state, &control);
            trajectory.set_row(i + 1, &current_state.transpose());
        }
        
        trajectory
    }
    
    /// Compute collision cost with obstacles
    pub fn compute_collision_cost(&self, trajectory: &DMatrix<f64>) -> f64 {
        let mut cost = 0.0;
        
        for i in 0..trajectory.nrows() {
            let x = trajectory[(i, 0)];
            let y = trajectory[(i, 1)];
            
            for obstacle in &self.obstacles {
                match obstacle {
                    Obstacle::Circle { x: ox, y: oy, radius } => {
                        let dx: f64 = x - ox;
                        let dy: f64 = y - oy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let min_dist = radius + self.robot_radius + self.safety_distance;
                        
                        if dist < min_dist {
                            cost += 500.0 * (-2.0 * (dist - min_dist)).exp();
                        }
                    }
                    Obstacle::Rectangle { x_min, x_max, y_min, y_max } => {
                        let margin = self.robot_radius + self.safety_distance;
                        if x >= x_min - margin && x <= x_max + margin &&
                           y >= y_min - margin && y <= y_max + margin {
                            cost += 1000.0;
                        }
                    }
                }
            }
        }
        
        cost
    }
    
    /// Cost function for MPC optimization
    pub fn cost_function(&self, controls: &DVector<f64>, current_state: &Vector3<f64>, reference: &Vector3<f64>) -> f64 {
        let trajectory = self.predict_trajectory(current_state, controls);
        let mut cost = 0.0;
        
        // Stage costs
        for i in 0..self.horizon {
            let state = Vector3::new(trajectory[(i, 0)], trajectory[(i, 1)], trajectory[(i, 2)]);
            let mut error = state - reference;
            error[2] = Self::normalize_angle(error[2]); // Wrap angle error
            
            // State cost
            cost += error.dot(&(self.q.component_mul(&error)));
            
            // Control cost
            let control_idx = i * 2;
            let control = Vector2::new(controls[control_idx], controls[control_idx + 1]);
            cost += control.dot(&(self.r.component_mul(&control)));
        }
        
        // Terminal cost
        let final_state = Vector3::new(
            trajectory[(self.horizon, 0)], 
            trajectory[(self.horizon, 1)], 
            trajectory[(self.horizon, 2)]
        );
        let mut final_error = final_state - reference;
        final_error[2] = Self::normalize_angle(final_error[2]);
        cost += final_error.dot(&(self.qf.component_mul(&final_error)));
        
        // Obstacle avoidance cost
        cost += self.compute_collision_cost(&trajectory);
        
        cost
    }
    
    /// Simple gradient-based optimization (replacing scipy.optimize.minimize)
    pub fn solve(&mut self, current_state: &Vector3<f64>, reference: &Vector3<f64>) -> (Vector2<f64>, DMatrix<f64>) {
        // Initialize control sequence
        let mut controls = if let Some(ref u_prev) = self.u_prev {
            // Warm start: shift previous solution
            let mut u0 = DVector::zeros(self.horizon * 2);
            for i in 0..self.horizon - 1 {
                u0[i * 2] = u_prev[(i + 1) * 2];
                u0[i * 2 + 1] = u_prev[(i + 1) * 2 + 1];
            }
            // Repeat last control
            u0[(self.horizon - 1) * 2] = u_prev[(self.horizon - 1) * 2];
            u0[(self.horizon - 1) * 2 + 1] = u_prev[(self.horizon - 1) * 2 + 1];
            u0
        } else {
            DVector::zeros(self.horizon * 2)
        };
        
        // Simple gradient descent optimization
        let learning_rate = 0.01;
        let max_iterations = 50;
        let epsilon = 1e-6;
        
        for _iter in 0..max_iterations {
            let current_cost = self.cost_function(&controls, current_state, reference);
            
            // Compute numerical gradient
            let mut gradient = DVector::zeros(self.horizon * 2);
            for i in 0..self.horizon * 2 {
                let mut controls_plus = controls.clone();
                controls_plus[i] += epsilon;
                let cost_plus = self.cost_function(&controls_plus, current_state, reference);
                gradient[i] = (cost_plus - current_cost) / epsilon;
            }
            
            // Update controls
            for i in 0..self.horizon * 2 {
                controls[i] -= learning_rate * gradient[i];
                
                // Apply constraints
                if i % 2 == 0 {
                    // Linear velocity
                    controls[i] = controls[i].clamp(self.v_min, self.v_max);
                } else {
                    // Angular velocity
                    controls[i] = controls[i].clamp(-self.omega_max, self.omega_max);
                }
            }
        }
        
        // Store for next iteration
        self.u_prev = Some(controls.clone());
        
        // Extract first control
        let optimal_control = Vector2::new(controls[0], controls[1]);
        
        // Predict trajectory
        let predicted_trajectory = self.predict_trajectory(current_state, &controls);
        
        // Log predicted trajectory
        self.log_trajectory("robot/predicted_path", &predicted_trajectory, [0, 0, 255]);
        
        (optimal_control, predicted_trajectory)
    }
}

/// Scenario 1: Point-to-Point Navigation with Obstacle Avoidance
fn scenario_point_to_point() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 1: Point-to-Point Navigation with Obstacle Avoidance");
    println!("{}", "=".repeat(70));
    
    let mut mpc = MobileRobotMPC::new(0.1, 20);
    mpc.init_logging("mpc_point_to_point")?;
    mpc.set_weights(50.0, 2.0, 0.05);
    
    // Add obstacles
    mpc.add_circular_obstacle(3.0, 2.0, 0.5);
    mpc.add_circular_obstacle(5.5, 3.5, 0.6);
    mpc.add_circular_obstacle(6.0, 2.0, 0.4);
    mpc.add_circular_obstacle(7.0, 1.0, 0.4);
    
    // Initial and goal states
    let mut state = Vector3::new(0.0, 0.0, 0.0);
    let goal = Vector3::new(9.0, 3.0, 0.0);
    
    // Log static elements
    mpc.log_obstacles();
    mpc.log_goal(&goal);
    
    println!("Start: ({:.1}, {:.1})", state[0], state[1]);
    println!("Goal:  ({:.1}, {:.1})", goal[0], goal[1]);
    println!("\nSimulating...");
    
    let max_time = 25.0;
    let mut t = 0.0;
    let mut step = 0;
    let mut actual_path = Vec::new();
    
    while t < max_time {
        // Check if goal reached
        let dx: f64 = state[0] - goal[0];
        let dy: f64 = state[1] - goal[1];
        let dist_to_goal = (dx * dx + dy * dy).sqrt();
        if dist_to_goal < 0.3 {
            println!("\n✓ Goal reached at t={:.1}s!", t);
            break;
        }
        
        // Compute optimal control
        let (control, _pred_traj) = mpc.solve(&state, &goal);
        
        // Log current state and control
        mpc.log_robot_state("robot", &state, step);
        mpc.log_control(&control, step);
        
        // Apply control
        state = mpc.robot_dynamics(&state, &control);
        actual_path.push(state);
        
        t += mpc.dt;
        step += 1;
        
        if step % 30 == 0 {
            println!("  t={:.1}s: pos=({:.2}, {:.2}), heading={:.2} rad, dist={:.2}m", 
                     t, state[0], state[1], state[2], dist_to_goal);
        }
    }
    
    // Log final actual path
    if !actual_path.is_empty() {
        let mut path_matrix = DMatrix::zeros(actual_path.len(), 3);
        for (i, &state) in actual_path.iter().enumerate() {
            path_matrix.set_row(i, &state.transpose());
        }
        mpc.log_trajectory("robot/actual_path", &path_matrix, [255, 255, 0]);
    }
    
    Ok(())
}

/// Scenario 2: Path Following
fn scenario_path_following() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 2: Path Following (Figure-8 Trajectory)");
    println!("{}", "=".repeat(70));
    
    let mut mpc = MobileRobotMPC::new(0.1, 15);
    mpc.init_logging("mpc_path_following")?;
    mpc.set_weights(50.0, 2.0, 0.05);
    
    // Generate figure-8 reference trajectory
    let num_points = 200;
    let mut reference_path = Vec::new();
    
    for i in 0..num_points {
        let t = (i as f64) * 4.0 * PI / (num_points as f64);
        let x = 4.0 * t.sin();
        let y = 2.0 * (2.0 * t).sin();
        let theta = (8.0 * t.cos() * (2.0 * t).cos()) / (4.0 * t.cos().powi(2) + 4.0 * (2.0 * t).cos().powi(2)).sqrt();
        reference_path.push(Vector3::new(x, y, theta));
    }
    
    let mut state = Vector3::new(0.5, 0.5, 0.0);
    
    // Log reference path
    let mut ref_matrix = DMatrix::zeros(reference_path.len(), 3);
    for (i, &ref_point) in reference_path.iter().enumerate() {
        ref_matrix.set_row(i, &ref_point.transpose());
    }
    mpc.log_trajectory("reference_path", &ref_matrix, [255, 0, 255]);
    
    println!("Following figure-8 path...");
    
    let mut actual_path = Vec::new();
    for (i, ref_point) in reference_path.iter().enumerate().take(reference_path.len() - mpc.horizon) {
        let (control, _) = mpc.solve(&state, ref_point);
        
        // Log state and control
        mpc.log_robot_state("robot", &state, i as i64);
        mpc.log_control(&control, i as i64);
        
        state = mpc.robot_dynamics(&state, &control);
        actual_path.push(state);
        
        if i % 40 == 0 {
            let dx: f64 = state[0] - ref_point[0];
            let dy: f64 = state[1] - ref_point[1];
            let tracking_error = (dx * dx + dy * dy).sqrt();
            println!("  Progress: {:.0}%, tracking error: {:.3}m", 
                     100.0 * i as f64 / reference_path.len() as f64, tracking_error);
        }
    }
    
    // Log final actual path
    if !actual_path.is_empty() {
        let mut path_matrix = DMatrix::zeros(actual_path.len(), 3);
        for (i, &state) in actual_path.iter().enumerate() {
            path_matrix.set_row(i, &state.transpose());
        }
        mpc.log_trajectory("robot/actual_path", &path_matrix, [255, 255, 0]);
    }
    
    Ok(())
}

/// Scenario 3: Moving Target Tracking
fn scenario_moving_target() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 3: Moving Target Tracking");
    println!("{}", "=".repeat(70));
    
    let mut mpc = MobileRobotMPC::new(0.1, 20);
    mpc.init_logging("mpc_moving_target")?;
    mpc.set_weights(30.0, 5.0, 0.1);
    
    // Add obstacle
    mpc.add_circular_obstacle(5.0, 5.0, 0.6);
    mpc.log_obstacles();
    
    let mut state = Vector3::new(0.0, 0.0, 0.0);
    
    println!("Tracking moving target...");
    
    let max_time = 20.0;
    let mut t = 0.0;
    let mut step = 0;
    let mut actual_path = Vec::new();
    let mut target_path = Vec::new();
    
    while t < max_time {
        // Update target position (circular motion)
        let target_x = 5.0 + 3.0 * (0.3_f64 * t).cos();
        let target_y = 5.0 + 3.0 * (0.3_f64 * t).sin();
        let target_theta = 0.3 * t + PI / 2.0;
        let target = Vector3::new(target_x, target_y, target_theta);
        
        // Log target position
        mpc.log_goal(&target);
        target_path.push(target);
        
        // Compute control
        let (control, _) = mpc.solve(&state, &target);
        
        // Log robot state and control
        mpc.log_robot_state("robot", &state, step);
        mpc.log_control(&control, step);
        
        // Apply control
        state = mpc.robot_dynamics(&state, &control);
        actual_path.push(state);
        
        t += mpc.dt;
        step += 1;
        
        if step % 30 == 0 {
            let dx: f64 = state[0] - target[0];
            let dy: f64 = state[1] - target[1];
            let dist_to_target = (dx * dx + dy * dy).sqrt();
            println!("  t={:.1}s: distance to target: {:.2}m", t, dist_to_target);
        }
    }
    
    // Log final paths
    if !actual_path.is_empty() {
        let mut path_matrix = DMatrix::zeros(actual_path.len(), 3);
        for (i, &state) in actual_path.iter().enumerate() {
            path_matrix.set_row(i, &state.transpose());
        }
        mpc.log_trajectory("robot/actual_path", &path_matrix, [255, 255, 0]);
    }
    
    if !target_path.is_empty() {
        let mut target_matrix = DMatrix::zeros(target_path.len(), 3);
        for (i, &target) in target_path.iter().enumerate() {
            target_matrix.set_row(i, &target.transpose());
        }
        mpc.log_trajectory("target_path", &target_matrix, [0, 255, 255]);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("  MPC FOR MOBILE ROBOTS - RUST IMPLEMENTATION WITH RERUN");
    println!("{}", "=".repeat(70));
    println!("\nThis demo shows three real-world scenarios:");
    println!("  1. Point-to-point navigation with obstacle avoidance");
    println!("  2. Path following (figure-8 trajectory)");
    println!("  3. Moving target tracking");
    println!("\nVisualization will be shown in Rerun viewer.");
    println!("{}", "=".repeat(70));
    
    // Run scenarios
    scenario_point_to_point()?;
    scenario_path_following()?;
    scenario_moving_target()?;
    
    println!("\n{}", "=".repeat(70));
    println!("All scenarios completed successfully!");
    println!("Check the Rerun viewer for visualizations.");
    println!("{}", "=".repeat(70));
    
    Ok(())
}
