use mpc_controller::mpc::MobileRobotMPC;
use nalgebra::Vector3;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mpc = MobileRobotMPC::new(0.1, 20);
    let res = mpc.init_logging("mpc_ros2_node");
    if let Err(e) = res {
        eprintln!("Failed to initialize logging: {}", e);
        std::process::exit(1);
    }
    mpc.set_weights(20.0, 0.2, 0.05);
    mpc.set_physical_constraints(0.0, 0.5, 1.0);

    let mut state = Vector3::new(0.0, 0.0, 0.0);
    let goal = Vector3::new(5.0, 5.0, 0.0);

    for _ in 0..100 {
        let (control, _pred_traj) = mpc.solve(&state, &goal);
        state = mpc.robot_dynamics(&state, &control);
    }
    Ok(())
}