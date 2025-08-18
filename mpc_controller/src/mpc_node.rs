// 
// MPC Node
//
// Author: Sachin Kumar
// Date: 17-08-2025
//

use anyhow::{Error, Result};
use rclrs::*;
// use mpc_controller::mpc::MpcNode;

use std::sync::Arc;

use rclrs::{Node, Publisher, Subscription, Time, QOS_PROFILE_DEFAULT};
use std_msgs::msg::String;

#[allow(unused)]
pub struct MpcNode {
    node: Node,
    pub msg_pub: Arc<Publisher<String>>,
    msg_sub: Arc<Subscription<String>>,
}

impl MpcNode {
    pub fn new(node: Node) -> Self {
        let msg_pub = Arc::new(node.create_publisher::<String>("msg_pub").unwrap());
        let msg_sub = Arc::new(
            node.create_subscription::<String, _>("topic", move |msg: std_msgs::msg::String| {
                println!("I heard: '{}'", msg.data);
            })
            .unwrap(),
        );
        MpcNode {
            node,
            msg_pub,
            msg_sub,
        }
    }
}

fn main() -> Result<(), Error> {
    let context = Context::default_from_env()?;
    let executor = context.create_basic_executor();

    let node = executor.create_node("mpc_controller")?;

    let mpc = MpcNode::new(node);

    let mut message = std_msgs::msg::String::default();
    message.data = "learning rust".to_string();

    // let mut publish_count: u32 = 1;

    while context.ok() {
        // message.data = format!("Hello, world! {}", publish_count);
        println!("Publishing: [{}]", message.data);
        mpc.msg_pub.publish(&message)?;
        // publish_count += 1;
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(())
}

