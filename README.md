# wmpc: Waypoint MPC for UR5 with a MATLAB–ROS 2 Bridge

> **TL;DR**  
> This repo connects a **UR5e MATLAB dynamics simulator** with a **ROS 2 waypoint Model Predictive Controller (wMPC)** in Python.  
> You send a TCP goal pose → wMPC plans a joint-space trajectory (double integrator) → a bridge node sends base twist / trajectories to the MATLAB UR5e plant.

---

## 1. Overview

This project implements the waypoint MPC concept from:

- Beck et al., *Model Predictive Trajectory Optimization With Dynamically Changing Waypoints for Serial Manipulators*, IEEE RA-L, 2024. :contentReference[oaicite:0]{index=0}  
- Nguyen et al., *Language-driven Closed-loop Grasping with Model-predictive Trajectory Optimization*, Mechatronics, 2025. :contentReference[oaicite:1]{index=1}  

adapted to a **UR5e** robot model with:

- MATLAB R2024b UR5e plant (full rigid-body dynamics + PD + Jacobian control)
- ROS 2 (Humble) integration via a **MotionExecutor** bridge
- A **double-integrator waypoint MPC** implemented in Python (`wmpc_double_integrator.py`)
- A ROS 2 node `wmpc_node` that:
  - Receives TCP waypoint / goal poses as `geometry_msgs/PoseStamped`
  - Converts them to joint targets via an IK solver
  - Solves wMPC in joint space
  - Sends base-frame twist commands to the MATLAB plant

High-level dataflow:

```text
[ MATLAB ur5e_ros2_plant_bridge.m ]
       ↑              ↑
       | /joint_states|   (sensor_msgs/JointState)
       |              |
       | base twist / joint trajectory / wrench
       |              |
[ waypoint_control.MotionExecutor ]
       ↑
       |
[ waypoint_control.wmpc_node (Waypoint MPC) ]
       ↑                 ↑
       |                 |
 /wmpc/waypoint_pose     /wmpc/goal_pose
 (PoseStamped in base_link frame, TCP pose)

## 2. Repository Structure
wmpc/
├── custom_libraries/
│   └── ik_solver.py            # UR5e IK / FK / Jacobian
├── matlab_r2024b/
│   └── ur5e_ros2_plant_bridge.m  # MATLAB UR5e plant + ROS2 bridge
├── waypoint_control/
│   ├── motion_utils_matlab_bridge.py  # MotionExecutor: ROS <-> MATLAB plant
│   ├── wmpc_double_integrator.py      # Core waypoint MPC (double integrator)
│   └── wmpc_node.py                   # ROS 2 node running wMPC + bridge
├── resource/                          # ROS 2 package resources
├── test/                              # Experiments / tests (placeholder)
├── package.xml                        # ROS 2 package definition
├── setup.cfg / setup.py               # Python package install config
└── README.md

