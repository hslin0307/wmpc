# wMPC: Waypoint MPC for UR5 with a MATLAB–ROS 2 Bridge

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
```

---

## 2. Repository Structure

```text
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
```

---

## 3. Theory 

### 3.1 State and Control

We use a **joint-space double integrator** model for the UR5e:

- **State**

$$
x =
\begin{bmatrix}
q \\
\dot q
\end{bmatrix}
\in \mathbb{R}^{2m},\quad m = 6
$$

- **Control**

$$
u = \ddot q \in \mathbb{R}^m
$$

This is obtained by assuming a lower-level inverse dynamics / PD law compensates the nonlinear dynamics (mass, Coriolis, gravity), so the closed-loop behaves approximately as $\ddot q = u$. 

We intentionally use **double** instead of the **triple integrator** from the original wMPC paper, because our real UR5e did not have jerk control.

### 3.2 FOH Discretization (Double Integrator)

We assume **First-Order Hold (FOH)** on the input:

$$
u(t) = u_k + \frac{t}{h},(u_{k+1}-u_k), \quad t \in [0,h],
$$

with sampling time (h). Integrating the double-integrator dynamics over $[kh,(k+1)h]$ yields (per joint):

$$
\begin{aligned}
q_{k+1}   &= q_k + h\dot q_k + \tfrac{h^2}{3}u_k + \tfrac{h^2}{6}u_{k+1},\
\dot q_{k+1} &= \dot q_k + \tfrac{h}{2}u_k + \tfrac{h}{2}u_{k+1}.
\end{aligned}
$$

Matrix form:

$$
x_{k+1} = \Phi x_k + \Gamma_1 u_k + \Gamma_2 u_{k+1},
$$

where

$$
\Phi=
\begin{bmatrix}
I & hI\
0 & I
\end{bmatrix},\quad
\Gamma_1=
\begin{bmatrix}
\frac{h^2}{3}I\
\frac{h}{2}I
\end{bmatrix},\quad
\Gamma_2=
\begin{bmatrix}
\frac{h^2}{6}I\
\frac{h}{2}I
\end{bmatrix}.
$$

The code in `wmpc_double_integrator.py` constructs these matrices and uses them in the MPC constraints.

### 3.3 Waypoints and Horizon Splitting

We consider:

* A joint-space waypoint $q_w$
* A joint-space goal $q_g$

The core idea from Beck et al.:

* Plan with a **short receding horizon** of length $N$
* As soon as $q_w$ becomes reachable within the horizon (up to tolerance $\varepsilon$), split the horizon at index $N_s$
* First part $(0 \dots N_s-1)$: cost-to-go and possibly terminal constraints towards $q_w$
* Second part $(N_s \dots N-1)$: cost-to-go and terminal constraints towards $q_g$
* Gradually shrink $N$ as $q_g$ comes into the horizon to avoid tail oscillations 

We use a smooth 1-norm (“smooth L1”) cost in joint space to encourage more time-optimal profiles while keeping optimization well-behaved near waypoints and goal.

### 3.4 Constraints

For each MPC iteration:

- **Dynamics:**

$$
x_{k+1} = \Phi x_k + \Gamma_1 u_k + \Gamma_2 u_{k+1}
$$

- **Initial condition:**

$$
x_0 = [q_\text{meas}, \dot q_\text{meas}]
$$

- **Terminal “steady state”:**

$$
x_{N-1} = \Phi x_{N-1},\quad u_{N-1} = 0
$$

- **Box constraints:**

$$
q_{\min} \le q_k \le q_{\max},\
\dot q_{\min} \le \dot q_k \le \dot q_{\max},\
\ddot q_{\min} \le u_k \le \ddot q_{\max}
$$

- **Optional terminal sets:**

  * If waypoint/goal is reachable: enforce $|q_{N_s-1,i} - q_{w,i}| \le \varepsilon$ and/or $|q_{N-1,i} - q_{g,i}| \le \varepsilon$

---

## 4. Implementation Details

### 4.1 IK, FK, Jacobian (`custom_libraries/ik_solver.py`)

Key functions: 

* `forward_kinematics(dh_params, joint_angles) -> 4x4`
  UR5e forward kinematics from DH parameters.
* `compute_ik(position, rpy_deg, q_guess=None)`

  * Converts RPY (deg) to rotation matrix
  * Builds a 4×4 target pose
  * Solves a bounded L-BFGS-B optimization problem that minimizes position + rotation error.
* `compute_jacobian(joint_angles)`

  * Computes 6×6 geometric Jacobian at the TCP for given `q`.

Used in the pipeline:

* `wmpc_node.py`:

  * **IK**: convert incoming TCP `PoseStamped` to joint-space waypoint/goal.
  * **Jacobian**: map joint velocities to base twist.

---

### 4.2 MATLAB UR5e plant (`matlab_r2024b/ur5e_ros2_plant_bridge.m`)

This script builds a UR5e simulation in MATLAB:

* Imports URDF (`$HOME/ur5e.urdf`) and sets gravity
* Creates ROS 2 node `matlab_ur5e_plant` and topics:

  | Topic                                                  | Type                                           | Role                              |
  | ------------------------------------------------------ | ---------------------------------------------- | --------------------------------- |
  | `/scaled_joint_trajectory_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory`              | Position control / point-to-point |
  | `/forward_velocity_controller/commands`                | `Float64MultiArray` / `Twist` / `TwistStamped` | Base twist control                |
  | `/force_mode_controller/commands`                      | `Wrench` / `WrenchStamped`                     | Force control                     |
  | `/joint_states`                                        | `sensor_msgs/JointState`                       | Plant joint state feedback        |

Control modes inside the loop:

* **TRAJ**: PD control tracking a joint trajectory
* **TWIST**: base twist → Jacobian → desired joint velocity → velocity PD
* **WRENCH**: wrench → Jacobianᵀ → joint torques
* **HOLD**: no command → hold current `q` as `q_hold`, damp velocity to zero

Dynamics use `massMatrix`, `velocityProduct`, `gravityTorque` from MATLAB’s Robotics System Toolbox.

---

### 4.3 Motion Bridge (`waypoint_control/motion_utils_matlab_bridge.py`)

`MotionExecutor` provides a ROS-side abstraction over the MATLAB plant topics:

* **Constructor**:

  * Takes joint names, controller names, QoS, etc.
  * Subscribes to `/joint_states` for seeding IK and FK.
* **APIs**:

  * `send_tcp_pose(xyz_m, rpy_deg, seconds)`

    * Calls `compute_ik` → `q`
    * Publishes a single-point `JointTrajectory`
  * `send_tcp_velocity(v_tcp6, seconds)`

    * Uses FK to get current TCP pose
    * Applies adjoint transform to convert TCP twist ➜ base twist
    * Streams base twist to velocity controller
  * `send_base_velocity(v_base6, seconds)`

    * **Added for wMPC**: directly streams base twist (already in base frame)
  * `send_tcp_force(...)` / `send_cartesian_force(...)`
  * `send_joint_angles(q, seconds)`

    * Directly sends a joint trajectory (no heuristics here in this repo)

Internally, `_publish_velocity_base` and `_publish_force_base` handle high-rate streaming at `rate_hz` (default 200 Hz).

---

### 4.4 MPC Core (`waypoint_control/wmpc_double_integrator.py`)

Key components:

* `JointLimits`
* `WMPCConfig`
* `foh_double_integrator(dof, h)` – builds FOH matrices
* `smooth_l1(q, q_target, gamma)` – smooth 1-norm
* `WaypointMPC`:

  * Builds a CVXPY optimization problem with:

    * Decision variables: `x[k], u[k]` (k=0…N−1)
    * Constraints: dynamics, box constraints, terminal conditions
    * Cost: waypoint + goal cost + input regularization + (optional) collision cost
  * Uses the previous solution as warm-start
  * Returns `(u0, x_opt, u_opt)` where `u0` is the first control to apply

---

### 4.5 ROS 2 Node (`waypoint_control/wmpc_node.py`)

This node runs the closed-loop controller between ROS and MATLAB.

#### 4.5.1 Subscriptions

* `/joint_states` (`sensor_msgs/JointState`)

  * Maintains `self.q`, `self.qd` in the correct joint order.
* `/wmpc/waypoint_pose` (`PoseStamped`)

  * Converts pose → joint-space `q_waypoint` via IK
  * Sets `have_waypoint_tcp = True`
* `/wmpc/goal_pose` (`PoseStamped`)

  * Converts pose → joint-space `q_goal` via IK
  * Sets `have_goal_tcp = True` and **enables MPC** (`mpc_enabled = True`)

#### 4.5.2 MPC Step

Every `h` seconds (`self.h ≈ 0.05 s`), `spin_loop()`:

1. Calls `rclpy.spin_once` to process callbacks.
2. If `mpc_enabled=False` or no joint state yet → do nothing (robot stays in HOLD).
3. If enabled:

   * Call:

     ```python
     u0, x_opt, u_opt = self.mpc.solve(q_meas=q, qd_meas=qd,
                                       q_w=self.q_waypoint,
                                       q_g=self.q_goal)
     ```
   * Integrate:

     $\dot q_\text{cmd} = \dot q + h,u_0$
     
     clipped to UR5e velocity limits.
     
   * Compute base twist:

     ```python
     J = compute_jacobian(q)
     v_base = J @ qd_cmd
     ```

     Then clip linear / angular speed to match MATLAB plant limits.
   * Send command:

     ```python
     self.motion.send_base_velocity(v_base.tolist(), self.h)
     ```

Result: UR5e plant in MATLAB receives continuous base twist commands in TWIST mode, executing the wMPC-generated trajectory.

---

## 5. How to Run

### 5.1 Prerequisites

* **ROS 2** (tested with Humble)
* **Python 3** with:

  * `numpy`
  * `cvxpy`
  * `scipy`
* **MATLAB R2024b** (or similar) with:

  * Robotics System Toolbox
  * ROS Toolbox
* UR5e URDF at: `$HOME/ur5e.urdf`

### 5.2 Build the ROS 2 Package

In your ROS 2 workspace (e.g., `~/ros2_ws`):

```bash
cd ~/ros2_ws/src
git clone https://github.com/hsilin0307/wmpc.git
cd ..
colcon build --packages-select wmpc
source install/setup.bash
```

### 5.3 Start MATLAB plant

In MATLAB:

```matlab
cd <path-to-repo>/matlab_r2024b
ur5e_ros2_plant_bridge   % or run the script
```

You should see:

* A UR5e figure window
* Console messages indicating ROS 2 node creation and `/joint_states` publishing

### 5.4 Start the wMPC ROS 2 node

In a terminal:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run wmpc wmpc_node
```

The node will:

* Subscribe to `/joint_states`
* Wait in **idle mode** (`mpc_enabled=False`) until it receives a goal pose

### 5.5 Send a TCP goal (start motion)

Example (UR5e TCP goal at `[0.1, -0.5, 0.247]`, RPY `[0°, 180°, 0°]` in `base_link`):

```bash
ros2 topic pub /wmpc/goal_pose geometry_msgs/PoseStamped "{
  header: {frame_id: 'base_link'},
  pose: {
    position: {x: 0.1, y: -0.5, z: 0.247},
    orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}
  }
}" -1
```

* This calls `goal_pose_cb` → IK → `q_goal`
* Sets `mpc_enabled=True`
* From the next MPC update, the node starts sending base twists and the arm moves.

Optional: send a waypoint:

```bash
ros2 topic pub /wmpc/waypoint_pose geometry_msgs/PoseStamped "{
  header: {frame_id: 'base_link'},
  pose: {
    position: {x: 0.1, y: -0.5, z: 0.35},
    orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}
  }
}" -1
```

Then the trajectory will be shaped to first pass near this pre-grasp waypoint before reaching the final goal.

---

## 6. Typical Experimental Flow

1. **Launch plant**: Start MATLAB `ur5e_ros2_plant_bridge.m`.
2. **Launch controller**: Run `ros2 run wmpc wmpc_node`.
3. **Send goal**: Publish `/wmpc/goal_pose` (and optional `/wmpc/waypoint_pose`).
4. **Observe**:

   * MATLAB figure shows the UR5e following a smooth joint-space trajectory.
   * You can log `trajLog` in MATLAB or record `/joint_states` via `ros2 bag`.

For finer tuning:

* Adjust MPC parameters in `WMPCConfig` (horizon, tolerance, weights).
* Adjust joint limits in `JointLimits`.
* Adjust PD gains `Kp`, `Kd` in `ur5e_ros2_plant_bridge.m`.

---

## 7. Extending the Project

* **Joint-space only MPC**
  Ignore Pose topics and directly set `q_waypoint`, `q_goal` to joint configurations.
* **Collision avoidance**
  Add collision cost `l_col(x_k)` to the MPC (distance fields, capsules, etc.) as in the original wMPC paper. 
* **Real robot**
  Replace the MATLAB ROS topics in `motion_utils_matlab_bridge.py` with your hardware controller topics.
* **Language-driven multi-waypoint planning**
  Follow the structure in the language-driven grasping paper: language → perception → object pose / pre-grasp waypoints → sequence of waypoint/goal updates for wMPC. 

---

## 8. Citation

If you use this repository in scientific work, please consider citing:

* F. Beck et al., *Model Predictive Trajectory Optimization With Dynamically Changing Waypoints for Serial Manipulators*, IEEE RA-L, 2024. 
* H. H. Nguyen et al., *Language-driven Closed-loop Grasping with Model-predictive Trajectory Optimization*, Mechatronics, 2025. 

