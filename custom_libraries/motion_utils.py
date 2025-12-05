import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped, Wrench, Twist, PoseStamped
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTolerance

from control_msgs.action import FollowJointTrajectory
from ur_msgs.srv import SetForceMode
from std_srvs.srv import Trigger

from custom_libraries.ik_solver import compute_ik, compute_jacobian
from custom_libraries.controller_utils import ControllerManager
from custom_libraries.generallibraries import canonicalize_euler

class MotionExecutor:
    """Bulk container for all ROS2 motion command publishers in three forms: position, velocity, and force."""
    def __init__(self, node: Node, joint_names, 
                 position_controller, velocity_controller, force_controller, 
                 passthrough_controller, gripper, pusher=None):
        self.node = node
        self.joint_names = joint_names
        self.position_controller = position_controller
        self.velocity_controller = velocity_controller
        self.force_controller = force_controller
        self.passthrough_controller = passthrough_controller
        self.controller_manager = ControllerManager(self.node)
        self.gripper = gripper
        self.pusher = pusher
        self.auto_gripper = True
        print(self.auto_gripper)

        # Action IO
        self.traj_client = ActionClient(
            node, FollowJointTrajectory, f'/{self.position_controller}/follow_joint_trajectory'
            )
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
            )
        self.ee_pose_sub = self.node.create_subscription(
            PoseStamped, '/tcp_pose_broadcaster/pose', self.ee_pose_callback, 10
            )
        self.joint_vel_pub = self.node.create_publisher(
            Float64MultiArray, f'/{self.velocity_controller}/commands', 10
            )
        self.commanded_twist_pub = self.node.create_publisher(
            TwistStamped, '/robot/commanded_twist', 10
            )
        self.start_force_client = self.node.create_client(
            SetForceMode, '/force_mode_controller/start_force_mode'
            )
        self.stop_force_client = self.node.create_client(
            Trigger, '/force_mode_controller/stop_force_mode'
            )

        self.controller_manager.prestart_controllers()

        self.joint_positions = None
        self.ee_position = []
        self.ee_quat = []
        self.ee_euler = []

    def joint_state_callback(self, msg: JointState):
        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        if all(name in joint_map for name in self.joint_names):
            self.joint_positions = np.array([joint_map[name] for name in self.joint_names])

    def ee_pose_callback(self, msg: PoseStamped):
        # Currently publishes gripper based on subscribed z coordinate
        # So, if you want to set the gripper to some value, make the EE go down more.
        self.ee_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ee_quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                 msg.pose.orientation.z, msg.pose.orientation.w])
        self.ee_euler = canonicalize_euler(R.from_quat(self.ee_quat).as_euler('xyz', degrees=True))
        if self.auto_gripper:
            # If you wish to avoid gripper contact with the table
            self.gripper.update(self.ee_position)
        if self.pusher is not None:
            self.pusher.update(self.ee_position, self.ee_euler)

    def send_joint_angles(self, joint_angles, seconds=3.0):
        """Send a trajectory goal directly using specified joint angles."""
        return self._send_joint_trajectory(joint_angles, seconds, "Joint angles goal")


    def send_cartesian_position(self, position, rpy_deg, seconds=3.0):
        """Compute IK and send the corresponding joint trajectory."""
        joint_positions = compute_ik(position, rpy_deg)
        if joint_positions is None:
            self.node.get_logger().error("IK failed for Cartesian goal.")
            return False
        return self._send_joint_trajectory(joint_positions.tolist(), seconds, 
                                           f"Cartesian goal: pos={position}, rpy={rpy_deg}")

    def _send_joint_trajectory(self, joint_positions, seconds=3.0, description="Joint goal"):
        self.controller_manager.switch_to_controller(
            [self.position_controller],
            [self.velocity_controller, self.force_controller, self.passthrough_controller]
        )

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start = Duration(sec=int(seconds))
        traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=0, nanosec=500_000_000)
        goal.goal_tolerance = [
            JointTolerance(position=0.01, velocity=0.01, name=name)
            for name in self.joint_names
        ]

        self.traj_client.wait_for_server()
        self.node.get_logger().info(f"Sending {description}: {joint_positions}")
        future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)

        # Goal Response
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error(f"{description} rejected by controller.")
            return False
        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(self.node, result_future)
        self.node.get_logger().info(f"{description} completed.")
        return True

    def send_cartesian_velocity(self, v_cartesian: np.ndarray, duration):
        self.controller_manager.switch_to_controller(
            [self.velocity_controller], 
            [self.position_controller, self.force_controller, self.passthrough_controller]
            )
        start_time = time.time()
        last_progress = start_time
        self.node.get_logger().info(f"Starting velocity phase: {v_cartesian} for {duration}s")
        while rclpy.ok() and (time.time() - start_time) < duration:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            now = time.time()
            elapsed = now - start_time
            # Progress printouts
            if now - last_progress >= 1.0:
                print(f"Velocity segment running: {elapsed:.1f}/{duration:.1f} s")
                last_progress = now

            if self.joint_positions is None:
                self.node.get_logger().warn("Waiting for joint state...")
                continue
            try:
                J = compute_jacobian(self.joint_positions)
                q_dot = np.linalg.pinv(J) @ v_cartesian
            except Exception as e:
                self.node.get_logger().error(f"Jacobian computation failed: {e}")
                continue
            msg = Float64MultiArray()
            msg.data = q_dot.tolist()
            self.joint_vel_pub.publish(msg)
            self.report_velocity_command(v_cartesian[0:3], v_cartesian[3:6])

        self.stop_cartesian_velocity()

    def stop_cartesian_velocity(self):
        self.node.get_logger().info("Zeroing velocity.")
        msg = Float64MultiArray()
        msg.data = [0.0] * len(self.joint_names)
        self.joint_vel_pub.publish(msg)
        self.report_velocity_command([0.0] * 3, [0.0] * 3)

    def send_cartesian_velocity_trajectory(self, v_traj: list[np.ndarray], dt: float):
        self.controller_manager.switch_to_controller(
            [self.velocity_controller],
            [self.position_controller, self.force_controller, self.passthrough_controller]
        )
        self.node.get_logger().info(f"Starting continuous velocity trajectory with {len(v_traj)} steps")
        
        for v_cartesian in v_traj:
            if not rclpy.ok():
                break
            try:
                J = compute_jacobian(self.joint_positions)
                q_dot = np.linalg.pinv(J) @ v_cartesian
                msg = Float64MultiArray()
                msg.data = q_dot.tolist()
                self.joint_vel_pub.publish(msg)
                self.report_velocity_command(v_cartesian[0:3], v_cartesian[3:6])
            except Exception as e:
                self.node.get_logger().error(f"Jacobian computation failed: {e}")
            time.sleep(dt)
        
        self.stop_cartesian_velocity()

    def report_velocity_command(self, lin, ang):
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.node.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base'
        twist_msg.twist.linear.x, twist_msg.twist.linear.y, twist_msg.twist.linear.z = lin
        twist_msg.twist.angular.x, twist_msg.twist.angular.y, twist_msg.twist.angular.z = ang
        self.commanded_twist_pub.publish(twist_msg)

    def send_cartesian_force(self, wrench: list, duration, 
                        selection_vector = [True]*6,
                        vel_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5], 
                        pos_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5], 
                        damping=0.025, gain=0.5):
        self.send_cartesian_force_step(wrench, duration, selection_vector=selection_vector,
                                        vel_limits=vel_limits, pos_limits=pos_limits, damping=damping,
                                        gain=gain)
        self.stop_force_mode()
        return True

    def stop_force_mode(self):
        if not self.stop_force_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error("Force mode stop service not available.")
            return False

        future = self.stop_force_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()
        if not result or not result.success:
            self.node.get_logger().error("Failed to stop force mode.")
            return False

        self.node.get_logger().info("Force mode stopped.")
        return True
    
    def send_cartesian_force_step(self, wrench: list, duration, 
                        selection_vector = [True]*6,
                        vel_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5], 
                        pos_limits = [0.25, 0.25, 0.25, 0.5, 0.5, 0.5], 
                        damping=0.025, gain=0.5):
        """Activate UR Force Mode through service."""
        self.controller_manager.switch_to_controller([self.force_controller, self.passthrough_controller],
                                  [self.position_controller, self.velocity_controller])

        if not self.start_force_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error("Force mode start service not available.")
            return False

        req = SetForceMode.Request()
        task_frame = PoseStamped()
        task_frame.header.frame_id = "base"
        task_frame.pose.orientation.w = 1.0  # Identity rotation

        req.task_frame = task_frame
        req.selection_vector_x = selection_vector[0]
        req.selection_vector_y = selection_vector[1]
        req.selection_vector_z = selection_vector[2]
        req.selection_vector_rx = selection_vector[3]
        req.selection_vector_ry = selection_vector[4]
        req.selection_vector_rz = selection_vector[5]

        req.wrench = Wrench()
        req.wrench.force.x, req.wrench.force.y, req.wrench.force.z = wrench[0:3]
        req.wrench.torque.x, req.wrench.torque.y, req.wrench.torque.z = wrench[3:6]

        req.type = 2  # Force frame not transformed
        req.speed_limits = Twist()
        req.speed_limits.linear.x, req.speed_limits.linear.y, req.speed_limits.linear.z = vel_limits[0:3]
        req.speed_limits.angular.x, req.speed_limits.angular.y, req.speed_limits.angular.z = vel_limits[3:6]

        req.deviation_limits = pos_limits # Keeping it the same; don't care too much
        req.damping_factor = damping
        req.gain_scaling = gain

        future = self.start_force_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        result = future.result()
        if not result or not result.success:
            self.node.get_logger().error(f"Failed to start force mode: {getattr(result, 'message', '')}")
            return False

        self.node.get_logger().info("Force mode activated.")
        time.sleep(duration)
        return True

    def send_cartesian_force_trajectory(
        self,
        wrench_traj: list[list[float]],
        dt: float,
        selection_vector=[True]*6,
        vel_limits=[0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        pos_limits=[0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        damping=0.025,
        gain=0.5
    ):
        """Apply a sequence of wrenches continuously without stopping force mode between steps."""
        self.controller_manager.switch_to_controller(
            [self.force_controller, self.passthrough_controller],
            [self.position_controller, self.velocity_controller]
        )

        if not self.start_force_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error("Force mode start service not available.")
            return False

        self.node.get_logger().info(f"Starting continuous force trajectory with {len(wrench_traj)} steps")

        # Apply the sequence
        for wrench in wrench_traj:
            if not rclpy.ok():
                break

            # Send the new wrench to the topic or service your force mode listens to
            msg = Wrench()
            msg.force.x, msg.force.y, msg.force.z = wrench[0:3]
            msg.torque.x, msg.torque.y, msg.torque.z = wrench[3:6]
            self.send_cartesian_force_step(wrench, dt, selection_vector=selection_vector,
                                           vel_limits=vel_limits, pos_limits=pos_limits, damping=damping,
                                           gain=gain)

        # Stop force mode cleanly
        self.stop_force_mode()
        self.node.get_logger().info("Completed continuous force trajectory.")
        return True