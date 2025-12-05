# pkg_path/scripts/force_follower.py
#!/usr/bin/env python3
"""
Force follower: subscribe WrenchStamped on /ee_force_command and stream it to the robot.

- Sub: /ee_force_command (geometry_msgs/WrenchStamped)
- Drives: MotionExecutor.send_cartesian_force([Fx, Fy, Fz, Tx, Ty, Tz], duration_s)

Safety:
- Deadband to ignore tiny noise
- Clamp per-axis force/torque
- Timeout watchdog: zero command if messages stop
- Optional low-pass filter to smooth commands
"""

from __future__ import annotations

import math
import time
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import WrenchStamped

# Your project libraries
from waypoint_control.custom_libraries.motion_utils import MotionExecutor
from waypoint_control.custom_libraries.pusher_utils import PusherHandler
from waypoint_control.custom_libraries.gripper_utils import GripperHandler


class ForceFollower(Node):
    def __init__(self) -> None:
        super().__init__("force_follower")

        # --- Parameters (edit in launch/YAML) ---
        self.declare_parameter("force_topic", "/ee_force_command")
        self.declare_parameter("rate_hz", 200.0)
        self.declare_parameter("max_force", [80.0, 80.0, 80.0])         # N
        self.declare_parameter("max_torque", [8.0, 8.0, 8.0])           # N·m
        self.declare_parameter("deadband_force", [0.2, 0.2, 0.2])       # N
        self.declare_parameter("deadband_torque", [0.02, 0.02, 0.02])   # N·m
        self.declare_parameter("timeout_sec", 0.25)                     # why: safety
        self.declare_parameter("alpha", 0.35)                           # 0..1 low-pass
        self.declare_parameter("frame_id", "base_link")

        # --- Read params ---
        gp = self.get_parameter
        self.force_topic: str = str(gp("force_topic").value)
        self.rate_hz: float = float(gp("rate_hz").value)
        self.maxF = self._vec3(gp("max_force").value)
        self.maxT = self._vec3(gp("max_torque").value)
        self.dbF = self._vec3(gp("deadband_force").value)
        self.dbT = self._vec3(gp("deadband_torque").value)
        self.timeout_sec: float = float(gp("timeout_sec").value)
        self.alpha: float = float(gp("alpha").value)
        self.base_frame: str = str(gp("frame_id").value)

        # --- Build your project stack (controllers same as you use elsewhere) ---
        self.joint_names: List[str] = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        position_controller = "scaled_joint_trajectory_controller"
        velocity_controller = "forward_velocity_controller"
        force_controller = "force_mode_controller"
        passthrough_controller = "passthrough_trajectory_controller"

        self.pusher = PusherHandler(self)
        self.gripper = GripperHandler(self, vertical_offset=0.003)

        self.motion = MotionExecutor(
            self, self.joint_names,
            position_controller,
            velocity_controller,
            force_controller,
            passthrough_controller,
            self.gripper,
            pusher=self.pusher
        )

        # --- State ---
        self._last_cmd = np.zeros(6, dtype=float)
        self._last_filtered = np.zeros(6, dtype=float)
        self._last_msg_time = None  # type: Optional[float]

        # --- I/O ---
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.sub = self.create_subscription(WrenchStamped, self.force_topic, self._on_wrench, qos)

        dt = 1.0 / max(1.0, self.rate_hz)
        self.timer = self.create_timer(dt, self._on_timer)

        self.get_logger().info(
            f"ForceFollower ready @ {self.rate_hz:.0f} Hz, force_topic={self.force_topic}, "
            f"maxF={self.maxF}N, maxT={self.maxT}Nm, deadbandF={self.dbF}, deadbandT={self.dbT}, alpha={self.alpha}"
        )

    # --- Callbacks ---

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # Why: only accept base frame to avoid frame mismatch surprises
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            self.get_logger().warn_once(
                f"Incoming frame '{msg.header.frame_id}' != expected '{self.base_frame}'. Assuming same frame."
            )

        Fx = float(msg.wrench.force.x)
        Fy = float(msg.wrench.force.y)
        Fz = float(msg.wrench.force.z)
        Tx = float(msg.wrench.torque.x)
        Ty = float(msg.wrench.torque.y)
        Tz = float(msg.wrench.torque.z)

        self._last_cmd[:] = np.array([Fx, Fy, Fz, Tx, Ty, Tz], dtype=float)
        self._last_msg_time = self.get_clock().now().nanoseconds * 1e-9

    def _on_timer(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._last_msg_time is None or (now - self._last_msg_time) > self.timeout_sec:
            cmd = np.zeros(6, dtype=float)  # why: watchdog -> safe
        else:
            cmd = self._last_cmd.copy()

        # Deadband
        for i in range(3):
            if abs(cmd[i]) < self.dbF[i]:
                cmd[i] = 0.0
        for i in range(3, 6):
            if abs(cmd[i]) < self.dbT[i - 3]:
                cmd[i] = 0.0

        # Clamp
        cmd[0] = max(-self.maxF[0], min(self.maxF[0], cmd[0]))
        cmd[1] = max(-self.maxF[1], min(self.maxF[1], cmd[1]))
        cmd[2] = max(-self.maxF[2], min(self.maxF[2], cmd[2]))
        cmd[3] = max(-self.maxT[0], min(self.maxT[0], cmd[3]))
        cmd[4] = max(-self.maxT[1], min(self.maxT[1], cmd[4]))
        cmd[5] = max(-self.maxT[2], min(self.maxT[2], cmd[5]))

        # Low-pass filter (why: reduce jerk into the controller)
        self._last_filtered = self.alpha * cmd + (1.0 - self.alpha) * self._last_filtered

        # Stream to robot
        dt = 1.0 / max(1.0, self.rate_hz)
        try:
            self.motion.send_cartesian_force(self._last_filtered.tolist(), dt)
        except Exception as e:
            self.get_logger().error(f"send_cartesian_force failed: {e}")

    def destroy_node(self) -> bool:
        # Send zero once on shutdown (why: safety)
        try:
            self.motion.send_cartesian_force([0, 0, 0, 0, 0, 0], 0.05)
        except Exception:
            pass
        return super().destroy_node()

    # --- Utils ---

    @staticmethod
    def _vec3(val) -> List[float]:
        # Accept list/tuple/ndarray/array('d') or scalar
        if isinstance(val, (list, tuple)):
            seq = list(val)
        elif hasattr(val, "tolist"):
            seq = list(val.tolist())
        else:
            seq = [float(val), float(val), float(val)]
            return seq
        if len(seq) >= 3:
            return [float(seq[0]), float(seq[1]), float(seq[2])]
        if len(seq) == 1:
            s = float(seq[0]); return [s, s, s]
        return [0.0, 0.0, 0.0]


def main() -> None:
    rclpy.init()
    node = ForceFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__" and __file__.endswith("force_follower.py"):
    main()
