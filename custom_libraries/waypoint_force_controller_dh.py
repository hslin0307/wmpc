# pkg_path/scripts/waypoint_force_controller_dh.py
#!/usr/bin/env python3
"""
ROS 2 node: Waypoint -> Force/Torque controller using hard-coded DH for UR5e.

- Sub: /joint_states (sensor_msgs/JointState)
- Sub: /waypoint     (geometry_msgs/PoseStamped)   # position + orientation (quaternion, xyzw)
- Pub: /ee_force_command (geometry_msgs/WrenchStamped)

Assumptions
- 6-DOF UR5e kinematics; joint order must match `joint_names` parameter.
- Waypoint is in base frame; quaternion may be non-normalized (we normalize).
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


# ------------------------------- QoS -------------------------------

def _qos_transient_reliable_depth1() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,  # keep last waypoint for late-joiners
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _qos_default_sensor() -> QoSProfile:
    return QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)


# -------------------------- UR5e DH Model --------------------------

class UR5eDH:
    # Kinematics (meters, radians)
    a = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=float)
    d = np.array([0.1625, 0.0,    0.0,    0.1333, 0.0997, 0.0996], dtype=float)
    alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0], dtype=float)

    @staticmethod
    def A(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Standard DH homogeneous transform."""
        sa, ca = math.sin(alpha), math.cos(alpha)
        st, ct = math.sin(theta), math.cos(theta)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0.0,   sa,     ca,    d],
            [0.0,  0.0,    0.0,  1.0],
        ], dtype=float)

    @classmethod
    def fk(cls, q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward kinematics.
        Returns:
            T0n: 4x4
            origins: [o0..o6], each 3x1
            axes_z:  [z0..z5], joint axes in base frame
        """
        assert q.shape == (6,), "q must be (6,)"
        T = np.eye(4)
        origins = [T[0:3, 3].copy()]  # o0 at base
        axes_z = []
        for i in range(6):
            # Save the current coordinate system z axis (joint i axis)
            z_axis = T[0:3, 2].copy()
            axes_z.append(z_axis)
            # Multiply the DH
            T = T @ cls.A(cls.a[i], cls.alpha[i], cls.d[i], q[i])
            origins.append(T[0:3, 3].copy())
        return T, origins, axes_z

    @staticmethod
    def jacobian(origins: List[np.ndarray], axes_z: List[np.ndarray]) -> np.ndarray:
        """
        Geometric Jacobian at EE (frame-0). Revolute joints only.
        Jv[:,i] = z_i × (o_n - o_i)
        Jw[:,i] = z_i
        """
        o_n = origins[-1]
        J = np.zeros((6, 6), dtype=float)
        for i in range(6):
            z = axes_z[i]
            o_i = origins[i]
            J[0:3, i] = np.cross(z, (o_n - o_i))
            J[3:6, i] = z
        return J


# -------------------------- Quaternion utils --------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + y1*z2 - z1*y2 + x1*w2,
        w1*y2 + z1*x2 - x1*z2 + y1*w2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to quaternion (xyzw), numerically stable."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return quat_normalize(np.array([x, y, z, w], dtype=float))

def quat_orient_error(q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """
    Precise attitude error vector (rad): e = angle * axis, angle ∈ [0, π].
    qe = q_des ⊗ q_cur*; if qe.w < 0, negate to take the shortest path.
    """
    qd = quat_normalize(q_des)
    qc = quat_normalize(q_cur)
    qe = quat_mul(qd, quat_conj(qc))
    if qe[3] < 0.0:
        qe = -qe  # shortest path

    v = qe[0:3]
    w = float(qe[3])
    v_norm = float(np.linalg.norm(v))

    # angle = 2*atan2(||v||, w) ∈ [0, π]
    angle = 2.0 * math.atan2(v_norm, max(min(w, 1.0), -1.0))

    if v_norm < 1e-9 or angle < 1e-9:
        return np.zeros(3, dtype=float)

    axis = v / v_norm
    return angle * axis


# --------------------------- Controller Node ---------------------------

class WaypointForcePublisherDH(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_force_controller_dh")

        # Parameters (position)
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("kp_xyz", [150.0, 150.0, 150.0])  # N/m
        self.declare_parameter("kd_xyz", [40.0, 40.0, 40.0])     # N/(m/s)
        self.declare_parameter("force_limit", 80.0)              # N
        self.declare_parameter("tolerance", 0.003)               # m
        self.declare_parameter("deadband", 0.01)                 # m/s

        # Parameters (orientation)
        self.declare_parameter("kp_rot", [6.0, 6.0, 6.0])        # N·m/rad
        self.declare_parameter("kd_rot", [0.4, 0.4, 0.4])        # N·m/(rad/s)
        self.declare_parameter("torque_limit", 8.0)              # N·m
        self.declare_parameter("tolerance_rot", 0.03)            # rad (~1.7 deg)
        self.declare_parameter("deadband_rot", 0.1)              # rad/s

        # Misc
        self.declare_parameter("publish_rate_hz", 200.0)         # Hz
        self.declare_parameter("base_frame", "base_link")        # frame id for output

        # Read params
        self.joint_names: List[str] = list(self.get_parameter("joint_names").get_parameter_value().string_array_value)
        ax = self.get_parameter("kp_xyz").get_parameter_value().double_array_value or [150.0, 150.0, 150.0]
        print(f"type={ax}")
        self.kp_xyz = self._vec3(self.get_parameter("kp_xyz").get_parameter_value().double_array_value or [150.0, 150.0, 150.0])
        self.kd_xyz = self._vec3(self.get_parameter("kd_xyz").get_parameter_value().double_array_value or [40.0, 40.0, 40.0])
        self.kp_rot = self._vec3(self.get_parameter("kp_rot").get_parameter_value().double_array_value or [6.0, 6.0, 6.0])
        self.kd_rot = self._vec3(self.get_parameter("kd_rot").get_parameter_value().double_array_value or [0.4, 0.4, 0.4])
        self.force_limit = float(self.get_parameter("force_limit").value)
        self.torque_limit = float(self.get_parameter("torque_limit").value)
        self.tol = float(self.get_parameter("tolerance").value)
        self.tol_rot = float(self.get_parameter("tolerance_rot").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.deadband_rot = float(self.get_parameter("deadband_rot").value)
        self.rate = float(self.get_parameter("publish_rate_hz").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        # Runtime state
        self.q = np.zeros(6, dtype=float)
        self.qdot = np.zeros(6, dtype=float)
        self.have_state = False

        # Goal: position (3,) and orientation (quat xyzw)
        self.goal_pos: Optional[np.ndarray] = None
        self.goal_quat: Optional[np.ndarray] = None
        self._settle_ticks = 0

        # I/O
        self.pub_force = self.create_publisher(WrenchStamped, "/ee_force_command", _qos_default_sensor())
        self.sub_js = self.create_subscription(JointState, "/joint_states", self._on_joint_state, _qos_default_sensor())
        self.sub_waypoint = self.create_subscription(PoseStamped, "/waypoint", self._on_waypoint, _qos_transient_reliable_depth1())

        dt = 1.0 / max(self.rate, 1.0)
        self.timer = self.create_timer(dt, self._on_timer)

        self.get_logger().info(
            f"DH UR5e controller ready. joints={self.joint_names}, "
            f"Kp_xyz={self.kp_xyz}, Kd_xyz={self.kd_xyz}, F_lim={self.force_limit}N, tol={self.tol}m; "
            f"Kp_rot={self.kp_rot}, Kd_rot={self.kd_rot}, Tau_lim={self.torque_limit}Nm, tol_rot={self.tol_rot}rad; "
            f"deadband(v)={self.deadband} m/s, deadband_rot={self.deadband_rot} rad/s, rate={self.rate}Hz"
        )

        # Diagnostics
        self._printed_joint_map = False

    # ----------------------- Callbacks -----------------------

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            self.get_logger().warning("JointState empty or missing position.")
            return

        name_to_pos = {n: v for n, v in zip(msg.name, msg.position)}
        name_to_vel = {n: v for n, v in zip(msg.name, msg.velocity or [])}

        missing = [jn for jn in self.joint_names if jn not in name_to_pos]
        if missing:
            self.get_logger().warning(f"JointState missing joints: {missing}")
            return

        if not self._printed_joint_map:
            self.get_logger().info("Joint order: " + ", ".join(self.joint_names))
            self.get_logger().info("Incoming names: " + ", ".join(msg.name))
            self._printed_joint_map = True

        vals = [float(name_to_pos[jn]) for jn in self.joint_names]
        vels = [float(name_to_vel.get(jn, 0.0)) for jn in self.joint_names]

        self.q[:] = np.asarray(vals, dtype=float)
        self.qdot[:] = np.asarray(vels, dtype=float)
        self.have_state = True

    def _on_waypoint(self, msg: PoseStamped) -> None:
        # Position
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        # Orientation (xyzw); default to identity if all-zero
        q = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
        if not np.isfinite(q).all() or np.allclose(q, 0.0):
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        q = quat_normalize(q)

        self.goal_pos = p
        self.goal_quat = q
        self._settle_ticks = 0

        frame = msg.header.frame_id or ""
        if frame and frame != self.base_frame:
            self.get_logger().warning(f"Waypoint frame '{frame}' != base_frame '{self.base_frame}'. No TF here.")
        self.get_logger().info(f"New waypoint: p={np.round(self.goal_pos,3)}, q(xyzw)={np.round(self.goal_quat,4)}")

    def _on_timer(self) -> None:
        if not self.have_state or self.goal_pos is None or self.goal_quat is None:
            self._publish_wrench(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        # FK + Jacobian
        T, origins, axes_z = UR5eDH.fk(self.q)
        R = T[0:3, 0:3]
        ee_pos = T[0:3, 3]
        q_cur = rot_to_quat(R)          # current orientation (xyzw)

        J = UR5eDH.jacobian(origins, axes_z)
        v_lin = J[0:3, :] @ self.qdot   # m/s
        omega = J[3:6, :] @ self.qdot   # rad/s

        # Errors
        e_pos = self.goal_pos - ee_pos
        e_rot = quat_orient_error(self.goal_quat, q_cur)  # angle-axis vector (rad)

        err_pos = float(np.linalg.norm(e_pos))
        err_rot = float(np.linalg.norm(e_rot))
        vel_lin = float(np.linalg.norm(v_lin))
        vel_rot = float(np.linalg.norm(omega))

        # Arrival check: pos + rot + velocity deadbands
        if (err_pos < self.tol) and (err_rot < self.tol_rot) and (vel_lin < self.deadband) and (vel_rot < self.deadband_rot):
            if self._settle_ticks <= 0:
                self._settle_ticks = 5  # avoid chatter at goal
            self._publish_wrench(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._settle_ticks -= 1
            if self._settle_ticks <= 0:
                self.get_logger().info("Reached waypoint (pos + ori). Holding.")
                self.goal_pos = None
                self.goal_quat = None
            return

        # PD (position)
        F = np.array([
            self.kp_xyz[0]*e_pos[0] - self.kd_xyz[0]*v_lin[0],
            self.kp_xyz[1]*e_pos[1] - self.kd_xyz[1]*v_lin[1],
            self.kp_xyz[2]*e_pos[2] - self.kd_xyz[2]*v_lin[2],
        ], dtype=float)
        F = np.clip(F, -self.force_limit, self.force_limit)

        # PD (orientation)
        Tau = np.array([
            self.kp_rot[0]*e_rot[0] - self.kd_rot[0]*omega[0],
            self.kp_rot[1]*e_rot[1] - self.kd_rot[1]*omega[1],
            self.kp_rot[2]*e_rot[2] - self.kd_rot[2]*omega[2],
        ], dtype=float)
        Tau = np.clip(Tau, -self.torque_limit, self.torque_limit)

        self._publish_wrench(float(F[0]), float(F[1]), float(F[2]),
                             float(Tau[0]), float(Tau[1]), float(Tau[2]))

    # ----------------------- Helpers -----------------------

    def _publish_wrench(self, fx: float, fy: float, fz: float, tx: float, ty: float, tz: float) -> None:
        now: Time = self.get_clock().now().to_msg()
        msg = WrenchStamped(header=Header(stamp=now, frame_id=self.base_frame))
        msg.wrench.force.x = fx
        msg.wrench.force.y = fy
        msg.wrench.force.z = fz
        msg.wrench.torque.x = tx
        msg.wrench.torque.y = ty
        msg.wrench.torque.z = tz
        self.pub_force.publish(msg)

    @staticmethod
    def _vec3(val) -> Tuple[float, float, float]:
        print(isinstance(val, (list, tuple)))
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return (float(val[0]), float(val[1]), float(val[2]))
        s = float(val) if not isinstance(val, (list, tuple)) else float(val[0])
        return (s, s, s)


def main():
    rclpy.init()
    node = WaypointForcePublisherDH()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__" and __file__.endswith("waypoint_force_controller_dh.py"):
    main()
