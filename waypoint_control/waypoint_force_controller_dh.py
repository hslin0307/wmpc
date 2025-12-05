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
from array import array as array_t
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState


# ------------------------------- QoS -------------------------------

def _qos_transient_reliable_depth1() -> QoSProfile:
    q = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    return q


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
 
        T = np.array([
            [ct,    -st * ca,  st * sa,  a * ct],
            [st,     ct * ca, -ct * sa,  a * st],
            [0.0,          sa,       ca,      d],
            [0.0,         0.0,      0.0,    1.0],
        ], dtype=float)
        return T

    @classmethod
    def fk(cls, q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        # 為何：後面 Jacobian 需要每節 z 軸與原點座標
        """
        Forward kinematics.
        Returns:
            T0n: 4x4
            origins: [o0..o6], each 3x1
            axes_z:  [z0..z5], joint axes in base frame
        """
        assert q.shape == (6,), "q must be (6,)"
        T = np.eye(4, dtype=float)
        origins: List[np.ndarray] = []
        axes_z: List[np.ndarray] = []

        # o0, z0
        origins.append(T[0:3, 3].copy())
        axes_z.append(T[0:3, 2].copy())

        i = 0
        while i < 6:
            # Multiply the DH
            T = T @ cls.A(cls.a[i], cls.alpha[i], cls.d[i], float(q[i]))
            origins.append(T[0:3, 3].copy())
            axes_z.append(T[0:3, 2].copy())
            i += 1

        # 去掉最後多加的 z_n（Jacobian 只用到 z0..z5）
        axes_z = axes_z[0:6]
        return T, origins, axes_z


    @staticmethod
    def jacobian(origins: List[np.ndarray], axes_z: List[np.ndarray]) -> np.ndarray:
        """
        Geometric Jacobian at EE (frame-0). Revolute joints only.
        Jv[:,i] = z_i × (o_n - o_i)
        Jw[:,i] = z_i
        """
        J = np.zeros((6, 6), dtype=float)
        o_n = origins[-1]

        i = 0
        while i < 6:
            z = axes_z[i]
            o_i = origins[i]
            p = o_n - o_i
            J[0:3, i] = np.cross(z, p)
            J[3:6, i] = z
            i += 1

        return J


# -------------------------- Quaternion utils --------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1 = q1[0]; y1 = q1[1]; z1 = q1[2]; w1 = q1[3]
    x2 = q2[0]; y2 = q2[1]; z2 = q2[2]; w2 = q2[3]

    x = w1 * x2 + y1 * z2 - z1 * y2 + x1 * w2
    y = w1 * y2 + z1 * x2 - x1 * z2 + y1 * w2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w], dtype=float)

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to quaternion (xyzw), numerically stable."""
    m00 = float(R[0, 0]); m01 = float(R[0, 1]); m02 = float(R[0, 2])
    m10 = float(R[1, 0]); m11 = float(R[1, 1]); m12 = float(R[1, 2])
    m20 = float(R[2, 0]); m21 = float(R[2, 1]); m22 = float(R[2, 2])

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
        qe = -qe

    vx = float(qe[0]); vy = float(qe[1]); vz = float(qe[2]); w = float(qe[3])
    v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)

    angle = 2.0 * math.atan2(v_norm, max(min(w, 1.0), -1.0))
    if v_norm < 1e-9 or angle < 1e-9:
        return np.zeros(3, dtype=float)

    ax = vx / v_norm
    ay = vy / v_norm
    az = vz / v_norm

    return np.array([angle * ax, angle * ay, angle * az], dtype=float)




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
        self.declare_parameter("force_limit", 3.0)              # N
        self.declare_parameter("tolerance", 0.003)               #m 
        self.declare_parameter("deadband", 0.01)                 # m/s

        # Parameters (orientation)
        self.declare_parameter("kp_rot", [6.0, 6.0, 6.0])        # N·m/rad
        self.declare_parameter("kd_rot", [0.4, 0.4, 0.4])        # N·m/(rad/s)
        self.declare_parameter("torque_limit", 1.0)              # N·m
        self.declare_parameter("tolerance_rot", 0.03)            # rad (~1.7 deg)
        self.declare_parameter("deadband_rot", 0.1)              # rad/s

        # Misc
        self.declare_parameter("publish_rate_hz", 200.0)         # Hz
        self.declare_parameter("base_frame", "base_link")        # frame id for output

        # Read params
        self.joint_names: List[str] = list(self.get_parameter("joint_names").get_parameter_value().string_array_value)
        az = self.get_parameter("kp_xyz").get_parameter_value().double_array_value or [150.0, 150.0, 150.0]
        
        print(f"type={type(az)}")
        print(f"type={az[0]}")   
        print(f"type={type(az[0])}")
        
        kp_xyz_raw = self.get_parameter("kp_xyz").get_parameter_value().double_array_value or [150.0, 150.0, 150.0]
        kd_xyz_raw = self.get_parameter("kd_xyz").get_parameter_value().double_array_value or [40.0, 40.0, 40.0]
        kp_rot_raw = self.get_parameter("kp_rot").get_parameter_value().double_array_value or [6.0, 6.0, 6.0]
        kd_rot_raw = self.get_parameter("kd_rot").get_parameter_value().double_array_value or [0.4, 0.4, 0.4]

        self.kp_xyz = self._vec3(kp_xyz_raw)
        self.kd_xyz = self._vec3(kd_xyz_raw)
        self.kp_rot = self._vec3(kp_rot_raw)
        self.kd_rot = self._vec3(kd_rot_raw)

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
        ee = T[0:3, 3]
        q_cur = rot_to_quat(R)          # current orientation (xyzw)

        J = UR5eDH.jacobian(origins, axes_z)
        v = J[0:3, :] @ self.qdot   # m/s
        w = J[3:6, :] @ self.qdot   # rad/s

        # Errors
        e_pos = self.goal_pos - ee
        e_rot = quat_orient_error(self.goal_quat, q_cur)

        # Arrival check: pos + rot + velocity deadbands
        if (float(np.linalg.norm(e_pos)) < self.tol and
            float(np.linalg.norm(e_rot)) < self.tol_rot and
            float(np.linalg.norm(v)) < self.deadband and
            float(np.linalg.norm(w)) < self.deadband_rot):
            if self._settle_ticks <= 0:
                self._settle_ticks = 5  # avoid chatter at goal
            self._publish_wrench(0, 0, 0, 0, 0, 0)
            self._settle_ticks -= 1
            if self._settle_ticks <= 0:
                self.goal_pos = None
                self.goal_quat = None
            return

        # PD (position)
        Fx = self.kp_xyz[0] * e_pos[0] - self.kd_xyz[0] * v[0]
        Fy = self.kp_xyz[1] * e_pos[1] - self.kd_xyz[1] * v[1]
        Fz = self.kp_xyz[2] * e_pos[2] - self.kd_xyz[2] * v[2]

        Fx = max(-self.force_limit, min(self.force_limit, Fx))
        Fy = max(-self.force_limit, min(self.force_limit, Fy))
        Fz = max(-self.force_limit, min(self.force_limit, Fz))

        # PD (orientation)
        Tx = self.kp_rot[0] * e_rot[0] - self.kd_rot[0] * w[0]
        Ty = self.kp_rot[1] * e_rot[1] - self.kd_rot[1] * w[1]
        Tz = self.kp_rot[2] * e_rot[2] - self.kd_rot[2] * w[2]

        Tx = max(-self.torque_limit, min(self.torque_limit, Tx))
        Ty = max(-self.torque_limit, min(self.torque_limit, Ty))
        Tz = max(-self.torque_limit, min(self.torque_limit, Tz))

        self._publish_wrench(float(Fx), float(Fy), float(Fz), float(Tx), float(Ty), float(Tz))

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

        # 序列
        if isinstance(val, (list, tuple, np.ndarray, array_t)):
            if isinstance(val, np.ndarray):
                seq = val.tolist()
            else:
                seq = list(val)

            if len(seq) >= 3:
                x = float(seq[0]); y = float(seq[1]); z = float(seq[2])
                return (x, y, z)

            if len(seq) == 1:
                s = float(seq[0])
                return (s, s, s)

            return (0.0, 0.0, 0.0)

        # 單一數值
        s = float(val)
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
