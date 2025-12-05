# file: waypoint_control/motion_utils_matlab_bridge.py
# SPDX-License-Identifier: Apache-2.0
"""
Matlab Bridge MotionExecutor (TCP-first, lab-aligned):
- 對外一律以 TCP (tool0) frame 提供目標：pose / velocity / force
- 內部以你們提供的 ik_solver 做 IK 與 FK，並將 TCP 量轉為 BASE 再發到 MATLAB plant
- 無夾爪；保留與舊版相容的函式名稱
"""

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Literal, List, Tuple
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Wrench, WrenchStamped, Twist, TwistStamped

# === Lab IK/FK (from your ik_solver.py) ===
# compute_ik(position[R^3], rpy_deg[R^3], q_guess=None) -> np.ndarray(6) or None
# forward_kinematics(dh_params, q) -> 4x4 homogeneous
from custom_libraries.ik_solver import compute_ik, forward_kinematics, dh_params  # type: ignore

VelMsgType = Literal['array', 'twist', 'twist_stamped']
ForceMsgType = Literal['wrench', 'wrench_stamped']


# ----------------- small math helpers -----------------
def _deg2rad(v: Sequence[float]) -> List[float]:
    return [math.radians(float(x)) for x in v]

def _rotm_from_rpy_deg(rpy_deg: Sequence[float]) -> List[List[float]]:
    r, p, y = _deg2rad(rpy_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # ZYX
    return [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ]

def _skew(p: Sequence[float]) -> List[List[float]]:
    x, y, z = p
    return [[0, -z,  y],
            [z,  0, -x],
            [-y, x,  0]]

def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    m, n, k = len(A), len(B[0]), len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]

def _matvec(A: List[List[float]], v: Sequence[float]) -> List[float]:
    return [sum(A[i][j]*v[j] for j in range(len(v))) for i in range(len(A))]

def _adjoint(R: List[List[float]], p: Sequence[float]) -> List[List[float]]:
    # Adj(base<-tcp) = [[R, 0],[p^ R, R]]
    Z = [[0.0, 0.0, 0.0] for _ in range(3)]
    p_hat = _skew(p)
    pR = _matmul(p_hat, R)
    upper = [R[0]+Z[0], R[1]+Z[1], R[2]+Z[2]]
    lower = [pR[0]+R[0], pR[1]+R[1], pR[2]+R[2]]
    return upper + lower

def _transpose(A: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*A)]

def _solve_6x6(A: List[List[float]], b: Sequence[float]) -> List[float]:
    """Tiny dense 6x6 solver (Gauss-Jordan, no pivots since A well-conditioned here)."""
    n = 6
    M = [list(map(float, A[i] + [b[i]])) for i in range(n)]
    for i in range(n):
        # pivot
        piv = M[i][i] if abs(M[i][i]) > 1e-12 else 1e-12
        inv = 1.0 / piv
        for j in range(i, n+1):
            M[i][j] *= inv
        # eliminate
        for r in range(n):
            if r == i: continue
            factor = M[r][i]
            for c in range(i, n+1):
                M[r][c] -= factor * M[i][c]
    return [M[i][n] for i in range(n)]


class MotionExecutor:
    """
    TCP-first MotionExecutor for MATLAB plant.
    - 外部所有接口：TCP frame（tool0）
    - 內部：IK/FK + Adjoint 轉成 BASE，再發到三個 topics
    - 無夾爪（gripper 參數僅保留簽名，不使用）
    """

    def __init__(
        self,
        node: Node,
        joint_names: Sequence[str],
        position_controller: str,
        velocity_controller: str,
        force_controller: str,
        passthrough_controller: Optional[str],
        gripper,  # ignored
        *,
        pusher=None,
        rate_hz: float = 200.0,
        vel_msg_type: VelMsgType = 'array',         # 'array' | 'twist' | 'twist_stamped'
        force_msg_type: ForceMsgType = 'wrench',    # 'wrench' | 'wrench_stamped'
        tcp_frame: str = 'tool0',
        explicit_topics: Optional[dict] = None
    ) -> None:

        self.node = node
        self.joint_names = list(joint_names)
        self.rate_hz = float(rate_hz)
        self.vel_msg_type = vel_msg_type
        self.force_msg_type = force_msg_type
        self.tcp_frame = tcp_frame
        self.gripper = None
        self.pusher = pusher

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        topics = {
            'traj': f'/{position_controller}/joint_trajectory',
            'vel':  f'/{velocity_controller}/commands',
            'force': f'/{force_controller}/commands',
            'js': '/joint_states',
        }
        if explicit_topics:
            topics.update({k: v for k, v in explicit_topics.items() if v})

        # publishers
        self.pub_traj = node.create_publisher(JointTrajectory, topics['traj'], qos)
        if self.vel_msg_type == 'array':
            self.pub_vel = node.create_publisher(Float64MultiArray, topics['vel'], qos)
        elif self.vel_msg_type == 'twist':
            self.pub_vel = node.create_publisher(Twist, topics['vel'], qos)
        else:
            self.pub_vel = node.create_publisher(TwistStamped, topics['vel'], qos)

        if self.force_msg_type == 'wrench':
            self.pub_force = node.create_publisher(Wrench, topics['force'], qos)
        else:
            self.pub_force = node.create_publisher(WrenchStamped, topics['force'], qos)

        # subscribe joint_states for IK seed & FK
        self._q = [0.0]*len(self.joint_names)
        self._q_lock = threading.Lock()
        self.js_sub = node.create_subscription(JointState, topics['js'], self._on_joint_state, qos)

        # fallback: record last commanded TCP
        self._last_tcp_R = [[1,0,0],[0,1,0],[0,0,1]]
        self._last_tcp_p = [0.45, 0.0, 0.25]

        self.node.get_logger().info(
            f"[MatlabBridge TCP] traj={topics['traj']} vel={topics['vel']} force={topics['force']} "
            f"(vel_msg={self.vel_msg_type}, force_msg={self.force_msg_type}, rate={self.rate_hz}Hz, tcp={self.tcp_frame})"
        )

    # --------------------- Public TCP-first APIs --------------------------

    def send_tcp_pose(self, xyz_m: Sequence[float], rpy_deg: Sequence[float], seconds: float) -> None:
        """TCP pose -> IK -> JointTrajectory."""
        pos = [float(x) for x in xyz_m]
        rpy = [float(x) for x in rpy_deg]
        q_seed = self._get_q()
        q = compute_ik(pos, rpy, q_guess=q_seed)  # uses your solver
        if q is None:
            self.node.get_logger().error("[MatlabBridge] IK failed for TCP pose.")
            return
        self._publish_traj(q, seconds)
        self._last_tcp_R = _rotm_from_rpy_deg(rpy)
        self._last_tcp_p = pos

    def send_tcp_velocity(self, v_tcp6: Sequence[float], seconds: float) -> None:
        """TCP twist (6D) -> Adjoint -> BASE twist -> publish."""
        v_tcp = [float(x) for x in v_tcp6]  # [vx vy vz wx wy wz] in TCP
        R, p = self._base_from_fk()
        Adj = _adjoint(R, p)
        v_base = _matvec(Adj, v_tcp)
        self._publish_velocity_base(v_base, seconds)

    def send_base_velocity(self, v_base6: Sequence[float], seconds: float) -> None:
        """BASE twist (6D) -> 直接 publish 到 velocity controller.

        wMPC 這種已經在 BASE frame 設計好的控制律可以呼叫這個函式，
        避免再做一次 TCP→BASE 的轉換。
        """
        v = [float(x) for x in v_base6]
        if len(v) != 6:
            self.node.get_logger().warn(
                f"[MatlabBridge] send_base_velocity expects 6 values, got {len(v)}"
            )
            return
        self._publish_velocity_base(v, seconds)


    def send_tcp_force(self, f_tcp6: Sequence[float], seconds: float) -> None:
        """TCP wrench (6D) -> (Adj^T)^{-1} -> BASE wrench -> publish."""
        f_tcp = [float(x) for x in f_tcp6]
        R, p = self._base_from_fk()
        Adj = _adjoint(R, p)
        # Solve Adj^T * f_base = f_tcp  -> f_base
        f_base = _solve_6x6(_transpose(Adj), f_tcp)
        self._publish_force_base(f_base, seconds)

    # ----------------- Backward-compatible wrappers -----------------------

    def send_joint_angles(self, arg: Iterable[float], seconds: float) -> None:
        vals = [float(x) for x in arg]
        if len(vals) == 6:
            self._publish_traj(vals, seconds)
            # # heuristic：像角度就當關節角；不然視為 [x y z r p y_deg] 的 TCP 姿態
            # if any(abs(v) > 3.5 for v in vals):    # not angles -> treat as TCP pose
            #     self.send_tcp_pose(vals[:3], vals[3:], seconds)
            # else:
            #     self._publish_traj(vals, seconds)
        else:
            self.node.get_logger().warn("[MatlabBridge] send_joint_angles expects 6 values.")

    def send_cartesian_velocity(self, velocity_6d: Sequence[float], seconds: float) -> None:
        self.send_tcp_velocity(velocity_6d, seconds)

    def send_cartesian_force(self, force_6d: Sequence[float], seconds: float) -> None:
        self.send_tcp_force(force_6d, seconds)

    # --------------------------- internals --------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        m = {n: p for n, p in zip(msg.name, msg.position)}
        new_q = []
        miss = []
        for n in self.joint_names:
            if n in m: new_q.append(float(m[n]))
            else:      miss.append(n); new_q.append(0.0)
        if miss:
            self.node.get_logger().warn_once(f"[MatlabBridge] joint_states missing: {miss}")
        with self._q_lock:
            self._q = new_q

    def _get_q(self) -> List[float]:
        with self._q_lock:
            return list(self._q)

    def _base_from_fk(self) -> Tuple[List[List[float]], List[float]]:
        """FK(q) -> BASE←TCP: (R, p)"""
        q = self._get_q()
        try:
            T = forward_kinematics(dh_params, q)  # 4x4 numpy
            R = [[float(T[0,0]), float(T[0,1]), float(T[0,2])],
                 [float(T[1,0]), float(T[1,1]), float(T[1,2])],
                 [float(T[2,0]), float(T[2,1]), float(T[2,2])]]
            p = [float(T[0,3]), float(T[1,3]), float(T[2,3])]
            return R, p
        except Exception:
            # fallback to last commanded TCP
            return self._last_tcp_R, self._last_tcp_p

    # ---- publishers ----
    def _publish_traj(self, q: Sequence[float], seconds: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q]
        sec = int(seconds); nsec = int((seconds - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        msg.points = [pt]
        self.pub_traj.publish(msg)
        self.node.get_logger().info(f"[MatlabBridge] Traj T={seconds:.3f}s q={[f'{x:.3f}' for x in q]}")

    def _publish_velocity_base(self, v6: Sequence[float], seconds: float) -> None:
        v = [float(x) for x in v6]
        end_t = time.monotonic() + float(seconds)
        period = 1.0 / self.rate_hz
        while time.monotonic() < end_t and rclpy.ok():
            if self.vel_msg_type == 'array':
                self.pub_vel.publish(Float64MultiArray(data=v))
            elif self.vel_msg_type == 'twist':
                m = Twist()
                m.linear.x, m.linear.y, m.linear.z = v[:3]
                m.angular.x, m.angular.y, m.angular.z = v[3:]
                self.pub_vel.publish(m)
            else:
                m = TwistStamped()
                m.header.frame_id = 'base_link'
                m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = v[:3]
                m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = v[3:]
                self.pub_vel.publish(m)
            time.sleep(period)
        self.node.get_logger().info(f"[MatlabBridge] Vel(base) v={[f'{x:.3f}' for x in v]} dur={seconds:.3f}s")

    def _publish_force_base(self, f6: Sequence[float], seconds: float) -> None:
        f = [float(x) for x in f6]
        end_t = time.monotonic() + float(seconds)
        period = 1.0 / self.rate_hz
        while time.monotonic() < end_t and rclpy.ok():
            if self.force_msg_type == 'wrench':
                w = Wrench()
                w.force.x, w.force.y, w.force.z = f[:3]
                w.torque.x, w.torque.y, w.torque.z = f[3:]
                self.pub_force.publish(w)
            else:
                ws = WrenchStamped()
                ws.header.frame_id = 'base_link'
                ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z = f[:3]
                ws.wrench.torque.x, ws.wrench.torque.y, ws.wrench.torque.z = f[3:]
                self.pub_force.publish(ws)
            time.sleep(period)
        self.node.get_logger().info(f"[MatlabBridge] Force(base) f={[f'{x:.2f}' for x in f]} dur={seconds:.3f}s")
