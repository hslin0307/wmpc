# file: waypoint_control/wmpc_node.py
from __future__ import annotations

import time
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .wmpc_double_integrator import (
    JointLimits,
    WMPCConfig,
    WaypointMPC,
)
from custom_libraries.ik_solver import compute_jacobian
from .motion_utils_matlab_bridge import MotionExecutor


def deg2rad(arr: List[float]) -> np.ndarray:
    return np.deg2rad(np.asarray(arr, dtype=float))


class WMPCNode(Node):
    def __init__(self) -> None:
        super().__init__("wmpc_node")

        self.dof = 6
        self.h = 0.05

        self.joint_names: List[str] = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        q_min = deg2rad([-360, -360, -360, -360, -360, -360])
        q_max = deg2rad([360, 360, 360, 360, 360, 360])

        qd_max_val = np.array([3.14, 3.14, 3.14, 3.2, 3.2, 3.2])
        qd_min = -qd_max_val
        qd_max = qd_max_val

        qdd_max_val = np.array([10.0, 10.0, 10.0, 15.0, 15.0, 20.0])
        qdd_min = -qdd_max_val
        qdd_max = qdd_max_val

        limits = JointLimits(
            q_min=q_min,
            q_max=q_max,
            qd_min=qd_min,
            qd_max=qd_max,
            qdd_min=qdd_min,
            qdd_max=qdd_max,
        )

        cfg = WMPCConfig(
            dof=self.dof,
            h=self.h,
            N_max=25,
            eps=deg2rad([2.0] * self.dof).mean(),
            gamma=0.1,
            sigma=10.0,
            d_min=deg2rad([5.0] * self.dof).mean(),
            N_min_goal=6,
            w_input_reg=1e-3,
        )

        self.mpc = WaypointMPC(cfg, limits)

        self.motion = MotionExecutor(
            node=self,
            joint_names=self.joint_names,
            position_controller="scaled_joint_trajectory_controller",
            velocity_controller="forward_velocity_controller",
            force_controller="force_mode_controller",
            passthrough_controller=None,
            gripper=None,
            rate_hz=200.0,
            vel_msg_type="array",
            force_msg_type="wrench",
            tcp_frame="tool0",
            explicit_topics=None,
        )

        self.q: Optional[np.ndarray] = None
        self.qd: Optional[np.ndarray] = None

        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            10,
        )

        self.q_home = deg2rad([0, -90, 90, -90, -90, 0])
        self.q_waypoint = deg2rad([30, -70, 80, -100, -90, 20])
        self.q_goal = deg2rad([60, -60, 70, -110, -80, 40])

        self.last_mpc_time = time.monotonic()
        self._warned_no_js = False

        self.get_logger().info("WMPCNode initialized (double-integrator wMPC for UR5).")

    # ----------------- ROS callbacks -----------------
    def joint_state_cb(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q_list = []
        qd_list = []
        for jn in self.joint_names:
            idx = name_to_idx.get(jn, None)
            if idx is None:
                q_list.append(0.0)
                qd_list.append(0.0)
            else:
                q_list.append(float(msg.position[idx]))
                if msg.velocity:
                    qd_list.append(float(msg.velocity[idx]))
                else:
                    qd_list.append(0.0)
        self.q = np.asarray(q_list, dtype=float)
        self.qd = np.asarray(qd_list, dtype=float)

    # ------------- MPC + command generation ----------
    def _compute_base_twist(self, q: np.ndarray, qd: np.ndarray, u0: np.ndarray) -> np.ndarray:
        qd_cmd = qd + self.h * u0

        vlim = np.array([3.14, 3.14, 3.14, 3.2, 3.2, 3.2], dtype=float)
        qd_cmd = np.clip(qd_cmd, -vlim, vlim)

        J = compute_jacobian(q)
        v_base = J @ qd_cmd

        VLIN_MAX = 0.25
        VANG_MAX = 1.5

        v_lin = v_base[:3]
        v_ang = v_base[3:]

        lin_norm = np.linalg.norm(v_lin)
        if lin_norm > VLIN_MAX and lin_norm > 1e-9:
            v_lin = v_lin * (VLIN_MAX / lin_norm)

        ang_norm = np.linalg.norm(v_ang)
        if ang_norm > VANG_MAX and ang_norm > 1e-9:
            v_ang = v_ang * (VANG_MAX / ang_norm)

        v_base[:3] = v_lin
        v_base[3:] = v_ang

        return v_base

    def _mpc_step(self) -> None:
        if self.q is None or self.qd is None:
            if not self._warned_no_js:
                self.get_logger().warn("Waiting for joint_states before running MPC.")
                self._warned_no_js = True
            return

        q = self.q.copy()
        qd = self.qd.copy()

        try:
            u0, x_opt, u_opt = self.mpc.solve(
                q_meas=q,
                qd_meas=qd,
                q_w=self.q_waypoint,
                q_g=self.q_goal,
            )
        except Exception as exc:
            self.get_logger().error(f"wMPC solve failed: {exc}")
            return

        v_base = self._compute_base_twist(q, qd, u0)

        self.get_logger().info(
            f"MPC: ||u0||={np.linalg.norm(u0):.3f}, "
            f"||v_base||={np.linalg.norm(v_base):.3f}"
        )

        # 這裡改用公開的 BASE 速度接口
        self.motion.send_base_velocity(v_base.tolist(), float(self.h))


    # -------------------- main loop -------------------
    def spin_loop(self) -> None:
        self.get_logger().info(f"Starting WMPC loop with dt={self.h:.3f}s.")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.monotonic()
            if now - self.last_mpc_time >= self.h:
                self._mpc_step()
                self.last_mpc_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WMPCNode()
    try:
        node.spin_loop()
    except KeyboardInterrupt:
        node.get_logger().info("WMPCNode interrupted by user.")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
