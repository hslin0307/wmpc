# file: over_controller_nogripper.py
import numpy as np
import rclpy
from rclpy.node import Node

from motion_utils_matlab_bridge import MotionExecutor
from custom_libraries.pusher_utils import PusherHandler
from custom_libraries.actionlibrariesmax import move, velocity, force, append_new_act


class OverController(Node):
    """
    Minimal controller without any gripper dependency.
    """
    def __init__(self) -> None:
        super().__init__('motion_planner')

        # Joint names (UR5/UR5e 6-DOF)
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        # Controller names
        self.position_controller = 'scaled_joint_trajectory_controller'
        self.velocity_controller = 'forward_velocity_controller'
        self.force_controller = 'force_mode_controller'
        self.passthrough_controller = 'passthrough_trajectory_controller'

        # Submodules
        self.pusher = PusherHandler(self)

        # Motion executor (no gripper -> pass None)
        # why: MotionExecutor 期望一個 gripper 參數；此處顯式給 None 以取消夾爪功能
        self.motion = MotionExecutor(
            self,
            self.joint_names,
            self.position_controller,
            self.velocity_controller,
            self.force_controller,
            self.passthrough_controller,
            None,  # gripper disabled
            pusher=self.pusher
        )

        # Motion plan (example sequence)
        move_1 = move([0.1, -0.5, 0.247], [0, 180, 0], 4.0)
        move_2 = velocity(np.array([0.0, -0.02, 0.0, 0.0, 0.0, 0.0]), 5.0)
        move_3 = velocity(np.array([0.0,  0.02, 0.0, 0.0, 0.0, 0.0]), 5.0)
        move_4 = force([0.0, 0.0, -10.0, 0.0, 0.0, 0.0], 3.0)

        self.acts = {}
        # self.acts = append_new_act(self.acts, move_1)
        self.acts = append_new_act(self.acts, move_2)
        # self.acts = append_new_act(self.acts, move_3)
        # self.acts = append_new_act(self.acts, move_4)

        self.active_goal_handle = None
        self.executing = False
        self.i = 0
        self.execute_next_act()

    def execute_next_act(self) -> None:
        if self.i >= len(self.acts):
            self.get_logger().info("Done with current list. Waiting for more...")
            self.executing = False
            return
        act_name = list(self.acts)[self.i]
        self.i += 1
        self.execute_act(act_name)

    def execute_act(self, act_name: str) -> None:
        """Execute one act, then trigger next."""
        act = self.acts[act_name]
        act_type = act["type"]
        self.get_logger().info(f"▶ Executing {act_name}: {act_type}")
        self._execute_single_act(act)
        self.execute_next_act()

    def _execute_single_act(self, act: dict) -> None:
        """Execute a single act immediately (non-queued)."""
        act_type = act["type"]
        self.executing = True

        if act_type == "position":
            joint_angles = act["joint_angles"]
            seconds = act["time_from_start"]
            self.motion.send_joint_angles(joint_angles, seconds)

        elif act_type == "velocity":
            v = act["velocity"]
            seconds = act["duration"]
            self.motion.send_cartesian_velocity(v, seconds)

        elif act_type == "force":
            f = act["force"]
            seconds = act["duration"]
            self.motion.send_cartesian_force(f, seconds)

        else:
            raise ValueError(f"Invalid Act Type: {act_type}")

        self.executing = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OverController()
    try:
        rclpy.spin(node)
    except (RuntimeError, SystemExit):
        node.get_logger().info("Shutting down")
    node.get_logger().info("Script complete.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
