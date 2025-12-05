"""
Publish a waypoint with position + orientation.
Usage examples:
  # RPY in degrees
  ros2 run waypoint_control waypoint_cli --x 0.45 --y -0.15 --z 0.25 --rpy 0 180 0 --deg --frame base_link
  # Quaternion xyzw
  ros2 run waypoint_control waypoint_cli --x 0.45 --y -0.15 --z 0.25 --quat 0 1 0 0 --frame base_link
"""

import argparse
import math
import time
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


# QoS helper ----------------------------------------------------------

def make_transient_local_qos() -> QoSProfile:
    # Why: keep last waypoint so new subscribers receive it immediately
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    return qos


# Math helpers --------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n <= 0.0:
        # Why: safe identity fallback
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def rpy_to_quat(roll, pitch, yaw, degrees=True):
    if degrees:
        roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    w = cr*cp*cy + sr*sp*sy
    return quat_normalize(np.array([x,y,z,w], dtype=float))


# ROS2 node -----------------------------------------------------------

class WaypointCLINode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("waypoint_cli")

        qos = make_transient_local_qos()
        self.pub = self.create_publisher(PoseStamped, "/waypoint", qos)

        msg = PoseStamped()
        msg.header.frame_id = args.frame

        # Position
        msg.pose.position.x = float(args.x)
        msg.pose.position.y = float(args.y)
        msg.pose.position.z = float(args.z)

        # Orientation
        if args.quat is not None:
            # Expect 4 numbers: x y z w
            q_in: List[float] = [float(v) for v in args.quat]
            q = quat_normalize(np.array(q_in, dtype=float))
        elif args.rpy is not None:
            # Expect 3 numbers: roll pitch yaw
            r = float(args.rpy[0])
            p = float(args.rpy[1])
            y = float(args.rpy[2])
            q = rpy_to_quat(r, p, y, in_degrees=args.deg)
        else:
            # Default identity
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        # Publish a few times with small sleeps (simple and robust)
        count = 0
        while count < args.repeat:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            count = count + 1
            time.sleep(args.interval)

        # Give a small grace period for DDS delivery then exit
        time.sleep(0.05)


# Main ----------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Publish one PoseStamped waypoint to /waypoint")
    ap.add_argument("--x", type=float, required=True, help="Waypoint X (meters)")
    ap.add_argument("--y", type=float, required=True, help="Waypoint Y (meters)")
    ap.add_argument("--z", type=float, required=True, help="Waypoint Z (meters)")
    ap.add_argument("--frame", type=str, default="base_link", help="Frame id")

    # Orientation (choose one)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--quat", nargs=4, metavar=("x", "y", "z", "w"), help="Orientation quaternion (xyzw)")
    group.add_argument("--rpy", nargs=3, metavar=("roll", "pitch", "yaw"), help="Orientation RPY")

    # RPY units
    ap.add_argument("--deg", action="store_true", help="RPY is in degrees (default if --rpy used)")
    ap.add_argument("--rad", dest="deg", action="store_false", help="RPY is in radians")
    ap.set_defaults(deg=True)

    # Publish behavior
    ap.add_argument("--repeat", type=int, default=5, help="Publish count")
    ap.add_argument("--interval", type=float, default=0.05, help="Seconds between publishes")
    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    rclpy.init()
    node = WaypointCLINode(args)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__" and __file__.endswith("waypoint_cli.py"):
    main()