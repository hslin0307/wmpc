# file: tool_pose_publisher.py
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from custom_libraries.ik_solver import forward_kinematics, dh_params  # your lab file
import numpy as np

def rotm_to_quat(R):
    # R: 3x3
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t+1.0)*2
        w = 0.25*s
        x = (R[2,1]-R[1,2])/s
        y = (R[0,2]-R[2,0])/s
        z = (R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i==0:
            s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w = (R[2,1]-R[1,2])/s
            x = 0.25*s
            y = (R[0,1]+R[1,0])/s
            z = (R[0,2]+R[2,0])/s
        elif i==1:
            s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w = (R[0,2]-R[2,0])/s
            x = (R[0,1]+R[1,0])/s
            y = 0.25*s
            z = (R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w = (R[1,0]-R[0,1])/s
            x = (R[0,2]+R[2,0])/s
            y = (R[1,2]+R[2,1])/s
            z = 0.25*s
    return (x,y,z,w)

def rotm_to_rpy_deg(R):
    # ZYX
    sy = -R[2,0]
    cy = math.sqrt(max(0.0, 1 - sy*sy))
    if cy > 1e-8:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.asin(sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        # near gimbal lock
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.asin(sy)
        yaw   = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

class ToolPosePub(Node):
    def __init__(self):
        super().__init__('tool_pose_publisher')
        self.joint_order = [
            "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
            "wrist_1_joint","wrist_2_joint","wrist_3_joint"
        ]
        self.sub = self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.pub = self.create_publisher(PoseStamped, '/tool_pose', 10)
        self.get_logger().info("tool_pose_publisher: /joint_states -> /tool_pose (PoseStamped)")

    def on_js(self, msg: JointState):
        # map positions by name
        m = {n:p for n,p in zip(msg.name, msg.position)}
        if not all(j in m for j in self.joint_order):
            return
        q = np.array([m[j] for j in self.joint_order], dtype=float)

        T = forward_kinematics(dh_params, q)  # 4x4
        p = T[:3,3].astype(float)
        R = T[:3,:3].astype(float)

        # publish PoseStamped
        ps = PoseStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = 'base_link'
        x,y,z,w = rotm_to_quat(R)
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = p.tolist()
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = x,y,z,w
        self.pub.publish(ps)

        # log xyz & rpy(deg) 每隔一段再印，這裡簡單直接印
        rpy_deg = rotm_to_rpy_deg(R)
        self.get_logger().info(f"TCP pos (m): [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]  rpy(deg): [{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]")

def main():
    rclpy.init()
    node = ToolPosePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
