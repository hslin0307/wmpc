# file: pusher_handler_nogripper.py
from functools import partial
from geometry_msgs.msg import PointStamped, Vector3Stamped
import numpy as np

def EE_pose_to_pushers_2D(pose, pusher_sep: float = 0.04):
    """
    Compute two lateral pusher contact points in XY plane around EE center.
    Why: 去除 gripper 依賴仍需左右推點，採用簡單左右偏移模型。
    pose: ((x,y,z), (roll, pitch, yaw)) 期望弧度。
    pusher_sep: 左右點總間距(m)。
    """
    (x, y, z), (roll, pitch, yaw) = pose  # roll/pitch 未用，僅以 yaw 近似 2D 佈置
    # 左右法向（與 heading 垂直）：n = [-sin(yaw), cos(yaw)]
    nx, ny = -np.sin(yaw), np.cos(yaw)
    half = 0.5 * pusher_sep
    p_left  = (x + nx * half,  y + ny * half,  z)
    p_right = (x - nx * half,  y - ny * half,  z)
    return p_left, p_right

class PusherHandler:
    """Pusher 相關介面（已移除 gripper 相依；發布左右推點與接收建議位姿/法向）"""
    def __init__(self, node, frame_id: str = "base", pusher_sep: float = 0.04):
        self.node = node
        self.frame_id = frame_id
        self.pusher_sep = float(pusher_sep)

        # 改名：不再使用 /gripper_* 詞頭
        self.pusher_pub_l = self.node.create_publisher(PointStamped, "/pusher_left", 10)
        self.pusher_pub_r = self.node.create_publisher(PointStamped, "/pusher_right", 10)

        self.recommended_topics = [
            ('pusher_1_position', '/recommended_pusher_1/position', PointStamped),
            ('pusher_2_position', '/recommended_pusher_2/position', PointStamped),
            ('pusher_1_normal',   '/recommended_pusher_1/normal',   Vector3Stamped),
            ('pusher_2_normal',   '/recommended_pusher_2/normal',   Vector3Stamped)
        ]
        self.recommended_subscriptions = {}
        for name, topic, msg_type in self.recommended_topics:
            cb = partial(self.push_recommend_callback, name=name)
            sub = self.node.create_subscription(msg_type, topic, cb, 10)
            self.recommended_subscriptions[name] = sub

        self.recommend_data = {
            "pusher_1": {"position": None, "normal": None},
            "pusher_2": {"position": None, "normal": None},
        }

    def push_recommend_callback(self, msg, name):
        _, number, attr = name.split("_")
        self.recommend_data[f"pusher_{number}"][attr] = msg

    def update(self, ee_position, ee_euler):
        """
        根據 EE 位姿更新左右推點。euler 預期為 (roll,pitch,yaw) [rad]。
        Why: 無 gripper 服務，仍需發布左右接觸點供外部可視化/使用。
        """
        p1, p2 = EE_pose_to_pushers_2D((ee_position, ee_euler), self.pusher_sep)
        now = self.node.get_clock().now().to_msg()

        for pub, pos in zip([self.pusher_pub_l, self.pusher_pub_r], [p1, p2]):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.point.x, msg.point.y, msg.point.z = pos
            pub.publish(msg)

    def extract_pusher_locations(self):
        """
        由推薦資料提取兩推點與建議 yaw（以 pusher_1 normal 為基準）。
        """
        if any([self.recommend_data["pusher_1"]["position"] is None,
                self.recommend_data["pusher_2"]["position"] is None,
                self.recommend_data["pusher_1"]["normal"] is None,
                self.recommend_data["pusher_2"]["normal"] is None]):
            print("Improper data detected. Skipping...")
            return

        x = self.recommend_data["pusher_1"]["normal"].vector.x
        y = self.recommend_data["pusher_1"]["normal"].vector.y
        # 原註解：normals 指向外側；若需相機反向可在外層再做偏移
        normal_yaw = np.arctan2(y, x)        # rad
        given_yaw  = np.degrees(normal_yaw) - 90.0

        pusher_1 = [
            self.recommend_data["pusher_1"]["position"].point.x,
            self.recommend_data["pusher_1"]["position"].point.y,
            self.recommend_data["pusher_1"]["position"].point.z
        ]
        pusher_2 = [
            self.recommend_data["pusher_2"]["position"].point.x,
            self.recommend_data["pusher_2"]["position"].point.y,
            self.recommend_data["pusher_2"]["position"].point.z
        ]
        return pusher_1, pusher_2, given_yaw
