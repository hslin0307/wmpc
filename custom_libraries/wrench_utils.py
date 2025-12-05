from geometry_msgs.msg import WrenchStamped

class WrenchHandler:
    """Small class for wrench sensor measurements. Formats messages for three wrench types."""
    def __init__(self, smoothed_wrench_pub, baseline_wrench_pub, normed_wrench_pub, node):
        self.smoothed_wrench_pub = smoothed_wrench_pub
        self.baseline_wrench_pub = baseline_wrench_pub
        self.normed_wrench_pub = normed_wrench_pub
        self.node = node

    def update(self, force_offset, torque_offset, smoothed_force, smoothed_torque):
        msg = WrenchStamped()
        now = self.node.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        # Send off baseline
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = force_offset
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = torque_offset
        self.baseline_wrench_pub.publish(msg)

        # Send off smoothed
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = smoothed_force
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = smoothed_torque
        self.smoothed_wrench_pub.publish(msg)

        # And submit the difference
        normed_force = smoothed_force - force_offset
        normed_torque = smoothed_torque - torque_offset
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = normed_force
        msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = normed_torque
        self.normed_wrench_pub.publish(msg)
