def canonicalize_euler(orientation, tol=1):
    """Forces euler angles near the form (-180, 0, yaw') to take the equivalent form (0, 180, yaw)"""
    roll, pitch, yaw = orientation
    if abs(pitch) < tol and abs(abs(roll) - 180) < tol:
        return (0.0, 180.0, (yaw % 360)-180)
    else:
        return orientation

def pose_text(pose):
    position, euler = pose
    return f"""XYZ: {1000*position[0]:.1f}, {1000*position[1]:.1f}, {1000*position[2]:.1f} mm
RPY: {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f} deg"""

def pushers_text(pusher_1_pos, pusher_2_pos):
    return f"""
    Pusher 1: ({1000*pusher_1_pos[0]:.1f}, {1000*pusher_1_pos[1]:.1f}, {1000*pusher_1_pos[2]:.1f}) mm
    Pusher 2: ({1000*pusher_2_pos[0]:.1f}, {1000*pusher_2_pos[1]:.1f}, {1000*pusher_2_pos[2]:.1f}) mm"""

def vector3_text(vec, prec=2):
    return f"({vec[0]: .{prec}f}, {vec[1]: .{prec}f}, {vec[2]: .{prec}f})"