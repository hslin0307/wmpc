from builtin_interfaces.msg import Duration
from custom_libraries.ik_solver import compute_ik
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

HOME_POSE = [0.065, -0.385, 0.481, 0, 180, 0]  # XYZRPY

def home():
    return move(HOME_POSE[0:3], HOME_POSE[3:6], 4)

def move(position, rpy, seconds):
    if len(position) != 3:
        raise ValueError(f"Expected 3D position, got {position}")
    joint_angles = compute_ik(position, rpy)
    if joint_angles is None:
        return []
    return {
        "type": "position",
        "joint_angles": [float(x) for x in joint_angles],
        "time_from_start": seconds
    }

def gripper_width_from_height(position, gripper):
    adjusted_height = position[2] - gripper.vertical_offset
    width = gripper.height_to_gripper_width(adjusted_height)
    return gripper_width(str(width))

def gripper_width(cmd):
    """cmd is a string, and may be 'open', 'close', or a numerical string from 0 to 1100"""
    return {
        "type": "gripper",
        "cmd": cmd
        }

def velocity(v_cartesian, seconds):
    return {
        "type": "velocity",
        "velocity": v_cartesian,
        "duration": seconds
        }

def force(f_cartesian, seconds):
    return {
        "type": "force",
        "force": f_cartesian,
        "duration": seconds
        }

def pick_and_place(block_pose, slot_pose):
    """
    block_pose and slot_pose are each (position, rpy), where position = [x, y, z]
    and EE orienttaion is [r, p, y]
    """
    block_hover = block_pose[0].copy() ## copying positions
    block_hover[2] += 0.1  # hover 10cm above block

    slot_hover = slot_pose[0].copy()
    slot_hover[2] += 0.1  # hover 10cm above slot

    segment_duration = 6 # specify segment_duration

    return {
        "act0": home(),
        "act1": move(  block_hover, block_pose[1], segment_duration), # hovers on block 
        "act2": move(block_pose[0], block_pose[1], segment_duration), # descends to grip position, 
        "act3": move(block_pose[0], block_pose[1], segment_duration), # gripper close
        "act4": move(  block_hover, block_pose[1], segment_duration), # holds block and hovers 
        "act5": move(   slot_hover,  slot_pose[1], segment_duration), # holds block and moves in 2D to hover on slot
        "act6": move( slot_pose[0],  slot_pose[1], segment_duration), # holds block and descends into slot,
        "act7": move( slot_pose[0],  slot_pose[1], segment_duration), # gripper open
        "act8": home() # homing
    }

def spin_around(target_pose, height):
    """
    target_pose is (position, rpy), where position = [x, y, z] and only x, y are considered
    """
    target_position = target_pose[0].copy() # copying positions
    target_position[2] = height  # Set height to given value
    yaws = range(0, 360, 45)
    segment_duration = 3 # specify segment_duration
    return {
        "act0": move(target_position, [0, 180, yaws[0]], segment_duration),
        "act1": move(target_position, [0, 180, yaws[1]], segment_duration),
        "act2": move(target_position, [0, 180, yaws[2]], segment_duration),
        "act3": move(target_position, [0, 180, yaws[3]], segment_duration),
        "act4": move(target_position, [0, 180, yaws[4]], segment_duration),
        "act5": move(target_position, [0, 180, yaws[5]], segment_duration),
        "act6": move(target_position, [0, 180, yaws[6]], segment_duration),
        "act7": move(target_position, [0, 180, yaws[7]], segment_duration),
    }

def hover_over(target_pose, height):
    """
    target_pose is (position, rpy), where position = [x, y, z] and only x, y are considered
    """
    target_position = target_pose[0].copy() # copying positions
    target_position[2] = height  # Set height to given value
    fixed_roll = 0
    fixed_pitch = 180
    yaw = target_pose[1][2] # in degrees
    null_rot = [0, 180, 0]
    target_rot = [fixed_roll, fixed_pitch, yaw]
    segment_duration = 2 # specify segment_duration
    # print(block_hover, target_rot)
    (x, y, z) = target_position
    print(f"Made target pose of <{x:.3f}, {y:.3f}, {z:.3f}> @ rpy [{fixed_roll:.1f}, {fixed_pitch:.1f}, {yaw:.1f}]")

    return {
        "act0": move(target_position,null_rot,segment_duration), # hovers over target 
        "act1": move(target_position,target_rot,segment_duration), # hovers over target, matching angle
    }

def append_new_act(traj_dict, trajectory):
    # Extract the numeric suffixes from keys and find the max
    existing_keys = [key for key in traj_dict.keys() if key.startswith("act")]
    if existing_keys:
        max_index = max(int(key[3:]) for key in existing_keys if key[3:].isdigit())
    else:
        max_index = 0

    new_key = f"act{max_index + 1}"
    traj_dict[new_key] = trajectory
    return traj_dict