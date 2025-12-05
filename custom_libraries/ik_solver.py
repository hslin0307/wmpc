import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# UR5e DH parameters
dh_params = [
    (0,  0.1625,  0,     np.pi/2),  
    (0,  0,      -0.425,  0),       
    (0,  0,      -0.3922, 0),       
    (0,  0.1333,  0,     np.pi/2),  
    (0,  0.0997,  0,    -np.pi/2),  
    (0,  0.0996,  0,     0)
]

def rpy_to_matrix(rpy):
    return R.from_euler('xyz', rpy, degrees=True).as_matrix()

def dh_transform(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1]
    ])

def forward_kinematics(dh_params, joint_angles):
    T = np.eye(4)
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = np.dot(T, T_i)
    return T

def ik_objective(q, target_pose):
    T_fk = forward_kinematics(dh_params, q)
    pos_error = np.linalg.norm(T_fk[:3, 3] - target_pose[:3, 3])
    rot_error = np.linalg.norm(T_fk[:3, :3] - target_pose[:3, :3])
    return 1.0 * pos_error + 0.1 * rot_error

def compute_ik(position, rpy, q_guess=None, max_tries=5, dx=0.001):

    if q_guess is None:
        q6 = -(np.mod(rpy[2] + 180, 360) - 180) # Adjust initial guess based on given yaw.
        q_guess = np.radians([85, -80, 90, -90, -90, q6])

    original_position = np.array(position)

    for i in range(max_tries):
        # Try small x-shift each iteration
        perturbed_position = original_position.copy()
        perturbed_position[0] += i * dx  # only nudging x

        target_pose = np.eye(4)
        target_pose[:3, 3] = perturbed_position
        target_pose[:3, :3] = rpy_to_matrix(rpy)

        joint_bounds = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi)
        ]

        result = minimize(ik_objective, q_guess, args=(target_pose,), method='L-BFGS-B', bounds=joint_bounds)

        if result.success:
            # print(f"IK succeeded at position: {perturbed_position}")
            # print(f"IK Cost is {ik_objective(result.x, target_pose)}")
            return result.x

    print(f"IK failed after {max_tries} attempts. Tried perturbing from {original_position}.")
    return None


def compute_jacobian(joint_angles):
    """
    Compute the 6x6 Jacobian matrix at the given joint configuration using DH parameters.
    Returns:
        J: A 6x6 numpy array
    """
    n = len(joint_angles)
    T = np.eye(4)
    origins = [T[:3, 3]]
    z_axes = [T[:3, 2]]

    Ts = []  # Store intermediate transforms

    # Compute transformation matrices up to each joint
    for i in range(n):
        theta, d, a, alpha = dh_params[i]
        T_i = dh_transform(joint_angles[i] + theta, d, a, alpha)
        T = T @ T_i
        Ts.append(T.copy())
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    J = np.zeros((6, n))

    o_n = origins[-1]
    for i in range(n):
        z = z_axes[i]
        o_i = origins[i]
        J[:3, i] = np.cross(z, o_n - o_i)   # Linear velocity part
        J[3:, i] = z                        # Angular velocity part

    return J
