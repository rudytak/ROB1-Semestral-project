import numpy as np

def euler_xyz_to_matrix(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype = float)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx

def matrix_to_euler_xyz(R):
    R = np.asarray(R)

    pitch = np.arcsin(np.clip(-R[2,0], -1, 1))

    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll=np.arctan2(-R[1,2], R[1,1])
        yaw=0.0

    return np.array([roll,pitch,yaw])