from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from helper import euler_xyz_to_matrix
import time

from ctu_crs import CRS97
from camera import take_calibration_pictures, save_camera_calibration, load_camera_calibration, detect_ArUco, estimate_aruco_world_positions

robot = CRS97()
robot.initialize(home=True)
# robot.initialize(home=False)

# # Taking calibration pictures
# take_calibration_pictures(robot, divisions=(3,5,5))
# take_calibration_pictures(robot, divisions=(2,2,2))

# # Calibrating camera (getting P,K matrices)
# calib_results = save_camera_calibration()
# print(calib_results)

# Load camera calibration data:
calib_results = load_camera_calibration()
print('P = ', calib_results['P'])
print('K = ', calib_results['K'])
print('R_CB = ', calib_results['R_CB'])
print('t_CB = ', calib_results['t_CB'])
print("-"*50)

# Perform ArUco detection
start_t = time.time()
ts,xs,ys,zs = [],[],[],[]
cx, cy, cz = [], [], []
while time.time() - start_t < 15:
    # detect_ArUco(robot.grab_image())
    img = robot.grab_image()
    res = estimate_aruco_world_positions(img, calib_results)
    # res = list(filter(lambda r: r["id"] == 1, res))

    # if len(res) > 0:
    #     vec = (res[0]['T_BM'][:3,3])

    #     ts.append(time.time() - start_t)
    #     xs.append(vec[0])
    #     ys.append(vec[1])
    #     zs.append(vec[2])

    #     print(f'Took img {len(ts)} at {ts[-1]}')

    if len(res) == 2:
        det1 = list(filter(lambda r: r["id"] == 1, res))[0]
        det2 = list(filter(lambda r: r["id"] == 2, res))[0]
        
        vec1 = (det1['T_BM'][:3,3])
        vec2 = (det2['T_BM'][:3,3])

        ts.append(time.time() - start_t)
        xs.append(vec1[0])
        ys.append(vec1[1])
        zs.append(vec1[2])

        cx.append((vec1[0] + vec2[0]) / 2)
        cy.append((vec1[1] + vec2[1]) / 2)
        cz.append((vec1[2] + vec2[2]) / 2)

        print(f'Took img {len(ts)} at {ts[-1]}')

cx = np.mean(cx)
cy = np.mean(cy)
cz = np.mean(cz)
center = np.array([cx,cy,cz])

angle = np.atan2(xs - cx, ys - cy)
omega = np.diff(angle) / np.diff(ts)

omega_scalar = -np.mean(list(filter(lambda o: np.abs(o) < 5, omega)))
print(f"Avg angular speed {omega_scalar} rad/s")

start_t = time.time()
while True:
    t = time.time() - start_t

    target_position = center + np.array([0.05 * np.cos(omega_scalar *t), 0.05* np.sin(omega_scalar * t),0.05])
    target_rotation =  euler_xyz_to_matrix(
        (0, -np.pi, np.pi/2)
    )
    # Create homogeneous SE(3) transformation
    target_pose = np.eye(4)
    target_pose[:3, :3] = target_rotation # target rotation matrix
    target_pose[:3, 3] = target_position # target position

    # Move to the target position using IK
    ik_sols = robot.ik(target_pose)
    # filter solution for reachability
    ik_sols = list(filter(lambda q: robot.in_limits(q), ik_sols))
    
    print(len(ik_sols))
    if len(ik_sols) > 0:
        # Select the closest position
        # q0 = robot.get_q()
        q0 = robot.q_home
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()
        # time.sleep(.1)

robot.close()

# plt.figure()
# # plt.scatter(ts, cx, c="r", label="X")
# # plt.scatter(ts, cy, c="g", label="Y")
# # plt.scatter(ts, cz, c="b", label="Z")
# plt.scatter(ts, angle, c="orange", label="deg/s")
# plt.scatter(ts[1:], omega, c="blue", label="rad/s")
# plt.legend()
# plt.show()
