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


measurement_start_t = start_t

xy = np.column_stack((cx, cy))
sample_times = np.asarray(ts)

# Fit a circle to the tracked midpoint's XY positions.
origin = xy.mean(axis=0)
uv = xy - origin

A = np.column_stack((2 * uv, np.ones(len(uv))))
b = np.sum(uv**2, axis=1)

solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
if rank < 3:
    raise ValueError("Need positions covering a curved arc to fit the circle.")

center_xy = origin + solution[:2]
offsets = xy - center_xy
radius = np.mean(np.linalg.norm(offsets, axis=1))

# Fit angular speed AND phase, using the same tracked point.
angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
relative_times = sample_times - sample_times[-1]
omega_scalar, phase_last = np.polyfit(relative_times, angles, 1)

center = np.array([center_xy[0], center_xy[1], np.mean(cz)])
height_offset = 0.1

start_t = time.time()

while True:
    # Time elapsed since the last observation.
    dt = time.time() - measurement_start_t - sample_times[-1]
    theta = phase_last + omega_scalar * dt

    p_BH_target = center + np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        height_offset,
    ])

    R_BF = euler_xyz_to_matrix((0, -np.pi, np.pi / 2))
    p_FH = np.array([-0.135, 0.135, 0.0])

    T_BF_target = np.eye(4)
    T_BF_target[:3, :3] = R_BF
    T_BF_target[:3, 3] = p_BH_target - R_BF @ p_FH

    ik_sols = robot.ik(T_BF_target)
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
