import time

import cv2 as cv
import numpy as np
from camera import (
    detect_ArUco,
    estimate_aruco_world_positions,
    load_camera_calibration,
    save_camera_calibration,
    take_calibration_pictures,
)
from ctu_crs import CRS97
from helper import euler_xyz_to_matrix
from matplotlib import pyplot as plt
from maze_d import TOTAL_LENGTH
from maze_localization import MazePredictor

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
print("P = ", calib_results["P"])
print("K = ", calib_results["K"])
print("R_CB = ", calib_results["R_CB"])
print("t_CB = ", calib_results["t_CB"])
print("-" * 50)

predictor = MazePredictor()
# finding position of rotational table
predictor.estimate_rotational_table_geometry(robot, calib_results, num_measurements=10, delay=0.1)

# estimating angular velocity
omega = predictor.estimate_angular_velocity(robot, calib_results, duration=4.0, delay=0.05)

# anchoring prediction
predictor.anchor_state(robot, calib_results, omega)


print(f"Rotational table pos: [{predictor.center_x:.3f}, {predictor.center_y:.3f}, {predictor.center_z:.3f}]")
print(f"Omega: {predictor.omega:.3f} rad/s")
print(f"Relative rotation of maze (theta0): {predictor.theta0:.3f} rad")

start_t = time.time()
s = 0.0
s_change_speed = 0.005
time_delay_s = 0.1

while s <= TOTAL_LENGTH:
    t = time.time()
    elapsed = t - start_t

    s = s_change_speed * elapsed
    s = min(s, 0.8 * TOTAL_LENGTH)

    target_pose = predictor.get_target_pose(s, time_delay_s=time_delay_s)
    print(f"Elapsed: {elapsed:.2f} .... Target_pose: {target_pose} .... s: {s:.4f} m")

    # IK
    ik_sols = robot.ik(target_pose)
    ik_sols = list(filter(lambda q: robot.in_limits(q), ik_sols))

    print(len(ik_sols))

    if len(ik_sols) > 0:
        q0 = robot.q_home
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()

robot.close()
