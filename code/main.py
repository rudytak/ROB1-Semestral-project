import signal
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
from helper import euler_xyz_to_matrix, print_pose
from matplotlib import pyplot as plt
from maze_a import TOTAL_LENGTH_A
from maze_d import TOTAL_LENGTH_D
from maze_localization import MazePredictor

# --- CONFIGURATION ---
# maze choice:
# 0 - static z offset
# 1 - maze A
# 2 - maze B not implemented
# 3 - maze C not implemented
# 4 - maze D
MAZE = 0
s_change_speed = 0.005  # speed of solving the maze m/s
time_delay_s = 0.1  # delay of prediction

match MAZE:
    case 0:
        total_length = 10.0
        s_change_speed = 1.0

    case 1:
        total_length = TOTAL_LENGTH_A

    case 2:
        pass

    case 3:
        pass

    case 4:
        total_length = TOTAL_LENGTH_D

T_FLANGE_TO_HOOP = np.eye(4)
T_FLANGE_TO_HOOP[:3, :3] = euler_xyz_to_matrix((0, -np.pi, 0))
T_FLANGE_TO_HOOP[:3, 3] = [-0.135, 0.0, 0.0066]

T_HOOP_TO_FLANGE = np.linalg.inv(T_FLANGE_TO_HOOP)

running = True


def signal_handler(sig, frame):
    global running
    print("\nEnding...")
    running = False


# connecting Ctrl+C with changing running
signal.signal(signal.SIGINT, signal_handler)

robot = CRS97()
# robot.initialize(home=True)
robot.initialize(home=False)
robot.move_to_q(robot.q_home)

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
predictor.estimate_rotational_table_geometry(robot, calib_results, num_measurements=50, delay=0.1)

# estimating angular velocity
omega = predictor.estimate_angular_velocity(robot, calib_results, duration=4.0, delay=0.05)

# anchoring prediction
predictor.anchor_state(robot, calib_results, omega)


print(f"Rotational table pos: [{predictor.center_x:.3f}, {predictor.center_y:.3f}, {predictor.center_z:.3f}]")
print(f"Omega: {predictor.omega:.3f} rad/s")
print(f"Relative rotation of maze (theta0): {predictor.theta0:.3f} rad")
print(f"Radius: {predictor.radius}")

start_t = time.time()
s = 0.0
while running:
    t = time.time()
    elapsed = t - start_t

    s = s_change_speed * elapsed
    s = min(s, total_length)

    target_pose_hoop = predictor.get_target_pose(s, maze=MAZE, time_delay_s=time_delay_s)

    # target_pose_hoop_fixed = np.eye(4)
    # target_pose_hoop_fixed[:3, 3] = target_pose_hoop[:3, 3]
    # target_pose_flange = target_pose_hoop_fixed @ T_HOOP_TO_FLANGE

    target_pose_flange = target_pose_hoop @ T_HOOP_TO_FLANGE

    # IK
    ik_sols = robot.ik(target_pose_flange)
    ik_sols = list(filter(lambda q: robot.in_limits(q), ik_sols))

    print_pose(target_pose_flange, "flange")
    print_pose(target_pose_hoop, "hoop")

    print(len(ik_sols))

    if len(ik_sols) > 0:
        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - robot.get_q()))

        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()

    else:
        time.sleep(0.1)

robot.move_to_q(robot.q_home)
robot.wait_for_motion_stop()
robot.close()
