from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from helper import euler_xyz_to_matrix
import time

from ctu_crs import CRS97
from camera import take_calibration_pictures, save_camera_calibration, load_camera_calibration, detect_ArUco, estimate_aruco_world_positions


from maze import maze_b as maze


PREDICTION_LEAD = 0.08  # seconds

def maze_base_pose(p1, p2) -> np.ndarray:
    """Maze-base pose from the two marker centres, in robot coordinates.
    marker id 1 upper-left, marker id 2
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    z_axis = np.array([0.0, 0.0, 1.0])

    diagonal = p1 - p2
    diagonal[2] = 0.0

    length = np.linalg.norm(diagonal)

    diagonal /= length

    # Correct for the diagonal marker arrangement.
    y_axis = diagonal + np.cross(diagonal, z_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = (p1 + p2) / 2.0

    return T

def main():
    R_Bmaze_last = None

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

    # Collect maze-base positions and orientations.
    start_t = time.monotonic()
    ts, phis = [], []
    cx, cy, cz = [], [], []

    while time.monotonic() - start_t < 15:
        img = robot.grab_image()
        frame_time = time.monotonic()
        res = estimate_aruco_world_positions(img, calib_results)

        detections = {d["id"]: d for d in res}

        if 1 in detections and 2 in detections:
            vec1 = detections[1]["T_BM"][:3, 3]
            vec2 = detections[2]["T_BM"][:3, 3]

            T_Bmaze_measured = maze_base_pose(vec1, vec2)

            # Heading of the maze's +X axis in the robot's XY plane.
            phi = np.arctan2(
                T_Bmaze_measured[1, 0],
                T_Bmaze_measured[0, 0],
            )

            midpoint = T_Bmaze_measured[:3, 3]

            ts.append(frame_time)
            phis.append(phi)
            cx.append(midpoint[0])
            cy.append(midpoint[1])
            cz.append(midpoint[2])

            print(
                f"Took img {len(ts)} "
                f"at {frame_time - start_t:.3f} s"
            )

    if len(ts) < 3:
        raise ValueError("Not enough observations of markers 1 and 2.")


    
    sample_times = np.asarray(ts)
    xy = np.column_stack((cx, cy))

    # Remove jumps between +pi and -pi.
    measured_phis = np.unwrap(np.asarray(phis))

    # Fit phi(t) = phi_last + omega * (t - last_time).
    last_time = sample_times[-1]
    relative_times = sample_times - last_time

    omega_scalar, phi_last = np.polyfit(
        relative_times, measured_phis, 1
    )

    # Fit:
    # maze_position_xy = table_center_xy + Rz(phi) @ offset_xy
    #
    # Unknowns: center_x, center_y, offset_x, offset_y.
    c = np.cos(measured_phis)
    sn = np.sin(measured_phis)

    A = np.zeros((2 * len(ts), 4))

    # X equations.
    A[0::2, 0] = 1.0
    A[0::2, 2] = c
    A[0::2, 3] = -sn

    # Y equations.
    A[1::2, 1] = 1.0
    A[1::2, 2] = sn
    A[1::2, 3] = c

    solution, _, rank, singular_values = np.linalg.lstsq(
        A, xy.reshape(-1), rcond=None
    )

    if (
        rank < 4
        or singular_values[-1] / singular_values[0] < 1e-3
    ):
        raise ValueError(
            "Too little observed rotation to estimate the table centre."
        )

    # Rotation-axis position, using mean marker height.
    center = np.array([
        solution[0],
        solution[1],
        np.mean(cz),
    ])

    # Vector from rotation axis to maze origin,
    # expressed along the maze's local axes.
    offset_local = np.array([
        solution[2],
        solution[3],
        0.0,
    ])

    print(f"Table centre [m]: {center}")
    print(f"Angular speed [rad/s]: {omega_scalar:.5f}")
    print(f"Last fitted angle [deg]: {np.degrees(phi_last):.2f}")
    print(f"Axis-to-maze offset [m]: {offset_local}")


    s = 0.0
    tilt = False

    # Current tool offset.
    p_FH = np.array([-0.135, 0.135, 0.0])

    R_HF = euler_xyz_to_matrix((0, -np.pi, np.pi / 2))

    T_FH = np.eye(4)
    T_FH[:3, :3] = R_HF.T
    T_FH[:3, 3] = p_FH
    T_HF = np.linalg.inv(T_FH)

    try:
        while True:
            dt = time.monotonic() - last_time

            phi = phi_last + omega_scalar * (dt + PREDICTION_LEAD)

            # Predict orientation and position using the same angle.
            R_Bmaze = euler_xyz_to_matrix((0.0, 0.0, phi))
            p_Bmaze = center + R_Bmaze @ offset_local

            T_Bmaze = np.eye(4)
            T_Bmaze[:3, :3] = R_Bmaze
            T_Bmaze[:3, 3] = p_Bmaze

            # Local maze target -> robot-base hoop target -> flange target.
            T_maze_H = maze.hoop_pose(s, tilt=tilt)
            T_BH_target = T_Bmaze @ T_maze_H
            T_BF_target = T_BH_target @ T_HF

            ik_sols = robot.ik(T_BF_target)
            ik_sols = [
                q for q in ik_sols
                if robot.in_limits(q)
            ]

            print(len(ik_sols))

            if len(ik_sols) > 0:
                q0 = robot.q_home
                closest_solution = min(
                    ik_sols,
                    key=lambda q: np.linalg.norm(q - q0),
                )

                robot.move_to_q(closest_solution)
                robot.wait_for_motion_stop()

    finally:
        robot.close()





if __name__ == "__main__":
    main()

















# EOF
