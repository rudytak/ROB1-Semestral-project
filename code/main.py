from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from helper import euler_xyz_to_matrix
import time

from ctu_crs import CRS97
from camera import take_calibration_pictures, save_camera_calibration, load_camera_calibration, detect_ArUco, estimate_aruco_world_positions


from maze import maze_b as maze


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

        # if len(res) == 2:
        #     det1 = list(filter(lambda r: r["id"] == 1, res))[0]
        #     det2 = list(filter(lambda r: r["id"] == 2, res))[0]
            
        #     vec1 = (det1['T_BM'][:3,3])
        #     vec2 = (det2['T_BM'][:3,3])

        #     ts.append(time.time() - start_t)
        #     xs.append(vec1[0])
        #     ys.append(vec1[1])
        #     zs.append(vec1[2])

        #     cx.append((vec1[0] + vec2[0]) / 2)
        #     cy.append((vec1[1] + vec2[1]) / 2)
        #     cz.append((vec1[2] + vec2[2]) / 2)

        #     print(f'Took img {len(ts)} at {ts[-1]}')

        detections = {d["id"]: d for d in res}

        if 1 in detections and 2 in detections:
            vec1 = detections[1]["T_BM"][:3, 3]
            vec2 = detections[2]["T_BM"][:3, 3]

            T_Bmaze_measured = maze_base_pose(vec1, vec2)
            R_Bmaze_last = T_Bmaze_measured[:3, :3].copy()

            ts.append(time.time() - start_t)
            xs.append(vec1[0])
            ys.append(vec1[1])
            zs.append(vec1[2])

            midpoint = T_Bmaze_measured[:3, 3]
            cx.append(midpoint[0])
            cy.append(midpoint[1])
            cz.append(midpoint[2])

            print(f"Took img {len(ts)} at {ts[-1]}")

    if R_Bmaze_last is None or len(ts) < 3:
        raise ValueError("Not enough observations of markers 1 and 2.")

    measurement_start_t = start_t

    xy = np.column_stack((cx, cy))
    sample_times = np.asarray(ts)

    # Fit a circle to the tracked midpoint's XY positions.
    origin = xy.mean(axis=0)
    uv = xy - origin

    A = np.column_stack((2 * uv, np.ones(len(uv))))
    b = np.sum(uv**2, axis=1)

    solution, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

    center_xy = origin + solution[:2]
    offsets = xy - center_xy
    radius = np.mean(np.linalg.norm(offsets, axis=1))

    angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
    relative_times = sample_times - sample_times[-1]
    omega_scalar, phase_last = np.polyfit(relative_times, angles, 1)

    center = np.array([center_xy[0], center_xy[1], np.mean(cz)])


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
            dt = (time.time() - measurement_start_t - sample_times[-1])

            delta_phi = omega_scalar * (dt + 0.08)
            theta = phase_last + delta_phi

            # Predicted maze-base position.
            p_Bmaze = center + np.array([
                radius * np.cos(theta),
                radius * np.sin(theta),
                0.0])

            # Predicted maze-base orientation.
            R_delta = euler_xyz_to_matrix((0.0, 0.0, delta_phi))
            R_Bmaze = R_delta @ R_Bmaze_last

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



# height_offset = 0.4

# start_t = time.time()

# progress = 1.0  

# while True:
#     # Time elapsed since the last observation.
#     dt = time.time() - measurement_start_t - sample_times[-1]
#     theta = phase_last + omega_scalar * (dt + 0.08)

#     p_BH_target = center + np.array([
#         radius * np.cos(theta),
#         radius * np.sin(theta),
#         max(0.1, height_offset * progress),
#     ])

#     progress -= 0.01

#     R_BF = euler_xyz_to_matrix((0, -np.pi, np.pi / 2))
#     p_FH = np.array([-0.135, 0.135, 0.0])

#     T_BF_target = np.eye(4)
#     T_BF_target[:3, :3] = R_BF
#     T_BF_target[:3, 3] = p_BH_target - R_BF @ p_FH

#     ik_sols = robot.ik(T_BF_target)
#     # filter solution for reachability
#     ik_sols = list(filter(lambda q: robot.in_limits(q), ik_sols))
    
#     print(len(ik_sols))
#     if len(ik_sols) > 0:
#         # Select the closest position
#         # q0 = robot.get_q()
#         q0 = robot.q_home
#         closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

#         robot.move_to_q(closest_solution)
#         robot.wait_for_motion_stop()
#         # time.sleep(.1)

# robot.close()

# plt.figure()
# # plt.scatter(ts, cx, c="r", label="X")
# # plt.scatter(ts, cy, c="g", label="Y")
# # plt.scatter(ts, cz, c="b", label="Z")
# plt.scatter(ts, angle, c="orange", label="deg/s")
# plt.scatter(ts[1:], omega, c="blue", label="rad/s")
# plt.legend()
# plt.show()
