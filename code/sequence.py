import time
import numpy as np

from camera import estimate_aruco_world_positions
from helper import euler_xyz_to_matrix, print_pose


# PREDICTION_LEAD = 2.4
PREDICTION_LEAD = 1

OBSERVATION_TIME = 15.0
START_DELAY = 8.0
PATH_STEP = 0.0003
# PATH_STEP = 0.04
# PATH_STEP = 0.0
END_MARGIN = 0.03
BASE_X_OFFSET = 0.135



T_FLANGE_TO_HOOP = np.eye(4)
T_FLANGE_TO_HOOP[:3, :3] = euler_xyz_to_matrix((0, np.pi, 3*np.pi/8))
T_FLANGE_TO_HOOP[:3, 3] = [-0.135, 0.0, 0.0066]

T_HOOP_TO_FLANGE = np.linalg.inv(T_FLANGE_TO_HOOP)

def maze_base_pose(p1, p2):
    """Calculate the maze-base pose from the marker centres."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    z_axis = np.array([0.0, 0.0, 1.0])

    diagonal = p1 - p2
    diagonal[2] = 0.0

    length = np.linalg.norm(diagonal)
    if length < 1e-9:
        raise ValueError(
            "Marker centres have no usable XY separation."
        )

    diagonal /= length

    y_axis = diagonal + np.cross(diagonal, z_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack(
        (x_axis, y_axis, z_axis)
    )
    T[:3, 3] = (p1 + p2) / 2.0

    return T



def calculate(robot, calib_results, duration=OBSERVATION_TIME):
    """Observe the maze and return fitted table-motion parameters."""
    start_t = time.monotonic()

    ts, phis = [], []
    cx, cy, cz = [], [], []

    while time.monotonic() - start_t < duration:
        img = robot.grab_image()
        frame_time = time.monotonic()

        results = estimate_aruco_world_positions(
            img, calib_results
        )
        detections = {d["id"]: d for d in results}

        if 1 not in detections or 2 not in detections:
            continue

        p1 = detections[1]["T_BM"][:3, 3]
        p2 = detections[2]["T_BM"][:3, 3]

        T_Bmaze = maze_base_pose(p1, p2)

        phi = np.arctan2(
            T_Bmaze[1, 0],
            T_Bmaze[0, 0],
        )

        ts.append(frame_time)
        phis.append(phi)

        cx.append(T_Bmaze[0, 3])
        cy.append(T_Bmaze[1, 3])
        cz.append(T_Bmaze[2, 3])

        print(
            f"Took img {len(ts)} "
            f"at {frame_time - start_t:.3f} s"
        )

    if len(ts) < 3:
        raise ValueError(
            "Not enough observations of markers 1 and 2."
        )

    sample_times = np.asarray(ts)
    measured_phis = np.unwrap(np.asarray(phis))
    xy = np.column_stack((cx, cy))

    last_time = sample_times[-1]
    relative_times = sample_times - last_time

    omega_scalar, phi_last = np.polyfit(
        relative_times, measured_phis, 1
    )

    # maze_xy = table_center_xy + Rz(phi) @ offset_xy
    c = np.cos(measured_phis)
    sn = np.sin(measured_phis)

    A = np.zeros((2 * len(ts), 4))

    A[0::2, 0] = 1.0
    A[0::2, 2] = c
    A[0::2, 3] = -sn

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
            "Too little observed rotation "
            "to estimate the table centre."
        )

    model = {
        "center": np.array([
            solution[0],
            solution[1],
            np.mean(cz),
        ]),
        "offset_local": np.array([
            solution[2],
            solution[3],
            0.0,
        ]),
        "omega_scalar": omega_scalar,
        "phi_last": phi_last,
        "last_time": last_time,
    }

    print(f"Table centre [m]: {model['center']}")
    print(f"Angular speed [rad/s]: {omega_scalar:.5f}")

    return model



def target_poses(maze, model, s, phi, tilt=False):
    """Return maze, hoop, and flange poses in robot coordinates."""
    R_Bmaze = euler_xyz_to_matrix((0.0, 0.0, phi))

    T_Bmaze = np.eye(4)
    T_Bmaze[:3, :3] = R_Bmaze
    T_Bmaze[:3, 3] = (
        model["center"]
        + R_Bmaze @ model["offset_local"]
    )
    T_Bmaze[0, 3] += BASE_X_OFFSET

    T_maze_H = maze.hoop_pose(s, tilt=tilt)

    T_BH = T_Bmaze @ T_maze_H
    T_BF = T_BH @ T_HOOP_TO_FLANGE

    return T_Bmaze, T_BH, T_BF


def check_sols(robot, maze, model, tilt=False):
    """Check the maze entrance at two extreme table positions."""
    center = np.asarray(model["center"], dtype=float).copy()
    center[0] += BASE_X_OFFSET

    T_maze_H = maze.hoop_pose(0.0, tilt=tilt)
    entrance_offset = (np.asarray(model["offset_local"], dtype=float)+ T_maze_H[:3, 3])

    center_distance = np.linalg.norm(center[:2])
    radius = np.linalg.norm(entrance_offset[:2])


    direction_angle = np.arctan2(center[1], center[0])
    offset_angle = np.arctan2(entrance_offset[1], entrance_offset[0])

    phi_far = direction_angle - offset_angle
    phi_near = phi_far + np.pi

    passed = True

    for label, phi in (("nearest", phi_near), ("farthest", phi_far)):
        _, T_BH, T_BF = target_poses(maze, model, s=0.0, phi=phi, tilt=tilt)

        valid_solutions = [q for q in robot.ik(T_BF) if robot.in_limits(q)]

        reachable = bool(valid_solutions)
        passed = passed and reachable

        print(
            f"Maze entrance — {label}: "
            f"{'PASS' if reachable else 'FAIL'} "
            f"({len(valid_solutions)} solutions within limits)"
        )
        print_pose(T_BH, "hoop")
        print_pose(T_BF, "flange")

    return passed


def run_sequence(robot, maze, model, tilt=False):
    """Run one outward-and-return traversal."""
    s_max = maze.TOTAL_LENGTH
    s_turn = s_max - END_MARGIN

    if not 0.0 < s_turn <= s_max:
        raise ValueError("END_MARGIN leaves no usable maze path.")

    s = 0.0
    s_step = PATH_STEP

    move_start = time.monotonic()


    while True:
        dt = time.monotonic() - model["last_time"]

        phi = (model["phi_last"] + model["omega_scalar"] * (dt + PREDICTION_LEAD))

        T_B_to_maze_base, T_BH, T_BF_target = target_poses(maze, model, s, phi, tilt=tilt)

        print_pose(T_B_to_maze_base, "maze base")
        print_pose(T_BH, "hoop")
        print_pose(T_BF_target, "flange")

        print(
            f"\ns = {s:.4f} m | "
            f"step = {s_step:+.4f} m"
        )

        print(
            "T_BF_target:\n"
            f"{np.array2string(T_BF_target, precision=4, suppress_small=True)}"
        )

        ik_raw = list(robot.ik(T_BF_target))
        ik_sols = [q for q in ik_raw if robot.in_limits(q)]

        print(
            f"IK solutions: {len(ik_raw)} raw, "
            f"{len(ik_sols)} within limits"
        )

        if not ik_sols:
            print("No valid IK solution. Returning to the menu.")
            continue

        closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - robot.q_home))

        robot.move_to_q(closest_solution)
        robot.wait_for_motion_stop()

        if time.monotonic() - move_start < START_DELAY:
            continue

        if s_step > 0 and s >= s_turn:
            s_step = -PATH_STEP

        elif s_step < 0 and s <= 0.0:
            print(
                "Returned to the starting point. "
                "Sequence finished."
            )
            return

        s = min(s_turn, max(0.0, s + s_step))