import time
import numpy as np

from ctu_crs import CRS97
from helper import euler_xyz_to_matrix
from camera import (
    load_camera_calibration,
    estimate_aruco_world_positions,
)
from maze import maze_d as maze


PREDICTION_LEAD = 0.8       # Seconds ahead of the current time.
OBSERVATION_TIME = 15.0    # Seconds used to fit table motion.
START_DELAY = 8.0          # Delay before advancing along the maze.
PATH_STEP = 0.003          # 3 mm per completed movement.
END_MARGIN = 0.050         # Reverse 5 cm of path before the endpoint.


def maze_base_pose(p1, p2) -> np.ndarray:
    """Return the maze-base pose from the two marker centres."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    z_axis = np.array([0.0, 0.0, 1.0])

    diagonal = p1 - p2
    diagonal[2] = 0.0

    length = np.linalg.norm(diagonal)

    if length < 1e-9:
        raise ValueError("Marker centres have no usable XY separation.")

    diagonal /= length

    # Correct for the diagonal marker arrangement.
    y_axis = diagonal + np.cross(diagonal, z_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = (p1 + p2) / 2.0

    return T


def print_target_pose(T_BF_target, s, phi, returning):
    """Print the commanded flange pose in robot-base coordinates."""
    px, py, pz = T_BF_target[:3, 3]

    phi_deg = np.degrees(
        (phi + np.pi) % (2 * np.pi) - np.pi
    )

    direction = "RETURN" if returning else "FORWARD"

    print(
        f"\n[{direction}] "
        f"s = {s:.4f} m | "
        f"Predicted table angle = {phi_deg:.2f} deg"
    )

    print(
        f"Target flange position [m]: "
        f"x = {px:.4f}, y = {py:.4f}, z = {pz:.4f}"
    )

    # Top-left 3×3: flange orientation in robot-base coordinates.
    # Top-right column: flange position in metres.
    print(
        "T_BF_target:\n"
        f"{np.array2string(T_BF_target, precision=4, suppress_small=True)}"
    )


def main():
    s_max = maze.get_len()
    s_turn = s_max - END_MARGIN

    if not 0.0 < s_turn <= s_max:
        raise ValueError("END_MARGIN must leave a positive travel distance.")

    s = 0.0
    s_step = PATH_STEP
    tilt = True

    # Flange-to-hoop displacement, expressed in flange coordinates.
    p_FH = np.array([-0.135, 0.135, 0.0])

    # Reference convention:
    # At this known flange orientation, choose the upright hoop's
    # reference axes aligned with the robot-base axes.
    R_BH_reference = np.eye(3)
    R_BF_reference = euler_xyz_to_matrix(
        (0.0, -np.pi, np.pi / 2)
    )

    # Constant flange orientation relative to the chosen hoop frame.
    R_HF = R_BH_reference.T @ R_BF_reference

    # Rotate around the hoop normal to place the flange on the
    # outside of maze D's upper bend.
    R_spin = euler_xyz_to_matrix(
        (0.0, 0.0, -3 * np.pi / 4)
    )

    robot = CRS97()

    try:
        robot.initialize(home=True)

        calibration = load_camera_calibration()

        # Collect maze position and heading observations.
        observation_start = time.monotonic()

        ts, phis = [], []
        cx, cy, cz = [], [], []

        while time.monotonic() - observation_start < OBSERVATION_TIME:
            img = robot.grab_image()
            frame_time = time.monotonic()

            results = estimate_aruco_world_positions(
                img, calibration
            )
            detections = {d["id"]: d for d in results}

            if 1 not in detections or 2 not in detections:
                continue

            p1 = detections[1]["T_BM"][:3, 3]
            p2 = detections[2]["T_BM"][:3, 3]

            T_Bmaze_measured = maze_base_pose(p1, p2)

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
                f"Observation {len(ts)} "
                f"at {frame_time - observation_start:.3f} s"
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

        # Heading