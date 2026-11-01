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

        # Heading model:
        # phi(t) = phi_last + omega_scalar * (t - last_time).
        omega_scalar, phi_last = np.polyfit(
            relative_times, measured_phis, 1
        )

        # Position model:
        # maze_xy = table_center_xy + Rz(phi) @ offset_xy.
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
                "Too little observed rotation to estimate the table centre."
            )

        center = np.array([
            solution[0],
            solution[1],
            np.mean(cz),
        ])

        offset_local = np.array([
            solution[2],
            solution[3],
            0.0,
        ])

        print(f"\nTable centre [m]: {center}")
        print(f"Angular speed [rad/s]: {omega_scalar:.5f}")
        print(f"Axis-to-maze offset [m]: {offset_local}")
        print(f"Turnaround distance s [m]: {s_turn:.4f}")

        move_start = time.monotonic()

        while True:
            dt = time.monotonic() - last_time

            phi = (
                phi_last
                + omega_scalar * (dt + PREDICTION_LEAD)
            )

            # Predict the maze pose.
            R_Bmaze = euler_xyz_to_matrix(
                (0.0, 0.0, phi)
            )
            p_Bmaze = center + R_Bmaze @ offset_local

            # Maze D supplies the desired local hoop pose.
            T_maze_H = maze.hoop_pose(s, tilt=tilt)

            p_maze_H = T_maze_H[:3, 3]
            R_maze_H = T_maze_H[:3, :3]

            # Desired hoop centre in robot-base coordinates.
            p_BH_target = p_Bmaze + R_Bmaze @ p_maze_H

            # Table rotation + maze orientation + chosen hoop spin.
            R_BH_target = R_Bmaze @ R_maze_H @ R_spin

            # Convert the hoop orientation to flange orientation.
            R_BF = R_BH_target @ R_HF

            # Construct the complete flange target.
            T_BF_target = np.eye(4)
            T_BF_target[:3, :3] = R_BF
            T_BF_target[:3, 3] = (
                p_BH_target - R_BF @ p_FH
            )

            print_target_pose(
                T_BF_target,
                s,
                phi,
                returning=s_step < 0,
            )

            ik_sols = [
                q
                for q in robot.ik(T_BF_target)
                if robot.in_limits(q)
            ]

            print(f"Valid IK solutions: {len(ik_sols)}")

            if not ik_sols:
                print("No valid IK solution. Ending the run.")
                break

            # Prefer the solution closest to the current joint position.
            q_current = np.asarray(robot.get_q())

            closest_solution = min(
                ik_sols,
                key=lambda q: np.linalg.norm(q - q_current),
            )

            robot.move_to_q(closest_solution)
            robot.wait_for_motion_stop()

            # Track the approach point before advancing along the path.
            if time.monotonic() - move_start < START_DELAY:
                continue

            # Reverse only after reaching the turnaround target.
            if s_step > 0 and s >= s_turn:
                s_step = -PATH_STEP
                print("Turnaround reached. Returning.")

            # Finish only after reaching the starting target again.
            elif s_step < 0 and s <= 0.0:
                print("Returned to the starting point.")
                break

            s = min(s_turn, max(0.0, s + s_step))

    finally:
        robot.close()


if __name__ == "__main__":
    main()