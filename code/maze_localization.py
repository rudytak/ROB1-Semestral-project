import time

import numpy as np
from camera import estimate_aruco_world_positions
from maze_d import hoop_pose


def extract_markers_by_id(detections):
    """Map detected ArUco marker IDs to their 3D center positions in robot-base coordinates."""
    markers = {}
    for det in detections:
        if "id" in det and "center_base" in det:
            markers[int(det["id"])] = np.asarray(det["center_base"], dtype=np.float64)
    return markers


class MazePredictor:
    """Predicts orientation and target hoop pose of the rotating maze table in robot-base coordinates."""

    def __init__(self, center=None, omega=None):
        # Table geometry
        if center is not None:
            self.center_x = float(center[0])
            self.center_y = float(center[1])
            self.center_z = float(center[2])
        else:
            self.center_x = 0.0
            self.center_y = 0.0
            self.center_z = 0.0
        self.radius = 0.0

        # Motion state variables
        self.t0 = 0.0
        self.theta0 = 0.0
        self.omega = float(omega) if omega is not None else 0.0

    def estimate_rotational_table_geometry(
        self, robot, calibration, num_measurements=10, delay=0.1, marker_ids=(1, 2), fit_circle=True
    ):
        """Estimate rotation center and height of the table using ArUco markers."""
        maze_axis_points = []
        heights = []

        for _ in range(num_measurements):
            img = robot.grab_image()
            detections = estimate_aruco_world_positions(img, calibration)
            markers = extract_markers_by_id(detections)

            # Both markers must be visible to reliably compute the base center
            if marker_ids[0] in markers and marker_ids[1] in markers:
                center = (markers[marker_ids[0]] + markers[marker_ids[1]]) / 2.0
                maze_axis_points.append(center)
                heights.append(center[2])
            else:
                time.sleep(delay)
                continue

            time.sleep(delay)

        if len(maze_axis_points) < 3:
            print(
                f"Warning: Not enough valid measurements to estimate geometry: "
                f"{len(maze_axis_points)}/{num_measurements} (minimum 3 required)."
            )
            return False

        pts = np.asarray(maze_axis_points)
        x, y = pts[:, 0], pts[:, 1]
        self.center_z = float(np.mean(heights))

        if fit_circle:
            # Circle fitting using algebraic least-squares: (x - cx)^2 + (y - cy)^2 = r^2
            A = np.column_stack([x, y, np.ones_like(x)])
            b = x**2 + y**2
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            self.center_x = float(c[0] / 2.0)
            self.center_y = float(c[1] / 2.0)
            self.radius = float(np.sqrt(max(0.0, c[2] + self.center_x**2 + self.center_y**2)))
        else:
            # Base plate centered on rotation axis
            self.center_x = float(np.mean(x))
            self.center_y = float(np.mean(y))
            self.radius = 0.0

        return True

    def anchor_state(self, robot, calibration, estimated_omega=None, marker_ids=(1, 2)):
        """Capture ArUco markers to lock initial angle theta0 and timestamp t0."""
        img = robot.grab_image()
        detections = estimate_aruco_world_positions(img, calibration)
        markers = extract_markers_by_id(detections)

        if marker_ids[0] in markers and marker_ids[1] in markers:
            # Orientation vector from marker 2 to marker 1
            direction_vec = markers[marker_ids[0]] - markers[marker_ids[1]]
            self.theta0 = float(np.arctan2(direction_vec[1], direction_vec[0]))
        elif marker_ids[0] in markers:
            # Direction from table center to marker 1
            dx = markers[marker_ids[0]][0] - self.center_x
            dy = markers[marker_ids[0]][1] - self.center_y
            self.theta0 = float(np.arctan2(dy, dx))
        elif marker_ids[1] in markers:
            # Direction from table center to marker 2 (opposite side: offset by pi)
            dx = markers[marker_ids[1]][0] - self.center_x
            dy = markers[marker_ids[1]][1] - self.center_y
            self.theta0 = float(np.arctan2(dy, dx) + np.pi)
        else:
            print("Warning: Neither base marker was detected. State anchoring failed.")
            return False

        self.t0 = time.time()
        if estimated_omega is not None:
            self.omega = float(estimated_omega)

        return True

    def estimate_angular_velocity(self, robot, calibration, duration=3.0, delay=0.05, marker_id=1):
        """Estimate angular velocity (rad/s) by tracking marker rotation over time."""
        start_t = time.time()
        ts = []
        angles = []

        while time.time() - start_t < duration:
            img = robot.grab_image()
            detections = estimate_aruco_world_positions(img, calibration)
            markers = extract_markers_by_id(detections)

            if marker_id in markers:
                dx = markers[marker_id][0] - self.center_x
                dy = markers[marker_id][1] - self.center_y
                angles.append(np.arctan2(dy, dx))
                ts.append(time.time() - start_t)

            time.sleep(delay)

        if len(ts) < 3:
            print("Warning: Insufficient samples to estimate angular velocity.")
            return None

        # change interval to [-pi, pi]
        unwrapped_angles = np.unwrap(angles)
        diff_angles = np.diff(unwrapped_angles)
        diff_ts = np.diff(ts)

        valid_mask = diff_ts > 1e-4
        if not np.any(valid_mask):
            print("Warning: Zero time difference between consecutive samples.")
            return None

        omegas = diff_angles[valid_mask] / diff_ts[valid_mask]
        # Filter outliers
        filtered_omegas = [w for w in omegas if abs(w) < 10.0]

        if len(filtered_omegas) == 0:
            print("Warning: All angular velocity measurements were filtered as outliers.")
            return None

        self.omega = float(np.mean(filtered_omegas))
        return self.omega

    def _predict_angle(self, target_time):
        """Predict table orientation angle at a target timestamp."""
        dt = target_time - self.t0
        return self.theta0 + (self.omega * dt)

    def get_target_pose(self, s, time_delay_s=0.0):
        """Return 4x4 homogeneous transformation matrix of the hoop pose in robot-base coordinates."""
        target_time = time.time() + time_delay_s
        theta = self._predict_angle(target_time)

        # Transformation from robot base to table rotation center
        T_base_center = np.eye(4)
        T_base_center[:3, 3] = [self.center_x, self.center_y, self.center_z]

        # Table rotation around Z-axis
        c_t, s_t = np.cos(theta), np.sin(theta)
        T_rot = np.array([[c_t, -s_t, 0.0, 0.0], [s_t, c_t, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # Radius offset (if maze origin is shifted relative to rotation center)
        T_radius = np.eye(4)
        T_radius[0, 3] = self.radius

        # Hoop pose in local maze coordinates
        T_maze_hoop = np.eye(4)
        T_maze_hoop[2, 3] = 0.3  # 30 cm above maze base
        # T_maze_hoop = hoop_pose(s)

        return T_base_center @ T_rot @ T_radius @ T_maze_hoop
