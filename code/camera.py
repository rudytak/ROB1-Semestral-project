import numpy as np
from matplotlib import pyplot as plt
import json
import os
import cv2 as cv
from helper import euler_xyz_to_matrix, matrix_to_euler_xyz

# TODO: possibly use extra hoop point outside just centers?
# TODO: acquire K externally using a checkerboard or chAruUco? if hoop approach isn't good enough

METADATA_PATH = 'calibration_data/metadata.json'
CALIBRATION_PATH = 'calibration_data/camera_calib.json'

def take_calibration_pictures(robot, divisions = (3,3,5), target_rotation_RPY = (0, -np.pi, 0), x_range = (0.4, 0.525), y_range = (-0.17, 0.17), z_range = (0.049, 0.15)):

    def select_good_rotation(x,y,z):
        center = (
            (x_range[-1] - x_range[0])/2 + x_range[0],
            (y_range[-1] - y_range[0])/2 + y_range[0],
            (z_range[-1] - z_range[0])/2 + z_range[0],
        )

        if target_rotation_RPY == 'auto':
            if x >= center[0]:
                return euler_xyz_to_matrix(
                    (0, -np.pi, 0)
                )
            elif y < center[1]:
                return euler_xyz_to_matrix(
                    (0, -np.pi, -np.pi/2)
                )
            elif y >= center[1]:
                return euler_xyz_to_matrix(
                    (0, -np.pi, np.pi/2)
                )
            else:
                return euler_xyz_to_matrix(
                    (0, -np.pi, 0)
                )
        else:
            return euler_xyz_to_matrix(
                target_rotation_RPY
            )

    data_points = []
    hoop_arm_length = .135

    i = 0
    for x in np.linspace(x_range[0], x_range[-1], divisions[0]):
        for y in np.linspace(y_range[0], y_range[-1], divisions[1]):
            for z in np.linspace(z_range[0], z_range[-1], divisions[2]):
                i+=1

                # for orientation in viable_orientations:
                target_rotation = select_good_rotation(x,y,z)
                target_position = np.array([x,y,z]) - target_rotation @ np.array([-hoop_arm_length, 0, 0])

                # Create homogeneous SE(3) transformation
                target_pose = np.eye(4)
                target_pose[:3, :3] = target_rotation # target rotation matrix
                target_pose[:3, 3] = target_position # target position

                # Move to the target position using IK
                ik_sols = robot.ik(target_pose)
                # filter solution for reachability
                ik_sols = list(filter(lambda q: robot.in_limits(q), ik_sols))

                if len(ik_sols) > 0:
                    # Select the closest position
                    # q0 = robot.get_q()
                    q0 = robot.q_home
                    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

                    robot.move_to_q(closest_solution)
                    robot.wait_for_motion_stop()
                else:
                    print(f"WARN: position {x},{y},{z} is not reachable")
                    continue

                # Take the image
                img = robot.grab_image()

                # Save the image
                os.makedirs("calibration_data", exist_ok=True)
                filepath = f"calibration_data/calibration_img_{i}.png"
                cv.imwrite(filepath, img)
                # Save metadata
                final_pose = robot.fk(robot.get_q())
                data_points.append({
                    "filepath": filepath,
                    "RPY": matrix_to_euler_xyz(final_pose[:3, :3]).tolist(),
                    "translation_vector": final_pose[:3, 3].tolist()
                })

    # save the metadata json
    with open(METADATA_PATH, 'w') as f:
        json.dump(data_points, f, ensure_ascii=False, indent=4)

    return data_points

def load_calibration_data():
    if not os.path.isfile(METADATA_PATH):
        raise Exception("Calibration metadata missing! Run take_calibration_pictures() first")

    metadata = None
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    images = []
    for meta in metadata:
        img = cv.imread(meta["filepath"])
        images.append(img)

    return metadata, images

def detect_hoop_positions(images):
    # Detection super-parameters
    max_hue_dist = 40
    min_circle_area = 5000
    minimal_roundness_deviation = 1 # 100% (is pretty much ignored becase cv.arcLength seems to be under-reporting)

    # Measured hoop color samples
    circle_color_samples = [
        (48, 84, 111),
        (52, 78, 96),
        (53, 82, 112),
        (82, 104, 135),
        (88, 105, 117),
        (83, 111, 137)
    ]
    target_color_rgb = np.mean(circle_color_samples, axis=0).astype(np.uint8)
    target_color_hsv = cv.cvtColor(target_color_rgb.reshape(1, 1, 3), cv.COLOR_RGB2HSV).reshape(3,)

    # print(f"Target color in HSV: {target_color}, max hue distance: {max_hue_dist}")
    lower_color_bound = np.array([target_color_hsv[0] - max_hue_dist, 50, 50])
    upper_color_bound = np.array([target_color_hsv[0] + max_hue_dist, 255, 255])

    centers = []
    for img in images:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_color_bound, upper_color_bound)

        contours_all, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours_all if cv.contourArea(c) > min_circle_area]
        print(f"Found {len(contours_all)} contours in the image of which {len(contours)} are larger than {min_circle_area} pixels")

        # find the most circular contour
        f = lambda c: abs(4*np.pi*cv.contourArea(c) / (cv.arcLength(c, True)**2 + 1e-8) - 1)
        passing_contours = list(filter(lambda c: f(c) < minimal_roundness_deviation, contours))
        print(list(map(f, contours)), list(map(cv.contourArea, contours)), list(map(lambda c: cv.arcLength(c,True), contours)))
        if len(passing_contours) == 0:
            raise Exception("No sufficiently good circle has been found in the image!")

        best = min(passing_contours, key=f)
        center = np.mean(best.reshape(-1, 2), axis=0)

        # cv.drawContours(img, [best], -1, (255, 0, 0), 1)
        # plt.figure(figsize=(9, 6))
        # plt.imshow(img)
        # plt.scatter([center[0]], [center[1]])
        # plt.tight_layout()
        # plt.show()

        centers.append(center)
    centers = np.array(centers).reshape(-1, 2)

    return centers






def normalize_2d_points(points):
    """Normalize Nx2 image points to centroid 0, mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))

    if mean_dist < 1e-12:
        raise ValueError("Degenerate image points")

    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    points_h = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ points_h.T).T

    return normalized[:, :2], T

def normalize_3d_points(points):
    """Normalize Nx3 world points to centroid 0, mean distance sqrt(3)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = np.mean(np.linalg.norm(centered, axis=1))

    if mean_dist < 1e-12:
        raise ValueError("Degenerate world points")

    scale = np.sqrt(3.0) / mean_dist
    T = np.eye(4)
    T[:3, :3] *= scale
    T[:3, 3] = -scale * centroid

    points_h = np.column_stack([points, np.ones(len(points))])
    normalized = (T @ points_h.T).T

    return normalized[:, :3], T

def estimate_projection_matrix(world_points, image_points):
    """
    Estimate P from Nx3 world points and Nx2 image points using
    normalized DLT. Requires at least 6 generic non-coplanar points.
    """
    world_points = np.asarray(world_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    if len(world_points) != len(image_points):
        raise ValueError("3D and 2D correspondence counts differ")
    if len(world_points) < 6:
        raise ValueError("Need at least 6 generic 3D-2D correspondences")

    Xn, T3 = normalize_3d_points(world_points)
    xn, T2 = normalize_2d_points(image_points)

    Xh = np.column_stack([Xn, np.ones(len(Xn))])
    A = []

    for (X, Y, Z, W), (u, v) in zip(Xh, xn):
        A.append([X, Y, Z, W, 0, 0, 0, 0,
                  -u*X, -u*Y, -u*Z, -u*W])
        A.append([0, 0, 0, 0, X, Y, Z, W,
                  -v*X, -v*Y, -v*Z, -v*W])

    A = np.asarray(A)

    # Solve A p = 0 using SVD.
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=True)
    Pn = Vt[-1].reshape(3, 4)

    # Undo coordinate normalization:
    # x_normalized = T2 @ x; X_normalized = T3 @ X
    P = np.linalg.inv(T2) @ Pn @ T3

    # Arbitrary scale normalization
    P /= np.linalg.norm(P[2, :3])

    return P, singular_values

def calibrate_camera():
    # USES DLT
    metadata, images = load_calibration_data()

    # Robot-base coordinates of the observed hoop center
    world_points = np.asarray(
        [m["translation_vector"] for m in metadata],
        dtype=np.float64
    )

    # Detected hoop centers, one (u, v) pair per image
    image_points = detect_hoop_positions(images)
    image_points = np.asarray(image_points, dtype=np.float64)

    if image_points.shape != (len(world_points), 2):
        raise ValueError(
            f"Expected image points with shape "
            f"({len(world_points)}, 2), got {image_points.shape}. "
            "Check detect_hoop_positions()."
        )

    # Estimate projection matrix
    P, singular_values = estimate_projection_matrix(
        world_points, image_points
    )

    # Decompose P = K [R | t]
    K, R_CB, camera_center_h, *_ = cv.decomposeProjectionMatrix(P)

    # Normalize K to conventional form
    K = K / K[2, 2]

    # OpenCV gives the camera center as a homogeneous point
    camera_center_base = (
        camera_center_h[:3, 0] / camera_center_h[3, 0]
    )

    # Camera center C = -R.T @ t, so t = -R @ C
    t_CB = -R_CB @ camera_center_base

    # Reprojection error using the estimated P
    world_h = np.column_stack([
        world_points, np.ones(len(world_points))
    ])
    projected_h = (P @ world_h.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]

    errors = np.linalg.norm(projected - image_points, axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    print("Projection matrix P:\n", P)
    print("Intrinsic matrix K:\n", K)
    print("World-to-camera rotation R_CB:\n", R_CB)
    print("World-to-camera translation t_CB:\n", t_CB)
    print("Camera position in robot base:\n", camera_center_base)
    print(f"RMS reprojection error: {rms_error:.3f} pixels")

    return {
        "P": P,
        "K": K,
        "R_CB": R_CB,
        "t_CB": t_CB,
        "camera_position_base": camera_center_base,
        "rms_error": rms_error,
        "reprojection_errors": errors
    }

def save_camera_calibration():
    res = calibrate_camera()

    # save the metadata json
    with open(CALIBRATION_PATH, 'w') as f:
        json.dump({
            "P": res['P'].tolist(),
            "K": res['K'].tolist(),
            "R_CB": res['R_CB'].tolist(),
            "t_CB": res['t_CB'].tolist(),
            "camera_position_base": res['camera_position_base'].tolist(),
            "rms_error": res['rms_error'].tolist(),
            "reprojection_errors": res['reprojection_errors'].tolist()
        }, f, ensure_ascii=False, indent=4)

def load_camera_calibration():
    if not os.path.isfile(CALIBRATION_PATH):
        raise Exception("Camera calibration data missing! Run take_calibration_pictures() and save_camera_calibration() first")

    data = None
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)

    return {
        "P": np.array(data['P']),
        "K": np.array(data['K']),
        "R_CB": np.array(data['R_CB']),
        "t_CB": np.array(data['t_CB']),
        "camera_position_base": np.array(data['camera_position_base']),
        "rms_error": np.array(data['rms_error']),
        "reprojection_errors": np.array(data['reprojection_errors'])
    }




def detect_ArUco(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initialize ArUco detector
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    detector = cv.aruco.ArucoDetector(aruco_dict)

    corners, ids, rej = detector.detectMarkers(gray)
    
    # imgc = img.copy()
    # cv.aruco.drawDetectedMarkers(imgc,corners,ids)
    # plt.figure(figsize=(10,5))
    # plt.imshow(imgc)
    # plt.show()

    return corners, ids, rej


def estimate_aruco_world_positions(
    img,
    calibration,
    marker_size=0.04,
    dist_coeffs=None,
    dictionary_id=cv.aruco.DICT_4X4_50,
):
    """
    Detect ArUco markers and estimate their 3D poses in the robot-base frame.

    Parameters
    ----------
    img : np.ndarray
        BGR image containing the markers.
    calibration : dict
        Output dictionary from calibrate_camera(), containing K, R_CB, t_CB.
    marker_size : float
        Physical marker side length in metres (default 0.03 m).
    dist_coeffs : np.ndarray or None
        Camera distortion coefficients. None assumes zero distortion.
    dictionary_id : int
        OpenCV ArUco dictionary identifier.

    Returns
    -------
    list of dict
        One entry per detected marker, including its center in the robot
        base frame and its depth relative to the camera.
    """
    K = np.asarray(calibration["K"], dtype=np.float64)
    R_CB = np.asarray(calibration["R_CB"], dtype=np.float64)
    t_CB = np.asarray(calibration["t_CB"], dtype=np.float64).reshape(3)

    if marker_size <= 0:
        raise ValueError("marker_size must be positive.")

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    else:
        dist_coeffs = np.asarray(
            dist_coeffs, dtype=np.float64
        ).reshape(-1, 1)

    # Transformation from camera coordinates to robot-base coordinates.
    R_BC = R_CB.T
    t_BC = -R_BC @ t_CB

    T_BC = np.eye(4)
    T_BC[:3, :3] = R_BC
    T_BC[:3, 3] = t_BC

    # Detect markers.
    corners, ids, rejected = detect_ArUco(img)

    # Marker-local corners, centered at the marker origin.
    # Ordering matches OpenCV's SOLVEPNP_IPPE_SQUARE convention.
    half = marker_size / 2.0
    object_points = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    results = []

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        image_points = marker_corners.reshape(4, 2).astype(
            np.float64
        )

        success, rvec, tvec = cv.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            continue

        R_CM, _ = cv.Rodrigues(rvec)
        t_CM = tvec.reshape(3)

        # Marker pose in camera coordinates.
        T_CM = np.eye(4)
        T_CM[:3, :3] = R_CM
        T_CM[:3, 3] = t_CM

        # Marker pose in robot-base coordinates.
        T_BM = T_BC @ T_CM

        # The marker-local origin is its center.
        center_base = T_BM[:3, 3]
        center_camera = t_CM

        # Reproject the corners to check the pose fit.
        projected, _ = cv.projectPoints(
            object_points, rvec, tvec, K, dist_coeffs
        )
        projected = projected.reshape(4, 2)
        reprojection_error = np.sqrt(
            np.mean(np.sum(
                (projected - image_points) ** 2, axis=1
            ))
        )

        results.append({
            "id": int(marker_id),
            "center_base": center_base,
            "center_camera": center_camera,
            "depth": float(center_camera[2]),
            "range": float(np.linalg.norm(center_camera)),
            "R_BM": T_BM[:3, :3],
            "T_BM": T_BM,
            "reprojection_error_px": float(reprojection_error),
        })

    return results
