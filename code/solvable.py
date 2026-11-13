import os

import cv2 as cv
import numpy as np
from helper import euler_xyz_to_matrix

# Import constants used in your sequence to ensure math exactly matches
from sequence import BASE_X_OFFSET, T_HOOP_TO_FLANGE, PATH_STEP, target_poses


def generate_reachability_map(robot, image, P_matrix, testing_height=0.2, angle_step_deg=1, radius_step_px=10):
    out_img = image.copy()
    height, width = out_img.shape[:2]

    # creating homography for fixed height
    H = np.column_stack((P_matrix[:, 0], P_matrix[:, 1], P_matrix[:, 2] * testing_height + P_matrix[:, 3]))
    H_inv = np.linalg.inv(H)

    # projecting world origin to img coordinates
    origin_world_h = np.array([0.0, 0.0, 1.0])
    origin_img_h = H @ origin_world_h
    u0 = int(origin_img_h[0] / origin_img_h[2])
    v0 = int(origin_img_h[1] / origin_img_h[2])

    # making sure the whole picture is covered
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    dists = np.linalg.norm(corners - np.array([u0, v0]), axis=1)
    max_radius_px = int(np.max(dists)) + 10

    # testing parameters
    angles = np.deg2rad(np.arange(0, 360, angle_step_deg))

    # initializing empty masks
    overlay_green = np.zeros_like(image, dtype=np.uint8)
    overlay_red = np.zeros_like(image, dtype=np.uint8)

    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        in_img = False

        for r in range(0, max_radius_px, radius_step_px):
            u = int(u0 + r * cos_a)
            v = int(v0 + r * sin_a)

            # pixel is not in img
            if not (0 <= u < width and 0 <= v < height):
                if in_img:
                    break
                continue

            in_img = True

            # transforming point from img coordinates to world coordinates
            pt_img_h = np.array([u, v, 1.0])
            pt_world_h = H_inv @ pt_img_h

            X = pt_world_h[0] / pt_world_h[2]
            Y = pt_world_h[1] / pt_world_h[2]

            # transforming point to hoop
            t_bh = np.eye(4)
            t_bh[:3, 3] = [X, Y, testing_height]
            t_bh[0, 3] += BASE_X_OFFSET

            # transforming hoop to flange
            t_bf_target = t_bh @ T_HOOP_TO_FLANGE

            # generating ik solutions
            ik_raw = list(robot.ik(t_bf_target))
            ik_sols = [q for q in ik_raw if robot.in_limits(q)]

            # adding BGR to masks
            if len(ik_sols) > 0:
                cv.circle(overlay_green, (u, v), 3, (0, 255, 0), -1)  # green
            else:
                cv.circle(overlay_red, (u, v), 3, (0, 0, 255), -1)  # red

    mask_g = overlay_green[:, :, 1] > 0
    mask_r = overlay_red[:, :, 2] > 0

    # adding green pts with 20 % opacity
    out_img[mask_g] = cv.addWeighted(image, 0.8, overlay_green, 0.2, 0)[mask_g]
    # adding red pts with 80 % opacity
    out_img[mask_r] = cv.addWeighted(image, 0.2, overlay_red, 0.8, 0)[mask_r]

    return out_img, mask_g, mask_r


def check_maze_placement(
    base_img,
    mask_r,
    P_matrix,
    maze,
    model,
    tilt=False,
    rotation_steps=72,
):
    out_img = base_img.copy()
    img_h, img_w = mask_r.shape
    is_solvable = True

    # testing all table rotations
    angles = np.linspace(0, 2 * np.pi, rotation_steps, endpoint=False)

    p_step = max(0.04, PATH_STEP)
    # print(p_step, maze.TOTAL_LENGTH/p_step)
    # Extract structural points along the maze wire
    # We sample 50 points along the path defined by TOTAL_LENGTH
    s_values = np.linspace(0, maze.TOTAL_LENGTH, int(maze.TOTAL_LENGTH/p_step))

    for s in s_values:
        trajectory = []
        collision = False

        for phi in angles:
            _, T_BH, _ = target_poses(maze, model, s, phi, tilt=tilt)
            T_BH[0, 3] -= BASE_X_OFFSET
            pt_world = np.array([T_BH[0, 3], T_BH[1, 3], T_BH[2, 3], 1.0])

            # world frame to image pixels
            pt_img_h = P_matrix @ pt_world

            u = int(pt_img_h[0] / pt_img_h[2])
            v = int(pt_img_h[1] / pt_img_h[2])

            # print(f"u: {u}, v: {v}")
            # print(img_w, img_h)

            # within image bounds check
            if not (0 <= u < img_w and 0 <= v < img_h):
                continue

            trajectory.append([u, v])

            # red mask collision check
            if mask_r[v, u]:
                collision = True
                is_solvable = False

        # drawing continuous circular trajectory for each sampled maze point
        if len(trajectory) > 1:
            pts = np.array(trajectory, np.int32).reshape((-1, 1, 2))
            color = (0, 255, 255) if collision else (255, 0, 0)  # Yellow = collision, Blue = safe
            cv.polylines(out_img, [pts], isClosed=True, color=color, thickness=2)

    return out_img, is_solvable


def run_visualization_pipeline(robot, maze, model, P_matrix, tilt=False, use_cached=False):
    """
    Main entry point for generating the dynamic visualization.
    Called directly from main.py.
    """
    raw_img_path = "reachability_imgs/reachability_img_raw.png"
    map_img_path = "reachability_imgs/reachability_img_map"
    check_img_path = "reachability_imgs/reachability_maze_check"

    img = None

    unsol_cntr = 0

    # 1. Handle raw image acquisition (cached or fresh)
    if use_cached:
        if os.path.isfile(raw_img_path):
            print(f"Loading cached raw image from '{raw_img_path}'...")
            img = cv.imread(raw_img_path)
        else:
            print(f"Warning: Cached raw image '{raw_img_path}' not found.")

    if img is None:
        print("Grabbing a fresh image from the robot...")
        img = robot.grab_image()

        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(raw_img_path), exist_ok=True)
        cv.imwrite(raw_img_path, img)
        print(f"Saved raw image to '{raw_img_path}'")

    # 2. Approximate testing height using the observed table center height
    # Adding a small offset to represent the average height of the maze wire
    approximate_maze_height = 0.21 #model["center"][2] + 0.3

    p_step = max(0.04, PATH_STEP)
    s_values = np.linspace(0, maze.TOTAL_LENGTH, int(maze.TOTAL_LENGTH/p_step))
    for s in s_values:
        _, T_BH, _ = target_poses(maze, model, s, 0, tilt=tilt)
        z = round(T_BH[2,3], 4)

        # 3. Generate the background masks and map image
        print("Calculating reachability map...")
        reachability_img, mask_g, mask_r = generate_reachability_map(
            robot=robot,
            image=img,
            P_matrix=P_matrix,
            testing_height=z,
            angle_step_deg=1.5,
            radius_step_px=15,
        )
        map_name = f"{map_img_path}_{z}.png"
        cv.imwrite(map_name, reachability_img)
        print(f"Saved reachability map to '{map_img_path}'")

        # 4. Draw the rotational bounds of the maze on top of the map
        print("Evaluating maze collisions...")
        final_img, solvable = check_maze_placement(
            base_img=reachability_img,
            mask_r=mask_r,
            P_matrix=P_matrix,
            maze=maze,
            model=model,
            tilt=tilt,
            rotation_steps=72,  # 5 deg steps
        )

        # 5. Save the final output
        check_name = f"{check_img_path}_{z}.png"
        cv.imwrite(check_name, final_img)
        print(f"Saved maze check to '{check_img_path}'")

        if not solvable:
            unsol_cntr += 1
            return False

    return not (unsol_cntr > 0)


if __name__ == "__main__":
    print("Please run this tool via main.py by pressing [r] in the menu.")
