import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from camera import estimate_aruco_world_positions
from sequence import (BASE_X_OFFSET, PREDICTION_LEAD, maze_base_pose, target_poses)



def draw_overlay(img, calibration, detections, maze, model=None, s=0.0, tilt=False, frame_time=None, prediction=None):
    """Draw global coordinates [m] on a copy of the camera image."""
    img = img.copy()

    K = calibration["K"]
    R = calibration["R_CB"]
    t = np.asarray(calibration["t_CB"]).reshape(3)

    # OpenCV uses BGR colours.
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)

    label_boxes = []

    def pixel(p):
        """Convert a global 3D point into image coordinates."""
        camera = R @ np.asarray(p) + t

        if not np.isfinite(camera).all() or camera[2] <= 1e-6:
            return None

        uv = K @ camera
        uv = uv[:2] / uv[2]

        if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e6:
            return None

        return tuple(np.rint(uv).astype(int))

    def text(label, uv, color=white):
        width = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1 )[0][0]

        uv = (max(0, min(uv[0], img.shape[1] - width)), max(14, min(uv[1], img.shape[0] - 5)))

        # Try nearby positions to reduce overlapping labels.
        shifts = [0] + [sign * 18 * i for i in range(1, 13) for sign in (1, -1) ]

        for shift in shifts:
            x = uv[0]
            y = max(14, min(uv[1] + shift, img.shape[0] - 5))
            box = (x - 2, y - 14, x + width + 2, y + 4)

            overlaps = any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in label_boxes)

            if not overlaps:
                break

        uv = (x, y)
        label_boxes.append(box)

        # Black outline makes text readable over the image.
        cv.putText(img, label, uv, cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(img, label, uv, cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)

    def point(label, p, color=white, offset=(8, -10)):
        uv = pixel(p)

        if uv is not None:
            cv.circle(img, uv, 5, color, 2, cv.LINE_AA)
            text(f"{label} ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m", (uv[0] + offset[0], uv[1] + offset[1]), color)

        return uv

    def line(a, b, color):
        a = pixel(a)
        b = pixel(b)

        if a is not None and b is not None:
            cv.line(img, a, b, color, 2, cv.LINE_AA)

    def axes(label, pose, offset=(8, -10)):
        p = pose[:3, 3]
        point(label, p, offset=offset)

        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))# X: red Y: green Z: blue

        for i, (name, color) in enumerate(zip("XYZ", colors)):
            tip = p + 0.035 * pose[:3, i]
            line(p, tip, color)

            uv = pixel(tip)
            if uv is not None:
                text(name, uv, color)

    # Coordinate systems of the two detected markers.
    for marker_id in (1, 2):
        if marker_id in detections:
            axes(f"ArUco {marker_id}", detections[marker_id]["T_BM"])

    # Measured maze coordinate system and current path point.
    observed = None

    if 1 in detections and 2 in detections:
        observed = maze_base_pose(detections[1]["T_BM"][:3, 3], detections[2]["T_BM"][:3, 3])

        axes("Maze observed", observed, (8, 22))

        observed_step = observed @ maze.hoop_pose(s, tilt=tilt)
        point("Step observed", observed_step[:3, 3], yellow)

    phi = None
    if observed is not None:
        phi = np.arctan2(observed[1, 0], observed[0, 0])

    status = "w = unavailable (requires fitted motion model)"

    # Table centre, radius and controller targets.
    if model is not None:
        timestamp = (time.monotonic() if frame_time is None else frame_time )

        phi_model = (model["phi_last"] + model["omega_scalar"] * (timestamp - model["last_time"]))

        base, current, _ = target_poses(maze, model, s, phi_model, tilt=tilt,)
        
        current[0, 3] -= 0.135


        # During a sequence, prediction is the actual commanded hoop pose.

        _, prediction, _ = target_poses(maze, model, s, phi_model + model["omega_scalar"] * PREDICTION_LEAD, tilt=tilt)

        prediction[0, 3] -= 0.135

        point("Table center (fit)", model["center"], cyan, (8, 40))

        # The radius uses the measured frame, without the target X offset.
        base_point = (base[:3, 3] - np.array([BASE_X_OFFSET, 0.0, 0.0]))

        if observed is not None:
            base_point = observed[:3, 3]
        else:
            base_fit = base.copy()
            base_fit[:3, 3] = base_point
            axes("Maze estimated", base_fit, (8, 22))

        line(model["center"], base_point, cyan)

        middle = pixel((model["center"] + base_point) / 2)

        if middle is not None:
            radius = np.linalg.norm((base_point - model["center"])[:2])
            text(f"r = {radius:.3f} m", middle, cyan)

        point("Target now", current[:3, 3], cyan, (8, 24))
        point(f"Prediction +{PREDICTION_LEAD:g}s", prediction[:3, 3], magenta, (8, -28))
            
        line(current[:3, 3], prediction[:3, 3], magenta)

        status = f"w = {model['omega_scalar']:+.4f} rad/s"

        if phi is None:
            phi = phi_model

    phi_text = "unavailable"

    if phi is not None:
        source = "measured" if observed is not None else "estimated"
        phi_text = f"{np.degrees(phi) % 360:.1f} deg ({source})"

    text(f"{status} | phi = {phi_text} | s = {s:.4f} m", (12, 22))
    text("Global XYZ [m] | axes: X red, Y green, Z blue" f" | targets: X offset {BASE_X_OFFSET:+.3f} m", (12, 44))
    text("q / Esc: close image only | main menu: i toggles plotting", (12, 66))

    if observed is None:
        text("Markers 1/2 missing: observed maze/step unavailable", (12, 88), yellow)

    return img


class LiveView:
    def __init__(self, robot, calibration, maze, tilt=False):
        self.robot = robot
        self.calibration = calibration
        self.maze = maze
        self.tilt = tilt

        self.enabled = True
        self.opened = False
        self.title = "Maze camera / global POI"

        self.folder = (
            Path(__file__).resolve().parent
            / "recordings"
            / time.strftime("%Y%m%d_%H%M%S")
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        self.frame_number = 0

        self.plot_times = []
        self.plot_positions = []
        self.plot_phis = []
        self.plot_omega = None

        print(f"Saving images to: {self.folder}")

    def capture(self):
        """Capture and detect before calculating the robot prediction."""
        if not self.enabled:
            return None

        img = self.robot.grab_image()
        timestamp = time.monotonic()

        try:
            results = estimate_aruco_world_positions(img, self.calibration)
            detections = {d["id"]: d for d in results}
            return img, timestamp, detections

        except:
            return None


    def show(self, model=None, s=0.0, prediction=None, frame=None):
        if not self.enabled:
            return False

        if frame is None:
            frame = self.capture()
            if frame is None:
                return self.enabled 

        img, timestamp, detections = frame

        self.plot_times.append(timestamp)

        if 1 in detections and 2 in detections:
            base = maze_base_pose(
                detections[1]["T_BM"][:3, 3],
                detections[2]["T_BM"][:3, 3],
            )

            self.plot_positions.append(base[:3, 3].copy())

            phi = np.arctan2(base[1, 0], base[0, 0])
            phi = (phi + np.pi) % (2 * np.pi) - np.pi

            self.plot_phis.append(phi)
        else:
            # Missing detections appear as gaps in the plots.
            self.plot_positions.append([np.nan, np.nan, np.nan])
            self.plot_phis.append(np.nan)

        if model is not None:
            self.plot_omega = model["omega_scalar"]

        img = draw_overlay(img, self.calibration, detections, self.maze, model=model, s=s, tilt=self.tilt, frame_time=timestamp, prediction=prediction)

        if not self.opened:
            cv.namedWindow(self.title, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.title, 1200, 800)
            self.opened = True

        cv.imshow(self.title, img)

        filename = self.folder / f"frame_{self.frame_number:06d}.png"

        if not cv.imwrite(str(filename), img):
            raise OSError(f"Could not save {filename}")

        self.frame_number += 1

        key = cv.waitKey(1) & 0xFF

        if (key in (27, ord("q")) or cv.getWindowProperty(self.title, cv.WND_PROP_VISIBLE) < 1):
            self.close()

        return self.enabled

    def save_gif(self):
        files = sorted(self.folder.glob("frame_*.png"))

        if not files:
            return

        # Sample across the whole recording: at most 100 frames.
        indices = np.linspace(
            0, len(files) - 1,
            min(100, len(files)),
            dtype=int,
        )

        frames = []

        for i in indices:
            with Image.open(files[i]) as image:
                image = image.convert("RGB")
                image.thumbnail((800, 600))
                frames.append(image.copy())

        output = self.folder / "preview.gif"

        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )

        print(f"GIF saved: {output}")

    def save_plots(self):

        if not self.plot_times:
            return

        t = np.asarray(self.plot_times)
        t = t - t[0]

        positions = np.asarray(self.plot_positions)
        phi = np.asarray(self.plot_phis)

        # Break the line where the wrapped angle crosses ±pi.
        phi_line = phi.copy()
        jumps = np.abs(np.diff(phi)) > np.pi
        phi_line[np.flatnonzero(jumps) + 1] = np.nan

        fig, axes = plt.subplots(
            3, 1,
            figsize=(11, 9),
            sharex=True,
            constrained_layout=True,
        )

        # Global coordinates of the observed maze base.
        for i, (label, color) in enumerate(
            zip(("x", "y", "z"), ("red", "green", "blue"))
        ):
            axes[0].plot(
                t, positions[:, i],
                label=label,
                color=color,
            )

        axes[0].set_title("Observed maze base — global coordinates")
        axes[0].set_ylabel("Position [m]")
        axes[0].legend()

        # Wrapped orientation.
        axes[1].plot(t, phi_line, color="purple")
        axes[1].set_ylabel(r"$\varphi$ [rad]")
        axes[1].set_ylim(-np.pi, np.pi)
        axes[1].set_yticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )

        # Constant angular velocity from the fitted motion model.
        if self.plot_omega is not None:
            axes[2].plot(
                t,
                np.full(t.shape, self.plot_omega),
                color="darkorange",
                label=f"Fitted ω = {self.plot_omega:.5f} rad/s",
            )
            axes[2].legend()
        else:
            axes[2].text(
                0.5, 0.5,
                "Angular velocity unavailable",
                ha="center",
                transform=axes[2].transAxes,
            )

        axes[2].set_ylabel(r"$\omega$ [rad/s]")
        axes[2].set_xlabel("Time since first frame [s]")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        fig.savefig(self.folder / "maze_motion.svg")
        fig.savefig(self.folder / "maze_motion.png", dpi=200)

        # Keep the original numeric data for later analysis.
        omega = np.full(
            t.shape,
            np.nan if self.plot_omega is None else self.plot_omega,
        )

        np.savetxt(
            self.folder / "maze_motion.csv",
            np.column_stack((t, positions, phi, omega)),
            delimiter=",",
            header="time_s,x_m,y_m,z_m,phi_rad,omega_rad_s",
            comments="",
        )

        print(f"Plots and data saved to: {self.folder}")

        plt.show()
        plt.close(fig)

    def close(self):
        if self.opened:
            try:
                cv.destroyWindow(self.title)
            except cv.error:
                # The window manager may have already closed it.
                pass

            self.opened = False

        self.enabled = False