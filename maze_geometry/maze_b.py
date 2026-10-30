import math
import numpy as np

SEGMENT_LENGTHS = (
    0.050,                   # Upper vertical.
    0.070 * math.sqrt(2.0),   # Diagonal.
    0.080,                   # Lower vertical.
)
TOTAL_LENGTH = sum(SEGMENT_LENGTHS)


def position(s: float) -> tuple[float, float, float]:
    """Return the centreline position in local maze coordinates, in metres.

    Starts at (0.070, 0, 0.200) and ends at (0, 0, 0).
    """
    if not math.isfinite(s) or not 0.0 <= s <= TOTAL_LENGTH:
        raise ValueError(
            f"s must be finite and between 0 and {TOTAL_LENGTH} m"
        )

    # Upper vertical: travel downward.
    if s <= SEGMENT_LENGTHS[0]:
        return (0.070, 0.0, 0.200 - s)

    s -= SEGMENT_LENGTHS[0]

    # Diagonal: travel toward decreasing X and Z.
    if s <= SEGMENT_LENGTHS[1]:
        d = s / math.sqrt(2.0)
        return (0.070 - d, 0.0, 0.150 - d)

    s -= SEGMENT_LENGTHS[1]

    # Lower vertical: travel downward to the base.
    return (0.0, 0.0, 0.080 - s)


def rail_direction(s: float) -> tuple[float, float, float]:
    """Return the unit direction toward increasing s.

    At sharp corners, use the outgoing segment's direction.
    """
    position(s)  # Validate s.

    corner_1 = SEGMENT_LENGTHS[0]
    corner_2 = sum(SEGMENT_LENGTHS[:2])

    if s < corner_1:
        return (0.0, 0.0, -1.0)

    if s < corner_2:
        q = 1.0 / math.sqrt(2.0)
        return (-q, 0.0, -q)

    return (0.0, 0.0, -1.0)


def hoop_pose(s: float) -> np.ndarray:
    """Return the hoop's 4×4 pose in local maze coordinates.

    Hoop +Z points opposite travel.
    Hoop +Y stays aligned with maze +Y.
    """
    z_axis = -np.array(rail_direction(s), dtype=float)
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = position(s)

    return T

################################################
# DEBUG LIBRARY
################################################


def visualize() -> None:
    """Interactive plot for the parametrization of the maze with the relevant transfromations for
    the hoop going through.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    # Sample positions along the rail.
    count = 500
    s_values = [
        TOTAL_LENGTH * i / (count - 1)
        for i in range(count)
    ]

    # Include section boundaries to preserve the sharp corners.
    boundary = 0.0
    for length in SEGMENT_LENGTHS[:-1]:
        boundary += length
        s_values.append(boundary)

    s_values.sort()
    points = [position(s) for s in s_values]
    
    x, y, z = zip(*points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, color="black", linewidth=2,
            label="Rail centreline")
    ax.scatter(*points[0], color="blue", label="Start")
    ax.scatter(0, 0, 0, color="black", label="Base origin")

    # Fixed coordinate frame at the base
    length = 0.025
    axes = [
        ((length, 0, 0), "red", "X"),
        ((0, length, 0), "green", "Y"),
        ((0, 0, length), "blue", "Z"),
    ]

    for direction, color, label in axes:
        ax.quiver(
            0, 0, 0, *direction,
            color=color,
            arrow_length_ratio=0.2,
        )
        ax.text(*direction, label, color=color)

    # Moving point starts at the upper tip.
    px, py, pz = position(0)
    moving_point, = ax.plot(
        [px], [py], [pz],
        "o",
        color="orange",
        markersize=9,
        label="Selected position",
    )

    coordinate_text = fig.text(
        0.15, 0.13, "", family="monospace"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Maze D — move along the rail with the slider below")

    # Match box proportions to axis ranges: equal physical scale.
    ax.set_xlim(-0.04, 0.10)
    ax.set_ylim(-0.04, 0.04)
    ax.set_zlim(-0.02, 0.18)
    ax.set_box_aspect((0.14, 0.08, 0.20))

    ax.view_init(elev=20, azim=-60)
    ax.legend()

    slider_ax = fig.add_axes([0.15, 0.06, 0.65, 0.03])
    slider = Slider(
        slider_ax,
        "s [m]",
        0.0,
        TOTAL_LENGTH,
        valinit=0.0,
        valfmt="%.5f",
    )


    hoop_arrows = []
    hoop_labels = []

    def update(s):
        T = hoop_pose(s)

        p = T[:3, 3]      # Hoop origin.
        R = T[:3, :3]    # Columns are hoop X, Y, Z directions.

        moving_point.set_data_3d([p[0]], [p[1]], [p[2]])

        for arrow in hoop_arrows:
            arrow.remove()
        hoop_arrows.clear()

        for label in hoop_labels:
            label.remove()
        hoop_labels.clear()

        arrow_length = 0.025

        for i, (color, name) in enumerate([
            ("red", "Xh"),
            ("green", "Yh"),
            ("blue", "Zh"),
        ]):
            direction = R[:, i]

            arrow = ax.quiver(
                *p,
                *direction,
                length=arrow_length,
                normalize=True,
                color=color,
                linewidth=2,
                arrow_length_ratio=0.25,
            )
            hoop_arrows.append(arrow)

            tip = p + arrow_length * direction
            hoop_labels.append(
                ax.text(*tip, name, color=color)
            )

        coordinate_text.set_text(
            f"Position [m]: ({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})\n"
            f"Hoop +Z:      ({R[0, 2]:.3f}, "
            f"{R[1, 2]:.3f}, {R[2, 2]:.3f})"
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0.0)



    fig._maze_slider = slider

    plt.show()





################################################
# DEBUG
################################################
if __name__ == "__main__" :
    visualize()
