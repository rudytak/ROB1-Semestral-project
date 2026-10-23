import math

SEGMENT_LENGTHS = (0.05, 0.025 * math.pi, 0.06, 0.015 * math.sqrt(2.0), 0.03)
TOTAL_LENGTH = sum(SEGMENT_LENGTHS)


def position(s: float) -> tuple[float, float, float]:
    """Return (x, y, z) at arc length s in [0, TOTAL_LENGTH], in m.

    Starts at the top tip (0.085, 0, 0.155) and ends at the plate (0, 0, 0).
    """
    if not math.isfinite(s) or not 0.0 <= s <= TOTAL_LENGTH:
        raise ValueError(f's must be finite and between 0 and {TOTAL_LENGTH} m')
    if s <= 0.05:
        return (0.085 - s, 0.0, 0.155)
    s -= 0.05
    if s <= SEGMENT_LENGTHS[1]:
        a = s / 0.05
        return (0.035 - 0.05 * math.sin(a), 0.0, 0.105 + 0.05 * math.cos(a))
    s -= SEGMENT_LENGTHS[1]
    if s <= 0.06:
        return (-0.015, 0.0, 0.105 - s)
    s -= 0.06
    if s <= SEGMENT_LENGTHS[3]:
        d = s / math.sqrt(2.0)
        return (-0.015 + d, 0.0, 0.045 - d)
    s -= SEGMENT_LENGTHS[3]
    return (0.0, 0.0, 0.03 - s)




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


    def update(s):
        px, py, pz = position(s)

        moving_point.set_data_3d([px], [py], [pz])
        coordinate_text.set_text(
            f"x = {px:.5f} m   "
            f"y = {py:.5f} m   "
            f"z = {pz:.5f} m"
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
