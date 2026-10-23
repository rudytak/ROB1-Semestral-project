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

    x = [p[0] for p in points]
    z = [p[2] for p in points]

    # Front view
    fig, ax = plt.subplots()
    ax.plot(x, z, color="black", linewidth=2, label="Rail centerline")

    ax.scatter(0, 0, color="red", label="Base origin (0, 0, 0)")
    ax.scatter(x[0], z[0], color="blue", label="Start")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Maze D — front view")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    plt.show()





################################################
# DEBUG
################################################
if __name__ == "__main__" :
    visualize()
