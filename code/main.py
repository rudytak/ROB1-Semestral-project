from ctu_crs import CRS97
from camera import load_camera_calibration
from sequence import calculate, run_sequence, check_sols    


def print_menu(maze_name, home):
    print("\n" + "-" * 45)
    print(f"Selected maze: {maze_name.upper()}")
    print(f"Automatic homing before start: "
          f"{'ON' if home else 'OFF'}")
    print("[s] Select maze and start sequence")
    print("[b] Toggle automatic homing")
    print("[h] Go home now")
    print("[q] Go home, close robot, and quit")


def select_maze(current_maze=None):
    while True:
        print("\nSelect maze: [a] [b] [c] [d]")
        print(f"[Enter] Keep {current_maze.upper()}")
        print("[q] Cancel")

        choice = input("> ").strip().lower()

        if choice == "":
            return current_maze

        if choice in ("a", "b", "c", "d"):
            return choice

        if choice == "q":
            return None

        print("Please select a, b, c, d, or q.")


def connect_robot(home):
    robot = CRS97()

    try:
        robot.initialize(home=home)
    except BaseException:
        try:
            robot.close()
        except Exception:
            pass
        raise

    return robot


def go_home(robot):
    print("Going home...")
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()
    print("Robot is home.")


def main():
    maze_name = "a"
    home = False
    robot = None

    try:
        while True:
            print_menu(maze_name, home)
            key = input("> ").strip().lower()

            if key == "b":
                home = not home
                continue

            if key == "q":
                try:
                    if robot is None:
                        robot = connect_robot(home)

                    go_home(robot)

                except Exception as exc:
                    print(f"Could not go home: {exc}")

                break

            try:
                if key == "h":
                    if robot is None:
                        robot = connect_robot(home)

                    go_home(robot)

                elif key == "s":
                    selected = select_maze(maze_name)

                    if selected is None:
                        continue

                    maze_name = selected

                    maze = __import__(
                        f"maze.maze_{maze_name}",
                        fromlist=["hoop_pose"],
                    )

                    # Load the saved camera calibration.
                    calibration = load_camera_calibration()

                    if robot is None:
                        robot = connect_robot(home)
                    elif home:
                        go_home(robot)

                    # False for A/B/C, True for D.
                    tilt = maze_name == "d"

                    # Observe the maze and estimate table motion.
                    print("\nMeasuring table motion...")
                    model = calculate(robot, calibration)

                    # Reachability check
                    if not check_sols(robot, maze, model, tilt=tilt):
                        print("Entrance reachability check failed. Returning to the menu.")
                        continue

                    # Wait before starting the actual traversal.
                    print(
                        f"\nPreparation finished for maze "
                        f"{maze_name.upper()}."
                    )

                    while True:
                        choice = input(
                            "Press Enter to start, or q + Enter to cancel: "
                        ).strip().lower()

                        if choice in ("", "q"):
                            break

                        print("Press Enter without typing, or type q.")

                    if choice == "q":
                        print("Sequence cancelled. Returning to the menu.")
                        continue

                    run_sequence(
                        robot,
                        maze,
                        model,
                        tilt=tilt,
                    )

                    go_home(robot)

                else:
                    print("Unknown command. Use s, b, h, or q.")

            except Exception as exc:
                print(f"Action failed: {exc}")

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Closing the robot connection.")

    finally:
        if robot is not None:
            robot.close()

        print("Application closed.")


if __name__ == "__main__":
    main()