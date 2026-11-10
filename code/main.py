from camera import load_camera_calibration
from ctu_crs import CRS97
from sequence import calculate, check_sols, run_sequence


def print_menu(maze_name, home, use_cached):
    print("\n" + "-" * 45)
    print(f"Selected maze: {maze_name.upper()}")
    print(f"Automatic homing before start: {'ON' if home else 'OFF'}")
    print(f"Use cached image for visualization: {'ON' if use_cached else 'OFF'}")
    print("-" * 45)
    print("[s] Select maze and start sequence")
    print("[r] Visualize reachability for selected maze")
    print("[c] Toggle cached visualization image")
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
    use_cached = False  # Default is to take a fresh photo
    robot = None

    try:
        while True:
            print_menu(maze_name, home, use_cached)
            key = input("> ").strip().lower()

            if key == "b":
                home = not home
                continue

            if key == "c":
                use_cached = not use_cached
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

                elif key in ("s", "v"):
                    selected = select_maze(maze_name)

                    if selected is None:
                        continue

                    maze_name = selected

                    maze = __import__(
                        f"maze.maze_{maze_name}",
                        fromlist=["hoop_pose"],
                    )

                    # Load the saved camera calibration
                    calibration = load_camera_calibration()

                    if robot is None:
                        robot = connect_robot(home)
                    elif home:
                        go_home(robot)

                    # False for A/B/C, True for D.
                    tilt = maze_name == "d"

                    # Observe the maze and estimate table motion
                    print("\nMeasuring table motion...")
                    model = calculate(robot, calibration)

                    if key == "r":
                        # Run the visualization pipeline dynamically
                        print("\nGenerating reachability map...")
                        from solvable import run_visualization_pipeline

                        solvable = run_visualization_pipeline(
                            robot=robot,
                            maze=maze,
                            model=model,
                            P_matrix=calibration["P"],
                            tilt=tilt,
                            use_cached=use_cached,
                        )

                        print("VISUALIZATION COMPLETE")
                        print(f"Result: The maze trajectory is {'SAFE' if solvable else 'UNSAFE'}")
                        print("'reachability_maze_check.png' generated")
                        continue

                    # Reachability check for actual run (key == "s")
                    if not check_sols(robot, maze, model, tilt=tilt):
                        print("Entrance reachability check failed. Returning to the menu.")
                        continue

                    # Wait before starting the actual traversal.
                    print(f"\nPreparation finished for maze {maze_name.upper()}.")

                    while True:
                        choice = input("Press Enter to start, or q + Enter to cancel: ").strip().lower()

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
                    print("Unknown command. Use s, v, c, b, h, or q.")

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
