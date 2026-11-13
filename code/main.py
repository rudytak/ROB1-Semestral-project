from ctu_crs import CRS97
from camera import load_camera_calibration
from sequence import calculate, run_sequence, check_sols   
from image_view import LiveView 
from solvable import run_visualization_pipeline


def print_menu(maze_name, home, show_images, show_solv):
    print("\n" + "-" * 45)
    print(f"Selected maze: {maze_name.upper()}")
    print(f"[b] Toggle automatic homing: {'ON' if home else 'OFF'}")
    print(f"[i] Toggle Show images: {'ON' if show_images else 'OFF'}")
    print(f"[r] Toggle Visualize reachability for selected maze: {'ON' if show_solv else 'OFF'}")
    print("[s] Select maze and start sequence")
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
    show_images = False
    view = None
    show_solv = False
    
    try:
        while True:
            print_menu(maze_name, home, show_images, show_solv)
            key = input("> ").strip().lower()

            if key == "i":
                show_images = not show_images
                continue

            if key == "b":
                home = not home
                continue

            if key == "r":
                show_solv = not show_solv
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

                    view = LiveView(robot, calibration, maze, tilt) if show_images else None

                    # Observe the maze and estimate table motion.
                    print("\nMeasuring table motion...")
                    model = calculate(robot, calibration, view=view)
                    if model is None:
                        print("\nNo valid meas")
                        continue

                    # Reachability check
                    if not check_sols(robot, maze, model, tilt=tilt):
                        print("Entrance reachability check failed. Returning to the menu.")
                        if not show_solv: continue

                    print("\nGenerating reachability map...")

                    solvable = run_visualization_pipeline(
                        robot=robot,
                        maze=maze,
                        model=model,
                        P_matrix=calibration["P"],
                        tilt=tilt,
                        use_cached=False,
                    )

                    print("VISUALIZATION COMPLETE")
                    print(f"Result: The maze trajectory is {'SAFE' if solvable else 'UNSAFE'}")
                    print("'reachability_maze_check.png' generated")

                    if not solvable:
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
                    for _ in range(8):
                        run_sequence(robot, maze, model, tilt=tilt, view=view, maze_name=maze_name)

                    go_home(robot)
                    if view is not None:
                        view.save_gif()
                        view.save_plots()

                else:
                    print("Unknown command. Use s, b, i, h, or q.")

            except Exception as exc:
                print(f"Action failed: {exc}")

            finally:
                if view is not None:
                    view.close()

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Closing the robot connection.")

    finally:
        if robot is not None:
            robot.close()

        print("Application closed.")


if __name__ == "__main__":
    main()