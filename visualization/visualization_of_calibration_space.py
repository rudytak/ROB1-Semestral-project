#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-09-18
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

from pathlib import Path
import time

import numpy as np

from robotics_toolbox.robots import SpatialManipulator
from robotics_toolbox.core import SE3
from robotics_toolbox.render import RendererSpatial

urdf_path = Path(__file__).parent.joinpath("crs_6dof.urdf")
renderer = RendererSpatial()
robot = SpatialManipulator(urdf_path=urdf_path)


def init_table(
    maze="A", maze_translation=[0.5, 0, 0], maze_offset=[0, 0], maze_rotation=0
):
    path = None
    if maze == "A":
        path = "table_mazeA.urdf"
    elif maze == "B":
        path = "table_mazeB.urdf"
    elif maze == "C":
        path = "table_mazeC.urdf"
    elif maze == "D":
        path = "table_mazeD.urdf"
    else:
        raise ValueError("Invalid maze type. Choose 'A', 'B', 'C' or 'D'")

    table = SpatialManipulator(
        urdf_path=Path(__file__).parent.joinpath(path),
        base_pose=SE3(maze_translation, None),
    )
    table.q = [0, maze_offset[0], maze_offset[1], maze_rotation]
    return table


# table = init_table(
#     "B", maze_translation=[0.5, 0, 0], maze_offset=[0.01, 0.01], maze_rotation=0
# )

renderer.plot_manipulator(robot)
# renderer.plot_manipulator(table)

# Generate calibration grid
x = np.linspace(0.4, 0.525, 3)
y = np.linspace(-0.17, 0.17, 5)
z = np.linspace(0.049, 0.15, 5)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
grid_points = np.column_stack([
    X.ravel(), Y.ravel(), Z.ravel()
])

# Add blue points to Viser
blue = np.tile(
    np.array([0, 0, 255], dtype=np.uint8),
    (len(grid_points), 1)
)

renderer.server.scene.add_point_cloud(
    name="/calibration_grid",
    points=grid_points,
    colors=blue,
    point_size=0.0075,
)

# Interpolate between configurations
q0 = np.zeros(robot.dof)
q1 = np.array([0,1,2,0,1,0]) * np.pi/4

with renderer.animation(fps=10):
    for t in np.linspace(0, 1, num=50):
        robot.q = q0 + t * (q1 - q0)
        renderer.plot_manipulator(robot)

        # table.q[0] = t * 2 * np.pi
        # renderer.plot_manipulator(table)

renderer.wait_at_the_end()
