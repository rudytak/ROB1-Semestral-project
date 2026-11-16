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


table = init_table(
    "B", maze_translation=[0.5, 0, 0], maze_offset=[0.1, 0.1], maze_rotation=0
)

renderer.plot_manipulator(robot)
renderer.plot_manipulator(table)

"""Interpolate between the following two configurations"""
q0 = np.zeros(robot.dof)
q1 = np.ones(robot.dof)
print(robot.dof)
print(table.dof)

with renderer.animation(fps=10):
    for t in np.linspace(0, 1, num=50):
        robot.q = q0 + t * (q1 - q0)  # interpolate
        renderer.plot_manipulator(robot)

        table.q[0] = t * 2 * np.pi  # rotate table
        renderer.plot_manipulator(table)

renderer.wait_at_the_end()
