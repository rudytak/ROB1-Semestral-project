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
from robotics_toolbox.render import RendererSpatial

np.random.seed(0)


urdf_path = Path(__file__).parent.joinpath("crs_6dof.urdf")
renderer = RendererSpatial()
robot = None
last_mtime = None

try:
    while True:
        mtime = urdf_path.stat().st_mtime
        if mtime != last_mtime:
            last_mtime = mtime
            try:
                new_robot = SpatialManipulator(urdf_path=urdf_path)
            except Exception as e:  # half-saved or broken urdf: keep the old view
                print(f"Cannot load {urdf_path.name}: {e}")
            else:
                if robot is not None:
                    root, viz = renderer.manipulators.pop(robot)
                    viz.remove()
                    root.remove()
                robot = new_robot
                robot.q = (
                    np.ones(robot.dof) * np.pi / 4
                )  # ponytail: static pose, edit here
                renderer.plot_manipulator(robot)

                renderer.manipulators[robot][1]._joint_frames[-1].show_axes = True
                print(f"Reloaded {urdf_path.name}")
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
