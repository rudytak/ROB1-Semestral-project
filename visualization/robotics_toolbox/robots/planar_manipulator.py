#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        # todo HW02: implement fk for the flange

        return self.fk_all_links()[-1]  # return the last frame as the flange pose

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = [self.base_pose]

        for i in range(len(self.link_parameters)):
            # compute the pose of the i-th link frame
            transform = SE2()
            if self.structure[i] == "R":
                rot = SE2(translation=(0, 0), rotation=self.q[i])
                trans = SE2(translation=(self.link_parameters[i], 0), rotation=0)
            elif self.structure[i] == "P":
                rot = SE2(translation=(0, 0), rotation=self.link_parameters[i])
                trans = SE2(translation=(self.q[i], 0), rotation=0)
            else:
                raise ValueError(f"Unknown joint type {self.structure[i]}")
            frames.append(frames[-1] * rot * trans)

        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation

        frames = self.fk_all_links()
        flange_pose = frames[-1]
        angle_sum = (
            self.base_pose.rotation.angle
        )  # Initialize angle_sum with base rotation angle

        print(self.structure)
        print(self.q, angle_sum)
        print(self.link_parameters)

        for i in range(len(self.structure)):
            joint_pose = frames[i]
            print(i, joint_pose)

            if self.structure[i] == "R":
                # Revolute joint
                joint_position = joint_pose.translation
                flange_position = flange_pose.translation

                angle_sum += self.q[i]

                r = flange_position - joint_position
                jac[:, i] = np.array([-r[1], r[0], 1.0])
            elif self.structure[i] == "P":
                angle_sum += self.link_parameters[i]

                direction = np.array([np.cos(angle_sum), np.sin(angle_sum)])

                jac[:, i] = np.array([direction[0], direction[1], 0.0])

        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation

        for i in range(len(self.structure)):
            q = self.q.copy()
            flange_pose = self.set_configuration(q).flange_pose()

            q_plus = self.q.copy()
            q_plus[i] += delta
            flange_pose_plus = self.set_configuration(q_plus).flange_pose()

            self.set_configuration(q)
            print(flange_pose_plus, flange_pose)

            # Compute the finite difference approximation of the Jacobian column
            jac[0, i] = (
                np.array(flange_pose_plus.translation)[0]
                - np.array(flange_pose.translation)[0]
            ) / (delta)

            jac[1, i] = (
                np.array(flange_pose_plus.translation)[1]
                - np.array(flange_pose.translation)[1]
            ) / (delta)

            jac[2, i] = (
                np.array(flange_pose_plus.rotation.angle)
                - np.array(flange_pose.rotation.angle)
            ) / (delta)

        return jac

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW05 implement numerical IK

        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        # todo: HW05 implement analytical IK for RRR manipulator
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            pass
        return []

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
