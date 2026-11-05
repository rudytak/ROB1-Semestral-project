#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)

        theta = np.linalg.norm(v)
        x, y, z = v / theta
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return SO3(R)

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""

        tr = np.trace(self.rot)
        v = np.zeros(3)

        theta = np.arccos(0.5 * (tr - 1))

        if np.isclose(theta, 0.0):
            # omega není definována
            return np.zeros(3)

        elif np.isclose(theta, np.pi):
            theta = np.pi

            r33 = self.rot[2, 2]
            r22 = self.rot[1, 1]
            r11 = self.rot[0, 0]

            if r33 != -1:
                v[0] = self.rot[0, 2]
                v[1] = self.rot[1, 2]
                v[2] = self.rot[2, 2] + 1
                return 1 / np.sqrt(2 + 2 * r33) * np.transpose(v) * theta
            if r22 != -1:
                v[0] = self.rot[0, 1]
                v[1] = self.rot[1, 1] + 1
                v[2] = self.rot[2, 1]
                return 1 / np.sqrt(2 + 2 * r22) * np.transpose(v) * theta
            if r11 != -1:
                v[0] = self.rot[0, 0] + 1
                v[1] = self.rot[1, 0]
                v[2] = self.rot[2, 0]
                return 1 / np.sqrt(2 + 2 * r11) * np.transpose(v) * theta

        else:
            K = 1 / (2 * np.sin(theta)) * (self.rot - np.transpose(self.rot))
            v[0] = K[2, 1]
            v[1] = K[0, 2]
            v[2] = K[1, 0]

            return v * theta

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        return SO3(self.rot @ other.rot)

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        return SO3(np.transpose(self.rot))

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""

        c = np.cos(angle)
        s = np.sin(angle)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, s, c],
            ]
        )
        return SO3(Rx)

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""

        c = np.cos(angle)
        s = np.sin(angle)
        Ry = np.array(
            [
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c],
            ]
        )
        return SO3(Ry)

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""

        c = np.cos(angle)
        s = np.sin(angle)
        Rz = np.array(
            [
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1],
            ]
        )
        return SO3(Rz)

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""

        theta = 2 * np.arccos(q[3])
        K = q[0:3] / np.sin(theta / 2)

        return SO3.exp(K * theta)

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""

        qw = np.sqrt(1 + np.trace(self.rot)) / 2
        qx = (self.rot[2, 1] - self.rot[1, 2]) / (4 * qw)
        qy = (self.rot[0, 2] - self.rot[2, 0]) / (4 * qw)
        qz = (self.rot[1, 0] - self.rot[0, 1]) / (4 * qw)
        return [qx, qy, qz, qw]

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        return SO3.exp(angle * axis)

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        v = self.log()

        theta = np.linalg.norm(v)
        return (theta, v / theta)

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        Rcomp = np.eye(3,3)
        for i in range(len(seq)):
            R = np.eye(3,3)

            ang = angles[i]
            ax = seq[i]

            match ax:
                case "x":
                    R = SO3.rx(ang)
                case "y":
                    R = SO3.ry(ang)
                case "z":
                    R = SO3.rz(ang)

            Rcomp = Rcomp @ R.rot

        return SO3(Rcomp)

    def __hash__(self):
        return id(self)
