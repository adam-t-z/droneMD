"""3D shape / mesh surface sampling for drone formation positions."""

from __future__ import annotations

import numpy as np
import trimesh
from numpy.typing import NDArray


def sample_obj_surface(filepath: str, n_points: int, target_height: float = 1.0) -> NDArray:
    """Sample ``n_points`` evenly from an OBJ mesh surface, normalized for drone positions.

    The OBJ model is assumed to have the vertical axis on Y. Output is remapped
    to the drone coordinate frame: ``(X_obj, Z_obj, Y_obj) -> (drone_x, drone_y, drone_z)``.
    XY is centered and scaled to ``[-1.8, 1.8]``, and Z starts at ``target_height``.
    """
    mesh = trimesh.load(filepath, force="mesh")
    points, _ = trimesh.sample.sample_surface_even(mesh, n_points)

    points = points[:, [0, 2, 1]]

    points[:, :2] -= points[:, :2].mean(axis=0)
    xy_max = np.abs(points[:, :2]).max()
    if xy_max > 1e-6:
        points[:, :2] *= 1.8 / xy_max

    points[:, 2] -= points[:, 2].min()
    z_range = points[:, 2].max()
    if z_range > 3.0:
        points[:, 2] *= 3.0 / z_range
    points[:, 2] += target_height

    return points.astype(np.float32)
