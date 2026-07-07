"""Spawn pattern generators for initial drone positions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_spawn(
    pattern: str, n_drones: int, bounds: NDArray, params: dict
) -> tuple[NDArray, NDArray]:
    """Generate initial positions and velocities for a spawn pattern."""
    fn = _REGISTRY.get(pattern)
    if fn is None:
        raise ValueError(f"Unknown spawn pattern: {pattern}")
    pos = fn(n_drones, bounds, params)
    vel = np.zeros((n_drones, 3), dtype=np.float32)
    return pos, vel


def _random(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    height = float(params.get("height", 1.0))
    rng = np.random.default_rng(42)
    pos = np.zeros((n_drones, 3), dtype=np.float32)
    pos[:, 0] = rng.uniform(bounds[0], bounds[1], n_drones)
    pos[:, 1] = rng.uniform(bounds[2], bounds[3], n_drones)
    pos[:, 2] = height
    return pos


def _grid(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    height = float(params.get("height", 1.0))
    spacing = float(params.get("spacing", 0.5))
    cols = int(np.ceil(np.sqrt(n_drones)))
    rows = int(np.ceil(n_drones / cols))
    gx, gy = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    gx = (gx.flatten()[:n_drones]) - (gx.flatten()[:n_drones]).mean()
    gy = (gy.flatten()[:n_drones]) - (gy.flatten()[:n_drones]).mean()
    pos = np.zeros((n_drones, 3), dtype=np.float32)
    pos[:, 0] = gx
    pos[:, 1] = gy
    pos[:, 2] = height
    return pos


def _circle(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    height = float(params.get("height", 1.0))
    radius = float(params.get("radius", 1.5))
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    pos = np.zeros((n_drones, 3), dtype=np.float32)
    pos[:, 0] = radius * np.cos(angles)
    pos[:, 1] = radius * np.sin(angles)
    pos[:, 2] = height
    return pos


def _line(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    height = float(params.get("height", 1.0))
    spacing = float(params.get("spacing", 0.5))
    axis = params.get("axis", "x")
    half = (n_drones - 1) * spacing / 2
    vals = np.linspace(-half, half, n_drones)
    pos = np.zeros((n_drones, 3), dtype=np.float32)
    if axis == "x":
        pos[:, 0] = vals
    elif axis == "y":
        pos[:, 1] = vals
    else:
        pos[:, 0] = vals / np.sqrt(2)
        pos[:, 1] = vals / np.sqrt(2)
    pos[:, 2] = height
    return pos


def _sphere(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    radius = float(params.get("radius", 2.0))
    pos = np.zeros((n_drones, 3), dtype=np.float32)
    if n_drones == 1:
        pos[0, 2] = radius
        return pos
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n_drones):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_drones)
        phi = 2 * np.pi * i / golden_ratio
        pos[i, 0] = radius * np.sin(theta) * np.cos(phi)
        pos[i, 1] = radius * np.sin(theta) * np.sin(phi)
        pos[i, 2] = radius * np.cos(theta)
    return pos


def _from_points(n_drones: int, bounds: NDArray, params: dict) -> NDArray:
    points = params.get("points")
    if points is None:
        raise ValueError("'points' param required for 'points' spawn pattern")
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must be shape (N,3), got {arr.shape}")
    if arr.shape[0] != n_drones:
        raise ValueError(f"points count ({arr.shape[0]}) doesn't match n_drones ({n_drones})")
    return arr


_REGISTRY: dict[str, callable] = {
    "random": _random,
    "grid": _grid,
    "circle": _circle,
    "line": _line,
    "sphere": _sphere,
    "points": _from_points,
}
