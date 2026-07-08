"""Motion-primitive trajectories for primitive-driven simulation.

Ports the shape-forming/holding logic from
``hidden/swarmGPT-main/swarm_gpt/core/motion_primitives.py`` and adapts it to
DroneMD's meter units and the flocking engine's fixed-rate control loop.

Where the swarmGPT primitives emit a sparse ``{time: {drone_id: pos}}`` waypoint
dict (consumed by an external MPC), here we precompute a dense
``(n_drones, n_steps, 3)`` target trajectory sampled at the control frequency.
The engine feeds these targets straight into ``state_control`` so the physics
solver holds the shape continuously instead of letting Boids disperse it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

TrajectoryFn = Callable[..., NDArray]


def _assign_positions(pos: NDArray, des_pos: NDArray) -> NDArray:
    """Assign drones to the closest desired positions (Hungarian algorithm).

    Mirrors ``motion_primitives._assign_positions`` but in meters.
    """
    dist = np.linalg.norm(pos[:, None, :] - des_pos[None, :, :], axis=-1)
    return linear_sum_assignment(dist)[1]


def _two_phase_trajectory(
    spawn: NDArray,
    des: NDArray,
    assignment: NDArray,
    t_form: float,
    duration: float,
    control_freq: float,
    post_motion: Callable[[float], NDArray],
) -> NDArray:
    """Build a trajectory that eases into a formation then holds/moves.

    Phase 1 ``[0, t_form]``: cosine ease-in-out from ``spawn`` to the assigned
    slot ``des[assignment]`` (matches ``form_*`` arrival semantics).

    Phase 2 ``[t_form, duration]``: ``post_motion(tt)`` returns the ``(n, 3)``
    positions at elapsed time ``tt`` since formation — used for holds,
    rotations, rises, etc.

    Args:
        spawn: Spawn positions in meters, shape ``(n, 3)``.
        des: Desired slot positions in meters, shape ``(n, 3)``.
        assignment: Hungarian slot-per-drone assignment, shape ``(n,)``.
        t_form: Formation arrival time in seconds.
        duration: Total duration in seconds.
        control_freq: Control loop frequency in Hz.
        post_motion: ``callable(tt) -> (n, 3)`` positions for the hold/motion
            phase, where ``tt`` is seconds since formation completed.

    Returns:
        Target positions in meters, shape ``(n, n_steps, 3)``.
    """
    n = spawn.shape[0]
    n_steps = int(duration * control_freq)
    circle_pos = des[assignment]
    t_form = float(np.clip(t_form, 0.0, duration))
    spawn = spawn.astype(np.float64)
    traj = np.empty((n, n_steps, 3), dtype=np.float64)
    for step in range(n_steps):
        t = step / control_freq
        if t < t_form and t_form > 0:
            alpha = 0.5 - 0.5 * np.cos(np.pi * (t / t_form))
            traj[:, step, :] = spawn * (1.0 - alpha) + circle_pos * alpha
        else:
            tt = max(0.0, t - t_form)
            traj[:, step, :] = post_motion(tt)
    return traj


def _bounds_centroid(bounds: NDArray) -> tuple[float, float, float]:
    """Return the xy centroid and the half-extent of the bounds."""
    cx = (bounds[0] + bounds[1]) * 0.5
    cy = (bounds[2] + bounds[3]) * 0.5
    max_radius = min(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.5 * 0.9
    return float(cx), float(cy), float(max_radius)


def circle_trajectory(
    init_pos: NDArray,
    duration: float,
    control_freq: float,
    bounds: NDArray,
    radius: float,
    height: float,
    t_form: float,
    omega: float,
) -> NDArray:
    """Form a circle then hold/rotate it continuously.

    Phase 1 ``[0, t_form]``: each drone eases from its spawn position to its
    assigned slot on the circle (cosine ease-in-out), matching the
    ``form_circle`` arrival semantics from the primitive library.

    Phase 2 ``[t_form, duration]``: the circle holds its shape and rotates
    around the world origin at ``omega`` rad/s. With ``omega=0`` this is a
    pure stationary hold (the ``_formation_waypoints`` hold behaviour); with
    ``omega>0`` it is the ``spiral``/``rotate`` continuous-shape behaviour.

    Args:
        init_pos: Spawn positions in meters, shape ``(n, 3)``.
        duration: Total simulation duration in seconds.
        control_freq: Control loop frequency in Hz.
        bounds: ``[xmin, xmax, ymin, ymax]`` in meters.
        radius: Desired circle radius in meters (clamped to fit bounds).
        height: Circle plane height in meters.
        t_form: Time to form the circle in seconds.
        omega: Rotation rate after formation in rad/s.

    Returns:
        Target positions in meters, shape ``(n, n_steps, 3)``.
    """
    n = init_pos.shape[0]
    cx, cy, max_radius = _bounds_centroid(bounds)
    radius = float(np.clip(radius, 0.1, max_radius))
    slot_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    des = np.stack(
        [
            cx + radius * np.cos(slot_angles),
            cy + radius * np.sin(slot_angles),
            np.full(n, height, dtype=np.float64),
        ],
        axis=1,
    )
    assignment = _assign_positions(init_pos, des)
    assigned_angles = slot_angles[assignment]

    def post(tt: float) -> NDArray:
        ang = assigned_angles + omega * tt
        return np.stack(
            [
                cx + radius * np.cos(ang),
                cy + radius * np.sin(ang),
                np.full(n, height, dtype=np.float64),
            ],
            axis=1,
        )

    return _two_phase_trajectory(init_pos, des, assignment, t_form, duration, control_freq, post)


def star_trajectory(
    init_pos: NDArray,
    duration: float,
    control_freq: float,
    bounds: NDArray,
    radius: float,
    delta_radius: float,
    height: float,
    t_form: float,
    omega: float = 0.0,
) -> NDArray:
    """Form a two-spoke star then hold/rotate it continuously.

    Ports ``form_star``: half the drones go on an inner circle, half on an
    outer circle offset by half a spoke angle, producing a star with
    ``n//2`` spokes. An odd drone (if any) sits at the center. After
    formation the whole star rotates rigidly at ``omega`` rad/s.

    Args:
        init_pos: Spawn positions in meters, shape ``(n, 3)``.
        duration: Total simulation duration in seconds.
        control_freq: Control loop frequency in Hz.
        bounds: ``[xmin, xmax, ymin, ymax]`` in meters.
        radius: Inner circle radius in meters.
        delta_radius: Spacing between inner and outer circle in meters.
        height: Star plane height in meters.
        t_form: Time to form the star in seconds.
        omega: Rotation rate after formation in rad/s.

    Returns:
        Target positions in meters, shape ``(n, n_steps, 3)``.
    """
    n = init_pos.shape[0]
    cx, cy, max_radius = _bounds_centroid(bounds)
    radius = float(np.clip(radius, 0.1, max_radius))
    delta_radius = float(np.clip(delta_radius, 0.05, max(0.05, max_radius - radius)))
    dpc = max(2, n // 2)
    radii = [radius, radius + delta_radius]
    offsets = [0.0, 2 * np.pi / dpc]

    parts_xy: list[NDArray] = []
    for r, off in zip(radii, offsets):
        ang = np.linspace(0, 2 * np.pi, dpc, endpoint=False) + off
        parts_xy.append(np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1))
    if n != dpc * 2:
        parts_xy.append(np.array([[cx, cy]]))
    xy = np.vstack(parts_xy)[:n]
    des = np.concatenate([xy, np.full((n, 1), height)], axis=1)

    rel = des[:, :2] - np.array([cx, cy])
    slot_r = np.hypot(rel[:, 0], rel[:, 1])
    slot_ang = np.arctan2(rel[:, 1], rel[:, 0])
    assignment = _assign_positions(init_pos, des)
    asg_r = slot_r[assignment]
    asg_ang = slot_ang[assignment]

    def post(tt: float) -> NDArray:
        ang = asg_ang + omega * tt
        return np.stack(
            [cx + asg_r * np.cos(ang), cy + asg_r * np.sin(ang), np.full(n, height)], axis=1
        )

    return _two_phase_trajectory(init_pos, des, assignment, t_form, duration, control_freq, post)


def cone_trajectory(
    init_pos: NDArray,
    duration: float,
    control_freq: float,
    bounds: NDArray,
    delta_height: float,
    spacing: float,
    height: float,
    t_form: float,
    inverted: bool = False,
    omega: float = 0.0,
) -> NDArray:
    """Form a layered cone then hold/rotate it continuously.

    Ports ``form_cone``: one drone at the apex, then rings of 4, 8, 12, ...
    drones at increasing radius and decreasing (or increasing, if inverted)
    height. After formation the cone rotates rigidly around its central
    axis at ``omega`` rad/s.

    Args:
        init_pos: Spawn positions in meters, shape ``(n, 3)``.
        duration: Total simulation duration in seconds.
        control_freq: Control loop frequency in Hz.
        bounds: ``[xmin, xmax, ymin, ymax]`` in meters.
        delta_height: Vertical spacing between layers in meters.
        spacing: Target neighbor spacing determining each ring's radius.
        height: Apex height in meters (top if upright, base if inverted).
        t_form: Time to form the cone in seconds.
        inverted: If True, apex at the bottom and layers stack upward.
        omega: Rotation rate after formation in rad/s.

    Returns:
        Target positions in meters, shape ``(n, n_steps, 3)``.
    """
    n = init_pos.shape[0]
    cx, cy, max_radius = _bounds_centroid(bounds)
    dz = abs(delta_height) * (-1.0 if not inverted else 1.0)
    z_max = max(0.2, height)

    slot_r = [0.0]
    slot_ang = [0.0]
    slot_z = [z_max if not inverted else 0.2]
    z = slot_z[0]
    drones_left = n - 1
    drones_in_layer = 0
    while drones_left > 0:
        drones_in_layer += 4
        z += dz
        r = min(spacing / (2 * np.sin(np.pi / drones_in_layer)), max_radius)
        drones_left -= drones_in_layer
        if drones_left < 0:
            drones_in_layer = drones_left + drones_in_layer
            drones_left = 0
        ang = np.linspace(0, 2 * np.pi, drones_in_layer, endpoint=False)
        slot_r.extend([r] * drones_in_layer)
        slot_ang.extend(ang.tolist())
        slot_z.extend([z] * drones_in_layer)

    slot_r = np.asarray(slot_r, dtype=np.float64)
    slot_ang = np.asarray(slot_ang, dtype=np.float64)
    slot_z = np.clip(np.asarray(slot_z, dtype=np.float64), 0.1, z_max)
    des = np.stack([cx + slot_r * np.cos(slot_ang), cy + slot_r * np.sin(slot_ang), slot_z], axis=1)
    assignment = _assign_positions(init_pos, des)
    asg_r = slot_r[assignment]
    asg_ang = slot_ang[assignment]
    asg_z = slot_z[assignment]

    def post(tt: float) -> NDArray:
        ang = asg_ang + omega * tt
        return np.stack([cx + asg_r * np.cos(ang), cy + asg_r * np.sin(ang), asg_z], axis=1)

    return _two_phase_trajectory(init_pos, des, assignment, t_form, duration, control_freq, post)


def trajectory_velocities(traj: NDArray, control_freq: float) -> NDArray:
    """Finite-difference target velocities (m/s) for a position trajectory.

    Args:
        traj: ``(n, T, 3)`` positions in meters.
        control_freq: Sampling frequency in Hz.

    Returns:
        ``(n, T, 3)`` velocities; the last step repeats the previous.
    """
    vel = np.zeros_like(traj)
    if traj.shape[1] > 1:
        vel[:, :-1, :] = (traj[:, 1:, :] - traj[:, :-1, :]) * control_freq
        vel[:, -1, :] = vel[:, -2, :]
    return vel


_REGISTRY: dict[str, TrajectoryFn] = {
    "circle": circle_trajectory,
    "star": star_trajectory,
    "cone": cone_trajectory,
}


def build_trajectory(
    primitive: str,
    init_pos: NDArray,
    duration: float,
    control_freq: float,
    bounds: NDArray,
    params: dict,
) -> NDArray:
    """Dispatch to a registered primitive trajectory builder.

    Args:
        primitive: Primitive name (e.g. ``"circle"``).
        init_pos: Spawn positions in meters, shape ``(n, 3)``.
        duration: Total simulation duration in seconds.
        control_freq: Control loop frequency in Hz.
        bounds: ``[xmin, xmax, ymin, ymax]`` in meters.
        params: Primitive-specific parameters.

    Returns:
        Target positions in meters, shape ``(n, n_steps, 3)``.
    """
    fn = _REGISTRY.get(primitive)
    if fn is None:
        raise ValueError(f"Unknown motion primitive: {primitive}")
    return fn(init_pos, duration, control_freq, bounds, **params)
