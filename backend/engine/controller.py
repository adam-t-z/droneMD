"""Boids flocking controller — pure JAX computation with no simulator dependency."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_boids_kernel(
    separation_weight: float,
    alignment_weight: float,
    cohesion_weight: float,
    perception_radius: float,
    max_speed: float,
    max_force: float,
    boundary_mode: str,
    bounds: jnp.ndarray,
    obstacles: list,
):
    """Build and return a JIT-compiled Boids flocking computation.

    Returns a callable ``(pos, vel) -> target_vel`` that computes the desired
    velocity for each drone based on the Boids rules (separation, alignment,
    cohesion) plus boundary handling and obstacle avoidance.
    """

    @jax.jit
    def _compute(pos, vel):
        pos2 = pos[:, :2]
        vel2 = vel[:, :2]

        diff = pos2[:, None, :] - pos2[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)

        eye = jnp.eye(pos2.shape[0], dtype=bool)
        mask = (dist < perception_radius) & (~eye)

        # Separation
        sep_mask = mask & (dist < perception_radius * 0.5)
        safe_dist = jnp.maximum(dist, 0.1)
        sep_force = (
            diff / jnp.maximum(dist[..., None], 1e-6) * separation_weight / safe_dist[..., None]
        )
        separation = jnp.sum(jnp.where(sep_mask[..., None], sep_force, 0.0), axis=1)

        # Cohesion + Alignment
        counts = jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
        mean_pos = jnp.where(mask[..., None], pos2[None], 0.0).sum(axis=1) / counts
        mean_vel = jnp.where(mask[..., None], vel2[None], 0.0).sum(axis=1) / counts
        has_neighbors = mask.sum(axis=1, keepdims=True) > 0

        cohesion = (mean_pos - pos2) * cohesion_weight * 0.5
        alignment = (mean_vel - vel2) * alignment_weight * 0.5

        acc = separation + jnp.where(has_neighbors, cohesion + alignment, 0.0)

        # Boundary handling
        if boundary_mode == "wrap":
            half_w = (bounds[1] - bounds[0]) * 0.5
            half_h = (bounds[3] - bounds[2]) * 0.5
            cx = (bounds[0] + bounds[1]) * 0.5
            cy = (bounds[2] + bounds[3]) * 0.5

            dx = pos2[:, 0] - cx
            dy = pos2[:, 1] - cy
            acc = acc.at[:, 0].add(-jnp.sign(dx) * jnp.maximum(jnp.abs(dx) - half_w, 0.0) * 2.0)
            acc = acc.at[:, 1].add(-jnp.sign(dy) * jnp.maximum(jnp.abs(dy) - half_h, 0.0) * 2.0)

        elif boundary_mode in ("bounce", "hard"):
            margin = 0.3
            for axis in (0, 1):
                lo = bounds[axis * 2]
                hi = bounds[axis * 2 + 1]
                acc = acc.at[:, axis].add(
                    jnp.where(pos2[:, axis] < lo + margin, max_speed * 2.0, 0.0)
                )
                acc = acc.at[:, axis].add(
                    jnp.where(pos2[:, axis] > hi - margin, -max_speed * 2.0, 0.0)
                )

        # Obstacles
        if obstacles:
            obs = jnp.asarray(obstacles)
            obs_xy = obs[:, :2]
            obs_r = obs[:, 2]
            delta = pos2[:, None, :] - obs_xy[None]
            odist = jnp.linalg.norm(delta, axis=-1)
            radius = obs_r + 0.5
            active = (odist < radius) & (odist > 1e-6)
            push = (radius - odist) / radius
            obs_force = (
                delta / jnp.maximum(odist[..., None], 1e-6) * push[..., None] * max_speed * 3.0
            )
            acc += jnp.sum(jnp.where(active[..., None], obs_force, 0.0), axis=1)

        # Limit acceleration
        force_mag = jnp.linalg.norm(acc, axis=1, keepdims=True)
        acc = jnp.where(force_mag > max_force, acc * max_force / force_mag, acc)

        target = vel2 + acc * 0.5
        speed = jnp.linalg.norm(target, axis=1, keepdims=True)
        target = jnp.where(speed > max_speed, target * max_speed / speed, target)

        return jnp.concatenate(
            [target, jnp.zeros((target.shape[0], 1), dtype=target.dtype)], axis=1
        )

    return _compute
