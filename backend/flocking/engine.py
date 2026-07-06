"""Flocking engine wrapping crazyflow.Sim with a Boids controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.integration import Integrator

if TYPE_CHECKING:
    from numpy.typing import NDArray

from backend.flocking.schemas import SwarmConfig

logger = logging.getLogger(__name__)


class FlockingEngine:
    """Wraps crazyflow.Sim with a Boids flocking controller.

    Runs physics-accurate quadrotor dynamics (via crazyflow/MuJoCo on GPU)
    driven by a lightweight Boids velocity controller.
    """

    def __init__(self, config: SwarmConfig) -> None:
        self.config = config

        device = config.device
        if device == "gpu":
            try:
                jax.devices("gpu")
            except RuntimeError:
                logger.warning("GPU requested but not available; falling back to CPU")
                device = "cpu"

        self.sim = Sim(
            n_worlds=1,
            n_drones=config.n_drones,
            drone_model="cf21B_500",
            physics=Physics[config.physics],
            control=Control.state,
            integrator=Integrator[config.integrator],
            freq=config.freq,
            state_freq=config.state_freq,
            device=device,
        )

        self.sim.reset()
        dummy = np.zeros((1, self.sim.n_drones, 13), dtype=np.float32)
        self.sim.state_control(dummy)
        self.sim.step(self.sim.freq // self.sim.control_freq)
        self.sim.reset()

        self.separation_weight = config.separation_weight
        self.alignment_weight = config.alignment_weight
        self.cohesion_weight = config.cohesion_weight
        self.perception_radius = config.perception_radius
        self.max_speed = config.max_speed
        self.max_force = config.max_force
        self.boundary_mode = config.boundary_mode
        self.bounds = np.array(config.bounds)
        self.obstacles = config.obstacles

    def run(self, duration: float) -> dict:
        """Simulate for `duration` seconds, return playback dict."""
        for _ in self.run_stream(duration):
            pass
        return self._last_result

    def run_stream(self, duration: float):
        """Generator yielding (phase_name, percent) during simulation."""
        n_substeps = self.sim.freq // self.sim.control_freq
        n_control_steps = int(duration * self.sim.control_freq)
        height = float(self.config.height)

        yield ("Initializing simulation engine", 0)

        yield ("Generating initial drone positions", 10)
        rng = np.random.default_rng(42)
        init_pos = np.zeros((1, self.sim.n_drones, 3), dtype=np.float32)
        init_pos[0, :, 0] = rng.uniform(self.bounds[0], self.bounds[1], self.sim.n_drones)
        init_pos[0, :, 1] = rng.uniform(self.bounds[2], self.bounds[3], self.sim.n_drones)
        init_pos[0, :, 2] = height
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=self.sim.data.states.pos.at[0].set(init_pos[0])
            ),
            core=self.sim.data.core.replace(
                mjx_synced=self.sim.data.core.mjx_synced.at[...].set(False)
            ),
        )

        init_vel = np.zeros((1, self.sim.n_drones, 3), dtype=np.float32)
        init_vel[0, :, 0] = rng.uniform(-0.5, 0.5, self.sim.n_drones)
        init_vel[0, :, 1] = rng.uniform(-0.5, 0.5, self.sim.n_drones)
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                vel=self.sim.data.states.vel.at[0].set(init_vel[0])
            ),
            core=self.sim.data.core.replace(
                mjx_synced=self.sim.data.core.mjx_synced.at[...].set(False)
            ),
        )

        states = np.empty((n_control_steps, self.sim.n_drones, 13), dtype=np.float64)
        timestamps = np.empty(n_control_steps, dtype=np.float64)

        sim_start_pct = 15
        sim_end_pct = 80
        sim_range = sim_end_pct - sim_start_pct
        report_interval = max(1, n_control_steps // 10)

        for step in range(n_control_steps):
            if step % report_interval == 0:
                pct = sim_start_pct + (step / n_control_steps) * sim_range
                yield ("Running flocking simulation", int(pct))

            pos = np.asarray(self.sim.data.states.pos[0])
            vel = np.asarray(self.sim.data.states.vel[0])

            target_vel = self._compute_flocking(pos, vel)

            control = np.zeros((1, self.sim.n_drones, 13), dtype=np.float32)
            control[0, :, 0] = pos[:, 0] + target_vel[:, 0] * (n_substeps / self.sim.freq)
            control[0, :, 1] = pos[:, 1] + target_vel[:, 1] * (n_substeps / self.sim.freq)
            control[0, :, 2] = np.full(self.sim.n_drones, height, dtype=np.float32)
            control[0, :, 3:5] = target_vel[:, :2].astype(np.float32)
            control[0, :, 9] = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)

            self.sim.state_control(control)
            self.sim.step(n_substeps)

            quat = np.asarray(self.sim.data.states.quat[0])
            ang_vel = np.asarray(self.sim.data.states.ang_vel[0])
            state_vec = np.concatenate([pos, quat, vel, ang_vel], axis=-1)
            states[step] = state_vec
            timestamps[step] = step / self.sim.control_freq

        yield ("Running flocking simulation", sim_end_pct)

        self.sim.close()

        yield ("Computing collision and speed data", 90)
        pos_t = states[:, :, :3]
        diffs = pos_t[:, :, None, :] - pos_t[:, None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        close = (dists < 0.3) & (dists > 0)
        collisions_per_frame = [
            [(int(i), int(j)) for i, j in zip(*np.where(frame)) if i < j] for frame in close
        ]

        vel_t = states[:, :, 7:10]
        speeds = np.linalg.norm(vel_t, axis=-1)

        yield ("Finalizing playback data", 95)
        self._last_result = {
            "num_drones": self.sim.n_drones,
            "timestamps": timestamps,
            "states": states,
            "controls": np.empty((0, self.sim.n_drones, 6)),
            "waypoints": {},
            "solve_times": np.array([]),
            "overlays": {"collisions_per_frame": collisions_per_frame, "speeds": speeds.tolist()},
        }
        yield ("Simulation complete", 100)

    def _compute_flocking(self, pos: NDArray, vel: NDArray) -> NDArray:
        """Compute target velocity for each drone using Boids rules."""
        n = self.sim.n_drones
        target = np.zeros((n, 2), dtype=np.float64)
        pos_2d = pos[:, :2]
        vel_2d = vel[:, :2]

        sep_rad = self.perception_radius * 0.5

        for i in range(n):
            acc = np.zeros(2, dtype=np.float64)
            neighbors = 0
            mean_pos = np.zeros(2, dtype=np.float64)
            mean_vel = np.zeros(2, dtype=np.float64)

            for j in range(n):
                if i == j:
                    continue
                diff = pos_2d[i] - pos_2d[j]
                dist = np.sqrt(diff @ diff)
                if dist < self.perception_radius and dist > 1e-8:
                    if dist < sep_rad:
                        acc += (diff / dist) * self.separation_weight / max(dist, 0.1)
                    mean_pos += pos_2d[j]
                    mean_vel += vel_2d[j]
                    neighbors += 1

            if neighbors > 0:
                mean_pos /= neighbors
                mean_vel /= neighbors
                cohesion = (mean_pos - pos_2d[i]) * self.cohesion_weight * 0.5
                alignment = (mean_vel - vel_2d[i]) * self.alignment_weight * 0.5
                acc += cohesion + alignment

            if self.boundary_mode == "wrap":
                half_w = (self.bounds[1] - self.bounds[0]) * 0.5
                half_h = (self.bounds[3] - self.bounds[2]) * 0.5
                cx = (self.bounds[0] + self.bounds[1]) * 0.5
                cy = (self.bounds[2] + self.bounds[3]) * 0.5
                dx = pos_2d[i, 0] - cx
                dy = pos_2d[i, 1] - cy
                if abs(dx) > half_w:
                    acc[0] -= np.sign(dx) * (abs(dx) - half_w) * 2.0
                if abs(dy) > half_h:
                    acc[1] -= np.sign(dy) * (abs(dy) - half_h) * 2.0
            elif self.boundary_mode in ("bounce", "hard"):
                margin = 0.3
                for axis in (0, 1):
                    lo = self.bounds[axis * 2]
                    hi = self.bounds[axis * 2 + 1]
                    if pos_2d[i, axis] < lo + margin:
                        acc[axis] += self.max_speed * 2.0
                    elif pos_2d[i, axis] > hi - margin:
                        acc[axis] -= self.max_speed * 2.0

            for ox, oy, r in self.obstacles:
                dx = pos_2d[i, 0] - ox
                dy = pos_2d[i, 1] - oy
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r + 0.5 and dist > 1e-8:
                    push = (r + 0.5 - dist) / (r + 0.5)
                    acc[0] += (dx / dist) * push * self.max_speed * 3.0
                    acc[1] += (dy / dist) * push * self.max_speed * 3.0

            force_mag = np.sqrt(acc @ acc)
            if force_mag > self.max_force:
                acc = acc / force_mag * self.max_force

            target[i] = vel_2d[i] + acc * 0.5

        speed = np.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)
        too_fast = speed > self.max_speed
        target[too_fast] *= self.max_speed / speed[too_fast, np.newaxis]

        result = np.zeros((n, 3), dtype=np.float64)
        result[:, :2] = target
        return result
