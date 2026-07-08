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

from backend.engine.controller import make_boids_kernel
from backend.engine.primitives import build_trajectory, trajectory_velocities
from backend.routes.schemas import SwarmConfig

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

        self._compute_flocking = make_boids_kernel(
            separation_weight=self.separation_weight,
            alignment_weight=self.alignment_weight,
            cohesion_weight=self.cohesion_weight,
            perception_radius=self.perception_radius,
            max_speed=self.max_speed,
            max_force=self.max_force,
            boundary_mode=self.boundary_mode,
            bounds=self.bounds,
            obstacles=self.obstacles,
        )

        self.hold_targets = None
        if config.obj_points:
            pts = np.array(config.obj_points, dtype=np.float32)
            if pts.shape == (config.n_drones, 3):
                self.hold_targets = pts

    def run(self, duration: float) -> dict:
        """Simulate for `duration` seconds, return playback dict."""
        for _ in self.run_stream(duration):
            pass
        return self._last_result

    def run_stream(self, duration: float):
        """Generator yielding (phase_name, percent) during simulation."""
        n_substeps = self.sim.freq // self.sim.control_freq
        n_control_steps = int(duration * self.sim.control_freq)

        yield ("Initializing simulation engine", 0)

        yield ("Generating initial drone positions", 10)
        height = float(self.config.height)
        if self.hold_targets is not None:
            init_pos = self.hold_targets.copy()
        else:
            init_pos = np.zeros((self.sim.n_drones, 3), dtype=np.float32)
            rng = np.random.default_rng(42)
            init_pos[:, 0] = rng.uniform(self.bounds[0], self.bounds[1], self.sim.n_drones)
            init_pos[:, 1] = rng.uniform(self.bounds[2], self.bounds[3], self.sim.n_drones)
            init_pos[:, 2] = height
        init_vel = np.zeros((self.sim.n_drones, 3), dtype=np.float32)

        init_pos = init_pos[np.newaxis, ...]
        init_vel = init_vel[np.newaxis, ...]
        self.sim.data = self.sim.data.replace(
            states=self.sim.data.states.replace(
                pos=self.sim.data.states.pos.at[0].set(init_pos[0]),
                vel=self.sim.data.states.vel.at[0].set(init_vel[0]),
            ),
            core=self.sim.data.core.replace(
                mjx_synced=self.sim.data.core.mjx_synced.at[...].set(False)
            ),
        )

        primitive_traj = None
        primitive_vel = None
        if self.config.motion_primitive != "none":
            pp = self.config.primitive_params
            t_form = float(pp.get("t_form", min(3.0, duration * 0.3)))
            omega = float(pp.get("rotation", 0.3))
            common = {
                "duration": duration,
                "control_freq": self.sim.control_freq,
                "bounds": self.bounds,
            }
            if self.config.motion_primitive == "circle":
                params = {
                    "radius": float(pp.get("radius", 1.5)),
                    "height": height,
                    "t_form": t_form,
                    "omega": omega,
                }
            elif self.config.motion_primitive == "star":
                params = {
                    "radius": float(pp.get("radius", 1.2)),
                    "delta_radius": float(pp.get("delta_radius", 0.4)),
                    "height": height,
                    "t_form": t_form,
                    "omega": omega,
                }
            elif self.config.motion_primitive == "cone":
                params = {
                    "delta_height": float(pp.get("delta_height", 0.3)),
                    "spacing": float(pp.get("spacing", 0.5)),
                    "height": height,
                    "t_form": t_form,
                    "inverted": bool(pp.get("inverted", False)),
                    "omega": omega,
                }
            else:
                params = {}
            primitive_traj = build_trajectory(
                self.config.motion_primitive, init_pos[0], **common, params=params
            )
            primitive_vel = trajectory_velocities(primitive_traj, self.sim.control_freq)

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

            pos = self.sim.data.states.pos[0]
            vel = self.sim.data.states.vel[0]

            control = np.zeros((1, self.sim.n_drones, 13), dtype=np.float32)
            if primitive_traj is not None:
                target_pos = primitive_traj[:, step, :]
                target_vel_xy = primitive_vel[:, step, :2]
                control[0, :, 0] = target_pos[:, 0]
                control[0, :, 1] = target_pos[:, 1]
                control[0, :, 2] = target_pos[:, 2]
                control[0, :, 3:5] = target_vel_xy.astype(np.float32)
                control[0, :, 9] = np.arctan2(target_vel_xy[:, 1], target_vel_xy[:, 0] + 1e-8)
            elif self.hold_targets is not None:
                control[0, :, 0] = self.hold_targets[:, 0]
                control[0, :, 1] = self.hold_targets[:, 1]
                control[0, :, 2] = self.hold_targets[:, 2]
            else:
                target_vel = self._compute_flocking(pos, vel)
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
