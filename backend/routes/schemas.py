"""Pydantic schemas for the swarm simulation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SwarmConfig(BaseModel):
    n_drones: int = Field(default=15, ge=1, le=200)
    duration: float = Field(default=30.0, ge=5.0, le=120.0)
    separation_weight: float = Field(default=1.5, ge=0.0, le=5.0)
    alignment_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    cohesion_weight: float = Field(default=1.0, ge=0.0, le=5.0)
    perception_radius: float = Field(default=3.0, ge=0.5, le=10.0)
    max_speed: float = Field(default=2.0, ge=0.5, le=5.0)
    max_force: float = Field(default=0.5, ge=0.1, le=3.0)
    boundary_mode: str = Field(default="wrap", pattern="^(wrap|bounce|hard)$")
    bounds: list[float] = Field(default=[-2.0, 2.0, -2.0, 2.0])
    obstacles: list[dict] = Field(default_factory=list)
    device: str = Field(default="cpu", pattern="^(cpu|cuda|rocm|gpu)$")
    height: float = Field(default=1.0, ge=0.5, le=5.0)
    physics: str = Field(
        default="first_principles",
        pattern="^(first_principles|so_rpy|so_rpy_rotor|so_rpy_rotor_drag)$",
    )
    integrator: str = Field(default="euler", pattern="^(euler|rk4|symplectic_euler)$")
    freq: int = Field(default=500, ge=250, le=2000)
    state_freq: int = Field(default=100, ge=20, le=200)
    motion_primitive: str = Field(default="none", pattern="^(none|circle|star|cone)$")
    primitive_params: dict = Field(default_factory=dict)
    obj_points: list[list[float]] | None = Field(default=None)


class GpuMetrics(BaseModel):
    """Per-run GPU / compute benchmark data."""

    platform: str = Field(default="cpu")
    device_name: str = Field(default="CPU")
    device_count: int = Field(default=0)
    sim_time_seconds: float = Field(default=0.0)
    num_drones: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    physics_freq_hz: int = Field(default=500)
    control_freq_hz: int = Field(default=100)
    timesteps_per_second: float = Field(default=0.0)
    device_memory_mb: int | None = Field(default=None)


class BenchmarkHistory(BaseModel):
    """Historical benchmark runs for the dashboard."""

    platform: str = Field(default="cpu")
    device_name: str = Field(default="Unknown")
    device_count: int = Field(default=0)
    measurements: list[GpuMetrics] = Field(default_factory=list)
