"""FastAPI router for the swarm simulation endpoint."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from backend.engine import FlockingEngine, sample_obj_surface
from backend.routes.schemas import BenchmarkHistory, GpuMetrics, SwarmConfig
from backend.utils import generate_default_colors
from backend.utils.gpu_info import get_device_description, get_gpu_count, get_platform_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/swarm", tags=["swarm"])

_benchmark_history: list[GpuMetrics] = []


def _normalize_swarm_playback(sim_data: dict, config: SwarmConfig) -> dict:
    timestamps = np.asarray(sim_data["timestamps"], dtype=float)
    states = np.asarray(sim_data["states"], dtype=float)
    num_drones = int(sim_data["num_drones"])

    colors = generate_default_colors(num_drones, limit=1.0)

    sample_rate = 0.0
    if len(timestamps) > 1:
        sample_rate = float(1.0 / np.median(np.diff(timestamps)))

    overlays = sim_data.get("overlays")
    result = {
        "schemaVersion": 1,
        "numDrones": num_drones,
        "timestamps": timestamps.tolist(),
        "states": states.tolist(),
        "fields": {"pos": [0, 3], "quat": [3, 7], "vel": [7, 10], "angVel": [10, 13]},
        "bounds": {
            "min": [config.bounds[0], config.bounds[2], 0.0],
            "max": [config.bounds[1], config.bounds[3], config.height],
        },
        "colors": colors.tolist(),
        "sampleRate": sample_rate,
        "gpuPlatform": sim_data.get("gpu_platform", "cpu"),
        "deviceInfo": sim_data.get(
            "device_info", {"platform": "cpu", "device_name": "CPU", "device_kind": "cpu"}
        ),
        "gpuMetrics": sim_data.get("gpu_metrics"),
    }
    if overlays:
        result["overlays"] = overlays
    return result


@router.post("/simulate")
def simulate(config: SwarmConfig) -> dict:
    try:
        engine = FlockingEngine(config)
        sim_data = engine.run(config.duration)
        _store_benchmark(sim_data)
        playback = _normalize_swarm_playback(sim_data, config)
        return playback
    except Exception as exc:
        logger.exception("Swarm simulation failed")
        return {"error": str(exc)}


@router.post("/simulate/stream")
def simulate_stream(config: SwarmConfig) -> StreamingResponse:
    def generate():
        try:
            engine = FlockingEngine(config)
            for phase, percent in engine.run_stream(config.duration):
                yield json.dumps({"type": "progress", "phase": phase, "percent": percent}) + "\n"
            sim_data = engine._last_result
            _store_benchmark(sim_data)
            playback = _normalize_swarm_playback(sim_data, config)
            yield json.dumps({"type": "result", "data": playback}) + "\n"
        except Exception as exc:
            logger.exception("Streaming swarm simulation failed")
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/shapes/upload-obj")
def upload_obj(
    file: UploadFile = File(...), n_drones: int = Query(default=100, ge=1, le=200)
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".obj",):
        raise HTTPException(status_code=400, detail="Only .obj files are accepted")

    tmp_dir = tempfile.mkdtemp(prefix="dronemd_obj_")
    tmp_path = Path(tmp_dir) / f"upload{suffix}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        points = sample_obj_surface(str(tmp_path), n_drones)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to process OBJ: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"points": points.tolist(), "n_drones": n_drones}


@router.get("/benchmark")
def get_benchmark() -> BenchmarkHistory:
    """Return historical GPU benchmark data for the dashboard."""
    return BenchmarkHistory(
        platform=get_platform_name().lower(),
        device_name=get_device_description(get_platform_name()),
        device_count=get_gpu_count(),
        measurements=_benchmark_history[-20:],
    )


def _store_benchmark(sim_data: dict) -> None:
    """Persist a completed run's GPU metrics into the history."""
    gm = sim_data.get("gpu_metrics")
    if not gm:
        return
    try:
        _benchmark_history.append(
            GpuMetrics(
                platform=gm.get("platform", "cpu"),
                device_name=gm.get("device_name", "Unknown"),
                device_count=gm.get("device_count", 0),
                sim_time_seconds=gm.get("sim_time_seconds", 0.0),
                num_drones=gm.get("num_drones", 0),
                duration_seconds=gm.get("duration_seconds", 0.0),
                physics_freq_hz=gm.get("physics_freq_hz", 500),
                control_freq_hz=gm.get("control_freq_hz", 100),
                timesteps_per_second=gm.get("timesteps_per_second", 0.0),
                device_memory_mb=gm.get("device_memory_mb"),
            )
        )
    except Exception:
        logger.debug("Failed to store benchmark history", exc_info=True)
