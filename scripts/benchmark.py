#!/usr/bin/env python3
"""Run a GPU benchmark sweep and write results to a JSON file.

Usage:
    python scripts/benchmark.py                       # default sweep
    python scripts/benchmark.py --drones 10,50,100    # custom drone counts
    python scripts/benchmark.py --device cuda         # force CUDA backend
    python scripts/benchmark.py --device rocm         # force ROCm backend
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.engine.engine import FlockingEngine, resolve_device
from backend.routes.schemas import SwarmConfig
from backend.utils.gpu_info import (
    get_device_description,
    get_gpu_count,
    get_platform_name,
    query_process_memory_mb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DRONE_SWEEP = [10, 25, 50, 100, 200]


def run_single(drones: int, duration: float = 10.0, device: str = "gpu") -> dict:
    """Run a single benchmark and return the metrics dict."""
    config = SwarmConfig(
        n_drones=drones,
        duration=duration,
        device=device,
        physics="first_principles",
        integrator="euler",
        freq=500,
        state_freq=100,
    )

    engine = FlockingEngine(config)
    _ = engine.run(config.duration)

    return engine._gpu_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DroneMD GPU benchmark sweep")
    parser.add_argument(
        "--drones",
        type=str,
        default="",
        help="Comma-separated drone counts (default: 10,25,50,100,200)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Simulation duration per run in seconds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "cuda", "rocm", "gpu"],
        help="Compute device to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "gpu_measurements.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    drone_list: list[int]
    if args.drones:
        drone_list = [int(x.strip()) for x in args.drones.split(",") if x.strip()]
    else:
        drone_list = DEFAULT_DRONE_SWEEP

    cf_dev, platform = resolve_device(args.device)
    desc = get_device_description(platform)

    print(f"Device: {args.device} -> resolved to {platform} ({desc})")
    print(f"Drone sweep: {drone_list}")
    print(f"Duration per run: {args.duration}s")
    print(f"Output: {args.output}")
    print()

    results: list[dict] = []
    total_start = time.perf_counter()

    for n in drone_list:
        print(f"  Running {n} drones ... ", end="", flush=True)
        t0 = time.perf_counter()
        metrics = run_single(n, args.duration, args.device)
        elapsed = time.perf_counter() - t0
        tps = metrics.get("timesteps_per_second", 0)
        print(f"{elapsed:.1f}s wall ({tps:.0f} steps/s)")

        results.append(
            {
                "num_drones": n,
                "duration_seconds": args.duration,
                "wall_time_seconds": elapsed,
                "timesteps_per_second": tps,
                "platform": platform,
                "device_name": desc,
                "device_count": get_gpu_count(),
                "memory_mb": metrics.get("device_memory_mb") or query_process_memory_mb(),
            }
        )

    total_elapsed = time.perf_counter() - total_start
    print()
    print(f"Total: {total_elapsed:.1f}s for {len(drone_list)} runs")

    output = {
        "platform": platform,
        "device_name": desc,
        "device_count": get_gpu_count(),
        "sweep_total_seconds": round(total_elapsed, 1),
        "measurements": results,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
