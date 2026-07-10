#!/usr/bin/env python3

import time
import json

import numpy as np
import jax

from crazyflow.sim import Sim
from crazyflow.control import Control


def check_jax_rocm():

    print("=" * 60)
    print("JAX / ROCm CHECK")
    print("=" * 60)

    print("JAX version:", jax.__version__)

    devices = jax.devices()

    print("Devices:")
    for d in devices:
        print(" ", d)

    if not devices:
        raise RuntimeError(
            "No JAX devices found"
        )

    device_string = " ".join(
        str(d).lower()
        for d in devices
    )

    if "rocm" not in device_string:
        raise RuntimeError(
            f"Expected ROCm device, got: {device_string}"
        )

    print("✓ JAX ROCm GPU active")
    print()


def benchmark(
    worlds,
    drones,
    duration=10,
):

    print(
        f"Running {worlds} worlds x {drones} drones"
    )

    sim = Sim(
        n_worlds=worlds,
        n_drones=drones,
        control=Control.state,
    )


    sim.build_reset_fn()
    sim.build_step_fn()


    cmd = np.zeros(
        (
            worlds,
            drones,
            13,
        ),
        dtype=np.float32,
    )

    # hover command
    cmd[..., 2] = 0.4


    #
    # JAX compilation warmup
    #
    sim.reset()

    for _ in range(100):

        sim.state_control(cmd)

        sim.step(
            sim.freq //
            sim.control_freq
        )


    jax.block_until_ready(
        sim.data.states.pos
    )


    #
    # timed run
    #

    steps = int(
        duration *
        sim.freq
    )


    start = time.perf_counter()


    for _ in range(steps):

        sim.state_control(cmd)

        sim.step(
            sim.freq //
            sim.control_freq
        )


    jax.block_until_ready(
        sim.data.states.pos
    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    result = {

        "worlds": worlds,

        "drones_per_world": drones,

        "total_drones":
            worlds * drones,

        "physics_steps":
            steps,

        "wall_seconds":
            elapsed,

        "steps_per_second":
            steps / elapsed,

        "drone_steps_per_second":
            (
                steps *
                worlds *
                drones
            )
            /
            elapsed,

        "real_time_factor":
            duration / elapsed,

        "device":
            str(jax.devices()[0]),
    }


    sim.close()

    return result



def main():

    check_jax_rocm()


    # Maximum: 100 worlds x 500 drones
    tests = [
        (1, 100),
        (10, 100),
        (50, 100),
        (100, 100),
        (100, 500),
    ]


    results = []


    for w, d in tests:

        results.append(
            benchmark(
                w,
                d,
                duration=10,
            )
        )


    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        json.dumps(
            results,
            indent=2
        )
    )


    with open(
        "crazyflow_gpu_results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()