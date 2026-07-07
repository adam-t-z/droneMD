"""Generate default simulation playback fixture for the frontend."""

import json
import os
import numpy as np

from backend.flocking.engine import FlockingEngine
from backend.flocking.schemas import SwarmConfig
from backend.utils import generate_default_colors

OUTPUT = "web/public/data/default-playback.json"

config = SwarmConfig(
    n_drones=15,
    duration=20.0,
    device="cpu",
    physics="first_principles",
    integrator="euler",
    freq=500,
    state_freq=50,
    height=1.0,
    motion_primitive="none",
)

print(f"Running default simulation: {config.n_drones} drones, {config.duration}s...")
engine = FlockingEngine(config)
sim_data = engine.run(config.duration)

timestamps = np.asarray(sim_data["timestamps"], dtype=float)
states = np.asarray(sim_data["states"], dtype=float)
num_drones = int(sim_data["num_drones"])
colors = generate_default_colors(num_drones, limit=1.0)

sample_rate = 0.0
if len(timestamps) > 1:
    sample_rate = float(1.0 / np.median(np.diff(timestamps)))

playback = {
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
    "overlays": sim_data.get("overlays"),
}

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(playback, f)

file_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Saved: {OUTPUT} ({file_mb:.1f} MB)")
print(f"Frames: {len(timestamps)}, Drones: {num_drones}")
