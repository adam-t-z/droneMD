"""One-shot script to pre-sample points from male.obj for the frontend default."""

from __future__ import annotations

import json

from backend.engine.shapes import sample_obj_surface

POINTS = sample_obj_surface("male.obj", 100)

OUTPUT_PATH = "web/public/data/default-obj-points.json"

with open(OUTPUT_PATH, "w") as f:
    json.dump({"points": POINTS.tolist(), "n": 100}, f)

print(f"Wrote {len(POINTS)} points to {OUTPUT_PATH}")
