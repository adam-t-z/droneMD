"""DroneMD - GPU-accelerated drone swarm flocking simulation."""

from __future__ import annotations

import logging
import os
import sys

os.environ["SCIPY_ARRAY_API"] = "1"

_logger = logging.getLogger(__name__)

if not os.environ.get("JAX_PLATFORMS"):
    backend_override = os.environ.get("Dronemd_GPU_BACKEND")
    if backend_override:
        if backend_override == "rocm":
            os.environ["JAX_PLATFORMS"] = "rocm"
            _logger.info("GPU backend set to ROCm via Dronemd_GPU_BACKEND env var")
        elif backend_override == "cuda":
            os.environ["JAX_PLATFORMS"] = "cuda"
            _logger.info("GPU backend set to CUDA via Dronemd_GPU_BACKEND env var")
        elif backend_override == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            _logger.info("GPU backend forced to CPU via Dronemd_GPU_BACKEND env var")
    elif sys.platform == "darwin":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
