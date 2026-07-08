"""GPU / compute-platform introspection helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_platform_name() -> str:
    """Return a human-readable platform label.

    Returns one of ``"ROCm"``, ``"CUDA"``, or ``"CPU"``.
    """
    try:
        import jax
    except ImportError:
        return "CPU"

    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return "CPU"

    if not devices:
        return "CPU"

    platform = str(getattr(devices[0], "platform", "")).lower()
    if "rocm" in platform:
        return "ROCm"
    if "cuda" in platform:
        return "CUDA"
    return "CPU"


def get_gpu_count() -> int:
    """Return the number of available GPU devices (0 if none)."""
    try:
        import jax
    except ImportError:
        return 0

    try:
        return len(jax.devices("gpu"))
    except RuntimeError:
        return 0


def get_device_description(platform: str) -> str:
    """Return a human-readable device description string."""
    if platform.lower() == "cpu":
        return "CPU (no GPU available)"

    try:
        import jax

        devices = jax.devices("gpu")
        if devices:
            d = devices[0]
            kind = str(getattr(d, "device_kind", ""))
            if kind:
                return f"{kind} (via {platform.upper()})"
            return f"{platform.upper()} GPU"
    except (ImportError, RuntimeError):
        pass

    return f"{platform.upper()} GPU"


def query_process_memory_mb() -> int:
    """Return process RSS memory usage in MiB via psutil.

    This works on all platforms (CPU, CUDA, ROCm) and reflects the total
    resident memory used by the sim process including JAX / XLA allocations.
    """
    try:
        import psutil  # noqa: F811
    except ImportError:
        return 0

    try:
        return int(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        return 0
