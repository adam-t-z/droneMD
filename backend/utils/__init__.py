"""Utility modules for dronemd."""

from backend.utils.utils import discretize_bspline, generate_default_colors

__all__ = [
    "discretize_bspline",
    "generate_default_colors",
    "get_device_description",
    "get_gpu_count",
    "get_platform_name",
    "query_process_memory_mb",
]
