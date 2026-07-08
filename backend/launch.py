"""Launch script for the DroneMD flocking simulation."""

from __future__ import annotations

import logging
import os

import uvicorn

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def main(host: str = "127.0.0.1", port: int = 8000):
    """Launch the DroneMD browser API."""
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    logging.getLogger("jax").setLevel(logging.WARNING)

    from backend.api.server import app

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
