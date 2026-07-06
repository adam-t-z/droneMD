"""FastAPI server for the DroneMD flocking simulation."""

from __future__ import annotations

from pathlib import Path

import drone_models
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from backend.flocking.router import router as flocking_router

ROOT = Path(__file__).resolve().parents[2]
WEB_DIST_DIR = ROOT / "web" / "dist"
SWARM_BACKGROUND = ROOT / "dronemd" / "ui" / "swarm.png"
DRONE_ASSET_DIR = Path(drone_models.__file__).resolve().parent / "data" / "assets"


def create_app() -> FastAPI:
    """Create the DroneMD API app."""
    app = FastAPI(title="DroneMD API")
    app.include_router(flocking_router)

    @app.get("/api/assets/swarm.png")
    def swarm_background() -> FileResponse:
        if not SWARM_BACKGROUND.is_file():
            raise HTTPException(status_code=404, detail="Background image not found")
        return FileResponse(SWARM_BACKGROUND)

    @app.get("/api/assets/drone/{asset_path:path}")
    def drone_asset(asset_path: str) -> FileResponse:
        asset_root = DRONE_ASSET_DIR.resolve()
        candidate = (asset_root / asset_path).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(asset_root):
            raise HTTPException(status_code=404, detail="Drone asset not found")
        return FileResponse(candidate)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if not WEB_DIST_DIR.is_dir():
            raise HTTPException(status_code=404, detail="Frontend build not found")
        requested = (WEB_DIST_DIR / full_path).resolve()
        web_root = WEB_DIST_DIR.resolve()
        if requested.is_file() and requested.is_relative_to(web_root):
            headers = {}
            if requested.name == "index.html":
                headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return FileResponse(requested, headers=headers)
        return FileResponse(
            web_root / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return app


app = create_app()
