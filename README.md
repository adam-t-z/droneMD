# DroneMD

GPU-accelerated drone swarm simulation platform with physics-accurate Boids flocking running on AMD GPUs via JAX/MuJoCo.

Built for the **AMD Developer Hackathon: Stage II** (lablab.ai).

<p align="center">
  <img src="assets/drone-sim-trial.gif" alt="DroneMD swarm simulation" width="800">
</p>

## What It Does

DroneMD lets you configure a drone swarm and immediately see it fly — with real physics, on real GPU hardware.

1. **Physics-Accurate Engine** — Each drone runs as a physically simulated Crazyflie quadrotor via MuJoCo/crazyflow, with JIT-compiled JAX kernels for Boids steering (separation, alignment, cohesion, goal-seeking, collision avoidance).

2. **Rich Configuration** — Tweak every aspect of the swarm through sliders: Boids weights (separation, alignment, cohesion, goal, obstacle avoidance), boundary behavior, physics fidelity (4 levels), integrators, and motion primitives (circle, star, cone, OBJ-based formations).

3. **Real-Time 3D Preview** — Watch your swarm fly in the browser with Three.js rendering, drone trails, collision heatmaps, and a full-screen cinematic mode.

4. **Analytics & Export** — Post-simulation report with 13 metrics (speed, collisions, formation error, connectivity, energy, safety score). Export trajectories to CSV, JSON, ROS waypoints, or PDF.

## GPU Benchmarks (Measured on AMD Radeon RX 7600S)

Sweep across drone counts, 10 simulated seconds each:

| Drones | Wall Time (s) | Timesteps/sec | VRAM (MB) |
|--------|--------------|---------------|-----------|
| 10 | 5.80 | 312.0 | 1,867 |
| 25 | 5.88 | 289.1 | 3,046 |
| 50 | 6.88 | 265.7 | 5,068 |
| 100 | 8.33 | 244.2 | 7,897 |
| 200 | 13.54 | 212.0 | 13,449 |

All measurements captured with JAX CUDA backend. Full data in `data/gpu_measurements.json`.

## Architecture

```
React + Three.js Frontend (Vite, port 5173)
        |
FastAPI Server (port 8000, serves frontend + API)
        |
Simulation Engine (JAX GPU kernel)
    ├── MuJoCo Physics (crazyflow quadrotor model)
    ├── Boids Flocking (JIT-compiled JAX)
    ├── Motion Primitives (circle, star, cone)
    └── OBJ Mesh Sampler (custom formations)
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| GPU Compute | JAX (CUDA / ROCm), MuJoCo via crazyflow |
| Physics | Crazyflie quadrotor dynamics, Euler/Runge-Kutta integrators |
| Flocking | Boids algorithm: separation, alignment, cohesion, JIT-compiled |
| 3D Rendering | Three.js, STL drone models, dynamic trails, collision effects |
| Charts | Recharts (speed, formation error, connectivity, collision metrics) |
| Frontend | React 19, TypeScript, Vite 7 |
| Backend | FastAPI, Python 3.12, Pydantic |
| Export | CSV, JSON, ROS waypoints, PDF (HTML print) |

## Deploy to AMD Developer Cloud

One-command deployment to a fresh MI300X droplet:

```bash
# 1. Create a GPU Droplet on amd.digitalocean.com
#    Image: Quick Start (ROCm 7.2)
#    GPU: MI300X
#    Add your SSH key

# 2. Deploy
./scripts/deploy.sh <DROPLET_IP>

# 3. Open in browser
# http://<DROPLET_IP>:8000
```

The deploy script handles: code sync, Python venv + `uv sync`, ROCm JAX installation from repo.radeon.com, Node.js, frontend build, and FastAPI server startup on port 8000.


## Deploy Locally (CPU-fallback)

Install `uv`, a fast python package manager from [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) 


```bash
# Clone and enter
git clone <repo-url> && cd dronemd

# Install Python dependencies
uv sync

# Build frontend + start API server
make api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Make Targets

| Command | Description |
|---------|------------|
| `make api` | Build frontend + start FastAPI server on port 8000 |
| `make web-dev` | Start Vite dev server on port 5173 (hot-reload) |
| `make web-build` | Production build of the frontend |
| `make benchmark` | Run GPU benchmark sweep (10–200 drones) |
| `make benchmark-rocm` | Force ROCm backend for benchmark |

## Usage

1. Configure your swarm: adjust drone count, Boids weights, physics model, motion primitive, and boundary mode via the sliders in the web UI.
2. Optionally upload an OBJ mesh to fly drones in a custom formation.
3. Click **Simulate** and watch the real-time 3D preview.
4. Inspect the analytics report with charts and metrics.
5. Export the flight plan (CSV, JSON, ROS waypoints).
6. Export the report of safety and simulation (PDF, Text)

## Project Structure

```
dronemd/
  backend/
    launch.py                Entry point: uvicorn server
    api/server.py            FastAPI app (routes, static files, CORS)
    engine/
      engine.py              FlockingEngine: MuJoCo sim + Boids controller
      controller.py          Boids steering: JIT-compiled JAX kernel
      primitives.py          Motion primitives: circle, star, cone
      shapes.py              OBJ mesh surface sampling (trimesh)
    routes/
      router.py              API endpoints: /simulate, /stream, /upload-obj, /benchmark
      schemas.py             Pydantic models: SwarmConfig, GpuMetrics, BenchmarkHistory
    data/
      drones.toml            Physical Crazyflie URIs and home positions
      settings.yaml          Environment and safety parameters
    utils/
      gpu_info.py            GPU platform introspection (ROCm/CUDA/CPU)
      utils.py               B-spline discretization, color generation
  web/
    src/
      SwarmLab.tsx            Main UI: controls, presets, simulation, 3D preview
      Player.tsx              Three.js 3D drone renderer (trails, collisions, cinematic)
      ReportPanel.tsx         Post-simulation analytics with Recharts
      BenchmarkCard.tsx       GPU benchmark display with history table
      Onboarding.tsx          Interactive walkthrough tour
      export.ts              CSV, JSON, ROS, TXT, PDF report export
    tests/
      player.spec.js         Playwright e2e tests (desktop + mobile + deploy)
  scripts/
    deploy.sh                One-command deploy to AMD Developer Cloud
    benchmark.py             GPU benchmark sweep (10–200 drone counts)
    install_rocm_jax.sh      AMD ROCm JAX installation from repo.radeon.com
    generate_default_playback.py   Precompute demo playback data
    precompute_obj_points.py       Pre-sample OBJ surfaces for formations
  ros_ws/                    ROS workspace for motion capture tracking
  data/
    gpu_measurements.json    Saved GPU benchmark results
  assets/
    drone-sim-trial.gif      Animated simulation preview
```

## License

MIT
