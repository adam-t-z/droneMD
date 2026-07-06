# DroneMD

DroneMD is a GPU-accelerated drone swarm flocking simulation platform that combines large language models (LLMs) with Boids-based motion planning. It provides a web-based interface for configuring, simulating, and deploying drone swarms using natural language prompts, with physics-accurate quadrotor dynamics running on GPU via MuJoCo/JAX.

<p align="center">
  <img src="assets/drone-sim-trial.gif" alt="DroneMD simulation preview" width="800">
</p>

## Prerequisites

- **Python 3.12+** (see `.python-version`)
- **Node.js 20+** and **npm** (for the web frontend)
- **Linux x86_64** recommended (CUDA/JAX support)

### LLM Backend (choose one)

- **OpenAI** — set your API key:
  ```bash
  export OPENAI_API_KEY="sk-your-key-here"
  ```
- **Ollama (local)** — install the [Ollama CLI](https://ollama.com/download) and run `ollama serve`.

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd dronemd

# Install Python dependencies
uv sync

# Install web dependencies
npm --prefix web install

# Build the frontend and launch the API server
make api
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Usage

1. Select your LLM backend (**ChatGPT/OpenAI** or **Ollama**) in the web UI.
2. Configure flocking parameters via natural language (e.g., drone count, speed, cohesion).
3. Run the simulation and review the animated preview in the browser.
4. Iterate on parameters as needed.
5. Deploy the flight plan to physical Crazyflie drones.

## Make Targets

| Command | Description |
|---|---|
| `make api` | Build the web frontend and start the FastAPI server (`backend/launch.py`) on port 8000 |
| `make web-dev` | Start the Vite development server on port 5173 (hot-reload) |
| `make web-build` | Build the web frontend for production |

## Project Structure

```
├── backend/           # Python API and simulation engine
│   ├── api/           # FastAPI server
│   ├── flocking/      # Flocking simulation router, engine, and schemas
│   ├── core/          # Core simulation logic
│   ├── data/          # Drone configs and scene definitions
│   └── launch.py      # Entry point (uvicorn)
├── web/               # React/Vite frontend
├── Makefile           # Build and run targets
└── pyproject.toml     # Python project configuration
```

## Advanced Configuration

- `backend/data/drones.toml` — drone URIs and home positions
- `backend/data/settings.yaml` — environment and safety filter parameters

## Deployment

For physical drone deployment, start the motion capture tracking system and launch in deploy mode:

```bash
make api
```

Then use the "Deploy" button in the web interface to execute the flight plan on your Crazyflie swarm.

## License

MIT
