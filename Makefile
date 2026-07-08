api:
	npm --prefix web run build && uv run python backend/launch.py

web-dev:
	npm --prefix web run dev

web-build:
	npm --prefix web run build

benchmark:
	uv run python scripts/benchmark.py

benchmark-rocm:
	Dronemd_GPU_BACKEND=rocm uv run python scripts/benchmark.py --device rocm

benchmark-cuda:
	Dronemd_GPU_BACKEND=cuda uv run python scripts/benchmark.py --device cuda