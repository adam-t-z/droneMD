api:
	npm --prefix web run build && uv run python backend/launch.py

web-dev:
	npm --prefix web run dev

web-build:
	npm --prefix web run build