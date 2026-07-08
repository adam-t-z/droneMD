#!/bin/bash
# DroneMD Deploy Script
# ======================
# Deploys DroneMD to a fresh AMD Developer Cloud MI300X droplet.
# Handles everything: code sync, deps, ROCm JAX, backend.
#
# Usage:
#   cd dronemd
#   ./scripts/deploy.sh <DROPLET_IP>
#
# Prerequisites:
#   1. Create a GPU Droplet on amd.digitalocean.com:
#      - Image: Quick Start (ROCm 7.2)
#      - GPU: MI300X
#      - Add your SSH key (~/.ssh/id_ed25519.pub)
#   2. Run this script from the DroneMD project root
#
# What this script does:
#   - Syncs code to the droplet (excludes node_modules, .git, venv, .venv)
#   - Installs uv, python3.12-venv + Node.js 20 if missing
#   - Installs Python deps via uv sync + ROCm JAX from repo.radeon.com
#   - Installs frontend deps and builds the Vite frontend
#   - Starts FastAPI server on port 8000 (serves API + built frontend)
#
# Known issues this script accounts for:
#   - Port 8000 is occupied by the rocm Docker container → kill it first
#   - AMD ROCm JAX is not on PyPI → install from repo.radeon.com wheels
#   - SSH commands with `pkill` exit non-zero when no process found → `|| true` everywhere

set -e

# ---- Config ----
DROPLET_IP="${1:?Usage: ./scripts/deploy.sh <DROPLET_IP>}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@${DROPLET_IP}"
REMOTE_DIR="/root/dronemd"
SERVER_PORT=8000

echo "============================================"
echo "DroneMD Deploy to ${DROPLET_IP}"
echo "============================================"

# ---- Step 1: Wait for SSH ----
echo ""
echo "[1/7] Waiting for SSH..."
for i in $(seq 1 30); do
    if $SSH "echo ok" > /dev/null 2>&1; then
        echo "  Connected."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Cannot SSH into ${DROPLET_IP} after 60s"
        exit 1
    fi
    sleep 2
done

# ---- Step 2: Sync code ----
echo ""
echo "[2/7] Syncing project files..."
rsync -avz --quiet \
    --exclude 'node_modules' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude '.env' \
    --exclude 'dist' \
    --exclude 'dronemd.egg-info' \
    --exclude 'ros_ws' \
    --exclude 'hidden' \
    "$(pwd)/" "root@${DROPLET_IP}:${REMOTE_DIR}/"
echo "  Done."

# ---- Step 3: Install system deps ----
echo ""
echo "[3/7] Installing system dependencies..."
# Free port 8000 (occupied by rocm Docker container)
$SSH "docker stop rocm 2>/dev/null || true"

# Install Python venv + Node.js if missing
$SSH "apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq python3.12-venv > /dev/null 2>&1" || true

# SSH may drop during apt installs. Wait for it to come back.
sleep 5
for i in $(seq 1 15); do
    if $SSH "echo ok" > /dev/null 2>&1; then break; fi
    sleep 3
done

# Install Node.js 20
$SSH "command -v node > /dev/null 2>&1 || (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1 && apt-get install -y -qq nodejs > /dev/null 2>&1)"
echo "  Node: $($SSH 'node --version 2>/dev/null || echo "not found"')"

# Install uv for Python dep management
$SSH "command -v uv > /dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 && echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> /root/.bashrc)"
echo "  uv: $($SSH 'export PATH=\"\$HOME/.local/bin:\$PATH\" && uv --version 2>/dev/null || echo "not found"')"

# ---- Step 4: Python deps + ROCm JAX ----
echo ""
echo "[4/7] Installing Python dependencies and ROCm JAX..."
$SSH "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ${REMOTE_DIR} && uv sync 2>&1 | tail -3"
$SSH "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd ${REMOTE_DIR} && source .venv/bin/activate && bash scripts/install_rocm_jax.sh 2>&1 | tail -10"
echo "  Done."

# ---- Step 5: Frontend deps + build ----
echo ""
echo "[5/7] Installing frontend dependencies and building..."
$SSH "cd ${REMOTE_DIR}/web && npm install --silent 2>&1 | tail -3"
$SSH "cd ${REMOTE_DIR}/web && npm run build 2>&1 | tail -3"
echo "  Done."

# ---- Step 6: Start backend ----
echo ""
echo "[6/7] Starting server on port ${SERVER_PORT}..."
$SSH "pkill -f 'python backend/launch.py' 2>/dev/null || true"
$SSH "pkill -f 'uvicorn' 2>/dev/null || true"
sleep 1

$SSH "cd ${REMOTE_DIR} && Dronemd_GPU_BACKEND=rocm setsid nohup .venv/bin/python backend/launch.py --host=0.0.0.0 --port=${SERVER_PORT} > /tmp/dronemd.log 2>&1 &"
sleep 5

if $SSH "curl -s http://localhost:${SERVER_PORT} > /dev/null 2>&1"; then
    echo "  Server running."
else
    echo "  WARNING: Server may still be starting."
    echo "  Check: ssh root@${DROPLET_IP} 'tail -20 /tmp/dronemd.log'"
fi

# ---- Summary ----
echo ""
echo "============================================"
echo "DroneMD Deployed!"
echo "============================================"
echo ""
echo "  Open in browser: http://${DROPLET_IP}:${SERVER_PORT}"
echo ""
echo "  Services:"
echo "    Web UI + API:  http://${DROPLET_IP}:${SERVER_PORT}"
echo "    API only:      http://${DROPLET_IP}:${SERVER_PORT}/api/swarm/simulate"
echo "    GPU:           AMD MI300X (192GB HBM3)"
echo ""
echo "  Logs:"
echo "    Server:  ssh root@${DROPLET_IP} 'tail -f /tmp/dronemd.log'"
echo ""
echo "  Redeploy after code changes:"
echo "    ./scripts/deploy.sh ${DROPLET_IP}"
echo ""
echo "  Destroy the droplet when done to save credits!"
echo "============================================"
