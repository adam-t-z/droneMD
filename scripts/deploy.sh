#!/bin/bash
# DroneMD Deploy Script
# ======================
# Deploys DroneMD to a fresh AMD Developer Cloud MI300X droplet.
#
# The AMD cloud droplet runs all ROCm software inside a Docker container
# called "rocm". JAX + ROCm libraries live inside that container, NOT on
# the host.  We MUST work inside the container for GPU access.
#
# Usage:
#   cd dronemd
#   ./scripts/deploy.sh <DROPLET_IP>
#
# Prerequisites:
#   1. Create a GPU Droplet on amd.digitalocean.com:
#      - Image: Quick Start (ROCm 7.2) or JAX + ROCm image
#      - GPU: MI300X
#      - Add your SSH key (~/.ssh/id_ed25519.pub)
#   2. Run this script from the DroneMD project root
#
# What this script does:
#   - Pre-flight checks (GPU, Python, disk, Docker container)
#   - Syncs code to the droplet host
#   - Copies code into the rocm Docker container
#   - Installs Python deps + ROCm JAX inside the container
#   - Builds the Vite frontend (on host, needs Node.js)
#   - Starts FastAPI server inside the rocm container on port 8000
#
# Architecture note:
#   Host OS → runs Docker → rocm container (JAX + ROCm + GPU access)
#   Frontend is built on the host (Node.js), then served by the
#   FastAPI backend which runs inside the rocm container.
#
# Known issues:
#   - The rocm container must stay running — stopping it reboots the droplet
#   - Port 8000 in the container is used by a dev server → kill it first
#   - AMD ROCm JAX may be pre-installed in the container, or install from wheels
#   - uv is not available inside the container → use pip

set -e

# ---- Config ----
DROPLET_IP="${1:?Usage: ./scripts/deploy.sh <DROPLET_IP>}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@${DROPLET_IP}"
DOCKER_EXEC="docker exec rocm"
REMOTE_DIR="/root/dronemd"
SERVER_PORT=8000

echo "============================================"
echo "DroneMD Deploy to ${DROPLET_IP}"
echo "============================================"

# ---- Helper: wait for SSH to come back ----
wait_for_ssh() {
    local reason="${1:-unknown}"
    echo "  → Waiting for SSH to recover (${reason})..."
    for i in $(seq 1 20); do
        if $SSH "echo ok" > /dev/null 2>&1; then
            echo "  → SSH reconnected"
            return 0
        fi
        echo "    attempt ${i}/20..."
        sleep 3
    done
    echo "  ✗ ERROR: SSH did not recover."
    exit 1
}

# ---- Helper: run command inside the rocm container ----
roc() {
    $SSH "${DOCKER_EXEC} bash -c \"$1\"" 2>&1
}

# ---- Step 1: Wait for SSH ----
echo ""
echo "[1/8] Connecting to droplet..."
for i in $(seq 1 30); do
    if $SSH "echo ok" > /dev/null 2>&1; then
        echo "  ✓ SSH connected on attempt $i"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ✗ ERROR: Cannot SSH into ${DROPLET_IP} after 60s."
        echo "    - Droplet may still be provisioning"
        echo "    - Wrong IP address"
        echo "    - SSH key not added to droplet"
        exit 1
    fi
    sleep 2
done

# ---- Step 2: Pre-flight checks (host) ----
echo ""
echo "[2/8] Pre-flight checks — host..."

echo "  → Waiting for droplet provisioning to complete..."
# AMD cloud droplets show a "Please wait while we get your droplet ready..."
# message in command output until provisioning finishes.
for i in $(seq 1 30); do
    PROVISION_CHECK=$($SSH "echo READY" 2>&1)
    if echo "${PROVISION_CHECK}" | grep -q "READY"; then
        echo "    Droplet is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ✗ ERROR: Droplet did not finish provisioning after 90s."
        echo "    It may be stuck. Check the AMD cloud console."
        exit 1
    fi
    echo "    Still provisioning... (attempt ${i}/30)"
    sleep 3
done

echo "  → Checking system resources..."

OS_INFO=$($SSH "cat /etc/os-release 2>/dev/null | head -1 || uname -a" 2>&1)
echo "    OS: ${OS_INFO}"

DF_OUT=$($SSH "df -h / | tail -1" 2>&1)
echo "    Disk: ${DF_OUT}"
USED_PCT=$($SSH "df / | tail -1 | awk '{print \$5}' | tr -d '%'" 2>/dev/null || echo "0")
if [ "${USED_PCT}" -gt 90 ] 2>/dev/null; then
    echo "  ⚠ WARNING: Disk is ${USED_PCT}% full."
fi

echo "  → Checking rocm Docker container..."
# The rocm container may take 1-2 minutes to start after droplet creation.
# Retry several times before giving up.
CONTAINER_READY=0
for i in $(seq 1 10); do
    if $SSH "docker ps --format '{{.Names}}' 2>/dev/null | grep -q rocm" 2>/dev/null; then
        if $SSH "${DOCKER_EXEC} echo ok" > /dev/null 2>&1; then
            echo "    ✓ rocm container is running (found on attempt ${i})"
            CONTAINER_READY=1
            break
        fi
    fi
    if [ "$i" -eq 10 ]; then
        echo "  ⚠ rocm container not responsive after 10 attempts."
        echo "    Checking what Docker containers exist..."
        $SSH "docker ps -a 2>&1" | sed 's/^/      /'
        echo ""
        echo "    If the 'rocm' container doesn't exist, this droplet image"
        echo "    may not include the ROCm Docker setup."
        echo "    Check the AMD cloud console — you may need the"
        echo "    'Quick Start (ROCm 7.2)' or JAX + ROCm image."
        echo ""
        echo "    If the container exists but isn't running:"
        echo "      ssh root@${DROPLET_IP} 'docker start rocm'"
        exit 1
    fi
    echo "    Container not ready yet (attempt ${i}/10 — this is normal for new droplets)..."
    sleep 10
done

if [ "${CONTAINER_READY}" != "1" ]; then
    echo "  ✗ ERROR: Cannot communicate with rocm container."
    exit 1
fi

# Check GPU inside the container
echo "  → Checking GPU inside container (10s timeout)..."
GPU_OUT=$(timeout 15 $SSH "${DOCKER_EXEC} bash -c \"rocm-smi --showproductname 2>/dev/null\" || echo 'ROCM_CHECK_TIMEOUT'" 2>&1)
if echo "${GPU_OUT}" | grep -q "ROCM_CHECK_TIMEOUT"; then
    echo "  ⚠ rocm-smi timed out — the container may still be initializing."
    echo "    This is normal for freshly created droplets. Continuing..."
elif [ -z "${GPU_OUT}" ]; then
    echo "  ⚠ WARNING: rocm-smi returned no output. GPU may not be accessible."
else
    echo "${GPU_OUT}" | head -5 | sed 's/^/    /'
fi

# Check JAX inside the container
echo "  → Checking JAX inside container (15s timeout)..."
# Create a small check script to avoid quoting hell with bash -c + python -c
$SSH "${DOCKER_EXEC} bash -c 'cat > /tmp/check_jax.py << \"PYEOF\"
import jax
print(f\"JAX {jax.__version__}\")
devices = jax.devices(\"gpu\")
print(f\"GPU devices: {len(devices)}\")
for d in devices:
    print(f\"  {d.device_kind}\")
PYEOF
'" 2>/dev/null || true
JAX_INFO=$(timeout 20 $SSH "${DOCKER_EXEC} python3 /tmp/check_jax.py 2>&1" || echo 'JAX_CHECK_FAILED')
if echo "${JAX_INFO}" | grep -q "JAX_CHECK_FAILED"; then
    echo "  ⚠ WARNING: JAX check timed out or failed."
    echo "    JAX will be installed during Step 5."
elif echo "${JAX_INFO}" | grep -q "JAX_NOT_FOUND\|ModuleNotFoundError\|No module named"; then
    echo "  ⚠ WARNING: JAX is NOT pre-installed in the container."
    echo "    It will be installed during Step 5."
else
    echo "${JAX_INFO}" | sed 's/^/    /'
    echo "    ✓ JAX is pre-installed — skipping ROCm JAX install step"
    JAX_PREINSTALLED=1
fi

echo "  ✓ Pre-flight checks complete"

# ---- Step 3: Sync code to host ----
echo ""
echo "[3/8] Syncing project files to host..."
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
echo "  ✓ Code synced to host"

# ---- Step 4: Copy code into the rocm container ----
echo ""
echo "[4/8] Copying code into rocm container..."
echo "  → Why: The rocm container holds the GPU drivers and JAX."
echo "    Code must be inside the container for the backend to run."

$SSH "${DOCKER_EXEC} mkdir -p ${REMOTE_DIR}" || true

# Copy the code from host into the container
$SSH "docker cp ${REMOTE_DIR}/. rocm:${REMOTE_DIR}/" 2>&1
echo "  ✓ Code is inside the container"

# ---- Step 5: Python deps inside container ----
echo ""
echo "[5/8] Python dependencies inside rocm container..."
echo "  → Removing old venv inside container..."
roc "rm -rf ${REMOTE_DIR}/.venv" || true

echo "  → Creating venv and installing packages..."
echo "    Note: using pip directly since uv is not available in the container."

# Create a venv inside the container
roc "python3 -m venv ${REMOTE_DIR}/.venv"

# Install main project deps
echo "  → Installing project dependencies..."
roc ".venv/bin/pip install --quiet -e ${REMOTE_DIR}/. 2>&1" | tail -5

# Install ROCm JAX if not pre-installed
if [ "${JAX_PREINSTALLED:-0}" = "1" ]; then
    echo "  → Linking pre-installed JAX into venv..."
    # The container's system JAX needs to be accessible from the venv
    roc "rm -rf ${REMOTE_DIR}/.venv/lib/python*/site-packages/jax ${REMOTE_DIR}/.venv/lib/python*/site-packages/jaxlib* ${REMOTE_DIR}/.venv/lib/python*/site-packages/jax_rocm* 2>/dev/null || true"
    # Create .pth file to add system site-packages
    SITE_PKGS=$(roc "python3 -c 'import site; print(site.getsitepackages()[0])'")
    roc "echo '${SITE_PKGS}' > ${REMOTE_DIR}/.venv/lib/python3.*/site-packages/system_rocm.pth 2>/dev/null || true"
    echo "    ✓ System JAX linked into venv"
else
    echo "  → Installing ROCm JAX from repo.radeon.com..."
    echo "    (downloads ~200MB of wheels — this takes 2-3 minutes)"
    # The install_rocm_jax.sh script requires the venv to be active
    # and pip to be the venv's pip
    roc "cd ${REMOTE_DIR} && source .venv/bin/activate && bash scripts/install_rocm_jax.sh 2>&1" | tail -12
fi

# Verify JAX + GPU
echo "  → Verifying JAX GPU access from venv..."
JAX_VERIFY=$(roc "cd ${REMOTE_DIR} && .venv/bin/python -c '
import jax
print(f\"JAX {jax.__version__}\")
try:
    devices = jax.devices(\"gpu\")
    print(f\"GPU devices: {len(devices)}\")
    for d in devices:
        print(f\"  {d.device_kind}\")
except Exception as e:
    print(f\"GPU error: {e}\")
'" 2>&1)
echo "${JAX_VERIFY}" | sed 's/^/    /'

if echo "${JAX_VERIFY}" | grep -q "GPU devices: 0"; then
    echo "  ⚠ WARNING: JAX found 0 GPU devices."
    echo "    The backend will run on CPU — very slow."
    echo "    Check: rocm-smi inside the container."
fi
if echo "${JAX_VERIFY}" | grep -q "GPU error\|ModuleNotFoundError"; then
    echo "  ⚠ WARNING: JAX GPU check failed."
    echo "    Check: ssh root@${DROPLET_IP} 'docker exec rocm python3 -c \"import jax; print(jax.devices())\"'"
fi

echo "  ✓ Python dependencies installed"

# ---- Step 6: Install Node.js on host + build frontend ----
echo ""
echo "[6/8] Frontend build (on host)..."
echo "  → Why: Frontend build runs on the host since it only needs Node.js."
echo "    The built files are copied into the container afterwards."

echo "  → Checking Node.js on host..."
if $SSH "command -v node > /dev/null 2>&1"; then
    echo "    Node: $($SSH 'node --version 2>/dev/null')"
else
    echo "    Installing Node.js 20..."
    $SSH "(curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1 && apt-get install -y -qq nodejs > /dev/null 2>&1)" || {
        echo "  ✗ ERROR: Failed to install Node.js"
        exit 1
    }
    echo "    Node: $($SSH 'node --version 2>/dev/null')"
fi

echo "  → Installing npm packages..."
$SSH "cd ${REMOTE_DIR}/web && npm install 2>&1" | tail -3

echo "  → Building Vite frontend..."
$SSH "cd ${REMOTE_DIR}/web && npm run build 2>&1" | tail -5

echo "  → Copying built frontend into container..."
$SSH "${DOCKER_EXEC} mkdir -p ${REMOTE_DIR}/web/dist" || true
$SSH "docker cp ${REMOTE_DIR}/web/dist/. rocm:${REMOTE_DIR}/web/dist/" 2>&1
echo "  ✓ Frontend ready"

# ---- Step 7: Start backend inside container ----
echo ""
echo "[7/8] Starting server inside rocm container..."

# Kill anything on port 8000 inside the container
echo "  → Freeing port ${SERVER_PORT} inside container..."
roc "fuser -k ${SERVER_PORT}/tcp 2>/dev/null || true"
sleep 2

# Verify port is free
PORT_IN_USE=$(roc "ss -tlnp 'sport = :${SERVER_PORT}' 2>/dev/null || true")
if [ -n "${PORT_IN_USE}" ]; then
    echo "  ⚠ Port ${SERVER_PORT} is still in use:"
    echo "${PORT_IN_USE}" | sed 's/^/    /'
    echo "    The rocm container may have a Jupyter/Flask server on this port."
    echo "    Attempting force kill..."
    roc "kill \$(ss -tlnp 'sport = :${SERVER_PORT}' | grep -oP 'pid=\K[0-9]+') 2>/dev/null || true"
    sleep 2
fi

echo "  → Launching FastAPI server inside container..."
# Use setsid + nohup inside the container to keep it alive after SSH exits
roc "cd ${REMOTE_DIR} && Dronemd_GPU_BACKEND=rocm nohup .venv/bin/python backend/launch.py --host=0.0.0.0 --port=${SERVER_PORT} > /tmp/dronemd.log 2>&1 &" &
sleep 6

# Verify server started
if $SSH "${DOCKER_EXEC} curl -s http://localhost:${SERVER_PORT} > /dev/null 2>&1"; then
    echo "  ✓ Server running (HTTP 200)"
else
    echo "  ✗ Server did not respond."
    echo ""
    echo "  --- Last 30 lines of /tmp/dronemd.log ---"
    roc "tail -30 /tmp/dronemd.log 2>/dev/null || echo '(log file does not exist)'" | sed 's/^/  | /'
    echo "  --- End of log ---"
    echo ""
    echo "  Common causes:"
    echo "  1. Port ${SERVER_PORT} still in use inside the container"
    echo "  2. Missing dependencies (check log for ImportError)"
    echo "  3. JAX GPU init failure (check log for XLA errors)"
    echo "  4. Insufficient GPU memory (check log for OOM)"
    echo ""
    echo "  Manual debug:"
    echo "    ssh root@${DROPLET_IP}"
    echo "    docker exec -it rocm bash"
    echo "    cd ${REMOTE_DIR}"
    echo "    .venv/bin/python backend/launch.py --host=0.0.0.0 --port=${SERVER_PORT}"
fi

# ---- Step 8: Diagnostics ----
echo ""
echo "[8/8] Diagnostics..."

echo "  → Server process inside container:"
roc "ps aux | grep 'backend/launch.py' | grep -v grep" 2>&1 | sed 's/^/    /' || echo "    (no process found)"

echo "  → Memory (host):"
$SSH "free -h | head -2" 2>&1 | sed 's/^/    /'

echo "  → GPU memory inside container:"
roc "rocm-smi --showmemuse 2>/dev/null | grep -E '^[0-9]|GPU' | head -5" 2>&1 | sed 's/^/    /'

if $SSH "${DOCKER_EXEC} curl -s http://localhost:${SERVER_PORT} > /dev/null 2>&1"; then
    echo "  ✓ All checks passed"
else
    echo "  ⚠ Health check failed"
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
echo "  Logs (inside container):"
echo "    Server:  ssh root@${DROPLET_IP} 'docker exec rocm tail -f /tmp/dronemd.log'"
echo ""
echo "  Enter the container:"
echo "    ssh root@${DROPLET_IP}"
echo "    docker exec -it rocm bash"
echo "    cd ${REMOTE_DIR}"
echo "    source .venv/bin/activate"
echo ""
echo "  Redeploy after code changes:"
echo "    ./scripts/deploy.sh ${DROPLET_IP}"
echo ""
echo "  Destroy the droplet when done to save credits!"
echo "============================================"
