#!/usr/bin/env bash
set -euo pipefail

# install_rocm_jax.sh — Install AMD ROCm JAX from repo.radeon.com
#
# The AMD ROCm JAX packages are hosted on AMD's package repo, NOT on PyPI.
# This script installs them in the correct order (required by AMD):
#   1. pjrt wheel
#   2. plugin wheel
#   3. jaxlib
#   4. jax
#
# Prerequisites:
#   - ROCm 7.2+ installed on the system (verify with: rocm-smi)
#   - Python 3.12 virtual environment activated
#   - Existing JAX packages uninstalled first (script handles this)
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/install_rocm_jax.sh

ROCM_REPO="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
PJRT_URL="${ROCM_REPO}/jax_rocm7_pjrt-0.8.2%2Brocm7.2.1-py3-none-manylinux_2_28_x86_64.whl"
PLUGIN_URL="${ROCM_REPO}/jax_rocm7_plugin-0.8.2%2Brocm7.2.1-cp312-cp312-manylinux_2_28_x86_64.whl"

echo "=== Uninstalling existing JAX packages ==="
pip uninstall -y jax-rocm7-pjrt jax-rocm7-plugin jaxlib jax 2>/dev/null || true
pip uninstall -y jax-cuda12-pjrt jax-cuda12-plugin 2>/dev/null || true

echo ""
echo "=== Installing pjrt wheel ==="
pip install "${PJRT_URL}"

echo ""
echo "=== Installing plugin wheel ==="
pip install "${PLUGIN_URL}"

echo ""
echo "=== Installing jaxlib ==="
pip install jaxlib==0.8.2

echo ""
echo "=== Installing jax ==="
pip install jax==0.8.2

echo ""
echo "=== Verifying ROCm JAX ==="
python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices('gpu')
print(f'GPU devices: {len(devices)}')
for d in devices:
    print(f'  {d.device_kind} (platform={d.platform})')
print()
print('ROCm JAX is ready.')
"

echo ""
echo "=== Done ==="
echo "Set Dronemd_GPU_BACKEND=rocm to force ROCm (auto-detected by default)."
echo "Start the server with: make api"
