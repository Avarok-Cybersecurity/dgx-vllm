#!/usr/bin/env bash
set -Eeuo pipefail

# Stop script for dgx-vllm multi-node cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_NODE="${REMOTE_NODE:-10.10.10.2}"
REMOTE_USER="${REMOTE_USER:-nologik}"
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-/tmp/dgx-vllm-build}"

echo "=== Stopping DGX vLLM Cluster ==="

# Stop containers on head node
echo "Stopping containers on head node..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" down

# Stop containers on worker node
echo "Stopping containers on worker node ${REMOTE_NODE}..."
ssh "${REMOTE_USER}@${REMOTE_NODE}" \
  "cd ${REMOTE_BUILD_DIR} && docker-compose -f docker-compose.worker.yml down 2>/dev/null || true"

echo ""
echo "=== Cluster Stopped ==="
