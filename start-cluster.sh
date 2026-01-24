#!/usr/bin/env bash
set -Eeuo pipefail

# Start script for dgx-vllm multi-node cluster
# This script starts Ray head + vLLM server on head node and Ray worker on remote node

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_NODE="${REMOTE_NODE:-10.10.10.2}"
REMOTE_USER="${REMOTE_USER:-nologik}"
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-/tmp/dgx-vllm-build}"

echo "=== Starting DGX vLLM Cluster ==="
echo "Head Node: 10.10.10.1 (local)"
echo "Worker Node: ${REMOTE_NODE}"

# Stop any existing containers
echo ""
echo "Stopping existing containers..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" down 2>/dev/null || true

# Start Ray head node
echo ""
echo "Starting Ray head node..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d ray-head

echo "Waiting for Ray head to initialize (10 seconds)..."
sleep 10

# Start Ray worker on remote node
echo ""
echo "Starting Ray worker on remote node ${REMOTE_NODE}..."

# Copy docker-compose.worker.yml to remote if needed
ssh "${REMOTE_USER}@${REMOTE_NODE}" "mkdir -p ${REMOTE_BUILD_DIR}"
scp "${SCRIPT_DIR}/docker-compose.worker.yml" \
  "${REMOTE_USER}@${REMOTE_NODE}:${REMOTE_BUILD_DIR}/"

# Start worker container
ssh "${REMOTE_USER}@${REMOTE_NODE}" \
  "cd ${REMOTE_BUILD_DIR} && docker-compose -f docker-compose.worker.yml down 2>/dev/null || true && docker-compose -f docker-compose.worker.yml up -d"

echo "Waiting for Ray worker to connect (10 seconds)..."
sleep 10

# Check Ray cluster status
echo ""
echo "Checking Ray cluster status..."
docker exec dgx-vllm-head ray status 2>/dev/null || echo "Warning: Could not get Ray status"

# Start vLLM server
echo ""
echo "Starting vLLM server with TP=2..."
docker-compose -f "${SCRIPT_DIR}/docker-compose.yml" up -d vllm-server

echo ""
echo "=== Cluster Started ==="
echo ""
echo "Monitor logs:"
echo "  Head node: docker logs -f dgx-vllm-head"
echo "  Worker node: ssh ${REMOTE_USER}@${REMOTE_NODE} 'docker logs -f dgx-vllm-worker'"
echo "  vLLM server: docker logs -f dgx-vllm-server"
echo ""
echo "Check health:"
echo "  curl http://10.10.10.1:8888/health"
echo ""
echo "Stop cluster:"
echo "  ./stop-cluster.sh"
