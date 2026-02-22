#!/usr/bin/env bash
set -Eeuo pipefail

# Push script for dgx-vllm Docker image.
# All config is read from .env — edit that file to change repo/version.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env as the single source of truth
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
else
  echo "ERROR: .env file not found in ${SCRIPT_DIR}"
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-dgx-vllm}"
IMAGE_VERSION="${IMAGE_VERSION:-22}"
DOCKER_HUB_REPO="${DOCKER_HUB_REPO:-avarok/dgx-vllm-nvfp4-kernel}"
BUILD_DATE=$(date +%Y-%m-%d)

echo "=== Docker Hub Push ==="
echo "  Source: ${IMAGE_NAME}:latest"
echo "  Target: ${DOCKER_HUB_REPO}"
echo "  Version: v${IMAGE_VERSION}"
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1; then
  echo "ERROR: ${IMAGE_NAME}:latest not found. Run ./build.sh first."
  exit 1
fi

# Login
echo "Authenticating with Docker Hub..."
docker login
echo ""

# Tag
echo "Tagging..."
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:latest"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:v${IMAGE_VERSION}"

# Push
echo "Pushing (this may take 10-30 minutes)..."
docker push "${DOCKER_HUB_REPO}:latest"
docker push "${DOCKER_HUB_REPO}:v${IMAGE_VERSION}"

echo ""
echo "Published:"
echo "  - ${DOCKER_HUB_REPO}:latest"
echo "  - ${DOCKER_HUB_REPO}:v${IMAGE_VERSION}"
echo ""
echo "Pull: docker pull ${DOCKER_HUB_REPO}:v${IMAGE_VERSION}"
