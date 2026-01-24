#!/usr/bin/env bash
set -Eeuo pipefail

# Docker Hub push script for dgx-vllm v75
# Target repository: avarok/vllm-dgx-spark

IMAGE_NAME="dgx-vllm"
VERSION="75"
DOCKER_HUB_REPO="avarok/vllm-dgx-spark"
BUILD_DATE=$(date +%Y-%m-%d)

echo "=== Docker Hub Push Script ==="
echo "Source image: ${IMAGE_NAME}:latest"
echo "Target repo: ${DOCKER_HUB_REPO}"
echo "Version: v${VERSION}"
echo ""

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}:latest" >/dev/null 2>&1; then
  echo "ERROR: Image ${IMAGE_NAME}:latest not found"
  echo "Build the image first with: ./build.sh"
  exit 1
fi

# Login to Docker Hub
echo "Step 1: Docker Hub Authentication"
echo "Enter your Docker Hub credentials:"
docker login

if [ $? -ne 0 ]; then
  echo "ERROR: Docker Hub login failed"
  exit 1
fi

echo ""
echo "Step 2: Tagging images"
echo "Creating tags:"
echo "  - ${DOCKER_HUB_REPO}:latest"
echo "  - ${DOCKER_HUB_REPO}:v${VERSION}"
echo "  - ${DOCKER_HUB_REPO}:${VERSION}"
echo "  - ${DOCKER_HUB_REPO}:cutlass"
echo "  - ${DOCKER_HUB_REPO}:nvfp4"
echo "  - ${DOCKER_HUB_REPO}:${BUILD_DATE}"
echo ""

docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:latest"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:v${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:cutlass"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:nvfp4"
docker tag "${IMAGE_NAME}:latest" "${DOCKER_HUB_REPO}:${BUILD_DATE}"

echo "Step 3: Pushing to Docker Hub"
echo "This may take 10-30 minutes depending on network speed..."
echo ""

# Push all tags
docker push "${DOCKER_HUB_REPO}:latest"
docker push "${DOCKER_HUB_REPO}:v${VERSION}"
docker push "${DOCKER_HUB_REPO}:${VERSION}"
docker push "${DOCKER_HUB_REPO}:cutlass"
docker push "${DOCKER_HUB_REPO}:nvfp4"
docker push "${DOCKER_HUB_REPO}:${BUILD_DATE}"

if [ $? -eq 0 ]; then
  echo ""
  echo "=== Push Successful ==="
  echo ""
  echo "Images published to Docker Hub:"
  echo "  - ${DOCKER_HUB_REPO}:latest (recommended for most users)"
  echo "  - ${DOCKER_HUB_REPO}:v${VERSION} (version-pinned)"
  echo "  - ${DOCKER_HUB_REPO}:${VERSION} (short version)"
  echo "  - ${DOCKER_HUB_REPO}:cutlass (feature tag)"
  echo "  - ${DOCKER_HUB_REPO}:nvfp4 (NVFP4 complete integration)"
  echo "  - ${DOCKER_HUB_REPO}:${BUILD_DATE} (date-stamped)"
  echo ""
  echo "Pull command:"
  echo "  docker pull ${DOCKER_HUB_REPO}:latest"
  echo ""
  echo "Usage example:"
  echo "  docker run --rm --gpus all --network host \\"
  echo "    -e MODEL=your-model-id \\"
  echo "    -v \$HOME/.cache/huggingface:/root/.cache/huggingface \\"
  echo "    ${DOCKER_HUB_REPO}:latest serve"
  echo ""
else
  echo ""
  echo "=== Push Failed ==="
  echo "Check your network connection and Docker Hub credentials"
  exit 1
fi
