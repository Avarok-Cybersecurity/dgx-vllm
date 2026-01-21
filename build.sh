#!/usr/bin/env bash
set -Eeuo pipefail

# Build script for dgx-vllm Docker image v15
# Features: CUTLASS support, vLLM main-latest, PyTorch nightly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="dgx-vllm"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_VERSION="${IMAGE_VERSION:-15}"
REMOTE_NODE="${REMOTE_NODE:-10.10.10.2}"
REMOTE_USER="${REMOTE_USER:-nologik}"

echo "=== Building dgx-vllm Docker Image v${IMAGE_VERSION} ==="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Features: CUTLASS support, NVFP4 quantization"
echo "Build time: 30-60 minutes"
echo "Directory: ${SCRIPT_DIR}"
echo ""

# Show version info
echo "Component Versions:"
echo "  vLLM: main branch (latest)"
echo "  PyTorch: nightly (CUDA 13.0)"
echo "  FlashInfer: latest pre-release"
echo "  XGrammar: latest stable"
echo "  CUTLASS: enabled (FP4/FP6/FP8)"
echo ""

# Build on local node
echo "Building on local node..."
cd "${SCRIPT_DIR}"

# Log build start time
START_TIME=$(date +%s)

docker build \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -t "${IMAGE_NAME}:v${IMAGE_VERSION}" \
  -t "${IMAGE_NAME}:cutlass" \
  . \
  --progress=plain \
  2>&1 | tee "build-v${IMAGE_VERSION}.log"

BUILD_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))
BUILD_TIME_MIN=$((BUILD_TIME / 60))
BUILD_TIME_SEC=$((BUILD_TIME % 60))

if [ $BUILD_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "✓ Local build successful!"
  echo "Build time: ${BUILD_TIME_MIN}m ${BUILD_TIME_SEC}s"
  echo ""
  docker images | grep "${IMAGE_NAME}"
  echo ""
  echo "Tagged as:"
  echo "  - ${IMAGE_NAME}:latest"
  echo "  - ${IMAGE_NAME}:v${IMAGE_VERSION}"
  echo "  - ${IMAGE_NAME}:cutlass"
else
  echo "✗ Local build failed! Check build-v${IMAGE_VERSION}.log for details."
  exit 1
fi

# Ask if user wants to build on remote node
echo ""
read -p "Build on remote node (${REMOTE_NODE})? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Building on remote node ${REMOTE_NODE}..."

  # Copy build context to remote node
  echo "Copying build context to ${REMOTE_USER}@${REMOTE_NODE}..."
  ssh "${REMOTE_USER}@${REMOTE_NODE}" "mkdir -p /tmp/dgx-vllm-build"
  rsync -avz --progress \
    "${SCRIPT_DIR}/" \
    "${REMOTE_USER}@${REMOTE_NODE}:/tmp/dgx-vllm-build/"

  # Build on remote node
  echo "Running docker build on remote node..."
  ssh "${REMOTE_USER}@${REMOTE_NODE}" \
    "cd /tmp/dgx-vllm-build && docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:v${IMAGE_VERSION} -t ${IMAGE_NAME}:cutlass ." \
    2>&1 | tee "build-v${IMAGE_VERSION}-remote.log"

  if [ $? -eq 0 ]; then
    echo "✓ Remote build successful!"
  else
    echo "✗ Remote build failed! Check build-v${IMAGE_VERSION}-remote.log for details."
    exit 1
  fi

  # Clean up remote build context
  echo "Cleaning up remote build context..."
  ssh "${REMOTE_USER}@${REMOTE_NODE}" "rm -rf /tmp/dgx-vllm-build"
fi

echo ""
echo "=== Build Complete (Version ${IMAGE_VERSION}) ==="
echo "Next steps:"
echo "  1. Test the image locally"
echo "  2. Verify CUTLASS support"
echo "  3. Push to Docker Hub (see push.sh)"
