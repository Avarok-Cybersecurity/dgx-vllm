#!/usr/bin/env bash
set -Eeuo pipefail

# Build script for dgx-vllm Docker image.
# All versions and config are read from .env — edit that file to change anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env as the single source of truth
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
else
  echo "ERROR: .env file not found in ${SCRIPT_DIR}"
  echo "Copy .env.example to .env and configure versions."
  exit 1
fi

# Allow CLI overrides (e.g., IMAGE_VERSION=23 ./build.sh)
IMAGE_NAME="${IMAGE_NAME:-dgx-vllm}"
IMAGE_VERSION="${IMAGE_VERSION:-22}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== Building dgx-vllm Docker Image v${IMAGE_VERSION} ==="
echo ""
echo "Configuration (from .env):"
echo "  Base image:   ${BASE_IMAGE}"
echo "  vLLM commit:  ${VLLM_COMMIT}"
echo "  PyTorch:      ${TORCH_VERSION}"
echo "  Image tag:    ${IMAGE_NAME}:v${IMAGE_VERSION}"
echo ""

cd "${SCRIPT_DIR}"

START_TIME=$(date +%s)

docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg IMAGE_VERSION="${IMAGE_VERSION}" \
  --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
  --build-arg TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -t "${IMAGE_NAME}:v${IMAGE_VERSION}" \
  . \
  --progress=plain \
  2>&1 | tee "build-v${IMAGE_VERSION}.log"

BUILD_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

if [ $BUILD_EXIT_CODE -eq 0 ]; then
  echo ""
  echo "Build successful! (${BUILD_TIME}s)"
  echo ""
  echo "Tagged as:"
  echo "  - ${IMAGE_NAME}:${IMAGE_TAG}"
  echo "  - ${IMAGE_NAME}:v${IMAGE_VERSION}"
  echo ""
  echo "Next: ./push.sh"
else
  echo "Build failed. See build-v${IMAGE_VERSION}.log"
  exit 1
fi
