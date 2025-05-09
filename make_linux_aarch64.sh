#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Docker image name to build/use
DOCKER_IMAGE_NAME="duckdb-lm-diskann-builder-linux-arm64-local"
# Path to the new local Dockerfile relative to the project root
DOCKERFILE_PATH="./scripts/Dockerfile.local_linux_aarch64"
# Docker build context is now the project root, as the Dockerfile is there
DOCKER_BUILD_CONTEXT="."

# vcpkg arguments for docker build (these are important!)
# You should use the same vcpkg commit your CI uses for max reproducibility
VCPKG_URL_ARG="https://github.com/microsoft/vcpkg.git"
# Replace "main" with the specific vcpkg commit SHA used by your CI if known
# Using "master" as it's the default branch for the vcpkg repository
VCPKG_COMMIT_ARG="master"

# Extra toolchains (if your Dockerfile uses this ARG and you need specific ones)
# Example: ";python3;rust;" - must start and end with ';' if not empty.
# The linux_arm64/Dockerfile uses this to conditionally install python3, fortran, etc.
EXTRA_TOOLCHAINS_ARG=";python3;" # Ensure Python3 is installed as per Dockerfile logic

# --- Script Logic ---

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker does not seem to be running, please start it and try again."
  exit 1
fi

echo ">>> Building Docker image: ${DOCKER_IMAGE_NAME}..."
echo ">>> Using Dockerfile: ${DOCKERFILE_PATH}"
echo ">>> Build context: ${DOCKER_BUILD_CONTEXT}"

# Build the Docker image
# The --platform linux/arm64 might be redundant if your Docker Desktop is already
# configured for ARM64 and the base image is ARM64, but explicit can be safer.
docker build \
    --platform linux/arm64 \
    --build-arg "vcpkg_url=${VCPKG_URL_ARG}" \
    --build-arg "vcpkg_commit=${VCPKG_COMMIT_ARG}" \
    --build-arg "extra_toolchains=${EXTRA_TOOLCHAINS_ARG}" \
    -f "${DOCKERFILE_PATH}" \
    -t "${DOCKER_IMAGE_NAME}" \
    "${DOCKER_BUILD_CONTEXT}"

echo ""
echo ">>> Docker image built."
echo ">>> Running build inside Docker container..."

# The WORKDIR in your CI Dockerfile is /duckdb_build_dir
# We mount the current project directory to /duckdb_build_dir inside the container
# We also mount a local directory for ccache persistence.
# The CI Dockerfile sets ENV CCACHE_DIR=/ccache_dir
PROJECT_ROOT_DIR="$(pwd)"
CCACHE_DIR_HOST="${PROJECT_ROOT_DIR}/.cache/docker_linux_arm64" # Specific to this build type

mkdir -p "${CCACHE_DIR_HOST}" # Ensure host ccache directory exists

# Define the specific toolchain flags for ARM64 GCC, mirroring CI
# Note: Your CI also includes -DCMAKE_Fortran_COMPILER. Added it for completeness,
# remove if your project doesn't use Fortran.
ARM64_GCC_TOOLCHAIN_FLAGS="-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_Fortran_COMPILER=aarch64-linux-gnu-gfortran"

# Add ccache launchers to the toolchain flags for the Docker build
# Also explicitly add -G Ninja to ensure it's used, as CI sets ENV GEN=ninja
TOOLCHAIN_FLAGS_WITH_CCACHE_AND_NINJA="${ARM64_GCC_TOOLCHAIN_FLAGS} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -G Ninja"

docker run \
    --rm \
    --platform linux/arm64 \
    -v "${PROJECT_ROOT_DIR}:/duckdb_build_dir" \
    -v "${CCACHE_DIR_HOST}:/ccache_dir" \
    -e "TOOLCHAIN_FLAGS=${TOOLCHAIN_FLAGS_WITH_CCACHE_AND_NINJA}" \
    -w "/duckdb_build_dir" \
    "${DOCKER_IMAGE_NAME}" \
    bash -c "echo 'User: $(whoami)' && \
             echo 'CCACHE_DIR: ${CCACHE_DIR}' && \
             echo 'TOOLCHAIN_FLAGS: ${TOOLCHAIN_FLAGS}' && \
             ls -la /ccache_dir && \
             make clean && \
             make" # The Makefile should pick up TOOLCHAIN_FLAGS

echo ""
echo ">>> Docker build finished."
