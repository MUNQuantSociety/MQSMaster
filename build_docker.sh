#!/usr/bin/env bash
# Exit immediately if a command fails, if an undefined variable is used, or if a pipe fails.
set -euo pipefail

# [Task] Script for Docker Image: automate docker build -t trading-bot:latest .
# Image tag to use (default is trading-bot:latest if you don't pass one).
IMAGE_TAG="${1:-trading-bot:latest}"
# Find the folder where this script lives (the repo root).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The Dockerfile should be in the repo root.
DOCKERFILE="$ROOT_DIR/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
  # If Dockerfile is missing, we cannot build an image.
  echo "Dockerfile not found at $DOCKERFILE"
  echo "Add a Dockerfile at the repo root, then re-run this script."
  exit 1
fi

# Build the Docker image from the repo root and apply the tag.
docker build -t "$IMAGE_TAG" "$ROOT_DIR"
