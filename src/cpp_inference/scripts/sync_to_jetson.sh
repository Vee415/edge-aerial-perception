#!/bin/bash
# Sync C++ inference source to Jetson and build
# Usage: ./scripts/sync_to_jetson.sh [build|clean]
set -e

JETSON_HOST="vee@192.168.55.1"
JETSON_DIR="\$HOME/drone_project/src/cpp_inference"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing C++ inference to Jetson ==="
rsync -avz --exclude='build' --exclude='.git' \
    "${LOCAL_DIR}/" \
    "${JETSON_HOST}:${JETSON_DIR}/"

echo "=== Done syncing ==="

if [ "$1" = "build" ]; then
    echo "=== Building on Jetson ==="
    ssh "${JETSON_HOST}" "cd ${JETSON_DIR} && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4"
    echo "=== Build complete ==="
elif [ "$1" = "clean" ]; then
    echo "=== Cleaning build on Jetson ==="
    ssh "${JETSON_HOST}" "cd ${JETSON_DIR} && rm -rf build"
    echo "=== Clean complete ==="
fi