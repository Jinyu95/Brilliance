#!/usr/bin/env bash
# Pull the Python base image from a mirror and tag it locally.
# Run this ONCE before "docker compose up --build".
#
# Usage:  bash scripts/pull_base.sh

# Ensure the JuTrack.jl submodule is populated
if [ ! -f "JuTrack.jl/Project.toml" ]; then
    echo "Initialising JuTrack.jl submodule..."
    git submodule update --init --recursive || {
        echo "ERROR: git submodule update failed. Clone with --recurse-submodules."
        exit 1
    }
fi

IMAGE="python:3.11-slim"
MIRRORS=(
    "docker.m.daocloud.io/library/python:3.11-slim"
    "docker.nju.edu.cn/library/python:3.11-slim"
    "docker.mirrors.sjtug.sjtu.edu.cn/library/python:3.11-slim"
    "dockerhub.icu/library/python:3.11-slim"
)

for src in "${MIRRORS[@]}"; do
    echo "Trying $src ..."
    if docker pull "$src" 2>/dev/null; then
        docker tag "$src" "$IMAGE"
        echo "OK: tagged $src as $IMAGE"
        echo ""
        echo "Now run:  docker compose up --build"
        exit 0
    fi
    echo "  failed, trying next mirror..."
done

echo ""
echo "ERROR: all mirrors failed."
echo "Configure Docker Desktop manually:"
echo "  Settings -> Docker Engine -> add to JSON:"
echo '  "registry-mirrors": ["https://docker.m.daocloud.io"]'
exit 1
