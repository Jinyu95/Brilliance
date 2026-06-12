# Pull the Python base image from a mirror and tag it locally.
# Run this ONCE before "docker compose up --build".
#
# Usage:  .\scripts\pull_base.ps1

# Ensure the JuTrack.jl submodule is populated
if (-not (Test-Path "JuTrack.jl\Project.toml")) {
    Write-Host "Initialising JuTrack.jl submodule..."
    git submodule update --init --recursive
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: git submodule update failed. Make sure you cloned with --recurse-submodules."
        exit 1
    }
}

$IMAGE = "python:3.11-slim"
$MIRRORS = @(
    "docker.m.daocloud.io/library/python:3.11-slim",
    "docker.nju.edu.cn/library/python:3.11-slim",
    "docker.mirrors.sjtug.sjtu.edu.cn/library/python:3.11-slim",
    "dockerhub.icu/library/python:3.11-slim"
)

foreach ($src in $MIRRORS) {
    Write-Host "Trying $src ..."
    docker pull $src 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        docker tag $src $IMAGE
        Write-Host "OK: tagged $src as $IMAGE"
        Write-Host ""
        Write-Host "Now run:  docker compose up --build"
        exit 0
    }
    Write-Host "  failed, trying next mirror..."
}

Write-Host ""
Write-Host "ERROR: all mirrors failed."
Write-Host "Configure Docker Desktop manually:"
Write-Host "  Settings -> Docker Engine -> add to JSON:"
Write-Host '  "registry-mirrors": ["https://docker.m.daocloud.io"]'
exit 1
