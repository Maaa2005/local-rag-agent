#!/bin/bash
set -e

COMPOSE_VERSION="v2.36.1"
DEST="/usr/local/lib/docker/cli-plugins/docker-compose"

echo "=== Creating plugin directory ==="
sudo mkdir -p /usr/local/lib/docker/cli-plugins

echo "=== Downloading Docker Compose ${COMPOSE_VERSION} ==="
sudo curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" -o "$DEST"

echo "=== Making executable ==="
sudo chmod +x "$DEST"

echo "=== Verifying ==="
docker compose version
