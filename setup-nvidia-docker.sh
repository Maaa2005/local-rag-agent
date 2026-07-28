#!/bin/bash
set -e

echo "=== Step 1: NVIDIA GPG key ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "=== Step 2: Add repository ==="
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "=== Step 3: Install ==="
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

echo "=== Step 4: Configure Docker runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker

echo "=== Step 5: Restart Docker ==="
sudo service docker restart

echo "=== Done! Testing GPU in Docker ==="
sudo docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
