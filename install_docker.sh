#!/usr/bin/env bash
# install_docker.sh — Instala Docker CE + Compose plugin en Ubuntu 20.04+
# Uso: bash install_docker.sh
set -euo pipefail

echo "=== [1/5] Actualizando apt e instalando dependencias ==="
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg

echo "=== [2/5] Añadiendo GPG key oficial de Docker ==="
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "=== [3/5] Añadiendo repositorio oficial de Docker ==="
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "=== [4/5] Instalando Docker CE + Compose plugin ==="
sudo apt-get update -y
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

echo "=== [5/5] Añadiendo $USER al grupo docker ==="
sudo usermod -aG docker "$USER"

echo ""
echo "============================================"
docker --version
docker compose version
echo "============================================"
echo ""
echo "DONE. Cierra y vuelve a abrir la sesión SSH"
echo "para que el grupo 'docker' tenga efecto."
echo "O ejecuta:  newgrp docker"
