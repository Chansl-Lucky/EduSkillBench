#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
elif sudo -n true 2>/dev/null; then
  SUDO="sudo -n"
else
  echo "This one-time host setup needs sudo. Run it interactively from your shell:" >&2
  echo "  code/evaluation/install_docker_ubuntu.sh" >&2
  exit 1
fi

$SUDO apt-get update
$SUDO apt-get install -y docker.io docker-compose-v2
$SUDO systemctl enable --now docker
$SUDO usermod -aG docker "${SUDO_USER:-$USER}"

echo "Docker installed. Log out/in once so the new docker group is active."
echo "Then verify with: docker info"
