#!/usr/bin/env bash
# =============================================================================
# PetRand: Raspberry Pi Automated Appliance Deployment Script
# Tested on Raspberry Pi OS 64-bit (Debian Bookworm) & Ubuntu 22.04/24.04 LTS
# =============================================================================
set -euo pipefail

echo "================================================================="
echo " PetRand TRNG: Single-Board Appliance Installer"
echo "================================================================="

INSTALL_DIR="${HOME}/petrand"
VENV_DIR="${INSTALL_DIR}/venv"

# 1. Update package lists and install system packages
echo "[*] Updating apt repository and installing hardware dependencies..."
sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    v4l-utils \
    libportaudio2 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    curl

# 2. Add current user to video and audio groups
echo "[*] Configuring user permissions for camera (/dev/video0) and sound..."
sudo usermod -aG video,audio "${USER}"

# 3. Create Python virtual environment (PEP 668 compliant)
echo "[*] Setting up isolated Python virtual environment at ${VENV_DIR}..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

# 4. Install Python dependencies
echo "[*] Installing PetRand core library, NumPy, SciPy, and OpenCV..."
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install numpy scipy opencv-python-headless sounddevice

# If repository is in current directory, install editable package
if [ -f "pyproject.toml" ]; then
    "${VENV_DIR}/bin/pip" install -e .
fi

# 5. Run Power-On Self-Test (POST) to verify cryptographic & health algorithms
echo "[*] Running power-on known-answer self-tests..."
"${VENV_DIR}/bin/python3" -m petrand.cli self-test

# 6. Install and enable systemd daemon
SERVICE_FILE="/etc/systemd/system/petrand.service"
if [ -f "deploy/petrand.service" ]; then
    echo "[*] Installing systemd service unit to ${SERVICE_FILE}..."
    sudo cp deploy/petrand.service "${SERVICE_FILE}"
    # Replace default user and paths with actual user and directory
    sudo sed -i "s|User=pi|User=${USER}|g" "${SERVICE_FILE}"
    sudo sed -i "s|Group=pi|Group=$(id -gn)|g" "${SERVICE_FILE}"
    sudo sed -i "s|/home/pi/petrand|${INSTALL_DIR}|g" "${SERVICE_FILE}"
    
    sudo systemctl daemon-reload
    sudo systemctl enable petrand.service
    echo "[+] PetRand systemd service installed and enabled!"
    echo "    Start service with: sudo systemctl start petrand"
    echo "    Check logs with:    sudo journalctl -u petrand -f"
fi

echo "================================================================="
echo " PetRand Installation Complete!"
echo " Web Dashboard: http://127.0.0.1:8080/ on the Pi (from another computer: ssh -L 8080:127.0.0.1:8080 pi@<pi-address>)"
echo "================================================================="
