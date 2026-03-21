#!/usr/bin/env bash
# setup.sh — one-shot install for hypha-whisper-node on Jetson devices
# Supports: Jetson Orin Nano, Jetson AGX Orin (64GB), JetPack 6.x, CUDA 12.x, Python 3.10
#
# Usage:
#   chmod +x setup.sh
#   sudo ./setup.sh          # full install (system packages + Python deps)
#   ./setup.sh --no-sudo     # Python-only (skip apt steps; system pkgs must be pre-installed)
#
# Supported hardware:
#   - Microphone: ReSpeaker 4 Mic Array v2.0 (USB)
#   - Speaker: Dell AC511 USB SoundBar OR HDMI/DisplayPort monitor audio
#
# After install, configure secrets:
#   sudo mkdir -p /etc/hypha-whisper
#   sudo tee /etc/hypha-whisper/config.env <<'EOF'
#   HYPHA_SERVER=https://hypha.aicell.io/
#   HYPHA_WORKSPACE=my-workspace
#   HYPHA_WORKSPACE_TOKEN=my-token
#   EOF
#   sudo chmod 600 /etc/hypha-whisper/config.env
#
# To install systemd service:
#   sudo ./setup.sh --install-service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
CUSPARSELT_DEB="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/libcusparselt0_0.6.3.2-1_arm64.deb"

NO_SUDO=false
INSTALL_SERVICE=false
for arg in "$@"; do
  [[ "$arg" == "--no-sudo" ]] && NO_SUDO=true
  [[ "$arg" == "--install-service" ]] && INSTALL_SERVICE=true
done

info()  { echo "[setup] $*"; }
fatal() { echo "[setup] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1 — System packages
# ---------------------------------------------------------------------------
if ! $NO_SUDO; then
  [[ "$EUID" -eq 0 ]] || fatal "Run with sudo (or pass --no-sudo to skip apt steps)"
  info "Installing system packages..."
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    portaudio19-dev \
    ffmpeg \
    libsndfile1
  info "System packages installed."
fi

# ---------------------------------------------------------------------------
# 2 — pip (user-level)
# ---------------------------------------------------------------------------
if ! python3 -m pip --version &>/dev/null; then
  info "Bootstrapping pip..."
  curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 - --user
fi

# ---------------------------------------------------------------------------
# 3 — libcusparseLt (missing on JP6.2)
# ---------------------------------------------------------------------------
CUSPARSELT_SO="$HOME/.local/lib/libcusparseLt.so.0"
if [[ ! -f "$CUSPARSELT_SO" ]]; then
  info "Installing libcusparseLt (missing on JP6.2)..."
  TMP=$(mktemp -d)
  curl -fsSL "$CUSPARSELT_DEB" -o "$TMP/cusparselt.deb"
  (cd "$TMP" && ar x cusparselt.deb && tar -xf data.tar.xz)
  mkdir -p "$HOME/.local/lib"
  cp "$TMP"/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0* "$HOME/.local/lib/"
  rm -rf "$TMP"
  info "libcusparseLt installed to ~/.local/lib/"
fi

# Ensure LD_LIBRARY_PATH is set in ~/.bashrc
LD_LINE='export LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
if ! grep -qF 'libcusparseLt' "$HOME/.bashrc" 2>/dev/null; then
  echo "$LD_LINE" >> "$HOME/.bashrc"
  info "Added LD_LIBRARY_PATH to ~/.bashrc"
fi
export LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# 4 — PyTorch (NVIDIA JetPack 6.1 wheel — works on CUDA 12.6 / JP6.2)
# ---------------------------------------------------------------------------
if ! python3 -c "import torch" &>/dev/null; then
  info "Installing PyTorch from NVIDIA wheel (may take a few minutes)..."
  python3 -m pip install --user "$PYTORCH_WHEEL"
  info "PyTorch installed."
fi

# ---------------------------------------------------------------------------
# 5 — Python dependencies
# ---------------------------------------------------------------------------
info "Installing Python dependencies from requirements.txt..."
python3 -m pip install --user -r "$SCRIPT_DIR/requirements.txt"
info "Python dependencies installed."

# ---------------------------------------------------------------------------
# 6 — Secrets directory (create if missing)
# ---------------------------------------------------------------------------
if $NO_SUDO; then
  info "Skipping /etc/hypha-whisper setup (--no-sudo mode)"
else
  if [[ ! -f /etc/hypha-whisper/config.env ]]; then
    mkdir -p /etc/hypha-whisper
    cat > /etc/hypha-whisper/config.env <<'ENVEOF'
# Hypha Whisper Node configuration
# Edit this file, then restart: systemctl restart hypha-whisper
HYPHA_SERVER=https://hypha.aicell.io/
HYPHA_WORKSPACE=
HYPHA_WORKSPACE_TOKEN=
ENVEOF
    chmod 600 /etc/hypha-whisper/config.env
    info "Created /etc/hypha-whisper/config.env — fill in your credentials"
  else
    info "/etc/hypha-whisper/config.env already exists, skipping"
  fi
fi

# ---------------------------------------------------------------------------
# 7 — Install systemd service (optional)
# ---------------------------------------------------------------------------
if $INSTALL_SERVICE; then
  if [[ "$EUID" -ne 0 ]]; then
    fatal "--install-service requires sudo"
  fi
  info "Installing systemd services..."
  
  # Get current user (the one who ran sudo)
  SERVICE_USER="${SUDO_USER:-$USER}"
  SERVICE_HOME="$(eval echo ~$SERVICE_USER)"
  WORKING_DIR="$SCRIPT_DIR"
  
  # Process service template files
  for service in hypha-whisper.service hypha-whisper-watchdog.service; do
    if [[ -f "$SCRIPT_DIR/deploy/$service" ]]; then
      info "Processing $service..."
      sed -e "s|%USER%|$SERVICE_USER|g" \
          -e "s|%HOME%|$SERVICE_HOME|g" \
          -e "s|%WORKING_DIR%|$WORKING_DIR|g" \
          "$SCRIPT_DIR/deploy/$service" > "/etc/systemd/system/$service"
      info "Installed /etc/systemd/system/$service"
    fi
  done
  
  systemctl daemon-reload
  info "Systemd services installed."
  info ""
  info "To start the service:"
  info "  sudo systemctl enable --now hypha-whisper"
  info ""
  info "To check status:"
  info "  journalctl -u hypha-whisper -f"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
info "Setup complete."
info ""
info "Next steps:"
info "  1. Edit /etc/hypha-whisper/config.env with your Hypha credentials"
if ! $INSTALL_SERVICE; then
  info "  2. Install the systemd service:"
  info "       sudo ./setup.sh --install-service"
  info "  3. Check status: journalctl -u hypha-whisper -f"
else
  info "  2. Start the service: sudo systemctl enable --now hypha-whisper"
  info "  3. Check status: journalctl -u hypha-whisper -f"
fi
