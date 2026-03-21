#!/usr/bin/env bash
# scripts/run_hardware_tests.sh
#
# Run hardware tests (loopback, stress) safely:
#   1. Stop hypha-whisper service if running
#   2. Run pytest with @hardware marker
#   3. Restart the service if it was running before
#
# Usage:
#   ./scripts/run_hardware_tests.sh                  # all hardware tests
#   ./scripts/run_hardware_tests.sh -k loopback      # filter by name
#   ./scripts/run_hardware_tests.sh -k "wer or rms"  # multiple filters
#
# Prerequisites:
#   - Speaker: Dell AC511 USB SoundBar OR HDMI/DisplayPort monitor speakers
#   - ReSpeaker 4 Mic Array plugged in
#   - tests/fixtures/darkness.mp3 present (auto-downloaded if missing)
#   - sudo access to systemctl (or run as root)

set -euo pipefail
cd "$(dirname "$0")/.."

AUDIO_FILE="tests/fixtures/darkness.mp3"
AUDIO_URL="http://www.moviesoundclips.net/movies1/darkknightrises/darkness.mp3"

# ── Download test audio if missing ─────────────────────────────────────────
if [ ! -f "$AUDIO_FILE" ]; then
    echo "[setup] Downloading test audio to $AUDIO_FILE ..."
    mkdir -p "$(dirname "$AUDIO_FILE")"
    wget -q --show-progress -O "$AUDIO_FILE" "$AUDIO_URL"
    echo "[setup] Download complete."
fi

# ── Service management ──────────────────────────────────────────────────────
SERVICE_WAS_RUNNING=false
if systemctl is-active --quiet hypha-whisper 2>/dev/null || \
   systemctl is-active --quiet hypha-whisper-watchdog 2>/dev/null; then
    SERVICE_WAS_RUNNING=true
    # Check if we can sudo without a password
    if sudo -n systemctl stop hypha-whisper &>/dev/null && \
       sudo -n systemctl stop hypha-whisper-watchdog &>/dev/null; then
        echo "[setup] Stopping hypha-whisper and watchdog services..."
        sudo systemctl stop hypha-whisper hypha-whisper-watchdog
        echo "[setup] Services stopped."
    else
        echo ""
        echo "[ERROR] hypha-whisper is running but passwordless sudo is not configured."
        echo "  Option 1: Stop manually:  sudo systemctl stop hypha-whisper hypha-whisper-watchdog"
        echo "  Option 2: Add sudoers rule:"
        echo "    echo \"$USER ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper, /bin/systemctl start hypha-whisper-watchdog, /bin/systemctl stop hypha-whisper-watchdog\" \\"
        echo "        | sudo tee /etc/sudoers.d/hypha-whisper-tests"
        exit 1
    fi
else
    echo "[setup] hypha-whisper services are not running — proceeding."
fi

restore_service() {
    if [ "$SERVICE_WAS_RUNNING" = true ]; then
        echo ""
        echo "[teardown] Restarting hypha-whisper service..."
        sudo systemctl start hypha-whisper  # Wants= brings up watchdog automatically
        echo "[teardown] Service restarted."
    fi
}

# Always restore on exit (including Ctrl+C)
trap restore_service EXIT

# ── Check hardware presence ─────────────────────────────────────────────────
echo ""
echo "[setup] Checking hardware..."
python3 - <<'PYCHECK'
import pyaudio, sys
pa = pyaudio.PyAudio()
devices = {pa.get_device_info_by_index(i)["name"]: i for i in range(pa.get_device_count())}
pa.terminate()

missing = []
respeaker = next((n for n in devices if "ReSpeaker" in n), None)

# Speaker detection with fallback (USB SoundBar or HDMI/DP audio)
speaker_candidates = ["Dell AC511", "HDMI", "DisplayPort", "alsa_output.pci"]
speaker_name = None
for candidate in speaker_candidates:
    speaker_name = next((n for n in devices if candidate in n), None)
    if speaker_name:
        break

if respeaker:
    print(f"  ✓ ReSpeaker: {respeaker}")
else:
    print("  ✗ ReSpeaker 4 Mic Array NOT FOUND")
    missing.append("ReSpeaker")

if speaker_name:
    print(f"  ✓ Speaker:   {speaker_name}")
else:
    print("  ✗ No speaker found (tried: USB SoundBar, HDMI, DisplayPort)")
    missing.append("Speaker")

if missing:
    print(f"\n[setup] Missing hardware: {missing}. Plug in and retry.")
    sys.exit(1)
PYCHECK

# ── Run tests ───────────────────────────────────────────────────────────────
echo ""
echo "[setup] Starting hardware tests..."
echo ""

python3 -m pytest tests/test_hardware_loopback.py -v -s -m hardware "$@"
