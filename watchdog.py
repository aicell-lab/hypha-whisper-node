"""
watchdog.py — Health watchdog for hypha-whisper systemd service.

Periodically polls the Hypha health endpoint. If the endpoint is unreachable
or returns a non-ok status, the systemd service is restarted.

Designed to run as its own systemd service (see deploy/hypha-whisper-watchdog.service).
Must run as root (or a user with sudo rights) so it can restart the main service.

Health endpoint: https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health
"""

import logging
import subprocess
import sys
import time

import httpx

HEALTH_URL = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health"
SERVICE_NAME = "hypha-whisper"
CHECK_INTERVAL = 60    # seconds between health checks
TIMEOUT = 15           # seconds for HTTP request
STOP_WAIT = 20         # seconds after stop before start, to let Hypha deregister
FAILURE_THRESHOLD = 2  # consecutive failures required before restarting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def check_health() -> bool:
    """Return True if the health endpoint reports ok, False otherwise."""
    try:
        response = httpx.get(HEALTH_URL, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                return True
            logging.warning("Health check returned non-ok status: %s", data)
        else:
            logging.warning("Health check HTTP %s from %s", response.status_code, HEALTH_URL)
    except Exception as exc:
        logging.warning("Health check failed: %s", exc)
    return False


def restart_service():
    """Stop then start the systemd service, with a pause between so Hypha can deregister."""
    logging.info("Restarting service: %s", SERVICE_NAME)
    for action in ("stop", "start"):
        result = subprocess.run(
            ["sudo", "systemctl", action, SERVICE_NAME],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error(
                "systemctl %s %s failed (code %s): %s",
                action, SERVICE_NAME, result.returncode, result.stderr.strip(),
            )
        else:
            logging.info("systemctl %s %s succeeded", action, SERVICE_NAME)
        if action == "stop":
            logging.info("Waiting %ss for Hypha server to deregister the old session…", STOP_WAIT)
            time.sleep(STOP_WAIT)


def main():
    logging.info("Watchdog started. Monitoring %s every %ss", HEALTH_URL, CHECK_INTERVAL)
    failures = 0
    while True:
        time.sleep(CHECK_INTERVAL)  # always wait first — gives service time to start up
        if check_health():
            logging.info("Health check OK")
            failures = 0
        else:
            failures += 1
            logging.error("Health check FAILED (%s/%s)", failures, FAILURE_THRESHOLD)
            if failures >= FAILURE_THRESHOLD:
                logging.error("Restarting service after %s consecutive failures", failures)
                restart_service()
                failures = 0


if __name__ == "__main__":
    main()
