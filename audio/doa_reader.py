"""
audio/doa_reader.py — Direction of Arrival (DOA) for ReSpeaker 4 Mic Array.

Reads the DOAANGLE register from the ReSpeaker firmware via USB vendor
ctrl_transfer (100 ms poll interval).

Root-cause note: firmware v4.0 (bcdDevice=0x400) appeared to STALL all
ctrl_transfer calls on Linux/Jetson — the actual bug was using wValue=0
instead of wValue=0xC0 (cmd = 0x80|offset|0x40 per tuning.py).  After
the fix both v3.0 and v4.0 firmware work correctly.

ReSpeaker USB Mic Array v2.0 mic geometry (circular, 4 mics at 90°):
    Mic 1:   0°  (channel 1)
    Mic 2:  90°  (channel 2)
    Mic 3: 180°  (channel 3)
    Mic 4: 270°  (channel 4)
    Radius: 46.5 mm

Requires pyusb: pip install --user pyusb
Requires USB device read/write access:
    echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"' \\
        | sudo tee /etc/udev/rules.d/99-respeaker.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
"""

import logging
import math
import struct
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

_VENDOR_ID  = 0x2886
_PRODUCT_ID = 0x0018
_POLL_INTERVAL_ACTIVE = 0.1   # seconds — used during active speech
_POLL_INTERVAL_IDLE   = 0.5   # seconds — used during silence
_HISTORY_LEN          = 20    # keep last 20 readings (2 s at 100 ms polling)
_DOA_ADDR             = 21    # DOAANGLE parameter index in ReSpeaker firmware


def _circular_mean(angles_deg: list) -> int:
    """Return circular mean of a list of angles (degrees, 0–359).

    Correct near the 0°/360° wraparound — unlike statistics.mean/median.
    Example: _circular_mean([350, 10]) == 0 (not 180).
    """
    sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
    mean = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return int(round(mean))


class DOAReader:
    """Poll DOA angle from ReSpeaker via USB vendor ctrl_transfer.

    Usage:
        doa = DOAReader()
        doa.start()
        angle = doa.current_direction()   # None if disabled or no data yet
        angle = doa.median_angle_since(t)
        doa.stop()
    """

    def __init__(self):
        self._dev = None
        self._enabled = False
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=_HISTORY_LEN)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._poll_interval = _POLL_INTERVAL_ACTIVE  # adjusted by set_active()

        self._init_device()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_device(self) -> None:
        try:
            import usb.core  # pyusb
        except ImportError:
            logger.warning("[DOAReader] pyusb not installed — USB DOA unavailable")
            return

        dev = usb.core.find(idVendor=_VENDOR_ID, idProduct=_PRODUCT_ID)
        if dev is None:
            logger.warning("[DOAReader] ReSpeaker USB device not found — USB DOA unavailable")
            return

        # Retry a few times — device may be in transient state after USB
        # re-enumeration or rapid service restarts.
        for attempt in range(3):
            try:
                # wValue=0xC0: cmd = 0x80 (read) | 0x00 (offset) | 0x40 (int type)
                # See tuning.py in respeaker/usb_4_mic_array for cmd calculation.
                data = dev.ctrl_transfer(0xC0, 0, 0xC0, _DOA_ADDR, 8)
                angle = struct.unpack('<ii', bytes(data))[0]
                logger.info("[DOAReader] USB DOA accessible — initial angle: %d°", angle)
                self._dev = dev
                self._enabled = True
                return
            except Exception as exc:
                if attempt < 2:
                    logger.debug("[DOAReader] ctrl_transfer attempt %d failed (%s), retrying...",
                                 attempt + 1, exc)
                    time.sleep(1.0)
                else:
                    logger.warning("[DOAReader] USB DOA unavailable after 3 attempts (%s).", exc)

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background USB polling (no-op if disabled)."""
        if not self._enabled:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="doa-reader")
        self._thread.start()
        logger.info("[DOAReader] USB polling thread started (active=%.0f ms, idle=%.0f ms)",
                    _POLL_INTERVAL_ACTIVE * 1000, _POLL_INTERVAL_IDLE * 1000)

    def stop(self) -> None:
        """Stop the DOA polling thread and release the USB device."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._dev is not None:
            try:
                import usb.util
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            self._dev = None
        self._enabled = False

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            angle = self._read_angle()
            if angle is not None:
                with self._lock:
                    self._history.append((time.monotonic(), angle))
            self._stop_event.wait(self._poll_interval)

    # ------------------------------------------------------------------
    # USB read
    # ------------------------------------------------------------------

    def _read_angle(self) -> Optional[int]:
        try:
            data = self._dev.ctrl_transfer(0xC0, 0, 0xC0, _DOA_ADDR, 8)
            return struct.unpack('<ii', bytes(data))[0] % 360
        except Exception as exc:
            logger.debug("[DOAReader] USB read error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_active(self, is_active: bool) -> None:
        """Adjust poll rate based on speech activity.

        Call with True when speech is detected (fast polling, 100 ms) and
        False during silence (slow polling, 500 ms) to reduce USB overhead.
        """
        self._poll_interval = _POLL_INTERVAL_ACTIVE if is_active else _POLL_INTERVAL_IDLE

    def median_angle_since(self, t: float) -> Optional[int]:
        """Return circular-mean DOA angle from readings since monotonic time *t*.

        Uses circular mean instead of arithmetic median to correctly handle
        the 0°/360° wraparound (e.g. angles 350° and 10° → 0°, not 180°).
        """
        if not self._enabled:
            return None
        # Copy under lock, compute outside to minimise lock contention.
        with self._lock:
            angles = [a for ts, a in self._history if ts >= t]
        if not angles:
            return None
        return _circular_mean(angles)

    def current_direction(self) -> Optional[int]:
        """Return stable current angle (circular mean over last 5 readings)."""
        if not self._enabled:
            return None
        # Copy under lock, compute outside.
        with self._lock:
            recent = list(self._history)[-5:]
        if len(recent) < 3:
            return None
        return _circular_mean([a for _, a in recent])
