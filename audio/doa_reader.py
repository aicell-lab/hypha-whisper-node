"""
audio/doa_reader.py — Direction of Arrival (DOA) for ReSpeaker 4 Mic Array.

Uses the official ReSpeaker tuning.py to read DOA from the XMOS firmware.
The firmware has built-in DOA algorithm that processes raw mic data on-chip.

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
import statistics
import struct
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# ReSpeaker USB VID/PID
_VENDOR_ID  = 0x2886
_PRODUCT_ID = 0x0018

# DOA parameter from tuning.py
_DOA_PARAM = ('DOAANGLE', 21, 0, 'int', 359, 0, 'ro', 'DOA angle. Current value.')


class DOAIntervalBuffer:
    """Buffer for storing DOA estimates as time intervals with duration-weighted lookup.
    
    This solves the speaker misattribution problem by calculating which angle
    had the longest overlap with a given time window (not just median/mode).
    
    Inspired by WhisperX's interval tree approach for speaker assignment.
    
    Usage:
        buf = DOAIntervalBuffer(maxlen=200)  # ~10s at 50ms chunks
        buf.add_interval(angle, start_time, end_time)  # Store as interval
        angle = buf.dominant_angle(query_start, query_end)  # Longest overlap wins
    """
    
    def __init__(self, maxlen: int = 200, poll_interval: float = 0.05):
        """
        Args:
            maxlen: Maximum number of intervals to store
            poll_interval: Expected time between DOA readings (seconds)
        """
        self._intervals = deque(maxlen=maxlen)  # List of (start, end, angle)
        self._lock = threading.Lock()
        self._poll_interval = poll_interval
        self._last_timestamp: Optional[float] = None
    
    def add(self, angle: int, timestamp: Optional[float] = None):
        """Add a DOA reading as an interval.
        
        Each reading represents the DOA for the interval [timestamp, timestamp+poll_interval).
        """
        if timestamp is None:
            timestamp = time.monotonic()
        
        with self._lock:
            # Each reading covers from this timestamp to the next (or poll_interval ahead)
            if self._last_timestamp is not None:
                # Use actual time since last reading
                interval_start = self._last_timestamp
                interval_end = timestamp
            else:
                # First reading - assume poll_interval duration
                interval_start = timestamp
                interval_end = timestamp + self._poll_interval
            
            self._intervals.append((interval_start, interval_end, angle))
            self._last_timestamp = timestamp
    
    def dominant_angle(self, query_start: float, query_end: float) -> Optional[int]:
        """Find the angle with longest total overlap with query interval.
        
        This is the key fix: instead of median/mode, we calculate which angle
        was active for the longest duration within the query window.
        
        Args:
            query_start: Start of query window (absolute monotonic time)
            query_end: End of query window (absolute monotonic time)
            
        Returns:
            Angle with longest total overlap, or None if no overlap
        """
        with self._lock:
            intervals = list(self._intervals)
        
        if not intervals:
            return None
        
        # Calculate overlap duration for each angle
        angle_durations: dict[int, float] = {}
        
        for seg_start, seg_end, angle in intervals:
            # Calculate intersection of [seg_start, seg_end) and [query_start, query_end)
            intersection_start = max(seg_start, query_start)
            intersection_end = min(seg_end, query_end)
            overlap_duration = intersection_end - intersection_start
            
            if overlap_duration > 0:
                angle_durations[angle] = angle_durations.get(angle, 0.0) + overlap_duration
        
        if not angle_durations:
            return None
        
        # Return angle with longest total overlap
        # If tie, max() picks the first one encountered
        return max(angle_durations.items(), key=lambda x: x[1])[0]
    
    def clear(self):
        """Clear the buffer and reset state."""
        with self._lock:
            self._intervals.clear()
            self._last_timestamp = None


# Keep DOABuffer for backwards compatibility (deprecated)
class DOABuffer(DOAIntervalBuffer):
    """Deprecated: Use DOAIntervalBuffer instead."""
    pass


class DOAReader:
    """Poll DOA angle from ReSpeaker firmware via USB using tuning.py.

    This uses the XMOS XVF-3000's built-in DOA algorithm which processes
    the 4 raw microphone channels on-chip. This is more accurate and
    efficient than software DOA estimation.

    Usage:
        doa = DOAReader()
        doa.start()
        angle = doa.current_direction()   # None if disabled or no data yet
        angle = doa.median_angle_since(t)
        doa.stop()
    """

    # USB control transfer constants from tuning.py
    TIMEOUT = 100000  # USB timeout in ms
    
    def __init__(self, poll_interval: float = 0.05):
        """Initialize DOA reader.
        
        Args:
            poll_interval: How often to poll DOA from firmware (seconds)
        """
        self._dev = None
        self._enabled = False
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=100)  # 5s at 50ms polling
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._poll_interval = poll_interval

        self._init_device()

    def _init_device(self) -> None:
        """Find and initialize the ReSpeaker USB device."""
        try:
            import usb.core  # pyusb
        except ImportError:
            logger.warning("[DOAReader] pyusb not installed — USB DOA unavailable")
            return

        dev = usb.core.find(idVendor=_VENDOR_ID, idProduct=_PRODUCT_ID)
        if dev is None:
            logger.warning("[DOAReader] ReSpeaker USB device not found — USB DOA unavailable")
            return

        try:
            # Test read DOA to verify device is working
            angle = self._read_angle_from_device(dev)
            if angle is not None:
                logger.info("[DOAReader] USB DOA accessible — initial angle: %d°", angle)
                self._dev = dev
                self._enabled = True
            else:
                logger.warning("[DOAReader] USB DOA read test failed")
        except Exception as exc:
            logger.warning("[DOAReader] USB DOA unavailable (%s).", exc)

    def _read_angle_from_device(self, dev) -> Optional[int]:
        """Read DOA angle from device using tuning.py protocol.
        
        This implements the same control transfer as tuning.py:
        - bmRequestType: CTRL_IN | CTRL_TYPE_VENDOR | CTRL_RECIPIENT_DEVICE
        - bRequest: 0
        - wValue: 0x80 | offset | (0x40 if int type)
        - wIndex: parameter id
        """
        try:
            import usb.util
            
            # DOAANGLE parameter: id=21, offset=0, type='int'
            param_id = _DOA_PARAM[1]  # 21
            param_offset = _DOA_PARAM[2]  # 0
            param_type = _DOA_PARAM[3]  # 'int'
            
            # Build command: 0x80 | offset | (0x40 for int type)
            cmd = 0x80 | param_offset
            if param_type == 'int':
                cmd |= 0x40
            
            # USB control transfer (same as tuning.py)
            response = dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,  # bRequest
                cmd,  # wValue
                param_id,  # wIndex
                8,  # length
                self.TIMEOUT
            )
            
            # Unpack response (two int32 values, first is the value)
            value = struct.unpack(b'ii', response.tobytes() if hasattr(response, 'tobytes') else bytes(response))[0]
            return int(value) % 360
            
        except Exception as exc:
            logger.debug("[DOAReader] USB read error: %s", exc)
            return None

    def _read_angle(self) -> Optional[int]:
        """Read current DOA angle from device."""
        if self._dev is None:
            return None
        return self._read_angle_from_device(self._dev)

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
        logger.info("[DOAReader] USB polling started (interval=%.0f ms)",
                    self._poll_interval * 1000)

    def stop(self) -> None:
        """Stop the DOA polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Close USB device
        if self._dev is not None:
            try:
                import usb.util
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
            self._dev = None
            self._enabled = False

    def _poll_loop(self) -> None:
        """Background thread that polls DOA from firmware."""
        while not self._stop_event.is_set():
            angle = self._read_angle()
            if angle is not None:
                with self._lock:
                    self._history.append((time.monotonic(), angle))
            self._stop_event.wait(self._poll_interval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return True if DOA reader is enabled and working."""
        return self._enabled

    def read(self) -> Optional[int]:
        """Read current DOA angle directly (blocking call).
        
        Returns:
            Angle in degrees (0-359) or None if not available.
        """
        if not self._enabled:
            return None
        return self._read_angle()

    def median_angle_since(self, t: float) -> Optional[int]:
        """Return median DOA angle from readings since monotonic time *t*."""
        if not self._enabled:
            return None
        with self._lock:
            angles = [a for ts, a in self._history if ts >= t]
        if not angles:
            return None
        return int(statistics.median(angles))

    def median_in_window(self, start_t: float, end_t: float) -> Optional[int]:
        """Return median DOA angle from readings in time window."""
        if not self._enabled:
            return None
        with self._lock:
            angles = [a for ts, a in self._history if start_t <= ts <= end_t]
        if not angles:
            return None
        return int(statistics.median(angles))

    def current_direction(self) -> Optional[int]:
        """Return stable current angle (mode over last 5 readings)."""
        if not self._enabled:
            return None
        with self._lock:
            recent = list(self._history)[-5:]
        if len(recent) < 3:
            return None
        angles = [a for _, a in recent]
        try:
            return statistics.mode(angles)
        except statistics.StatisticsError:
            return int(statistics.median(angles))
