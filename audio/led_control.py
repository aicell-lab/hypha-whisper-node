"""Simple LED control for ReSpeaker 4 Mic Array — turn off those annoying lights."""

import logging

logger = logging.getLogger(__name__)

# ReSpeaker USB VID/PID
_VID = 0x2886
_PID = 0x0018


def led_off():
    """Turn off all LEDs on the ReSpeaker 4 Mic Array."""
    try:
        import usb.core
        import usb.util

        dev = usb.core.find(idVendor=_VID, idProduct=_PID)
        if dev is None:
            return  # No ReSpeaker connected

        # Command 6 = custom pattern, 12 LEDs × 4 bytes (r, g, b, 0) all zeros = off
        data = bytes([0, 0, 0, 0] * 12)
        dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 6, 0x1C, data, 100000
        )
        usb.util.dispose_resources(dev)
        logger.info("[LED] ReSpeaker LEDs turned off")

    except Exception:
        pass  # Silently ignore if pyusb not installed or no permission
