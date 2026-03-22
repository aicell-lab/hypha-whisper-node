# Hardware Setup Guide

## Microphone

### Primary: ReSpeaker 4 Mic Array (USB)

**Model:** Seeed Studio ReSpeaker USB 4-Mic Array (v2.0 or v3.0)

**Features:**
- 4 PDM microphones in circular array (90° intervals)
- Radius: 46.5mm
- USB UAC1.0 audio interface
- Requires **6-channel firmware** for multi-channel DOA

**Firmware:**
- Default: 1-channel (beamformed output only)
- Required: 6-channel firmware for raw mic access

**Channel Layout (6-channel firmware):**
| Channel | Content | Use |
|---------|---------|-----|
| 0 | Beamformed audio | ASR input |
| 1 | Raw mic 1 (0°) | DOA estimation |
| 2 | Raw mic 2 (90°) | DOA estimation |
| 3 | Raw mic 3 (180°) | DOA estimation |
| 4 | Raw mic 4 (270°) | DOA estimation |
| 5 | Playback reference | AEC (unused) |

**Note:** The ReSpeaker is a **microphone array only**. It does NOT have a speaker output.

## Speaker (Required for Tests)

### Option 1: Dell AC511 USB SoundBar (Preferred)
- USB-powered soundbar
- Plug-and-play with Jetson

### Option 2: HDMI/DisplayPort Audio
- Monitor/TV with built-in speakers
- Connect via HDMI or DisplayPort cable

### Option 3: Other USB Audio
- Any USB speaker or headphone adapter

**Important:** For hardware loopback tests, you MUST have an external speaker connected. The ReSpeaker mic array does not provide audio output.

## Connection Diagram

```
[USB Speaker] → [Jetson Orin] ← [ReSpeaker 4-Mic Array]
                                    ↓
                              Captures audio for ASR + DOA
```

## Firmware Flashing

To enable multi-channel DOA, flash the 6-channel firmware:

```bash
cd /tmp
git clone https://github.com/respeaker/usb_4_mic_array.git
cd usb_4_mic_array
sudo python dfu.py --download 6_channels_firmware.bin
# Re-plug the ReSpeaker
```

## Verification

Check devices are detected:
```bash
# List capture devices (should show ReSpeaker)
arecord -l | grep ReSpeaker

# List playback devices (should show your speaker)
aplay -l | grep -E "(Dell|HDMI|USB)"
```

## DOA Algorithm

Uses SRP-PHAT (Steered Response Power - Phase Transform) algorithm on raw mic channels 1-4.

**Geometry:**
- Circular array, radius 46.5mm
- Mic positions: 0°, 90°, 180°, 270°

**Process:**
1. Capture 6-channel audio via PyAudio
2. Extract ch0 → Whisper ASR
3. Extract ch1-4 → SRP-PHAT DOA estimation
4. Estimate speaker direction at each commit
5. Register speakers by angle (30° threshold)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No ReSpeaker detected | Check USB connection, re-plug device |
| No 6-channel audio | Re-flash firmware, verify with `arecord -l` shows 6 channels |
| DOA returns '?' | Check firmware is 6-channel, verify raw mics are being passed to engine |
| Test produces no sound | Ensure external speaker (Dell USB or HDMI) is connected |
| High WER | Check speaker volume, mic distance, reduce background noise |

## Test Requirements

Hardware loopback tests require:
1. ReSpeaker 4-Mic Array with 6-channel firmware
2. External speaker (Dell USB SoundBar or HDMI audio)
3. Quiet environment (for WER test)
4. Speaker positioned ~0.5-1m from ReSpeaker
