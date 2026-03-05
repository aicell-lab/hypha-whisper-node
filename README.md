# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and Jetson Orin Nano.
Captures speech via ReSpeaker mic array, transcribes on-device, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

---

## Hardware

| Component | Details |
|---|---|
| Compute | Jetson Orin Nano |
| Microphone | ReSpeaker Mic Array v2.0 (USB) |
| Power | USB-C PD power bank |
| Enclosure | 3D printed shell (ventilation + antenna ports) |

**Device stack:** ReSpeaker (top) → Jetson Orin Nano (bottom) inside 3D printed shell.

---

## Features

- On-device Whisper transcription (works offline)
- Real-time transcript streaming via Hypha RPC
- ReSpeaker mic array with beamforming / noise suppression
- Integrates with the Hypha agent ecosystem


---

## Use Cases

- Live conference captioning
- AI agent voice interfaces
- Edge speech processing nodes
- Lab automation voice control

---

## License

MIT
