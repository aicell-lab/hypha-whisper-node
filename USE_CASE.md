# Hypha-Whisper-Node Use Cases

Local speech-to-text transcription using whisper-timestamped with optimized settings (beam_size=5, temperature=0.0).

---

## Use Case 1: Presentation Rehearsal Feedback

A researcher records their conference presentation rehearsal and wants objective feedback on speech delivery.

### Real Example

**Input:** 7-minute 43-second rehearsal audio (M4A format, 15.1MB)

**Transcription Output:**
```
[0.0s - 9.1s]  Good morning, everyone. My name is...
[9.1s - 16.9s]  ...from KTH. Today, I will present Agent Lens...
[158.0s - 166.8s]  So, this is our lab's part...
[216.5s - 222.9s]  So, here's our demo video.
[222.9s - 239.1s]  On this video, a user can talk with AI agents...
[328.0s - 338.2s]  So, with this demo video, then use cases...
[467.5s - 475.9s]  ...thanks for our collaborators and funding.
```

**Analysis Generated:**
- **Total timing:** 7:43 (within 8-minute limit)
- **Filler words:** "and" used 30+ times as sentence connector, "so" used 12 times, "okay" used 8 times
- **Section timing:** Demo description = 1 minute 30 seconds (recommended: 45 seconds)
- **Grammar issues:** "Current data cells are too small" → should be "Current datasets are too small"
- **Transitions:** "So" used 3 times to start new sections (overused)

**Pronunciation Issues Identified:**
- "acquisition" pronounced as "equation" (phonetic confusion)
- "Hypha" pronounced as "HIFA" (uncommon word)
- Project name "Agent Lens" transcribed as "Agent loans" or "H&N's" (base model limitation)

**Benefit:** Iterative improvement across multiple rehearsal versions without requiring a human listener for each round.

---

## Use Case 2: Transcription Quality Optimization

Improving accuracy for domain-specific terminology by tuning Whisper parameters.

### Problem

Base Whisper model (base.en) with default settings (beam_size=1) produces these errors on technical presentations:

| Original Speech | Transcription (beam_size=1) |
|-----------------|---------------------------|
| robotic arm | "rewarding arm" |
| cell circularity | "cell security" |
| metrics | "milkshakes" |
| Agent Lens | "Agent loans" |
| Hypha | "HIFA" |

### Solution Applied

```python
# Before (default in whisper_online.py)
result = whisper_timestamped.transcribe_timestamped(
    model, audio, language=language, 
    initial_prompt=init_prompt, beam_size=1, ...
)

# After (optimized)
result = whisper_timestamped.transcribe_timestamped(
    model, audio, language=language,
    initial_prompt=init_prompt, beam_size=5, temperature=0.0, ...
)
```

### Results Comparison

| Original Speech | beam_size=1 | beam_size=5 | Improvement |
|-----------------|-------------|-------------|-------------|
| robotic arm | "rewarding arm" | **"robotic arm"** | Fixed |
| cell circularity | "cell security" | **"cell circularity"** | Fixed |
| metrics | "milkshakes" | **"metrics"** | Fixed |
| Fucci system | "FUCHI system" | "FUCHI system" | Partial |
| acquisition | "equation" | "equation" | Base model limitation |
| Hypha | "HIFA" | "HIFA" | Requires larger model |

**Note:** Technical terms and common words improved significantly with beam_size=5. Proper nouns (project names like "Agent Lens") and phonetically similar words ("acquisition" vs "equation") still require larger models (small.en/medium.en) or domain-specific fine-tuning.

---

## Use Case 3: Voice Interface Testing

Validating voice commands for lab automation systems.

### Test Session

**Scenario:** Testing 6 common lab automation voice commands with optimized settings on a smart microscopy system.

| Voice Command | Transcription Result | Status |
|--------------|---------------------|--------|
| "take images" | "take images" | pass |
| "run autofocus" | "run autofocus" | pass |
| "acquire images" | "acquire images" | pass |
| "show segmentation" | "show segmentation" | pass |
| "cell metrics" | "cell metrics" | pass (fixed with beam_size=5) |
| "robotic arm" | "robotic arm" | pass (fixed with beam_size=5) |

### Code Implementation

```python
import whisper
import whisper_timestamped

# Load model
model = whisper.load_model("base.en")

# Process audio
result = whisper_timestamped.transcribe_timestamped(
    model, 
    audio_path,
    language="en",
    beam_size=5,
    temperature=0.0,
    initial_prompt=""
)

# Extract text with timestamps
segments = result["segments"]
for segment in segments:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.1f}s - {end:.1f}s] {text}")

# Output examples from testing:
# "take images, run autofocus, acquire images with these three channels"
# "show the segmentation"
# "cell size, cell circularity, and other metrics"
# "find cells in mitosis"
```

**Use Case:** Hands-free control of REEF Imaging Farm infrastructure where gloves prevent keyboard/mouse use. The voice interface connects to Hypha services for microscope control, segmentation, and analysis.

---

## Setup and Endpoint

### Local Setup

```bash
# Install dependencies
pip install openai-whisper whisper-timestamped

# Download model (auto-download on first use)
# Models: tiny.en, base.en (default), small.en, medium.en, large
```

### Usage Options

**Option 1: Direct Python (shown in examples above)**
```python
import whisper
import whisper_timestamped

model = whisper.load_model("base.en")
result = whisper_timestamped.transcribe_timestamped(
    model, audio_file, beam_size=5, temperature=0.0
)
```

**Option 2: Hypha-RPC Service**
- Register as Hypha service for distributed access
- Endpoint: `hypha.aicell.io` workspace
- Function: `transcribe(audio_data)` returns timestamped text
- Use case: Remote transcription from edge devices (Jetson Orin Nano)

**Option 3: HTTP ASGI Endpoint**
- POST audio file to `/transcribe` endpoint
- Returns JSON with segments and metadata
- Use case: Direct integration with web applications

---

*All transcription performed locally - no audio data sent to external APIs.*
