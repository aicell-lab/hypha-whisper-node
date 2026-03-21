# 🔒 Privacy Policy & Data Protection

**Last Updated:** March 2025

This document explains how **hypha-whisper-node** handles your voice data and protects your privacy.

---

## 🛡️ Core Privacy Principles

### 1. Local-First Processing
All speech recognition happens **on your device** (NVIDIA Jetson). Your voice never leaves the hardware for cloud-based transcription.

### 2. No Audio Storage
- ❌ **Audio recordings are NEVER saved to disk**
- ❌ **Audio is NOT logged** — not even temporarily
- Audio is captured in real-time, processed in chunks, and immediately discarded

### 3. No Transcript Storage
- ❌ **Transcribed text is NEVER saved to files**
- ❌ **Transcripts are NOT persisted** to any database or storage
- Transcript data is held in memory only and streamed to connected clients
- When the service stops or restarts, all transcript data is irrevocably lost

### 4. Ephemeral Data Processing
```
Microphone → Memory Buffer → Transcription → SSE Stream → Discard
     ↓            ↓               ↓              ↓           ↓
  (USB)     (temporary)     (temporary)    (live only)  (gone forever)
```

---

## 📋 What Data IS Logged

The system produces logs for operational monitoring. These logs contain:

| Logged | Not Logged |
|--------|------------|
| ✅ System startup/shutdown events | ❌ Audio recordings |
| ✅ Connection status to Hypha server | ❌ Raw audio buffers |
| ✅ Transcript **metadata** (timestamp, speaker angle) | ❌ Transcript **text content** |
| ✅ Speaker direction angles (e.g., `45°`) | ❌ Voice biometric data |
| ✅ Model loading status | ❌ Historical transcripts |
| ✅ Error messages | ❌ Any persistent user data |

### Log Destinations
When running as a systemd service, logs go to:
- **systemd journal** (`journalctl -u hypha-whisper`) — system-dependent retention
- **NOT to any file** by default

When running manually:
- **stdout/stderr** only

### Log Retention
- In-memory log buffer: **2000 entries maximum** (rolling, old entries auto-discarded)
- systemd journal retention depends on your system configuration (`/etc/systemd/journald.conf`)

---

## 🌐 Network Communication

### Hypha RPC Streaming
- Transcripts are streamed to connected clients via **Server-Sent Events (SSE)**
- This requires a network connection to your configured Hypha server
- **Only transcript text is transmitted** — never raw audio
- You control the Hypha server endpoint (can be self-hosted)

### Offline Mode
Run completely offline with no network transmission:
```bash
python3 main.py --server ""
```
In offline mode, transcripts are printed to stdout only.

---

## 🔍 Code Auditability

This project is **100% open source**. You can verify our privacy claims by:

1. **Reviewing the source code:**
   ```bash
   # Check for any file write operations
   grep -r "open\|write\|save" --include="*.py" . | grep -v "test\|benchmark"
   
   # Check logging statements
   grep -r "logger\." --include="*.py" transcribe/ audio/ rpc/
   ```

2. **Monitoring filesystem activity:**
   ```bash
   # Use strace to see all file operations
   sudo strace -e trace=openat,write -f python3 main.py --server ""
   ```

3. **Checking network connections:**
   ```bash
   # Monitor network traffic
   sudo tcpdump -i any -A | grep -i speech\|audio
   ```

---

## 📜 GDPR & Data Protection Compliance

### Your Rights Under GDPR
Since **no personal data is stored**, most GDPR rights are satisfied by design:

| GDPR Right | How We Comply |
|------------|---------------|
| **Right to Access** | No data stored — nothing to access |
| **Right to Erasure** | Data is ephemeral — automatically deleted on service stop |
| **Right to Portability** | Real-time streaming allows immediate access |
| **Right to Object** | Run in offline mode to prevent any transmission |

### Data Controller Responsibilities
If you deploy this system:
- You are the **Data Controller** for any transcripts your application stores
- This software acts as a **Data Processor** for the audio-to-text conversion only
- Configure your downstream systems to handle data according to your privacy policy

---

## 🏢 Third-Party Dependencies

This project uses the following external services/libraries:

| Dependency | Purpose | Data Shared |
|------------|---------|-------------|
| **Hypha RPC** (optional) | Remote streaming | Transcript text only (not audio) |
| **OpenAI Whisper** | Speech recognition | None (local model) |
| **PyTorch** | ML inference | None (local execution) |
| **NVIDIA CUDA** | GPU acceleration | None |

---

## 🔐 Security Recommendations

### For Maximum Privacy

1. **Run in Offline Mode:**
   ```bash
   python3 main.py --server ""
   ```

2. **Self-Host Hypha:** Deploy your own Hypha server to keep all data within your infrastructure

3. **Configure systemd Journal:** Limit log retention in `/etc/systemd/journald.conf`:
   ```ini
   [Journal]
   SystemMaxUse=100M
   MaxRetentionSec=1week
   ```

4. **Review Log Access:** Ensure log files are readable only by authorized users:
   ```bash
   sudo chmod 640 /var/log/journal/*
   ```

5. **Physical Security:** Since this runs on edge hardware (Jetson), secure physical access to the device

---

## 📝 Transparency Report

### No Telemetry
- ❌ No analytics or usage statistics collected
- ❌ No crash reports automatically sent
- ❌ No phone-home functionality

### No Cloud Dependencies (for core function)
- Whisper model downloads happen once during setup
- After setup, the system works entirely offline
- Optional: Hypha connection for remote streaming

---

## ❓ Frequently Asked Questions

**Q: Is my voice recorded and saved?**  
A: **No.** Your voice is captured in real-time, transcribed, and immediately discarded. No audio files are created.

**Q: Are my transcripts saved to a file?**  
A: **No.** Transcripts exist only in memory and are streamed to connected clients. When the service stops, all transcript data is gone.

**Q: Can someone access my past conversations?**  
A: **No.** There are no past conversations stored. Each session starts fresh with no history.

**Q: Does this send my data to OpenAI?**  
A: **No.** The Whisper model runs locally on your Jetson. No audio or text is sent to OpenAI's servers.

**Q: What if I want to keep a record of transcripts?**  
A: You can consume the SSE stream (`/transcript_feed`) and save transcripts in your own application. The choice to store data is yours.

---

## 📞 Privacy Concerns & Reporting

If you discover a potential privacy issue or have questions:

1. **Review the code** — it's all here in this repository
2. **Open an issue** on GitHub
3. **Email the maintainers** (see repository contributors)

---

## 🔄 Updates to This Policy

We will update this privacy policy as needed. Changes will be documented in the git history of this file.

---

## ✅ Summary

| Feature | Status |
|---------|--------|
| Audio stored | ❌ Never |
| Transcripts stored | ❌ Never |
| Cloud transcription | ❌ Never |
| Telemetry/analytics | ❌ None |
| Open source | ✅ 100% |
| Auditable | ✅ Yes |
| Works offline | ✅ Yes |
| Self-hostable | ✅ Yes |

**hypha-whisper-node is designed with privacy as a core feature, not an afterthought.**
