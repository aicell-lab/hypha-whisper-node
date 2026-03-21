#!/usr/bin/env python3
"""
scripts/benchmark_models.py — Benchmark Whisper models on Jetson AGX Orin.

Tests different Whisper model sizes using pre-recorded audio to measure:
- Word Error Rate (WER) - accuracy
- Inference latency (seconds)
- Real-time Factor (RTF) - processing speed vs audio duration
- Model loading time

Usage:
    cd ~/workspace/hypha-whisper-node
    source ./activate_env.sh
    python scripts/benchmark_models.py

    # Specific models:
    python scripts/benchmark_models.py --models tiny.en base.en small.en

    # Specific audio file:
    python scripts/benchmark_models.py --audio tests/test-audio-male.wav
"""

import argparse
import io
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import whisper

# Default models to test (English-only for consistency)
DEFAULT_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]

# Reference text for WER calculation (from test-audio-male.wav)
REFERENCE_TEXT = (
    "In microscopy laboratories, researchers often need to annotate experiments, "
    "describe observations, or record notes while working at the microscope, which "
    "can interrupt the workflow if done manually. A portable real-time speech-to-text "
    "device can capture spoken descriptions and automatically convert them into text "
    "during imaging sessions."
)


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)  # Remove punctuation except apostrophes
    return text.split()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using Levenshtein distance."""
    ref_words = normalize_text(reference)
    hyp_words = normalize_text(hypothesis)
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    # Dynamic programming for edit distance
    d = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words):
        prev = d[:]
        d[0] = i + 1
        for j, hyp_word in enumerate(hyp_words):
            cost = 0 if ref_word == hyp_word else 1
            d[j + 1] = min(prev[j] + cost, d[j] + 1, prev[j + 1] + 1)
    
    return d[len(hyp_words)] / len(ref_words)


def load_audio_file(audio_path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file using ffmpeg."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(audio_path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def benchmark_model(model_name: str, audio: np.ndarray, audio_duration: float) -> dict:
    """Benchmark a single Whisper model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    # Measure model loading time
    print(f"  Loading model...", end=" ", flush=True)
    load_start = time.time()
    try:
        model = whisper.load_model(model_name)
        load_time = time.time() - load_start
        print(f"✓ ({load_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None
    
    # Measure inference time
    print(f"  Running inference...", end=" ", flush=True)
    infer_start = time.time()
    try:
        result = model.transcribe(audio, language="en", fp16=True)
        infer_time = time.time() - infer_start
        print(f"✓ ({infer_time:.1f}s)")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None
    
    # Calculate metrics
    hypothesis = result["text"].strip()
    wer = compute_wer(REFERENCE_TEXT, hypothesis)
    rtf = infer_time / audio_duration  # Real-time factor
    
    metrics = {
        "model": model_name,
        "load_time": load_time,
        "infer_time": infer_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "wer": wer,
        "wer_percent": wer * 100,
        "transcript": hypothesis,
    }
    
    print(f"  Results:")
    print(f"    Load time:     {load_time:.1f}s")
    print(f"    Infer time:    {infer_time:.1f}s")
    print(f"    Audio length:  {audio_duration:.1f}s")
    print(f"    RTF:           {rtf:.2f}x ({'✓ Real-time' if rtf < 1.0 else '✗ Not real-time'})")
    print(f"    WER:           {wer*100:.1f}%")
    print(f"    Transcript:    {hypothesis[:80]}...")
    
    return metrics


def generate_summary_text(results: list, audio_duration: float, device_info: str) -> str:
    """Generate clean summary text for file output."""
    lines = []
    lines.append("=" * 70)
    lines.append("Whisper Model Benchmark Summary")
    lines.append("=" * 70)
    lines.append(f"Device: {device_info}")
    lines.append(f"Audio Duration: {audio_duration:.1f} seconds")
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Results table
    lines.append("Results:")
    lines.append("-" * 70)
    lines.append(f"{'Model':<15} {'Load(s)':<10} {'Infer(s)':<10} {'RTF':<8} {'WER%':<10} {'Status'}")
    lines.append("-" * 70)
    
    for r in results:
        if r["rtf"] < 1.0 and r["wer"] < 0.2:
            status = "Good"
        elif r["rtf"] < 1.5 and r["wer"] < 0.3:
            status = "OK"
        else:
            status = "Slow"
        lines.append(f"{r['model']:<15} {r['load_time']:<10.1f} {r['infer_time']:<10.1f} "
                    f"{r['rtf']:<8.2f} {r['wer_percent']:<10.1f} {status}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    
    # Filter models that can run in real-time (RTF < 1.0)
    realtime_models = [r for r in results if r["rtf"] < 1.0]
    
    if realtime_models:
        best = min(realtime_models, key=lambda x: x["wer"])
        lines.append(f"Best Model for Real-time: {best['model']}")
        lines.append("")
        lines.append(f"  Metrics:")
        lines.append(f"    - Word Error Rate: {best['wer_percent']:.1f}%")
        lines.append(f"    - Real-time Factor: {best['rtf']:.2f}x")
        lines.append(f"    - Inference Time: {best['infer_time']:.1f}s")
        lines.append(f"    - Load Time: {best['load_time']:.1f}s")
        lines.append("")
        lines.append(f"  Verdict: This model can process audio faster than real-time,")
        lines.append(f"           making it suitable for live streaming applications.")
    else:
        best = min(results, key=lambda x: x["wer"])
        lines.append("WARNING: No model achieves real-time processing (RTF < 1.0)")
        lines.append("")
        lines.append(f"Best Accuracy (but slow): {best['model']}")
        lines.append(f"  - WER: {best['wer_percent']:.1f}%")
        lines.append(f"  - RTF: {best['rtf']:.2f}x")
        lines.append("")
        lines.append("Consider using a smaller model for real-time applications.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("Reference Guide")
    lines.append("=" * 70)
    lines.append("Real-time Factor (RTF):")
    lines.append("  < 1.0  : Faster than real-time (suitable for streaming)")
    lines.append("  1.0-1.5: Near real-time (acceptable for batch processing)")
    lines.append("  > 1.5  : Too slow for real-time applications")
    lines.append("")
    lines.append("Word Error Rate (WER):")
    lines.append("  < 10%  : Excellent accuracy")
    lines.append("  10-20% : Good accuracy")
    lines.append("  20-30% : Acceptable accuracy")
    lines.append("  > 30%  : Poor accuracy")
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def print_summary(results: list, audio_duration: float):
    """Print benchmark summary table."""
    import torch
    device = f"{torch.cuda.get_device_name(0)} (CUDA: {torch.cuda.is_available()})"
    
    # Generate and print summary
    summary_text = generate_summary_text(results, audio_duration, device)
    print(summary_text)
    
    # Also save to file
    output_file = Path("scripts/benchmark_results.txt")
    output_file.write_text(summary_text)
    print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Whisper models on Jetson AGX Orin"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to test (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("tests/test-audio-male.wav"),
        help="Path to test audio file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"Whisper Model Benchmark - Jetson AGX Orin")
    print(f"{'='*80}")
    print(f"Test audio: {args.audio}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Device: CUDA (if available)")
    
    # Check CUDA
    import torch
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load audio
    try:
        audio_duration = get_audio_duration(args.audio)
        print(f"\nLoading audio ({audio_duration:.1f}s)...", end=" ", flush=True)
        audio = load_audio_file(args.audio)
        print(f"✓")
    except FileNotFoundError:
        print(f"\n✗ Audio file not found: {args.audio}")
        print(f"Please download or specify a different audio file.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        sys.exit(1)
    
    # Benchmark each model
    results = []
    for model_name in args.models:
        result = benchmark_model(model_name, audio, audio_duration)
        if result:
            results.append(result)
    
    if not results:
        print("\n✗ No models successfully tested!")
        sys.exit(1)
    
    # Print summary
    print_summary(results, audio_duration)
    
    # Save to JSON if requested
    if args.output:
        import json
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.output}")
    
    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
