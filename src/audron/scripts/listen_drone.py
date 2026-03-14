"""
Continuous microphone listener: run binary ONNX model on a 3-second sliding window, advanced every 0.1 s.

Requires: pip install onnxruntime sounddevice

Usage:
  python -m audron.scripts.listen_drone --onnx runs/binary_no_aug/best.onnx
  python -m audron.scripts.listen_drone --onnx runs/binary_no_aug/best.onnx --hop 0.1 --threshold 0.6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Optional deps
try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)
try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice", file=sys.stderr)
    sys.exit(1)


SAMPLE_RATE = 16000
WINDOW_DURATION_SEC = 3.0
HOP_DURATION_SEC = 0.1
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_SEC)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION_SEC)
DRONE_CLASS_ID = 1  # binary: 0=noise, 1=drone


def peak_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(waveform)))
    if peak < eps:
        return waveform.astype(np.float32)
    return (waveform / peak).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Listen to microphone and alert when drone is detected (binary ONNX model)."
    )
    parser.add_argument("--onnx", type=Path, required=True, help="Path to binary ONNX model (best.onnx).")
    parser.add_argument("--threshold", type=float, default=0.6, help="Drone probability threshold to trigger alert (0–1).")
    parser.add_argument("--alert-cooldown", type=float, default=10.0, help="Min seconds between alerts.")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Microphone sample rate (must match model).")
    parser.add_argument("--window", type=float, default=WINDOW_DURATION_SEC, help="Window length in seconds (model input, default 3 s).")
    parser.add_argument("--hop", type=float, default=HOP_DURATION_SEC, help="Hop between inferences in seconds (default 0.1 s).")
    parser.add_argument("--device", type=int, default=None, help="Input device ID (None = default microphone).")
    parser.add_argument("--quiet", action="store_true", help="Only print when drone detected (no per-window log).")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    window_samples = int(args.sample_rate * args.window)
    hop_samples = int(args.sample_rate * args.hop)
    if window_samples != WINDOW_SAMPLES:
        print(f"Warning: window has {window_samples} samples; model expects {WINDOW_SAMPLES} (3 s @ 16 kHz).", file=sys.stderr)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    last_alert_time = -1.0

    def process_window(window: np.ndarray) -> None:
        nonlocal last_alert_time
        window = window.astype(np.float32)
        if window.size != window_samples:
            window = np.resize(window, window_samples)
        window = peak_normalize(window)
        window = window.reshape(1, -1)

        logits = session.run(None, {input_name: window})[0][0]
        probs = softmax(logits)
        p_drone = float(probs[DRONE_CLASS_ID])
        pred = int(np.argmax(logits))

        now = time.monotonic()
        if not args.quiet:
            print(f"\r  p(drone)={p_drone:.3f}  pred={'drone' if pred == 1 else 'noise'}   ", end="", flush=True)

        if pred == 1 and p_drone >= args.threshold:
            if now - last_alert_time >= args.alert_cooldown:
                last_alert_time = now
                print(f"\n*** DRONE DETECTED (p={p_drone:.2f}) ***", flush=True)
                try:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

    print(f"Listening at {args.sample_rate} Hz, window={args.window}s, hop={args.hop}s, threshold={args.threshold}, cooldown={args.alert_cooldown}s.")
    print("Press Ctrl+C to stop.\n")

    block = min(256, hop_samples)
    stream = sd.InputStream(
        samplerate=args.sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=block,
        device=args.device,
    )
    buffer = np.array([], dtype=np.float32)
    hop_counter = 0
    stream.start()

    try:
        while True:
            data, _ = stream.read(block)
            new_samples = data.squeeze(axis=1)
            buffer = np.concatenate([buffer, new_samples])
            if len(buffer) > window_samples:
                buffer = buffer[-window_samples:]
            hop_counter += len(new_samples)

            while hop_counter >= hop_samples and len(buffer) >= window_samples:
                window = buffer[-window_samples:].copy()
                process_window(window)
                hop_counter -= hop_samples
                buffer = buffer[hop_samples:]

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
