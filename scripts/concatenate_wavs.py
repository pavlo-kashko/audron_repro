#!/usr/bin/env python3
"""Concatenate all WAV files in each of two folders into one WAV per folder."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

SR = 22050  # common sample rate for concatenation

UNKNOWN_DIR = Path(
    "/Users/pavlo/Desktop/EDH/audron_repro/data/raw/DroneAudioDataset/Binary_Drone_Audio/unknown"
)
YES_DRONE_DIR = Path(
    "/Users/pavlo/Desktop/EDH/audron_repro/data/raw/DroneAudioDataset/Binary_Drone_Audio/yes_drone"
)
OUT_DIR = Path(
    "/Users/pavlo/Desktop/EDH/audron_repro/data/raw/DroneAudioDataset/Binary_Drone_Audio/concatenated"
)


def load_resampled(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        n = int(round(len(data) * target_sr / sr))
        data = resample(data, n).astype(np.float32)
    return data


def concatenate_folder(src_dir: Path, out_path: Path) -> None:
    wavs = sorted(src_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {src_dir}")
    chunks = []
    for p in wavs:
        chunks.append(load_resampled(p, SR))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate(chunks).astype(np.float32)
    sf.write(out_path, combined, SR)
    print(f"Wrote {out_path} ({len(wavs)} files, {combined.shape[0] / SR:.1f}s)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    concatenate_folder(UNKNOWN_DIR, OUT_DIR / "unknown.wav")
    concatenate_folder(YES_DRONE_DIR, OUT_DIR / "yes_drone.wav")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
