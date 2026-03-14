from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from audron.utils.io import ensure_dir


@dataclass
class DroneSynthParams:
    name: str
    base_freq_hz: float
    num_harmonics: int
    amplitude_decay: float
    noise_std: float
    wind_scale: float


DEFAULT_SYNTH_CLASSES = [
    DroneSynthParams('quadcopter', 75.0, 8, 0.82, 0.020, 0.040),
    DroneSynthParams('hexacopter', 65.0, 10, 0.84, 0.022, 0.042),
    DroneSynthParams('octocopter', 55.0, 12, 0.86, 0.025, 0.045),
    DroneSynthParams('racing_drone', 120.0, 10, 0.80, 0.030, 0.038),
]


def generate_drone_waveform(
    params: DroneSynthParams,
    sample_rate: int = 16000,
    duration_sec: float = 3.0,
    modulation_alpha: float = 0.15,
    modulation_freq_hz: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate

    harmonic = np.zeros_like(t)
    phase = rng.uniform(0, 2 * math.pi)
    for k in range(1, params.num_harmonics + 1):
        amp = (params.amplitude_decay ** (k - 1)) * rng.uniform(0.85, 1.15)
        freq = params.base_freq_hz * k * rng.uniform(0.985, 1.015)
        harmonic += amp * np.sin(2 * math.pi * freq * t + rng.uniform(0, 2 * math.pi))

    modulation = 1.0 + modulation_alpha * np.sin(2 * math.pi * modulation_freq_hz * t + phase)
    gaussian_noise = rng.normal(0.0, params.noise_std, size=t.shape).astype(np.float32)

    wind_freq = rng.uniform(0.08, 0.35)
    wind = params.wind_scale * np.sin(2 * math.pi * wind_freq * t + rng.uniform(0, 2 * math.pi))
    wind += (params.wind_scale * 0.3) * np.sin(2 * math.pi * wind_freq * 2.3 * t + rng.uniform(0, 2 * math.pi))

    waveform = harmonic * modulation + gaussian_noise + wind
    peak = max(np.max(np.abs(waveform)), 1e-8)
    waveform = (waveform / peak).astype(np.float32)
    return waveform


def synth_dataset(
    out_dir: str | Path,
    train_per_class: int = 500,#65,
    val_per_class: int = 20,
    sample_rate: int = 16000,
    duration_sec: float = 3.0,
    seed: int = 42,
    classes: Iterable[DroneSynthParams] = DEFAULT_SYNTH_CLASSES,
) -> dict[str, list[dict]]:
    out_dir = ensure_dir(out_dir)
    train_rows, val_rows = [], []
    idx = 0
    for label_id, params in enumerate(classes):
        class_dir = ensure_dir(out_dir / params.name)
        total = train_per_class + val_per_class
        for j in range(total):
            waveform = generate_drone_waveform(
                params=params,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
                seed=seed + idx,
            )
            file_path = class_dir / f'{params.name}_{j:04d}.wav'
            sf.write(file_path, waveform, sample_rate)
            row = {'path': str(file_path.resolve()), 'label_name': params.name, 'label_id': label_id}
            if j < train_per_class:
                train_rows.append(row)
            else:
                val_rows.append(row)
            idx += 1
    return {'train': train_rows, 'val': val_rows}
