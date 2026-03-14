from __future__ import annotations

from pathlib import Path
from typing import Literal

import librosa
import numpy as np


PadMode = Literal['zero', 'repeat']


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def fit_audio_length(waveform: np.ndarray, target_length: int, pad_mode: PadMode = 'repeat') -> np.ndarray:
    if len(waveform) == target_length:
        return waveform.astype(np.float32)
    if len(waveform) > target_length:
        return waveform[:target_length].astype(np.float32)
    if len(waveform) == 0:
        return np.zeros(target_length, dtype=np.float32)
    if pad_mode == 'repeat':
        reps = int(np.ceil(target_length / len(waveform)))
        waveform = np.tile(waveform, reps)
        return waveform[:target_length].astype(np.float32)
    out = np.zeros(target_length, dtype=np.float32)
    out[: len(waveform)] = waveform
    return out


def peak_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(waveform)))
    if peak < eps:
        return waveform.astype(np.float32)
    return (waveform / peak).astype(np.float32)
