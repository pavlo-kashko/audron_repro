"""
State-of-the-art audio augmentations for robust drone vs noise classification.

- Additive Gaussian noise (SNR-based)
- Random gain (volume)
- Background noise mixing (environmental noise at random SNR)
- Time masking (zero or replace segment)
- Time stretch (then pad/trim to fixed length)
- Mixup (same-class: convex mix of two samples, same label)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np

from audron.data.audio import fit_audio_length, load_audio


def _rng() -> np.random.Generator:
    return np.random.default_rng()


def add_gaussian_noise(
    waveform: np.ndarray,
    snr_db_range: tuple[float, float] = (10.0, 30.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add white Gaussian noise at random SNR (dB). Higher SNR = less noise."""
    rng = rng or _rng()
    snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    signal_power = np.mean(waveform.astype(np.float64) ** 2)
    if signal_power < 1e-12:
        return waveform.astype(np.float32)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def random_gain(
    waveform: np.ndarray,
    gain_range: tuple[float, float] = (0.7, 1.3),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Multiply by random gain (volume augmentation)."""
    rng = rng or _rng()
    gain = float(rng.uniform(gain_range[0], gain_range[1]))
    return (waveform * gain).astype(np.float32)


def mix_background_noise(
    waveform: np.ndarray,
    noise_paths: list[Path],
    sample_rate: int,
    snr_db_range: tuple[float, float] = (0.0, 15.0),
    target_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix in a random background noise clip at random SNR. Noise is cropped or repeated to match length."""
    if not noise_paths:
        return waveform.astype(np.float32)
    rng = rng or _rng()
    path = Path(rng.choice(noise_paths))
    if not path.exists():
        return waveform.astype(np.float32)
    noise, _ = librosa.load(path, sr=sample_rate, mono=True)
    noise = noise.astype(np.float32)
    length = len(waveform)
    if target_length is not None:
        length = target_length
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    noise = noise[:length]
    if len(noise) < len(waveform):
        noise = np.pad(noise, (0, len(waveform) - len(noise)), mode="wrap")
    else:
        noise = noise[: len(waveform)]
    snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))
    signal_power = np.mean(waveform.astype(np.float64) ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-12) * np.sqrt(noise_power)
    return (waveform + noise).astype(np.float32)


def time_mask(
    waveform: np.ndarray,
    max_ratio: float = 0.15,
    fill: str = "zero",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero out or replace with noise a random contiguous segment (time masking, SpecAugment-style in time domain)."""
    rng = rng or _rng()
    length = len(waveform)
    mask_len = int(length * float(rng.uniform(0.02, max_ratio)))
    if mask_len < 2:
        return waveform.astype(np.float32)
    start = rng.integers(0, max(1, length - mask_len))
    out = waveform.astype(np.float32).copy()
    if fill == "zero":
        out[start : start + mask_len] = 0.0
    else:
        out[start : start + mask_len] = rng.normal(0, 0.01, size=mask_len).astype(np.float32)
    return out


def time_stretch(
    waveform: np.ndarray,
    rate_range: tuple[float, float] = (0.9, 1.1),
    target_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Time stretch (librosa); then trim/pad to target_length if given."""
    rng = rng or _rng()
    rate = float(rng.uniform(rate_range[0], rate_range[1]))
    stretched = librosa.effects.time_stretch(waveform.astype(np.float64), rate=rate)
    stretched = stretched.astype(np.float32)
    if target_length is not None:
        stretched = fit_audio_length(stretched, target_length, pad_mode="repeat")
    return stretched


def mixup_same_class(
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convex combination of two waveforms (same length). λ ~ Beta(alpha, alpha). Use for same-class pairs."""
    rng = rng or _rng()
    lam = float(rng.beta(alpha, alpha))
    lam = np.clip(lam, 0.2, 0.8)
    length = min(len(waveform_a), len(waveform_b))
    a = waveform_a[:length].astype(np.float32)
    b = waveform_b[:length].astype(np.float32)
    return (lam * a + (1 - lam) * b).astype(np.float32)


def apply_augmentation_pipeline(
    waveform: np.ndarray,
    data_cfg: dict[str, Any],
    target_length: int,
    sample_rate: int,
    noise_paths: list[Path] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a random subset of enabled augmentations according to data config. Order: gain, gaussian noise, background noise, time stretch, time mask."""
    rng = rng or _rng()
    w = waveform.astype(np.float32)

    aug = data_cfg.get("augmentation", {}) or {}
    if not aug.get("train", True):
        return w

    if aug.get("gain", {}).get("enabled", False):
        gr = aug["gain"].get("range", [0.7, 1.3])
        w = random_gain(w, (float(gr[0]), float(gr[1])), rng)

    if aug.get("gaussian_noise", {}).get("enabled", False):
        snr = aug["gaussian_noise"].get("snr_db", [10, 30])
        w = add_gaussian_noise(w, (float(snr[0]), float(snr[1])), rng)

    if aug.get("background_noise", {}).get("enabled", False) and noise_paths:
        snr = aug["background_noise"].get("snr_db", [0, 15])
        if rng.random() < float(aug["background_noise"].get("prob", 0.5)):
            w = mix_background_noise(w, noise_paths, sample_rate, (float(snr[0]), float(snr[1])), target_length, rng)

    if aug.get("time_stretch", {}).get("enabled", False):
        if rng.random() < float(aug["time_stretch"].get("prob", 0.3)):
            rate = aug["time_stretch"].get("rate", [0.9, 1.1])
            w = time_stretch(w, (float(rate[0]), float(rate[1])), target_length, rng)

    if aug.get("time_mask", {}).get("enabled", False):
        if rng.random() < float(aug["time_mask"].get("prob", 0.3)):
            max_ratio = float(aug["time_mask"].get("max_ratio", 0.15))
            w = time_mask(w, max_ratio, "zero", rng)

    return w


def collect_noise_paths(data_cfg: dict[str, Any], base_dir: Path | None = None) -> list[Path]:
    """Collect list of paths for background noise from data config (augmentation.background_noise.dir or .paths)."""
    aug = data_cfg.get("augmentation", {}) or {}
    bn = aug.get("background_noise", {}) or {}
    if not bn.get("enabled"):
        return []
    paths: list[Path] = []
    if "dir" in bn:
        d = Path(bn["dir"])
        if not d.is_absolute() and base_dir is not None:
            d = base_dir / d
        if d.is_dir():
            paths.extend(d.rglob("*.wav"))
    if "paths" in bn:
        for p in bn["paths"]:
            p = Path(p)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            if p.is_file():
                paths.append(p)
    return list(set(paths))
