from __future__ import annotations

import math
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn


class TorchAudioFrontend(nn.Module):
    """Pure-torch STFT/mel/MFCC frontend using librosa filter construction.

    This keeps the AUDRON branches driven by raw waveform tensors even though
    torchaudio is intentionally avoided here for portability in constrained environments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        n_mels: int = 64,
        n_mfcc: int = 13,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2.0

        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax,
        ).astype(np.float32)
        dct_basis = self._make_dct(n_mfcc, n_mels).astype(np.float32)
        self.register_buffer('window', torch.hann_window(self.win_length), persistent=False)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis), persistent=False)
        self.register_buffer('dct_basis', torch.from_numpy(dct_basis), persistent=False)

    @staticmethod
    def _make_dct(n_mfcc: int, n_mels: int) -> np.ndarray:
        basis = np.empty((n_mfcc, n_mels), dtype=np.float32)
        basis[0, :] = 1.0 / math.sqrt(n_mels)
        samples = np.arange(1, 2 * n_mels, 2, dtype=np.float32) * math.pi / (2.0 * n_mels)
        for i in range(1, n_mfcc):
            basis[i, :] = math.sqrt(2.0 / n_mels) * np.cos(i * samples)
        return basis

    def stft_power(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, T]
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        return power

    def mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self.stft_power(waveform)
        mel_basis = self.mel_basis.to(waveform.device)
        mel = torch.einsum('mf,bft->bmt', mel_basis, power)
        mel = torch.clamp(mel, min=1e-10)
        return mel

    def log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(waveform)
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))

    def mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        log_mel = self.log_mel_spectrogram(waveform)
        dct_basis = self.dct_basis.to(waveform.device)
        return torch.einsum('cm,bmt->bct', dct_basis, log_mel)
