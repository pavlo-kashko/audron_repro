from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from audron.models.attention import TemporalAttention
from audron.models.audio_frontend import TorchAudioFrontend


class MFCCExtractor(nn.Module):
    def __init__(self, frontend: TorchAudioFrontend, feature_dim: int = 128) -> None:
        super().__init__()
        self.frontend = frontend
        self.conv1 = nn.Conv1d(frontend.n_mfcc, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, feature_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(64)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.frontend.mfcc(waveform)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = x.mean(dim=-1)
        return x


class STFTCNNExtractor(nn.Module):
    def __init__(self, frontend: TorchAudioFrontend, out_dim: int = 256) -> None:
        super().__init__()
        self.frontend = frontend
        channels = [1, 32, 64, 128, 256]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ])
        self.conv = nn.Sequential(*blocks)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.frontend.log_mel_spectrogram(waveform).unsqueeze(1)
        x = self.conv(spec)
        x = self.adapt(x)
        x = self.head(x)
        return x


class RNNExtractor(nn.Module):
    def __init__(self, frontend: TorchAudioFrontend, n_mels: int = 64, hidden_size: int = 128, out_dim: int = 192) -> None:
        super().__init__()
        self.frontend = frontend
        self.rnn = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = TemporalAttention(hidden_size * 2)
        self.head = nn.Linear(hidden_size * 2, out_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.frontend.log_mel_spectrogram(waveform).transpose(1, 2)
        seq, _ = self.rnn(mel)
        context, _ = self.attn(seq)
        return self.head(context)


class AudioAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 48000, encoder_dims: Sequence[int] = (2048, 1024, 512, 160), dropout: float = 0.2) -> None:
        super().__init__()
        dims = [input_dim, *encoder_dims]
        enc_layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            enc_layers.append(nn.Linear(d_in, d_out))
            if d_out != encoder_dims[-1]:
                enc_layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [encoder_dims[-1], encoder_dims[-2], encoder_dims[-3], encoder_dims[-4], input_dim]
        dec_layers = []
        for i, (d_in, d_out) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            dec_layers.append(nn.Linear(d_in, d_out))
            if i < len(dec_dims) - 2:
                dec_layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.decoder = nn.Sequential(*dec_layers)
        self.embedding_dim = encoder_dims[-1]

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(waveform)
        recon = self.decoder(z)
        return z, recon
