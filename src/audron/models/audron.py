from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from audron.models.audio_frontend import TorchAudioFrontend
from audron.models.branches import AudioAutoencoder, MFCCExtractor, RNNExtractor, STFTCNNExtractor


@dataclass
class AudronOutputs:
    logits: torch.Tensor
    features: Dict[str, torch.Tensor]
    reconstruction: torch.Tensor


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Audron(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        data_cfg = cfg['data']
        model_cfg = cfg['model']
        self.enabled_branches: List[str] = model_cfg.get('enabled_branches', ['mfcc', 'stft', 'rnn', 'autoencoder'])
        self.input_samples = int(data_cfg['sample_rate'] * data_cfg['clip_duration_sec'])
        self.frontend = TorchAudioFrontend(
            sample_rate=data_cfg['sample_rate'],
            n_fft=model_cfg['frontend']['n_fft'],
            hop_length=model_cfg['frontend']['hop_length'],
            win_length=model_cfg['frontend']['win_length'],
            n_mels=model_cfg['frontend']['n_mels'],
            n_mfcc=model_cfg['frontend']['n_mfcc'],
            fmin=model_cfg['frontend'].get('fmin', 0.0),
            fmax=model_cfg['frontend'].get('fmax'),
        )
        self.mfcc_branch = MFCCExtractor(self.frontend, feature_dim=model_cfg['mfcc_dim'])
        self.stft_branch = STFTCNNExtractor(self.frontend, out_dim=model_cfg['stft_dim'])
        self.rnn_branch = RNNExtractor(self.frontend, n_mels=model_cfg['frontend']['n_mels'], hidden_size=model_cfg['rnn_hidden'], out_dim=model_cfg['rnn_dim'])
        self.autoencoder_branch = AudioAutoencoder(
            input_dim=self.input_samples,
            encoder_dims=tuple(model_cfg['autoencoder_dims']),
            dropout=model_cfg['autoencoder_dropout'],
        )

        fusion_in = 0
        if 'mfcc' in self.enabled_branches:
            fusion_in += model_cfg['mfcc_dim']
        if 'stft' in self.enabled_branches:
            fusion_in += model_cfg['stft_dim']
        if 'rnn' in self.enabled_branches:
            fusion_in += model_cfg['rnn_dim']
        if 'autoencoder' in self.enabled_branches:
            fusion_in += model_cfg['autoencoder_dims'][-1]

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = ClassificationHead(256, cfg['task']['num_classes'])

    def forward(self, waveform: torch.Tensor) -> AudronOutputs:
        features: Dict[str, torch.Tensor] = {}
        recon = waveform
        if 'mfcc' in self.enabled_branches:
            features['mfcc'] = self.mfcc_branch(waveform)
        if 'stft' in self.enabled_branches:
            features['stft'] = self.stft_branch(waveform)
        if 'rnn' in self.enabled_branches:
            features['rnn'] = self.rnn_branch(waveform)
        if 'autoencoder' in self.enabled_branches:
            z, recon = self.autoencoder_branch(waveform)
            features['autoencoder'] = z
        fused = torch.cat([features[name] for name in ['mfcc', 'stft', 'rnn', 'autoencoder'] if name in features], dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        return AudronOutputs(logits=logits, features=features, reconstruction=recon)
