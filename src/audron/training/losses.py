from __future__ import annotations

import torch
import torch.nn as nn


class AudronLoss(nn.Module):
    def __init__(self, reconstruction_weight: float = 0.1) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.reconstruction_weight = reconstruction_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, reconstruction: torch.Tensor, waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        cls = self.cross_entropy(logits, targets)
        recon = self.mse(reconstruction, waveform)
        total = cls + self.reconstruction_weight * recon
        return total, {
            'loss': float(total.detach().cpu()),
            'classification_loss': float(cls.detach().cpu()),
            'reconstruction_loss': float(recon.detach().cpu()),
        }
