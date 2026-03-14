from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, H]
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights
