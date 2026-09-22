from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1]).squeeze(-1)

