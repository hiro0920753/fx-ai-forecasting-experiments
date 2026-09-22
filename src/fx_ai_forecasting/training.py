from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SequenceData
from .model import LSTMRegressor


@dataclass(frozen=True)
class FittedModel:
    model: LSTMRegressor
    mean: np.ndarray
    std: np.ndarray
    target_mean: float
    target_std: float
    best_validation_mse: float
    epochs_run: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select(data: SequenceData, start: int, end: int) -> SequenceData:
    return SequenceData(
        x=data.x[start:end],
        y=data.y[start:end],
        current=data.current[start:end],
        timestamps=data.timestamps[start:end],
    )


def scale(data: SequenceData, mean: np.ndarray, std: np.ndarray) -> SequenceData:
    return SequenceData(
        x=((data.x - mean) / std).astype(np.float32),
        y=data.y,
        current=data.current,
        timestamps=data.timestamps,
    )


def make_loader(data: SequenceData, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(
    model: nn.Module,
    data: SequenceData,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in make_loader(data, 2048, False):
            values.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(values) * target_std + target_mean


def fit_lstm(
    train: SequenceData,
    valid: SequenceData,
    *,
    hidden_size: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
) -> FittedModel:
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train.x.mean(axis=(0, 1), keepdims=True)
    std = train.x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = scale(train, mean, std)
    valid_scaled = scale(valid, mean, std)

    target_mean = float(train.y.mean())
    target_std = float(train.y.std())
    if target_std < 1e-8:
        target_std = 1.0
    train_target = ((train.y - target_mean) / target_std).astype(np.float32)
    valid_target = ((valid.y - target_mean) / target_std).astype(np.float32)
    train_scaled = SequenceData(
        train_scaled.x, train_target, train_scaled.current, train_scaled.timestamps
    )
    valid_scaled = SequenceData(
        valid_scaled.x, valid_target, valid_scaled.current, valid_scaled.timestamps
    )

    model = LSTMRegressor(hidden_size=hidden_size, input_size=train.x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_run = 0

    for epoch in range(epochs):
        model.train()
        for x, y in make_loader(train_scaled, batch_size, True):
            optimizer.zero_grad()
            loss = loss_function(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        validation_prediction = predict(model, valid_scaled, device)
        validation_loss = float(np.mean((validation_prediction - valid_target) ** 2))
        epochs_run = epoch + 1
        print(f"epoch={epochs_run} validation_mse={validation_loss:.8f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    return FittedModel(model, mean, std, target_mean, target_std, best_loss, epochs_run)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    active = (actual != 0) | (predicted != 0)
    direction = float(np.mean(np.sign(actual[active]) == np.sign(predicted[active])))
    return {
        "rmse_log_return": float(np.sqrt(np.mean(error**2))),
        "mae_log_return": float(np.mean(np.abs(error))),
        "directional_accuracy": direction,
    }

