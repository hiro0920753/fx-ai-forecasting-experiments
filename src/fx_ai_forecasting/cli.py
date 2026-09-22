from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SequenceData, chronological_split, load_prices, make_sequences
from .model import LSTMRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a chronological FX forecasting experiment")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--timestamp-column")
    parser.add_argument("--close-column")
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/baseline")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(data: SequenceData, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, data: SequenceData, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader(data, 2048, False):
            predictions.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(predictions)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    nonzero = (actual != 0) | (predicted != 0)
    direction = float(np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero])))
    return {
        "rmse_log_return": float(np.sqrt(np.mean(error**2))),
        "mae_log_return": float(np.mean(np.abs(error))),
        "directional_accuracy": direction,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_prices(args.csv, args.timestamp_column, args.close_column)
    data = make_sequences(frame, args.lookback, args.horizon)
    train, valid, test = chronological_split(data)

    mean = train.x.mean(axis=(0, 1), keepdims=True)
    std = train.x.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def scaled(part: SequenceData) -> SequenceData:
        return SequenceData(
            x=((part.x - mean) / std).astype(np.float32),
            y=part.y,
            current=part.current,
            timestamps=part.timestamps,
        )

    train_scaled, valid_scaled, test_scaled = map(scaled, (train, valid, test))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in loader(train_scaled, args.batch_size, True):
            optimizer.zero_grad()
            loss = loss_function(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        valid_prediction = predict(model, valid_scaled, device)
        valid_loss = float(np.mean((valid_prediction - valid.y) ** 2))
        print(f"epoch={epoch + 1} validation_mse={valid_loss:.8f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    prediction = predict(model, test_scaled, device)
    random_walk = np.zeros_like(test.y)
    moving_average = test.x[:, -min(args.lookback, 12) :, 0].mean(axis=1) * args.horizon
    results = {
        "samples": {"train": len(train.y), "validation": len(valid.y), "test": len(test.y)},
        "configuration": {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "seed": args.seed,
            "device": str(device),
        },
        "lstm": metrics(test.y, prediction),
        "random_walk": metrics(test.y, random_walk),
        "moving_average_return": metrics(test.y, moving_average),
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "timestamp": test.timestamps,
            "actual_log_return": test.y,
            "predicted_log_return": prediction,
            "random_walk_log_return": random_walk,
            "moving_average_log_return": moving_average,
        }
    ).to_csv(output / "predictions.csv", index=False)
    torch.save(best_state, output / "model.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

