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

from .data import (
    SequenceData,
    chronological_split,
    load_prices,
    load_sequence_dataset,
    make_sequences,
)
from .model import LSTMRegressor
from .training import fit_lstm, predict as predict_fitted, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a chronological FX forecasting experiment")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="NPZ created by scripts/01_prepare_dataset.py")
    source.add_argument("--csv", help="Read a CSV and create sequences in memory")
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

    if args.dataset:
        data, lookback, horizon = load_sequence_dataset(args.dataset)
    else:
        frame = load_prices(args.csv, args.timestamp_column, args.close_column)
        data = make_sequences(frame, args.lookback, args.horizon)
        lookback, horizon = args.lookback, args.horizon
    train, valid, test = chronological_split(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fitted = fit_lstm(
        train,
        valid,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )
    test_scaled = scale(test, fitted.mean, fitted.std)
    prediction = predict_fitted(
        fitted.model,
        test_scaled,
        device,
        fitted.target_mean,
        fitted.target_std,
    )
    random_walk = np.zeros_like(test.y)
    moving_average = test.x[:, -min(lookback, 12) :, 0].mean(axis=1) * horizon
    results = {
        "samples": {"train": len(train.y), "validation": len(valid.y), "test": len(test.y)},
        "configuration": {
            "lookback": lookback,
            "horizon": horizon,
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
    torch.save(fitted.model.state_dict(), output / "model.pt")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

