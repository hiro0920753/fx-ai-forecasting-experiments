"""Run a concrete expanding walk-forward evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fx_ai_forecasting.data import SequenceData, load_prices, make_sequences
from fx_ai_forecasting.training import fit_lstm, predict, regression_metrics, scale, select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/USDJPY_M5.csv")
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--test-rows", type=int, default=2000)
    parser.add_argument("--step-rows", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/walk_forward")
    return parser.parse_args()


def split_train_valid(data: SequenceData) -> tuple[SequenceData, SequenceData]:
    valid_count = max(1, int(len(data.y) * 0.15))
    if len(data.y) - valid_count < 1:
        raise ValueError("Not enough training samples")
    return select(data, 0, len(data.y) - valid_count), select(data, len(data.y) - valid_count, len(data.y))


def main() -> None:
    args = parse_args()
    frame = load_prices(args.csv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in range(args.folds):
        train_end = args.train_rows + fold * args.step_rows
        test_end = train_end + args.test_rows
        if test_end > len(frame):
            print(f"stop: fold={fold + 1} needs {test_end} rows but CSV has {len(frame)}")
            break

        training_sequences = make_sequences(
            frame.iloc[:train_end].reset_index(drop=True), args.lookback, args.horizon
        )
        train, valid = split_train_valid(training_sequences)
        test_frame = frame.iloc[train_end - args.lookback:test_end].reset_index(drop=True)
        test = make_sequences(test_frame, args.lookback, args.horizon)
        fitted = fit_lstm(
            train,
            valid,
            epochs=args.epochs,
            seed=args.seed + fold,
            device=device,
        )
        test_scaled = scale(test, fitted.mean, fitted.std)
        prediction = predict(fitted.model, test_scaled, device)
        fold_metrics = regression_metrics(test.y, prediction)
        summaries.append(
            {
                "fold": fold + 1,
                "train_start": str(frame["timestamp"].iloc[0]),
                "train_end": str(frame["timestamp"].iloc[train_end - 1]),
                "test_start": str(test.timestamps[0]),
                "test_end": str(test.timestamps[-1]),
                "train_samples": len(train.y),
                "validation_samples": len(valid.y),
                "test_samples": len(test.y),
                **fold_metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold + 1,
                    "timestamp": test.timestamps,
                    "actual_log_return": test.y,
                    "predicted_log_return": prediction,
                }
            )
        )
        print(f"fold={fold + 1} metrics={fold_metrics}")

    if not summaries:
        raise RuntimeError("No fold could be evaluated. Reduce train/test rows or use more data.")
    summary = pd.DataFrame(summaries)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary.to_csv(output / "fold_summary.csv", index=False)
    predictions.to_csv(output / "oos_predictions.csv", index=False)
    aggregate = {
        "folds": len(summary),
        "mean_directional_accuracy": float(summary["directional_accuracy"].mean()),
        "mean_rmse_log_return": float(summary["rmse_log_return"].mean()),
        "total_oos_samples": int(summary["test_samples"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

