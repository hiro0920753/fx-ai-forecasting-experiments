"""Re-run the complete USDJPY forecasting study from raw Dukascopy M5 CSV files.

The script deliberately keeps 2025 onward untouched until a configuration has
been selected using 2023-2024. It writes every table, prediction, chart and an
Excel workbook used by the accompanying reproducibility guide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fx_ai_forecasting.data import SequenceData
from fx_ai_forecasting.training import fit_lstm, predict, regression_metrics, scale


@dataclass(frozen=True)
class Condition:
    family: str
    name: str
    timeframe: int = 5
    lookback_hours: int = 4
    horizon_minutes: int = 60
    features: tuple[str, ...] = ("return",)


FEATURE_SETS = {
    "price_only": ("return",),
    "price_rsi": ("return", "rsi"),
    "price_volatility": ("return", "volatility"),
    "price_macd_momentum": ("return", "macd", "momentum"),
    "all_basic": ("return", "rsi", "volatility", "macd", "momentum"),
    "all_acceleration": (
        "return", "rsi", "volatility", "macd", "momentum", "acceleration"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing YYYY_MM.csv")
    parser.add_argument("--output", default="artifacts/full_study")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--quick", action="store_true", help="One seed and fewer epochs")
    return parser.parse_args()


def load_monthly(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No monthly CSV files under {directory}")
    digest = hashlib.sha256()
    frames = []
    for path in files:
        digest.update(path.read_bytes())
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    required = {"time", "bid", "ask", "open", "high", "low", "close", "spread"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    numeric = ["bid", "ask", "open", "high", "low", "close", "spread"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    frame.attrs["sha256"] = digest.hexdigest()
    frame.attrs["files"] = len(files)
    return frame.reset_index(drop=True)


def resample(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 5:
        return frame.copy()
    indexed = frame.set_index("timestamp")
    result = indexed.resample(f"{minutes}min", label="left", closed="left").agg(
        time=("time", "first"), bid=("bid", "first"), ask=("ask", "first"),
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), spread=("spread", "median")
    ).dropna()
    return result.reset_index()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    log_close = np.log(out["close"])
    out["return"] = log_close.diff()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["volatility"] = out["return"].rolling(24).std()
    out["macd"] = log_close.ewm(span=12, adjust=False).mean() - log_close.ewm(
        span=26, adjust=False
    ).mean()
    out["momentum"] = log_close.diff(12)
    out["acceleration"] = out["return"].diff()
    return out.replace([np.inf, -np.inf], np.nan)


def make_data(frame: pd.DataFrame, condition: Condition) -> tuple[SequenceData, pd.DataFrame]:
    lookback = condition.lookback_hours * 60 // condition.timeframe
    horizon = condition.horizon_minutes // condition.timeframe
    if horizon < 1:
        raise ValueError("Horizon must be at least one bar")
    values = frame[list(condition.features)].to_numpy(dtype=np.float32)
    close = frame["close"].to_numpy(dtype=np.float64)
    # One additional bar is reserved for executable bid/ask prices. The model
    # observes through ``end - 1``; a trade can only enter at the next bar.
    count = len(frame) - lookback - horizon
    x = np.empty((count, lookback, len(condition.features)), np.float32)
    y = np.empty(count, np.float32)
    current = np.empty(count, np.float64)
    timestamps = np.empty(count, dtype="datetime64[ns]")
    meta = []
    for i in range(count):
        entry = i + lookback - 1
        target = entry + horizon
        entry_execution = entry + 1
        exit_execution = target + 1
        x[i] = values[i : entry + 1]
        y[i] = np.log(close[target] / close[entry])
        current[i] = close[entry]
        timestamps[i] = frame["timestamp"].iloc[target].tz_localize(None).to_datetime64()
        meta.append((
            frame["timestamp"].iloc[entry_execution], frame["timestamp"].iloc[exit_execution],
            frame["bid"].iloc[entry_execution], frame["ask"].iloc[entry_execution],
            frame["bid"].iloc[exit_execution], frame["ask"].iloc[exit_execution],
        ))
    valid = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y)
    data = SequenceData(x[valid], y[valid], current[valid], timestamps[valid])
    metadata = pd.DataFrame(
        np.asarray(meta, dtype=object)[valid],
        columns=["entry_time", "exit_time", "entry_bid", "entry_ask", "exit_bid", "exit_ask"],
    )
    return data, metadata.reset_index(drop=True)


def subset(data: SequenceData, mask: np.ndarray) -> SequenceData:
    return SequenceData(data.x[mask], data.y[mask], data.current[mask], data.timestamps[mask])


def split_data(data: SequenceData, holdout_start: str) -> tuple[SequenceData, SequenceData, SequenceData, np.ndarray, np.ndarray, np.ndarray]:
    times = pd.to_datetime(data.timestamps, utc=True)
    development = times < pd.Timestamp(holdout_start, tz="UTC")
    dev_indices = np.flatnonzero(development)
    validation_start = dev_indices[int(len(dev_indices) * 0.80)]
    train_mask = np.arange(len(times)) < validation_start
    valid_mask = (np.arange(len(times)) >= validation_start) & development
    test_mask = ~development
    return subset(data, train_mask), subset(data, valid_mask), subset(data, test_mask), train_mask, valid_mask, test_mask


def choose_threshold(actual: np.ndarray, predicted: np.ndarray) -> float:
    candidates = np.quantile(np.abs(predicted), [0.0, 0.25, 0.50, 0.75, 0.90])
    scores = []
    for threshold in candidates:
        active = np.abs(predicted) >= threshold
        score = float(np.sum(np.sign(predicted[active]) * actual[active])) if active.any() else -np.inf
        scores.append(score)
    return float(candidates[int(np.argmax(scores))])


def trade_metrics(meta: pd.DataFrame, prediction: np.ndarray, threshold: float) -> tuple[dict[str, float], pd.DataFrame]:
    direction = np.where(prediction > threshold, 1, np.where(prediction < -threshold, -1, 0))
    long_pips = (meta["exit_bid"].astype(float) - meta["entry_ask"].astype(float)) * 100
    short_pips = (meta["entry_bid"].astype(float) - meta["exit_ask"].astype(float)) * 100
    pips = np.where(direction > 0, long_pips, np.where(direction < 0, short_pips, 0.0))
    trades = pd.DataFrame({"exit_time": meta["exit_time"], "direction": direction, "pips": pips})
    trades = trades[trades["direction"] != 0].reset_index(drop=True)
    equity = trades["pips"].cumsum()
    drawdown = equity - equity.cummax()
    wins = trades.loc[trades["pips"] > 0, "pips"]
    losses = trades.loc[trades["pips"] < 0, "pips"]
    gross_loss = abs(float(losses.sum()))
    metrics = {
        "trades": int(len(trades)), "total_pips": float(trades["pips"].sum()),
        "win_rate": float((trades["pips"] > 0).mean()) if len(trades) else 0.0,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss else np.nan,
        "max_drawdown_pips": float(drawdown.min()) if len(drawdown) else 0.0,
        "average_pips": float(trades["pips"].mean()) if len(trades) else 0.0,
    }
    trades["equity_pips"] = equity
    return metrics, trades


def conditions() -> list[Condition]:
    items = [Condition("baseline", "M5_4h_60m")]
    items += [Condition("feature", name, features=value) for name, value in FEATURE_SETS.items()]
    items += [Condition("horizon", f"horizon_{minutes}m", horizon_minutes=minutes) for minutes in (30, 60, 90, 120)]
    items += [Condition("lookback", f"lookback_{hours}h", lookback_hours=hours) for hours in (1, 2, 4, 8, 12, 24)]
    items += [Condition("timeframe", "M5", timeframe=5), Condition("timeframe", "M15", timeframe=15)]
    return items


def draw_outputs(output: Path, results: pd.DataFrame, holdout_predictions: pd.DataFrame, trades: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for family, name, ylabel in [
        ("lookback", "lookback_comparison.png", "Holdout RMSE"),
        ("horizon", "horizon_comparison.png", "Holdout RMSE"),
        ("feature", "feature_comparison.png", "Holdout RMSE"),
    ]:
        part = results[results["family"] == family]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(part["name"], part["rmse_log_return"])
        ax.set(ylabel=ylabel, title=f"{family.title()} comparison")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout(); fig.savefig(output / name, dpi=180); plt.close(fig)
    sample = holdout_predictions.head(500)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sample["exit_time"], sample["actual_pips"], label="Actual", alpha=.75)
    ax.plot(sample["exit_time"], sample["predicted_pips"], label="LSTM", alpha=.75)
    ax.set(title="USDJPY: predicted and actual 60-minute move", ylabel="pips")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "prediction_example.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(trades["exit_time"], trades["equity_pips"])
    ax.set(title="Untouched holdout: cumulative pips after bid/ask cost", ylabel="pips")
    fig.tight_layout(); fig.savefig(output / "holdout_equity.png", dpi=180); plt.close(fig)


def run_walk_forward(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding quarterly OOS folds after the simple holdout experiment."""
    condition = Condition("wfo", "expanding_quarterly")
    data, meta = make_data(frame, condition)
    times = pd.to_datetime(data.timestamps, utc=True)
    fold_rows, all_trades = [], []
    for test_start in pd.date_range("2024-01-01", times.max(), freq="QS", tz="UTC"):
        test_end = test_start + pd.DateOffset(months=3)
        development = times < test_start
        testing = (times >= test_start) & (times < test_end)
        dev_indices = np.flatnonzero(development)
        if len(dev_indices) < 5000 or testing.sum() == 0:
            continue
        split = dev_indices[int(len(dev_indices) * 0.85)]
        train_mask = np.arange(len(times)) < split
        valid_mask = (np.arange(len(times)) >= split) & development
        train, valid, test = subset(data, train_mask), subset(data, valid_mask), subset(data, testing)
        fitted = fit_lstm(
            train, valid, epochs=args.epochs, patience=args.patience, seed=seed, device=device
        )
        valid_prediction = predict(
            fitted.model, scale(valid, fitted.mean, fitted.std), device,
            fitted.target_mean, fitted.target_std,
        )
        test_prediction = predict(
            fitted.model, scale(test, fitted.mean, fitted.std), device,
            fitted.target_mean, fitted.target_std,
        )
        threshold = choose_threshold(valid.y, valid_prediction)
        trade_summary, trades = trade_metrics(meta.loc[testing].reset_index(drop=True), test_prediction, threshold)
        fold_rows.append({
            "fold": len(fold_rows) + 1, "train_start": times[train_mask].min(),
            "train_end": times[valid_mask].max(), "test_start": test_start,
            "test_end": min(test_end, times.max()), **regression_metrics(test.y, test_prediction),
            **trade_summary, "threshold": threshold,
        })
        trades["fold"] = len(fold_rows)
        all_trades.append(trades)
    return pd.DataFrame(fold_rows), pd.concat(all_trades, ignore_index=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.quick:
        seeds, args.epochs = seeds[:1], min(args.epochs, 3)
    raw = load_monthly(Path(args.data_dir))
    frames = {minutes: add_features(resample(raw, minutes)) for minutes in (5, 15)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, prediction_tables = [], []
    baseline_trades = None
    baseline_predictions = None

    for condition in conditions():
        data, meta = make_data(frames[condition.timeframe], condition)
        train, valid, test, train_mask, valid_mask, test_mask = split_data(data, args.holdout_start)
        seed_metrics, seed_predictions, seed_valid_predictions = [], [], []
        for seed in seeds:
            fitted = fit_lstm(train, valid, epochs=args.epochs, patience=args.patience, seed=seed, device=device)
            valid_prediction = predict(fitted.model, scale(valid, fitted.mean, fitted.std), device, fitted.target_mean, fitted.target_std)
            test_prediction = predict(fitted.model, scale(test, fitted.mean, fitted.std), device, fitted.target_mean, fitted.target_std)
            seed_metrics.append(regression_metrics(test.y, test_prediction))
            seed_predictions.append(test_prediction)
            seed_valid_predictions.append(valid_prediction)
        prediction = np.mean(seed_predictions, axis=0)
        metrics = regression_metrics(test.y, prediction)
        valid_average = np.mean(seed_valid_predictions, axis=0)
        threshold = choose_threshold(valid.y, valid_average)
        test_meta = meta.loc[test_mask].reset_index(drop=True)
        trade_summary, trade_table = trade_metrics(test_meta, prediction, threshold)
        rows.append({**asdict(condition), **metrics, **trade_summary, "threshold": threshold, "test_samples": len(test.y)})
        if condition.family == "baseline":
            for baseline_name, baseline_prediction in (
                ("zero_change", np.zeros_like(test.y)),
                ("last_return", test.x[:, -1, 0]),
                ("moving_average_12", test.x[:, -min(12, test.x.shape[1]) :, 0].mean(axis=1)),
            ):
                baseline_metrics = regression_metrics(test.y, baseline_prediction)
                baseline_trade_summary, _ = trade_metrics(
                    test_meta, baseline_prediction, choose_threshold(valid.y, np.zeros_like(valid.y))
                )
                rows.append({
                    **asdict(condition), "family": "baseline_model", "name": baseline_name,
                    **baseline_metrics, **baseline_trade_summary, "threshold": 0.0,
                    "test_samples": len(test.y),
                })
        table = test_meta.copy()
        table["condition"] = condition.name
        table["actual_log_return"] = test.y
        table["predicted_log_return"] = prediction
        prediction_tables.append(table)
        if condition.family == "baseline":
            baseline_trades = trade_table
            baseline_predictions = table.copy()
            baseline_predictions["actual_pips"] = test.y * test.current * 100
            baseline_predictions["predicted_pips"] = prediction * test.current * 100

    results = pd.DataFrame(rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    results.to_csv(output / "condition_summary.csv", index=False)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    baseline_trades.to_csv(output / "holdout_trades.csv", index=False)
    draw_outputs(output, results, baseline_predictions, baseline_trades)
    wfo_summary, wfo_trades = run_walk_forward(frames[5], args, device, seeds[0])
    wfo_summary.to_csv(output / "wfo_fold_summary.csv", index=False)
    wfo_trades.to_csv(output / "wfo_trades.csv", index=False)
    audit = pd.DataFrame({
        "item": ["source_files", "source_sha256", "rows", "first_timestamp", "last_timestamp", "duplicate_timestamps"],
        "value": [raw.attrs["files"], raw.attrs["sha256"], len(raw), raw["timestamp"].min(), raw["timestamp"].max(), raw["timestamp"].duplicated().sum()],
    })
    gaps = raw[["timestamp"]].copy()
    gaps["gap_minutes"] = gaps["timestamp"].diff().dt.total_seconds().div(60)
    gaps = gaps[gaps["gap_minutes"] > 5].copy()
    gaps["previous_timestamp"] = gaps["timestamp"] - pd.to_timedelta(gaps["gap_minutes"], unit="m")
    gaps["likely_weekend"] = gaps["gap_minutes"] >= 24 * 60
    gaps.to_csv(output / "data_gaps.csv", index=False)
    with pd.ExcelWriter(output / "full_study.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, "Conditions", index=False)
        baseline_predictions.to_excel(writer, "Holdout predictions", index=False)
        baseline_trades.to_excel(writer, "Holdout trades", index=False)
        wfo_summary.to_excel(writer, "WFO folds", index=False)
        wfo_trades.to_excel(writer, "WFO trades", index=False)
        audit.to_excel(writer, "Data audit", index=False)
        gaps.to_excel(writer, "Data gaps", index=False)
    manifest = {"arguments": vars(args), "data_sha256": raw.attrs["sha256"], "device": str(device), "conditions": len(results)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
