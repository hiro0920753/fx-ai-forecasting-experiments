from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = ("timestamp", "time", "datetime", "date")
CLOSE_CANDIDATES = ("close", "bid_close", "ask_close", "mid_close")


@dataclass(frozen=True)
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    current: np.ndarray
    timestamps: np.ndarray


def _resolve_column(columns: list[str], explicit: str | None, candidates: tuple[str, ...]) -> str:
    lookup = {column.lower().strip(): column for column in columns}
    if explicit:
        key = explicit.lower().strip()
        if key not in lookup:
            raise ValueError(f"Column not found: {explicit}")
        return lookup[key]
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(f"Could not detect a column. Tried: {', '.join(candidates)}")


def load_prices(
    path: str | Path,
    timestamp_column: str | None = None,
    close_column: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_name = _resolve_column(list(frame.columns), timestamp_column, TIMESTAMP_CANDIDATES)
    close_name = _resolve_column(list(frame.columns), close_column, CLOSE_CANDIDATES)
    result = frame[[time_name, close_name]].rename(
        columns={time_name: "timestamp", close_name: "close"}
    )
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise", utc=True)
    result["close"] = pd.to_numeric(result["close"], errors="raise")
    result = result.sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps were found")
    if not np.isfinite(result["close"].to_numpy()).all():
        raise ValueError("Close prices contain non-finite values")
    if (result["close"] <= 0).any():
        raise ValueError("Close prices must be positive")
    return result


def make_sequences(frame: pd.DataFrame, lookback: int, horizon: int) -> SequenceData:
    if lookback < 2 or horizon < 1:
        raise ValueError("lookback must be >= 2 and horizon must be >= 1")
    close = frame["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    sample_count = len(frame) - lookback - horizon + 1
    if sample_count < 1:
        raise ValueError("Not enough rows for the requested lookback and horizon")

    x = np.empty((sample_count, lookback, 1), dtype=np.float32)
    y = np.empty(sample_count, dtype=np.float32)
    current = np.empty(sample_count, dtype=np.float64)
    timestamps = np.empty(sample_count, dtype="datetime64[ns]")
    time_values = frame["timestamp"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    for index in range(sample_count):
        end = index + lookback
        target = end + horizon - 1
        x[index, :, 0] = log_returns[index:end]
        current[index] = close[end - 1]
        y[index] = np.log(close[target] / close[end - 1])
        timestamps[index] = time_values[target]
    return SequenceData(x=x, y=y, current=current, timestamps=timestamps)


def load_sequence_dataset(path: str | Path) -> tuple[SequenceData, int, int]:
    archive = np.load(path)
    required = {"x", "y", "current", "timestamps", "lookback", "horizon"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"Dataset is missing arrays: {', '.join(sorted(missing))}")
    data = SequenceData(
        x=archive["x"].astype(np.float32),
        y=archive["y"].astype(np.float32),
        current=archive["current"].astype(np.float64),
        timestamps=archive["timestamps"].astype("datetime64[ns]"),
    )
    return data, int(archive["lookback"]), int(archive["horizon"])


def chronological_split(
    data: SequenceData,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> tuple[SequenceData, SequenceData, SequenceData]:
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1:
        raise ValueError("Split ratios must be between zero and one")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be less than one")
    n = len(data.y)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    if train_end < 1 or valid_end <= train_end or valid_end >= n:
        raise ValueError("Each chronological split must contain at least one sample")

    def select(start: int, end: int) -> SequenceData:
        return SequenceData(
            x=data.x[start:end],
            y=data.y[start:end],
            current=data.current[start:end],
            timestamps=data.timestamps[start:end],
        )

    return select(0, train_end), select(train_end, valid_end), select(valid_end, n)

